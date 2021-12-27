import argparse
import cv2
import dlib
import imutils
import time

import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream

from mylib import config, thread
from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject

t0 = time.time()


def run():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=False,
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
                    help="path to Caffe pre-trained model")
    ap.add_argument("-i", "--input", type=str,
                    help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str,
                    help="path to optional output video file")
    # confidence default 0.4
    ap.add_argument("-c", "--confidence", type=float, default=0.4,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=30,
                    help="# of skip frames between detections")
    args = vars(ap.parse_args())

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    if not args.get("input", False):
        print("[INFO] Starting the live stream..")
        vs = VideoStream(config.url).start()
        time.sleep(2.0)

    else:
        print("[INFO] Starting the video..")
        vs = cv2.VideoCapture(args["input"])

    writer = None

    W = None
    H = None

    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    totalFrames = 0
    totalRight = 0
    totalLeft = 0
    x = []
    empty = []
    empty1 = []

    fps = FPS().start()

    if config.Thread:
        vs = thread.ThreadingClass(config.url)

    while True:
        frame = vs.read()
        frame = frame[1] if args.get("input", False) else frame

        if args["input"] is not None and frame is None:
            break

        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

        # rectangles returned by either
        # (1) our object detector or
        # (2) the correlation trackers
        status = "Waiting"
        rects = []

        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if totalFrames % args["skip_frames"] == 0:
            # set the status and initialize our new set of object trackers
            status = "Detecting"
            trackers = []

            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > args["confidence"]:
                    idx = int(detections[0, 0, i, 1])

                    if CLASSES[idx] != "person":
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)
                    trackers.append(tracker)

        else:
            # loop over the trackers
            for tracker in trackers:
                status = "Running"

                tracker.update(rgb)
                pos = tracker.get_position()

                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                rects.append((startX, startY, endX, endY))

        cv2.line(frame, (W // 4, 0), (W // 4, H), (0, 0, 0), 3)
        cv2.putText(frame, "AQ Demo", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 3)

        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID, None)

            # First detection frame
            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                y = [c[0] for c in to.centroids]
                # [1, 2] - mean ==> [-0.5, 0.5]
                direction = centroid[0] - np.mean(y)
                to.centroids.append(centroid)

                if not to.counted:
                    if direction < 0 and centroid[0] < (W // 4):
                        totalLeft += 1
                        empty.append(totalLeft)
                        to.counted = True

                    elif direction > 0 and centroid[0] > (W // 4):
                        totalRight += 1
                        empty1.append(totalRight)
                        # First initialise to be not counted, then counted
                        to.counted = True

                    x = totalLeft + totalRight
                print("=== Total people inside: === ", x)

            trackableObjects[objectID] = to

            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
            cv2.rectangle()

        totalPeople = totalLeft + totalRight
        info = [
            ("Total People", totalPeople),
            ("In the region", totalRight),
            ("Out the region", totalLeft),
            ("Tracking", status),
        ]

        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if writer is not None:
            writer.write(frame)

        cv2.imshow("Tracking Screen", frame)
        key = cv2.waitKey(1) & 0xFF

        totalFrames += 1
        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
