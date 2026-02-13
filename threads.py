import queue
import threading
import time
import cv2 as cv

depth_queue = queue.Queue(maxsize=1)

def camera_thread(feed_queue, depth_queue,ret, frame, stop_event):
    while not stop_event.is_set():
        if not ret:
            continue

        if feed_queue.full():
            feed_queue.get_nowait()

        feed_queue.put(frame)

def depth_thread(feed_queue, depth_queue, depth, stop_event):
    while not stop_event.is_set():
        try:
            frame = feed_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        pointDepth = cv.reprojectImageTo3D(disp2, Q)

        if depth_queue.full():
            depth_queue.get_nowait()

        depth_queue.put(frame, depth)