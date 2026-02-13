import camera
import videos
import camera_calibration
import disparityMap
import depthMap
import os
import cv2 as cv
from pathlib import Path
import glob
import queue
import threading
import time
import matplotlib.pyplot as plt
import numpy as np

frame_queue = queue.Queue(maxsize=1)
depth_queue = queue.Queue(maxsize=1)

ans = input("Do you want to calibrate your cameras (y/n)")
if(ans == 'y'):
    #camera.camera()
    #camera.disparityCam()


    right_img = Path("images/right")
    left_img = Path("images/left")

    disparity_right = Path("images/disparityright")
    disparity_left = Path("images/disparityleft")

    right_images = list(right_img.glob("*.jpg"))  # or *.png
    left_images  = list(left_img.glob("*.jpg"))

    disp_right = list(disparity_right.glob("*.jpg"))
    disp_left = list(disparity_left.glob("*.jpg"))

    mtx1, dist1 = camera_calibration.calibrateCamera(right_images)
    mtx2, dist2 = camera_calibration.calibrateCamera(left_images)
    R, T, gray1, gray2, height, width = camera_calibration.stereocalibrate(mtx1, dist1, mtx2, dist2, right_images, left_images)
    left_mapx, left_mapy, right_mapx, right_mapy, Q = camera_calibration.stereoRectification(mtx2, dist2, mtx1, dist1, R, T, width, height, disp_right, disp_left)
    np.savez('stereo_calibration_data.npz', mtx1=mtx1, dist1=dist1,mtx2=mtx2, dist2=dist2,R=R, T=T, height = height, width = width, right_mapx = right_mapx, right_mapy = right_mapy, left_mapx = left_mapx, left_mapy = left_mapy, Q = Q)
else:
    data = np.load('stereo_calibration_data.npz')
    mtx1, dist1 = data['mtx1'], data['dist1']
    mtx2, dist2 = data['mtx2'], data['dist2']
    R, T = data['R'], data['T']
    width, height = data['width'], data['height']


cap = cv.VideoCapture(0, cv.CAP_DSHOW)  #right camera
cap2 = cv.VideoCapture(1, cv.CAP_DSHOW)  #left camera
stop_event = threading.Event()

threads = [
    threading.Thread(target=videos.feed, args=(frame_queue, stop_event)),
    threading.Thread(target=disparityMap.disparityMap, args=(frame_queue, depth_queue, stop_event)),
]

for t in threads:
    t.start()

for t in threads:
    t.join()

cap.release()
cv.destroyAllWindows()
