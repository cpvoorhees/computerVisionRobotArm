import camera
import camera_calibration
import disparityMap
import depthMap
import os
import cv2 as cv
from pathlib import Path
import glob
import time
import matplotlib.pyplot as plt
import numpy as np

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

ans = input("Do you want to calibrate your cameras (y/n)")
if(ans == 'y'):
    mtx1, dist1 = camera_calibration.calibrateCamera(right_images)
    mtx2, dist2 = camera_calibration.calibrateCamera(left_images)
    R, T, gray1, gray2, height, width = camera_calibration.stereocalibrate(mtx1, dist1, mtx2, dist2, right_images, left_images)
    np.savez('stereo_calibration_data.npz', mtx1=mtx1, dist1=dist1,mtx2=mtx2, dist2=dist2,R=R, T=T, height = height, width = width)
else:
    data = np.load('stereo_calibration_data.npz')
    mtx1, dist1 = data['mtx1'], data['dist1']
    mtx2, dist2 = data['mtx2'], data['dist2']
    R, T = data['R'], data['T']
    width, height = data['width'], data['height']




time1 = time.perf_counter()

rectified_left, rectified_right, Q = camera_calibration.stereoRectification(mtx1, dist1, mtx2, dist2, R, T, width, height, disp_right, disp_left)
    
disparity = disparityMap.disparityMap(rectified_right, rectified_left, Q)

depthMap.depthMap(disparity, Q)
depthMap.depthMapMeters(disparity,Q, mtx1, T)
time2 = time.perf_counter()

print(time2 - time1)