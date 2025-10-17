import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D as a3d

imageL = []
imageR = []
 
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*7,3), np.float32)
objp[:,:2] = np.mgrid[0:5,0:7].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
imagesLeft = glob.glob('C:\\Users\\chris\\OpenCV Practice\\images\\left\\*.jpg')
imagesRight =glob.glob('C:\\Users\\chris\\OpenCV Practice\\images\\right\\*.jpg')
imageL='C:\\Users\\chris\\OpenCV Practice\\images\\left\\*.jpg'
imageR='C:\\Users\\chris\\OpenCV Practice\\images\\right\\*.jpg'

def calibrate_camera(image_folder):
    for fname in image_folder:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (5,7), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
 
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
 
        # Draw and display the corners
            cv.drawChessboardCorners(img, (5,7), corners2, ret)
            #cv.imshow('img', img)
            #cv.waitKey(100)
    cv.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    return mtx, dist, dst, newcameramtx, roi, rvecs, tvecs

ret, mtxL,distL,mtxR,distR,R,T,E,F=cv.stereoCalibrate(objpoints,imgpoints,mtxL,distL,mtxR,distR,(700,700),criteria)