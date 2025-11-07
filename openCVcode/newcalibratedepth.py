import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D as a3d

#calibration image paths
leftcalibratepath='C:\\Users\\chris\\OpenCV Practice\\openCVcode\\images\\left\\*.jpg'
rightcalibratepath='C:\\Users\\chris\\OpenCV Practice\\openCVcode\\images\\right\\*.jpg'
leftdisparitypath='C:\\Users\\chris\\OpenCV Practice\\openCVcode\\images\\disparityleft\\imgLeftDisparity.jpg'
rightdisparitypath='C:\\Users\\chris\\OpenCV Practice\\openCVcode\\images\\disparityright\\imgRightDisparity.jpg'

#size of inside corners of chessboard
chessboardSize = [5,7]

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#creates points like (1,0,0), (2,1,0),...
objp = np.zeros((chessboardSize[0]*chessboardSize[1],3),np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[1],0:chessboardSize[0]].T.reshape(-1,2)

#array of points in 3d real world
objpoints = []
#array of 2d points on image
imgpointsL = []
imgpointsR = []

def calibrateCameraL(image_path):

    for fname in glob.glob(image_path):
        #gets an image
        img = cv.imread(fname)
        #makes image gray
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #find corners on chessboard
        ret, corners = cv.findChessboardCorners(gray,chessboardSize, None)
    #if corners can be found
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpointsL.append(corners2)
            print('true')
        else:
            print('false')

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpointsL, gray.shape[::-1], None, None)
    
    img = cv.imread(fname)
    h,w = img.shape[:2]
    print(h)
    print(w)
    nmtx, roi = cv.getOptimalNewCameraMatrix(mtx,dist, (w,h), 1, (w, h))

    return mtx, nmtx, dist, rvecs, tvecs, h, w ,imgpointsL

def calibrateCameraR(image_path):

    for fname in glob.glob(image_path):
        #gets an image
        img = cv.imread(fname)
        #makes image gray
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #find corners on chessboard
        ret, corners = cv.findChessboardCorners(gray,chessboardSize, None)
    #if corners can be found
        if ret:
            corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpointsR.append(corners2)
            print('true')

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpointsR, gray.shape[::-1], None, None)
    
    img = cv.imread(fname)
    h,w = img.shape[:2]
    nmtx, roi = cv.getOptimalNewCameraMatrix(mtx,dist, (w,h), 1, (w, h))

    return mtx, nmtx, dist, rvecs, tvecs, h, w ,imgpointsR
#get carmera matrices
mtxL,nmtxL,distL,rvecsL,tvecsL, h, w,imgpL = calibrateCameraL(leftcalibratepath)
mtxR,nmtxR,distR,rvecsR,tvecsR, h, w,imgpR = calibrateCameraR(rightcalibratepath)


ret, nmtxL,distL,nmtxR,distR,R,T,E,F = cv.stereoCalibrate(objpoints,imgpL,imgpR,nmtxL,distL,nmtxR,distR,(w,h))
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(mtxL, distL,mtxR, distR,(w,h), R, T,flags=cv.CALIB_ZERO_DISPARITY,alpha=0)
mapLx,mapLy = cv.initUndistortRectifyMap(mtxL,distL,R1,P1,(w,h),5)
mapRx,mapRy = cv.initUndistortRectifyMap(mtxR,distR,R1,P1,(w,h),5)

print('mapxl')
print(mapLx)
print('mapxr')
print(mapRx)
imgL=cv.imread(leftdisparitypath)
imgR=cv.imread(rightdisparitypath)


CAMERA_WIDTH = 640
CROP_WIDTH = 480
def cropHorizontal(image):
    return image[:,
            int((CAMERA_WIDTH-CROP_WIDTH)/2):
            int(CROP_WIDTH+(CAMERA_WIDTH-CROP_WIDTH)/2)]
imgL = cropHorizontal(imgL)
imgR = cropHorizontal(imgR)
cv.imshow('imgL',imgL)
cv.imshow('imgR',imgR)
cv.waitKey(0)
dstL=cv.remap(imgL,mapLx,mapLy,cv.INTER_LINEAR)
#dstL = cv.undistort(imgL, mtxL,distL, None, nmtxL)
dstR=cv.remap(imgR,mapRx,mapRy,cv.INTER_LINEAR)
#dstR=cv.undistort(imgR,mtxR,distR,None,nmtxR)
dstL = cv.cvtColor(dstL,cv.COLOR_BGR2GRAY)
dstR = cv.cvtColor(dstR,cv.COLOR_BGR2GRAY)

cv.imshow('img',dstL)
cv.imshow('imgr',dstR)
cv.waitKey(0)


cv.imshow('img',dstL)
cv.imshow('imgr',dstR)
cv.waitKey(0)
stereo = cv.StereoSGBM.create(minDisparity=-1, numDisparities=16, blockSize=5
                              ,speckleWindowSize=100)
disparity=stereo.compute(dstL,dstR)
plt.imshow(disparity,'gray')
plt.show()