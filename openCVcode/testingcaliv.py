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
chesslength = 2 #2 cm
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#creates points like (1,0,0), (2,1,0),...
objp = np.zeros((chessboardSize[0]*chessboardSize[1],3),np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[1],0:chessboardSize[0]].T.reshape(-1,2)
objp = objp * chesslength
#array of points in 3d real world
objpoints = []
#array of 2d points on image
imgpointsL = []
imgpointsR = []

def calibrateCamera(image_path,image_pathR):

    for fnameL in glob.glob(image_path):
        #gets an image
        img = cv.imread(fnameL)
        #makes image gray
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #find corners on chessboard
        ret, cornersL = cv.findChessboardCorners(gray,chessboardSize, None)
    for fnameR in glob.glob(image_pathR):
        #gets an image
        img = cv.imread(fnameR)
        #makes image gray
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #find corners on chessboard
        retR, cornersR = cv.findChessboardCorners(gray,chessboardSize, None)
    #if corners can be found
    #if corners can be found
    if ret and retR:
        objpoints.append(objp)
        corners2L = cv.cornerSubPix(gray,cornersL,(11,11),(-1,-1),criteria)
        imgpointsL.append(corners2L)
        corners2R = cv.cornerSubPix(gray,cornersR,(11,11),(-1,-1),criteria)
        imgpointsR.append(corners2R)

    ret, mtxL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, gray.shape[::-1], None, None)
    ret, mtxR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, gray.shape[::-1], None, None)

    
    img = cv.imread(fnameL)
    h,w = img.shape[:2]
    nmtxL, roi = cv.getOptimalNewCameraMatrix(mtxL,distL, (w,h), 1, (w, h))
    img = cv.imread(fnameR)
    nmtxR, roi = cv.getOptimalNewCameraMatrix(mtxR,distR, (w,h), 1, (w, h))
    return mtxL, nmtxL, distL,mtxR,nmtxR,distR,imgpointsL, imgpointsR, w, h

#get carmera matrices
mtxL,nmtxL,distL,mtxR,nmtxR,distR,imgpL,imgpR,w,h = calibrateCamera(leftcalibratepath, rightcalibratepath)


print('Left image points:\n')
print(imgpL)
print('Right image points:\n')
print(imgpR)
print('objectpoints:\n')
print(objpoints)
ret, mtxL,distL,mtxR,distR,R,T,E,F = cv.stereoCalibrate(objpoints,imgpL,imgpR,nmtxL,distL,nmtxR,distR,(w,h))
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(mtxL, distL,mtxR, distR,(w,h), R, T,flags=cv.CALIB_ZERO_DISPARITY,alpha=0)
mapLx,mapLy = cv.initUndistortRectifyMap(mtxL,distL,R1,P1,(w,h),5)
mapRx,mapRy = cv.initUndistortRectifyMap(mtxR,distR,R1,P1,(w,h),5)


imgL=cv.imread(leftdisparitypath)
grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)

dstL=cv.remap(grayL,mapLx,mapLy,cv.INTER_LINEAR)
imgR=cv.imread(rightdisparitypath)
grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
cv.imshow('img',grayR)
dstR=cv.remap(grayR,mapRx,mapRy,cv.INTER_LINEAR)
#dstL = cv.cvtColor(dstL,cv.COLOR_BGR2GRAY)
#dstR = cv.cvtColor(dstR,cv.COLOR_BGR2GRAY)

stereo = cv.StereoBM.create(numDisparities=16,blockSize=5)
disparity=stereo.compute(dstL,dstR)
plt.imshow(disparity,'gray')
plt.show()

