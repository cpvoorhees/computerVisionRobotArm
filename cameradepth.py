import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt

 
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*7,3), np.float32)
objp[:,:2] = np.mgrid[0:5,0:7].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
imagesLeft = glob.glob('C:\\Users\\chris\\OneDrive\\Pictures\\Camera Roll\\*.jpg')
imagesRight =glob.glob('C:\\Users\\chris\\OneDrive\\Desktop\\chessboardpic\\*.jpg')
imageL='C:\\Users\\chris\\OneDrive\\Pictures\\Camera Roll\\*.jpg'
imageR='C:\\Users\\chris\\OneDrive\\Desktop\\chessboardpic\\*.jpg'

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
 
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
 
    print( "total error: {}".format(mean_error/len(objpoints)) )

    return mtx, dist, dst, newcameramtx, roi

mtxL,distL,dstL,nmtxL,roiL=calibrate_camera(imagesLeft)
mtxR,distR,dstR,nmtxR,roiR=calibrate_camera(imagesRight)
imgL=cv.imread("WIN_20251010_16_02_29_Pro.jpg")
imgR=cv.imread("backpackright.jpeg")
dstL = cv.undistort(imgL, mtxL, distL, None, nmtxL)
dstR = cv.undistort(imgR, mtxR, distR, None, nmtxR)

cv.imwrite("leftundistort.png",dstL)
cv.imwrite("rightundistort.png",dstR)

undistimgL=cv.imread("leftundistort.png")
undistimgR=cv.imread("rightundistort.png")

resizedL = cv.resize(imgL, (700,700), interpolation= cv.INTER_LINEAR)
resizedR = cv.resize(imgR, (700,700), interpolation= cv.INTER_LINEAR)

resizedL=cv.imwrite("resizedLeft.png",resizedL)
resizedR=cv.imwrite("resizedRight.png",resizedR)

resizedL = cv.imread('resizedLeft.png', cv.IMREAD_GRAYSCALE)
resizedR = cv.imread('resizedRight.png', cv.IMREAD_GRAYSCALE)

stereo = cv.StereoBM.create(numDisparities=176, blockSize=5)
disparity = stereo.compute(resizedL,resizedR)
plt.imshow(disparity,'gray')
plt.show()
