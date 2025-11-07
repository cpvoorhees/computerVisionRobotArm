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
objp = objp * 2
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


 
imagesLeft = glob.glob('C:\\Users\\chris\\OpenCV Practice\\openCVcode\\images\\left\\*.jpg')
imagesRight =glob.glob('C:\\Users\\chris\\OpenCV Practice\\openCVcode\\images\\right\\*.jpg')
imageL='C:\\Users\\chris\\OpenCV Practice\\openCVcode\\images\\left\\*.jpg'
imageR='C:\\Users\\chris\\OpenCV Practice\\openCVcode\\images\\right\\*.jpg'

#imagesLeft = glob.glob('C:\\Users\\chris\\OneDrive\\Pictures\\Camera Roll\\*.jpg')
#imagesRight =glob.glob('C:\\Users\\chris\\OneDrive\\Desktop\\chessboardpic\\*.jpg')
#imageL='C:\\Users\\chris\\OneDrive\\Pictures\\Camera Roll\\*.jpg'
#imageR='C:\\Users\\chris\\OneDrive\\Desktop\\chessboardpic\\*.jpg'

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
 
    # crop the image
    #x, y, w, h = roi
    #dst = dst[y:y+h, x:x+w]
    
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
 
    print( "total error: {}".format(mean_error/len(objpoints)) )

    return mtx, dist, newcameramtx, roi, rvecs, tvecs, imgpoints

mtxL,distL,nmtxL,roiL, rL,tL, imgpointsL=calibrate_camera(imagesLeft)
mtxR,distR,nmtxR,roiR,rR,tR,imgpointsR=calibrate_camera(imagesRight)

imgL=cv.imread("C:\\Users\\chris\\OpenCV Practice\\openCVcode\\images\\disparityleft\\imgLeftDisparity.jpg", cv.IMREAD_GRAYSCALE)
imgR=cv.imread("C:\\Users\\chris\\OpenCV Practice\\openCVcode\\images\\disparityright\\imgRightDisparity.jpg",cv.IMREAD_GRAYSCALE)

width=imgL.shape[0]
height=imgL.shape[1]

flags = 0
flags = cv.CALIB_FIX_INTRINSIC
criteria_stereo = (cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER,30,0.001)
ret, mtxL,distL,mtxR,distR,R,T,E,F=cv.stereoCalibrate(objpoints,imgpointsL,imgpointsR,nmtxL,distL,nmtxR,distR,(width,height),criteria,flags = cv.CALIB_FIX_INTRINSIC)
rectifyScale = 1
rectL, rectR, projML, projMR, Q, roi_L, roi_R = cv.stereoRectify(mtxL, distL,mtxR, distR,(width,height), R, T,rectifyScale,(0,0))
#grayL = cv.cvtColor(imgL)
#grayR = cv.cvtColor(imgR)
stereoMapLx, stereoMapLy=cv.initUndistortRectifyMap(mtxL,distL,rectL,projML,imgL.shape[::1],cv.CV_16SC2)
stereoMapRx, stereoMapRy=cv.initUndistortRectifyMap(mtxR,distR,rectR,projMR,imgR.shape[::1],cv.CV_16SC2)


def downsample(image, reduce_factor):
    for i in range(0, reduce_factor):
        if len(image.shape) > 2:
            row,col = image.shape[:2]
        else:
            row,col = image.shape
        image = cv.pyrDown(image,dstsize=(col//2,row//2))

#downsample(imgL,3)
#downsample(imgR,3)
#imgL = cv.remap(imgL,stereoMapLx,stereoMapLy,cv.INTER_LANCZOS4,cv.BORDER_CONSTANT, 0)
#imgR = cv.remap(imgR,stereoMapRx,stereoMapRy,cv.INTER_LANCZOS4,cv.BORDER_CONSTANT, 0)

print(Q)
imgL = cv.undistort(imgL, mtxL, distL, None, None)
imgR = cv.undistort(imgR, mtxR, distR, None, None)


blocksize = 3
mindisp=-1
maxdisp=31
disp=maxdisp-mindisp



stereo = cv.StereoBM(numDisparities=16,blockSize=5)
disparity = stereo.compute(imgL,imgR)

plt.imshow(disparity,'gray')
plt.show()

h,w =imgR.shape[:2]

disparity_map = np.float32(np.divide(disparity,16.0))

img3d = cv.reprojectImageTo3D(disparity_map,Q,handleMissingValues=False)
colors = cv.cvtColor(imgR,cv.COLOR_BGR2RGB)

mask = disparity_map > disparity_map.min()
maskedp = img3d[mask]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x = maskedp[:,0]
y = maskedp[:,1]
z = maskedp[:,2]

ax.scatter(x,y,z)
plt.show()