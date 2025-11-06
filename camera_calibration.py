import numpy as np
import cv2 as cv
import camera
import matplotlib.pyplot as plt
from ultralytics import YOLO
import random

chessboard = (6, 9)
framesize = (640,480)
worldSize = 1      #size of chessboard squares


#############################################Indivual Camera Calibration###########################################################

def calibrateCamera(imageFolder):
    #termination criteria
    #epsilon- 0.001
    #max iterations = 100
    criteria = (cv.TermCriteria_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.001)

    #object points
    #find the total number or interal corners ignore outside edge
    #creates as many points as there are internal corners and intializes them to (0, 0, 0) with a precision of float32
    objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
    #fills in the first two columns of each coordinate the x and y values
    #creates a 2D mesh grid of x values and one of y values
    #.T puts the x and y values together as (x, y) points, every two values are grouped together as tuples
    #.reshape(-1,2) combines everything into one 2D array, -1 tells python to figure # of rows autmotically and 2 defines the number of col
    objp[:,:2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)
    #scales the coordinates
    objp = worldSize * objp

    #loads the image to get coordinates
    first_img = cv.imread(str(imageFolder[0]))

    #find the height and width of the images
    height = first_img[0].shape[0]
    width = first_img[0].shape[1]

    #pixel coordinates for the chessboard
    imgpoints = []          #2D points on the image points

    #3D coordinates for the object points
    objpoints = []


    #loops through each image from the image folder
    #frame will iterate through each image in the folder
    for img in imageFolder:

        #reads the window path object to a string
        frame = cv.imread(str(img))
        #creates a greyscale of the image
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #Finds the corners of the chess board based on the dimension that we give
        #Returns a bool that describes whether or not the image was properly processed(ret)
        #returns a matrix containing the x,y values of where the corners are(left to right)
        #We pass in the grayscale image from above, pass the size of the chessboard, and no flags needed yet
        c_ret1, corner1 = cv.findChessboardCorners(gray, chessboard, None)

        if c_ret1 == True:
            #takes the general corners that we found and refines the accuracy of the coordinates to sub Pixel level
            #Takes the gray scale image, and the corner coodinates we found
            #takes the window size or area around each corner in pixels (11,11) decreased the numbers for lower res, for blurry images increase for sharp images can decrease
            # takes the zero zone or the area in which we don't computer the drivaive or anything (-1,-1) mean we have no zero zone(can changes if we have distorted corners higher number higher disotortion)
            # Stopping criteria taken from the top of the code max_iter= 30, max_eps=0.001 
            corners = cv.cornerSubPix(gray, corner1, (11, 11), (-1, -1), criteria)       #optimized corners

            #draws the corners of the chess board onto frame1
            #takes the grayscale image of the chessboard, we define the size of the chess board (rows, columns)
            #put in your array of corners you found use the one you optimized with cornerSubPix
            #the last avlue is a boolean that lets the function know if the corners were found sucessfully or not
            #finally imshow displays the image that we just drew the corners on  (window name, image to be displayed)
            #waitkey is used to keep the image window open to look at
            cv.drawChessboardCorners(frame, chessboard, corners, c_ret1)
            cv.imshow('Image', frame)


            #We are now appending the 3D structure of our chessboard objp it's constant across all images sincou our chessboard doesn't change
            #We append this with each set of corner points so open CV can match the 3D grid to the 2D corner points
            #objp is a fixed grid of the physical locations of the chess board corners with z=0 because the board is flat
            objpoints.append(objp)

            #we append all of the corner points we found for each image to the two arrays we initialized
            imgpoints.append(corners)
    
    if len(objpoints) == 0 or len(imgpoints) == 0:
        raise RuntimeError("No chessboard corners were found in any image.")

    
    #Takes the 3D object points, the @D image point, the dimensions of the chess board, initial guess for camera matrix and distortion
    #returns projection error (how close the objpoints are to the image), intrinsinc matrix, distortion coefficents, rotational vectors, translational vectors
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)


 
    return mtx, dist




##############################################Stereo Camera Calibration####################################################

#start of stereo calibaration function
def stereocalibrate(mtx1, dist1, mtx2, dist2, frame1, frame2):
    #termination criteria
    #epsilon- 0.001
    #max iterations = 100
    criteria = (cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER, 100, 0.001)

    #object points
    #finds the total number of internal cornere(9*6 = 54)
    #creates that many points with 3 axis all intialized to 0 (54 points that ar (0, 0, 0))
    objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)  
    #fills in the first two columns of each coordinate the x and y values
    #creates a 2D mesh grid of x values and one of y values
    #.T puts the x and y values together as (x, y) points, every two values are grouped together as tuples
    #.reshape(-1,2) combines everything into one 2D array, -1 tells python to figure # of rows autmotically and 2 defines the number of col
    objp[:,:2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)
    #scales the coordinates
    objp = worldSize * objp

    #first image left
    firstImgR = cv.imread(str(frame1[0]))
    firstImgL = cv.imread(str(frame2[0]))
    #first image right
    #find the height and width of the images
    height, width = firstImgR.shape[:2]


    #pixel coordinates for the chessboard
    imgpoints_l = []          #2D points on the image points
    imgpoints_r = []

    #3D coordinates for the object points
    objpoints = []

    #loops through each image taken from left and right camera
    #frame1 is the iterator that will loop through all of the images from the right camera
    #frame2 is the iterator will loop through all of the images from the left camera
    for imgR, imgL in zip(frame1, frame2):

        frameR = cv.imread(str(imgR))
        frameL = cv.imread(str(imgL))
        #creates a greyscale of both images (converts all the colors in the image to gray)
        gray1 = cv.cvtColor(frameR, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frameL, cv.COLOR_BGR2GRAY)

        #Finds the corners of the chess board based on the dimension that we give
        #Returns a bool that describes whether or not the image was properly processed(ret)
        #returns a matrix containing the x,y values of where the corners are(left to right)
        #We pass in the grayscale image from above, pass the size of the chessboard, and no flags needed yet
        c_ret1, corner1 = cv.findChessboardCorners(gray1, chessboard, None)
        c_ret2, corner2 = cv.findChessboardCorners(gray2, chessboard, None)


        #checks if the chessboard was properly processed then runs the commands
        if c_ret1 == True & c_ret2 == True:
            #takes the general corners that we found and refines the accuracy of the coordinates to sub Pixel level
            #Takes the gray scale image, and the corner coodinates we found
            #takes the window size or area around each corner in pixels (11,11) decreased the numbers for lower res, for blurry images increase for sharp images can decrease
            # takes the zero zone or the area in which we don't computer the drivaive or anything (-1,-1) mean we have no zero zone(can changes if we have distorted corners higher number higher disotortion)
            # Stopping criteria taken from the top of the code max_iter= 30, max_eps=0.001 
            corners1 = cv.cornerSubPix(gray1, corner1, (11, 11), (-1, -1), criteria)       #optimized corners
            corners2 = cv.cornerSubPix(gray2, corner2, (11,11), (-1, -1), criteria)

            #draws the corners of the chess board onto frame1
            #takes the grayscale image of the chessboard, we define the size of the chess board (rows, columns)
            #put in your array of corners you found use the one you optimized with cornerSubPix
            #the last avlue is a boolean that lets the function know if the corners were found sucessfully or not
            #finally imshow displays the image that we just drew the corners on  (window name, image to be displayed)
            #waitkey is used to keep the image window open to look at
            cv.drawChessboardCorners(frameR, (5,8), corners1, c_ret1)
            cv.imshow('Right', frameR)

            cv.drawChessboardCorners(frameL, (5,8), corners2, c_ret2)
            cv.imshow('Left', frameL)
            cv.waitKey(500)

            #We are now appending the 3D structure of our chessboard objp it's constant across all images sincou our chessboard doesn't change
            #We append this with each set of corner points so open CV can match the 3D grid to the 2D corner points
            #objp is a fixed grid of the physical locations of the chess board corners with z=0 because the board is flat
            objpoints.append(objp)

            #we append all of the corner points we found for each image to the two arrays we initialized
            imgpoints_r.append(corners1)
            imgpoints_l.append(corners2)



    #This locks our stereo calibration from calibarting the intrinsic camera matrix or the distortion coefficents
    #We want to optimze our intrinsic matrix for our cameras indivually and only use stereo to optimize the extrinsic (focus on R and T)
    #This allows us to get more accurate optimization
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC

    #takes in object points (3D coordinates), left and right image points (left and right), the two intrinsic matrixesm the two distortions, the dimensions of the image,  the termination criteria, and the flags for the calibration
    #returns ret a bool if the 
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_l, imgpoints_r, mtx1, dist1, mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
    return R, T, gray1, gray2, height, width

#takes the intrinsic camera matrix of the first camera, the distortion of the first camera the same for the second
#takes the rotation 
def stereoRectification(mtx1, dist1, mtx2, dist2, R, T, width, height, gray1, gray2):
    #takes the first image in both windows path objects and converts them to string so the code can read it
    firstImg = cv.imread(str(gray1[0]))
    secImg = cv.imread(str(gray2[0]))
    #find the width and height of the image
    width, height = firstImg.shape[:2]

    #Takes the intrinsic matrix of 1 and 2, takes the distortion of 1 and 2, takes the width and height of the image, takes the rotations and translation matrix, and the flag
    # reutrns rectification transforms (what was done to correct the image)
    #the projection matrix for each camera in our new coordinate system
    #disparity to depth matrxi
    #regions of interest
    R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(mtx1, dist1, mtx2, dist2, (width, height), R, T, flags=cv.CALIB_ZERO_DISPARITY, alpha=0)

    #takes the intrinsic matrix of one camera, the rectification transformations, the prjection hatrix, image dimesions and defines the data type
    #then returns the undistored returns the destiantions of the x and y pixels of each image
    right_mapx, right_mapy = cv.initUndistortRectifyMap(mtx1, dist1, R1, P1, (width, height), cv.CV_16SC2)
    left_mapx, left_mapy = cv.initUndistortRectifyMap(mtx2, dist2,R2, P2, (width, height), cv.CV_16SC2)

    #appies the linear transformations we just found to the image to undistort and rectify the image
    rectified_right = cv.remap(firstImg, right_mapx, right_mapy, cv.INTER_LINEAR)
    rectified_left  = cv.remap(secImg,  left_mapx, left_mapy, cv.INTER_LINEAR)

    #draws lines over the image to see if everything is properly rectified
    for y in range(0, rectified_left.shape[0], 40):
        cv.line(rectified_left, (0, y), (rectified_left.shape[1], y), (255, 0, 0), 1)
        cv.line(rectified_right, (0, y), (rectified_right.shape[1], y), (255, 0, 0), 1)

    #comvines the lines drawn into one image that's side by side
    combined = np.hstack((rectified_left, rectified_right))
    #displays the compined image created
    cv.imshow("Rectified stero pair", combined)

    return rectified_left, rectified_right, Q





def depthMap(disparity, Q):

    disparity[disparity < 0] = np.nan
    print("Disparity min:", np.min(disparity))
    print("Disparity max:", np.max(disparity))
    pointDepth = cv.reprojectImageTo3D(disparity, Q)

    #for all rows and columns take on the depth or Z value which is at index 2
    depth = pointDepth[:,:,2]

    #masks the coordinates keeping only the valid depth values
    #for every depth which is zero or less which means the the object was too far away or there was no match
    depth_vis = np.nan_to_num(depth, nan=0, posinf=0, neginf=0)

    #normalized the depth value 
    #input: depth values, no existing array we want to write to, target min/mac range from 0 ro 255, and linear scaling
    depth_vis = cv.normalize(depth_vis, None, 0, 255, cv.NORM_MINMAX)
    #converts the float point values to unsigned 8-bit integers which in standar for open-cv display
    depth_vis = np.uint8(depth_vis)


    #opens a depth map showing normalized depth image
    cv.imshow("Depth Map", depth_vis)
    #pauses program until a key is pressed
    cv.waitKey(-1)

    depth_colored = cv.applyColorMap(depth_vis, cv.COLORMAP_PLASMA)
    cv.imshow("Colored Depth Map", depth_colored)
    cv.waitKey(-1)

    return depth

#finds the disparity map in meters 
#input: disparity map, Q projection matrix, right matrix, translation vectors
#output: returns depth map in meters
def depthMapMeters(disparity, Q, mtx1, T):
    f = mtx1[0,0]        
    B = np.linalg.norm(T)         
    disparity = disparity.astype(np.float32)
    disparity[disparity <= 0] = np.nan

    # Depth in meters
    depth_m = (f * B) / disparity

    # normalize for visualization
    depth_vis = cv.normalize(depth_m, None, 0, 255, cv.NORM_MINMAX)
    depth_vis = np.uint8(depth_vis)
    cv.imshow("Depth Map (Meters)", depth_vis)
    cv.waitKey(0)

    return depth_m


def YOLOintegration():
    def getColor(cls_num):
        random.seed(cls_num)
        return tuple(random.randint(0, 255) for _ in range(3))
    yolo = YOLO()