import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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
    #depth_vis = cv.normalize(depth_vis, None, 0, 255, cv.NORM_MINMAX)
    #converts the float point values to unsigned 8-bit integers which in standar for open-cv display
    depth_vis = np.uint8(depth_vis)


    #opens a depth map showing normalized depth image
    cv.imshow("Depth Map", depth_vis)
    #pauses program until a key is pressed
    cv.waitKey(-1)

    depth_colored = cv.applyColorMap(depth_vis, cv.COLORMAP_PLASMA)
    cv.imshow("Colored Depth Map", depth_colored)

    plt.imshow(depth_vis, 'gray')
    plt.show()

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

    cv.imshow("Depth Map (Meters)", depth_m)

    plt.imshow(depth_vis)
    plt.show()

    cv.waitKey(0)

    return depth_m
