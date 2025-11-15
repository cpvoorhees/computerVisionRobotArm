import camera
import camera_calibration
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def leftRightConsistency(disp_left, disp_right, threshold = 3.0):
    #find the height and width of the image
    h, w = disp_left.shape
    #creates a variable mask that is the same as disp_left but filled with 1
    #set everything inside to 8 bit integers
    mask = np.ones_like(disp_left, dtype=np.uint8)

    #this array will loop over all pixels in the left disparity map checking them all
    for y in range(h):
        for x in range(w):
            #d is the disparity value at pixel (x,y) in the left disparity map
            #shows how much the left pixel shifts to match the right pixel
            d = disp_left[y, x]

            #checks if corresponding picture in the right image exists
            #if <0 it would be out of the image so we would skip it
            if x - int(d) >= 0:
                #corresponding pixel in the right-to-left disparity map
                #the dsiprity map the takes right to left
                x_r = x - int(d)
                x_r = np.clip(x_r, 0, w-1)
                d_r = disp_right[y, x_r]

                #compares the left and right disparity
                #if the difference is greater than the threshold than mark it as invalid
                if abs(d -d_r) > threshold:
                    mask[y,x] = 0
            #if the pixel is out of bounds we also mark it as invalid
            else:
                mask[y,x] = 0





def disparityMap(rectified_right, rectified_left, Q):
    #takes the two images that have been mapped and converts them to grayscale
    rectified_right = cv.cvtColor(rectified_right, cv.COLOR_BGR2GRAY)
    rectified_left = cv.cvtColor(rectified_left, cv.COLOR_BGR2GRAY)

    #converst the datatype to uint8 because that's the only type openCV will use to display images
    rectified_left  = rectified_left.astype(np.uint8)
    rectified_right = rectified_right.astype(np.uint8)

    #prints the shape of the image in pixels to verify they are the same
    #used as a way to check both images come from similar quality cameraswhich helps for disparity maps
    print(rectified_left.shape, rectified_right.shape)

    #the intial parameters for the disparity map these are what we will change using the sliders
    #these are the intial values just to create our intialized disparity map
    numDisparities = 5
    blockSize = 5
    minDisparity = 0
    speckleWindowSize = 5
    speckleRange = 10
    disp12MaxDiff = 0
    P1 = 8 * 3 * 5 * 5
    P2 = 32 * 3 * 5 * 5
    preFilterCap = 0
    stereo = None

    #define the window name we want for the trackbars
    window_name = 'SGBM Parameters'


    #empty call back function
    def nothing(x):
        pass

    #the function that creates our disparity map with the inital parametrs
    stereo = cv.StereoSGBM_create(
        #shifts the disparity range If your images are rectified such that the corresponding points are always to the left in the right image
        minDisparity=minDisparity,

        #controls the depth resolution a higher number allows for a wider range  of depth values
        #gives more detail but takes more time to compute
        #must be divisible by 16
        numDisparities=numDisparities *16,

        #larger blocksize leads to a smoother disparity map that's more robust to noise, blurring boundaries and small deatils
        #smaller values preserves small details but are more sensitive to noise 
        #must be an odd number
        blockSize=blockSize,

        #penalty factors for disparity change bewteen neigborhood pixels control the smoothness of the map
        #Higher values encourage smoother transitions in disparity, potentially reducing noise but also potentially over-smoothing edges and details.
        P1=P1,
        P2=P2,

        #Defines the maximum allowed difference between the disparity calculated from left-to-right matching and right-to-left matching for a pixel to be considered valid.
        #Helps filter out inconsistent matches, improving the reliability of the disparity map, especially at object boundaries.
        disp12MaxDiff=disp12MaxDiff,

        #These parameters are used for speckle filtering, a post-processing step to remove small, isolated regions of disparity 
        #speckleWindowSize defines the window size for the filter
        #speckleRange defines the maximum disparity variation within that window to be considered speckle
        #help remove noise and artifacts that appear near object boundaries
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,

        #Truncates the input image pixel values after pre-filtering to a specific range
        #Can help normalize image intensity and reduce the impact of extreme pixel values on the matching process.
        preFilterCap = preFilterCap,

        #A margin by which the best matching cost should exceed the second-best matching cost for a disparity to be considered unique and valid.
        #Filters out ambiguous matches, leading to a more confident and cleaner disparity map, reducing noise and false positives.
        uniquenessRatio = 10,

        #Specifies the SGBM mode, offering different trade-offs between speed and accuracy.
        mode = cv.StereoSGBM_MODE_SGBM
    )
    


    # We're creating two new windows the first to display the trackbars, the second to display the disparity map
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.namedWindow('Disparity Map', cv.WINDOW_NORMAL)

    # Create trackbars for each SGBM parameter. The maximum values are illustrative.
    # The ranges need to be adjusted based on the input images and SGBM documentation.
    cv.createTrackbar('numDisparities (x16)', window_name, numDisparities, 16, nothing) # Max value 16 for numDisparities * 16 range
    cv.createTrackbar('blockSize (odd, >=5)', window_name, blockSize, 21, nothing)
    cv.createTrackbar('minDisparity', window_name, minDisparity, 10, nothing)
    cv.createTrackbar('speckleWindowSize', window_name, speckleWindowSize, 100, nothing)
    cv.createTrackbar('speckleRange', window_name, speckleRange, 100, nothing)
    cv.createTrackbar('disp12MaxDiff', window_name, disp12MaxDiff, 25, nothing)
    cv.createTrackbar('P1 (e.g., 8*c*bs*bs)', window_name, 8 * 3 * 5 * 5, 500, nothing) # Example range for P1
    cv.createTrackbar('P2 (e.g., 32*c*bs*bs)', window_name, 32 * 3 * 5 * 5, 1000, nothing) # Example range for P2
    cv.createTrackbar('prefiltercap: ', window_name, 0, 63, nothing)
    cv.createTrackbar('UniquenessRatio: ', window_name, 0, 30, nothing)
   
    # Main loop to get trackbar positions and update SGBM
    while True:
        # Get current positions of all trackbars
        numDisparities = cv.getTrackbarPos('numDisparities (x16)', window_name)
        bs = cv.getTrackbarPos('blockSize (odd, >=5)', window_name)
        md = cv.getTrackbarPos('minDisparity', window_name)
        sw = cv.getTrackbarPos('speckleWindowSize', window_name)
        sr = cv.getTrackbarPos('speckleRange', window_name)
        d12 = cv.getTrackbarPos('disp12MaxDiff', window_name)
        P1 = cv.getTrackbarPos('P1 (e.g., 8*c*bs*bs)', window_name)
        P2 = cv.getTrackbarPos('P2 (e.g., 32*c*bs*bs)', window_name)
        un = cv.getTrackbarPos('UniquenessRatio: ', window_name)
        pf = cv.getTrackbarPos("prefiltercap: ", window_name)
       
        # Update the SGBM object with the new parameters
        stereo = cv.StereoSGBM_create(
            minDisparity      = md,
            numDisparities    = numDisparities * 16,
            blockSize         = bs,
            P1                = P1,
            P2                = P2,
            disp12MaxDiff     = d12,
            uniquenessRatio   = un,
            speckleWindowSize = sw,
            speckleRange      = sr,
            preFilterCap      = pf,
            mode              = cv.StereoSGBM_MODE_SGBM_3WAY
        )
  
        
        right_matcher = cv.ximgproc.createRightMatcher(stereo)

        disp_left  = stereo.compute(rectified_left, rectified_right).astype(np.float32) / 16.0
        disp_right = right_matcher.compute(rectified_right, rectified_left).astype(np.float32) / 16.0

        # mask = leftRightConsistency(disp_left, disp_right)
        # disp_left_filterd = disp_left.copy()
        # disp_left_filterd[mask == 0] = 0

        # --- WLS Filtering ---
        wls = cv.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
        wls.setLambda(8000)
        wls.setSigmaColor(1.5)

        filtered = wls.filter(disp_left, rectified_left, disparity_map_right=disp_right)

        #create a confidence map to show which pixels are good
        confidence_map = wls.getConfidenceMap()
        threshold = 150  # or adjust dynamically
        filtered_masked = filtered.copy()
        filtered_masked[confidence_map < threshold] = 0

        # Normalize for display
        disp_vis = cv.normalize(disp_left, None, 0, 255, cv.NORM_MINMAX)
        disp_vis = np.uint8(disp_vis)

        filtered_vis = cv.normalize(filtered, None, 0, 255, cv.NORM_MINMAX)
        filtered_vis = np.uint8(filtered_vis)

        conf_vis = cv.normalize(confidence_map, None, 0, 255, cv.NORM_MINMAX)
        conf_vis = np.uint8(conf_vis)

        cv.imshow("Confidence Map", conf_vis)
        cv.imshow("Disparity Map", disp_vis)
        cv.imshow("Filtered Disparity (WLS)", filtered_vis)

        # Exit loop if 'Esc' key is pressed
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break

    #plots the disparity map so we can see teh different depths in pizels
    plt.imshow(filtered_vis, 'gray')
    plt.show()
    return filtered

