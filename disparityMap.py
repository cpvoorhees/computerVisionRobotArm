import camera
import camera_calibration
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



def disparityMap(rectified_right, rectified_left, Q):
    #takes the two images that have been mapped and converts them to grayscale
    rectified_right = cv.cvtColor(rectified_right, cv.COLOR_BGR2GRAY)
    rectified_left = cv.cvtColor(rectified_left, cv.COLOR_BGR2GRAY)

    #converst the datatype to uint8 because that's the only type openCV will use to display images
    rectified_left  = rectified_left.astype(np.uint8)
    rectified_right = rectified_right.astype(np.uint8)

    print(rectified_left.shape, rectified_right.shape)

    numDisparities = 5 * 16
    blockSize = 5
    minDisparity = 0
    speckleWindowSize = 5
    speckleRange = 10
    disp12MaxDiff = 0
    P1 = 0
    P2 = 0
    stereo = None
    window_name = 'SGBM Parameters'
    # Use safe numDisparities
    #finds the height and width of the image

    #empty call back function
    def nothing(x):
        pass

    init_numDisp = 5          # actual = x16
    init_blockSize = 5
    init_minDisp = 0
    init_speckleWindow = 5
    init_speckleRange = 10
    init_disp12 = 0
    init_P1 = 8 * 3 * 5 * 5
    init_P2 = 32 * 3 * 5 * 5

    stereo = cv.StereoSGBM_create(
        minDisparity=init_minDisp,
        numDisparities=init_numDisp,
        blockSize=init_blockSize,
        P1=init_P1,
        P2=init_P2,
        disp12MaxDiff=init_disp12,
        speckleWindowSize=init_speckleWindow,
        speckleRange=init_speckleRange,
        preFilterCap = 0,
        mode = cv.StereoSGBM_MODE_SGBM
    )
            # You may need to add other parameters like preFilterCap, uniquenessRatio, mode etc.
    

    # In a real application, you would re-read your left/right images here and compute disparity
    # disp = sgbm.compute(imgL, imgR).astype(np.float32) / 16.0
    # cv2.imshow("Disparity Map", disp)

    # Create a black image, a window for controls, and another for the result
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
       
        # Update the SGBM object with the new parameters
        stereo = cv.StereoSGBM_create(
            minDisparity      = md,
            numDisparities    = numDisparities,
            blockSize         = bs,
            P1                = P1,
            P2                = P2,
            disp12MaxDiff     = d12,
            uniquenessRatio   = 0,
            speckleWindowSize = sw,
            speckleRange      = sr,
            preFilterCap      = 0,
            mode              = cv.StereoSGBM_MODE_SGBM_3WAY
        )
        # Add your image processing code here
        # (Load/process images, compute disparity, display result)
        # ...
        #computes the disparity and converts the data type
        
        right_matcher = cv.ximgproc.createRightMatcher(stereo)

        disp_left  = stereo.compute(rectified_left, rectified_right).astype(np.float32) / 16.0
        disp_right = right_matcher.compute(rectified_right, rectified_left).astype(np.float32) / 16.0

        # --- WLS Filtering ---
        wls = cv.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
        wls.setLambda(8000)
        wls.setSigmaColor(1.5)

        filtered = wls.filter(disp_left, rectified_left, disparity_map_right=disp_right)

        # Normalize for display
        disp_vis = cv.normalize(disp_left, None, 0, 255, cv.NORM_MINMAX)
        disp_vis = np.uint8(disp_vis)

        filtered_vis = cv.normalize(filtered, None, 0, 255, cv.NORM_MINMAX)
        filtered_vis = np.uint8(filtered_vis)

        cv.imshow("Disparity Map", disp_vis)
        cv.imshow("Filtered Disparity (WLS)", filtered_vis)

        # Exit loop if 'Esc' key is pressed
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break

        #plots the disparity map so we can see teh different depths in pizels
    plt.imshow(filtered, 'gray')
    plt.show()

        
            
        
            
    return filtered