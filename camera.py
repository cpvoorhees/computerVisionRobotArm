import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
 
imgL = cv.imread('IMG_2182.jpeg', cv.IMREAD_GRAYSCALE)
imgR = cv.imread('IMG_2181.jpeg', cv.IMREAD_GRAYSCALE)

stereo = cv.StereoBM.create(numDisparities=160, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()

