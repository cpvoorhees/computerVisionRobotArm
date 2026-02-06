import cv2
import main
import camera_calibration
import disparityMap
import threading
import os
import time

def feed():
 

    #starts up both cameras and turns the video on
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  #right camera
    cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)  #left camera

    num=0

    #checks if camera is on and set up
    if not cap.isOpened():
        print("Error videostream is not working")
        exit()

    while True:
        #reads the frames generated, img is the image data generated, ret is a bool for if any data was read
        ret, img1 = cap.read(0)
        ret, img2 = cap.read(1)

        if((ret) == False):
            print("Stopped receiving frames")
            break

        #waits 3 seconds for a key press and saves the character pressed
        key = cv2.waitKey(30)

        #if the key pressed if q then the program is terminated
        if key == ord('q'):
            break

        #shows image taken in seperate window
        cv2.imshow('Img 1', img1)          
        cv2.imshow('Img 2', img2)

        re_img1 = cv2.remap(img1, main.right_mapx, main.right_mapy, cv2.INTER_LINEAR)
        re_img2 = cv2.remap(img2, main.left_mapx, main.left_mapy, cv2.INTER_LINEAR)

        disparityMap.disparityMap(re_img1,re_img2, main.Q)

        #depthMap.depthMap(disparity, Q)
        #depthMap.depthMapMeters(disparity,Q, mtx1, T)
        


    #terminates the process and destroys all data used by openCV
    cap.release()

    cv2.destroyAllWindows()



