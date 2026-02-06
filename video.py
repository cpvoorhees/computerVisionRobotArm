import cv2
import disparityMap
import threading
import os
import time

def video():
    imageL = []
    imageR = []

    #starts up both cameras and turns the video on
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  #right camera
    #cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)  #left camera

    num=0

    #checks if camera is on and set up
    if not cap.isOpened():
        print("Error videostream is not working")
        exit()

    while True:
        #reads the frames generated, img is the image data generated, ret is a bool for if any data was read
        ret, img1 = cap.read(0)

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
        #cv2.imshow('Img 2', img2)

        disparityMap()


    #terminates the process and destroys all data used by openCV
    cap.release()

    cv2.destroyAllWindows()



video()