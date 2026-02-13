import cv2
import camera_calibration
import disparityMap
import depthMap
import threading
import queue
import os
import time
import numpy as np
import YOLO_code
from ultralytics import YOLO
from matplotlib import pyplot


data = np.load('stereo_calibration_data.npz')
right_mapx, right_mapy = data['right_mapx'], data['right_mapy']
left_mapx, left_mapy = data['left_mapx'], data['left_mapy']
Q = data['Q']

feed_queue = queue.Queue(maxsize=1)


def feed(frame_queue, depth_queue, stop_event, cap, cap2):
    #starts up both cameras and turns the video on

    num=0

    #checks if camera is on and set up
    if not cap.isOpened():
        print("Error videostream is not working")
        exit()

    while True:
        #reads the frames generated, img is the image data generated, ret is a bool for if any data was read
        while not stop_event.is_set():
            ret, img1 = cap.read(0)
            ret, img2 = cap2.read(1)

            if((ret) == False):
                print("Stopped receiving frames")
                break

            #waits 3 seconds for a key press and saves the character pressed
            key = cv2.waitKey(30)

            #if the key pressed if q then the program is terminated
            if key == ord('q'):
                break

            #shows image taken in seperate window
            #cv2.imshow('Img 1', img1)          
            #cv2.imshow('Img 2', img2)

            re_img1 = cv2.remap(img1, right_mapx, right_mapy, cv2.INTER_LINEAR)
            re_img2 = cv2.remap(img2, left_mapx, left_mapy, cv2.INTER_LINEAR)

            detection = YOLO_code.run_YOLO(re_img2)

            if frame_queue.full():
                frame_queue.get_nowait()

            frame_queue.put(re_img2)

            
            
            depth_map = disparityMap.disparityMap(re_img2,re_img1, Q, feed_queue, depth_queue, stop_event)

            #depth_map = depthMap.depthMap(disparity, Q)

            for det in detection:
                depth = YOLO_code.get_object_depth(depth_map, det["bbox"])
                if depth is None:
                    continue

                x1, y1, x2, y2 = det["bbox"]

                label = f'{det["class"]} {depth/1000:.2f} m'
                cv2.rectangle(re_img2, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(re_img2, label, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            cv2.imshow("YOLO Detection", re_img2)
            #depthMap.depthMapMeters(disparity,Q, mtx1, T)
        


    #terminates the process and destroys all data used by openCV
    cap.release()

    cv2.destroyAllWindows()



