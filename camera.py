import cv2

#starts up both cameras and turns the video on
cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

num=0

#checks if camera is on and set up
if not cap.isOpened():
    print("Error videostream is not working")
    exit()

while True:
    #reads the frames generated, img is the image data generated, ret is a bool for if any data was read
    ret, img1 = cap.read()
    ret2, img2 = cap2.read()

    if((ret & ret2) == False):
        print("Stopped receiving frames")
        break

    #waits 3 seconds for a key press and saves the character pressed
    key = cv2.waitKey(3)

    #if the key pressed if q then the program is terminated
    if key == ord('q'):
        break

    #if the key pressed if s then the image is captured and saved by both cameras
    elif key == ord('s'):
        cv2.imwrite("images/right/imgRight" + str(num) + ".jpg", img1)      #writes the image data of the right/left camera to the images file
        cv2.imwrite("images/left/imgLeft" + str(num), img2)
        print("Image captured sucessfully")
        num = num + 1

    #shows image taken in seperate window
    cv2.imshow('Img 1', img1)          
    cv2.imshow('Img 2', img2)

#terminates the process and destroys all data used by openCV
cap.release()
cap2.release()

cv2.destroyAllWindows()