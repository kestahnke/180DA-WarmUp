# refereces:
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
# https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
# https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
# https://automaticaddison.com/real-time-object-tracking-using-opencv-and-a-webcam/

import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([90,50,50])
    upper_blue = np.array([130,255,255])

    # Threshold HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame, frame, mask= mask)

    #convert to grayscale for contours
    gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)

    # get contours
    contours, hierarchy = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    areas = [cv.contourArea(c) for c in contours]
    #if no contours
    if len(areas) < 1:
        #show resulting frame
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else: 
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        x,y,w,h = cv.boundingRect(cnt)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        # Display the resulting frame
        cv.imshow('frame',frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

# When everything is done, release the capture
cap.release()
cv.destroyAllWindows()
