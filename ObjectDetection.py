import numpy as np
import cv2
import time

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 390)

time.sleep(2)

    
while(cam.isOpened()):
    
    ret, img = cam.read()
    
    if ret == False:
        break
        
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    loweryellow = np.array([20,100,100])
    upperyellow = np.array([60,255,255])
    mask1 = cv2.inRange(hsv , loweryellow , upperyellow)
    
    kernel = np.ones((3,3))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN,kernel,iterations=4)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE,kernel,iterations=4)
    mask1 = cv2.dilate(mask1,kernel,iterations = 1)
    res1 = cv2.bitwise_and(img, img, mask=mask1)
    
    
    cv2.imshow('Final Object Detect' , res1)
    cv2.imshow('Masked' , mask1)
    cv2.imshow('Original' , img)
    k=cv2.waitKey(10)
    if k==27:
        break
        
cam.release()
cv2.destroyAllWindows()