#!/usr/bin/env python
# coding: utf-8


import cv2 
import numpy as np
import time


#Various different test videos including in the repository, to change which video is ran, uncomment it and comment out
#the other videos.


#Video captured into a video object
#cap = cv2.VideoCapture("test1.mp4")
#cap = cv2.VideoCapture("test2.m4v")
#cap = cv2.VideoCapture("test3.mp4")
#cap = cv2.VideoCapture("test4.mp4")
cap = cv2.VideoCapture("test5.mp4")


#read first frame of the video
_, frameStart = cap.read() 

#resizing the input video to increase frame rate by working with a smaller input
resizeDim = 600
maxDim = max(frameStart.shape)
scale = resizeDim/maxDim
frameStart = cv2.resize(frameStart,None,fx=scale,fy=scale)

#converting the first resized frame to grayscale
previousGray = cv2.cvtColor(frameStart,cv2.COLOR_BGR2GRAY)

#Creating a empty matrix, size of the input frame with all pixels set to 255
mask = np.zeros_like(frameStart)
mask[..., 1] = 255

#While loop until the video object is open
while(cap.isOpened()):
        
    #Read the next frame into the variable frame, and ret being a boolean value if the read was successful
    ret, frame = cap.read()
    
    #If the read failed, being at the end of the video or a crash, break out of the loop and proceed to end
    if(ret == False):
        break

        
    #Converting to grayscale and resizing
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameGray = cv2.resize(frameGray,None,fx=scale,fy=scale)
    
    
    #Calculate the gunnar-farneback flow 
    flow = cv2.calcOpticalFlowFarneback(previousGray, frameGray, None, 0.5, 5, 12, 10, 5, 1.1, 0) 

    #Convert the flow into polar coordinates
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    #image hue according to flow direction
    mask[...,0] = angle * 180/np.pi/2
    #image magnitude to flow magnitude 
    mask[...,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    

    #convert HSV(Hue,Saturation,Value) to RGB to Grayscale
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    grayer = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    #Erosion, opening and then closing on the flow to reduce blurring between objects and reduce noise
    kernel = np.ones((5,5),np.uint8)
    grayer = cv2.erode(grayer,kernel,iterations = 3)
    grayer = cv2.morphologyEx(grayer, cv2.MORPH_OPEN, kernel)
    grayer = cv2.morphologyEx(grayer, cv2.MORPH_CLOSE, kernel)
    

    #resizing the output frame
    frame = cv2.resize(frame,None,fx=scale,fy=scale)
    
    #converting the grayscale image back to rgb resulting in a white overlay for the tracking
    rgb = cv2.cvtColor(grayer,cv2.COLOR_GRAY2RGB)
    
    #Overlaying the white over the initial image, creating a outline for tracking.
    denseflow = cv2.addWeighted(frame, 1, rgb, 5,0)
    
    #Showing the results
    cv2.imshow('Processed Video Stream', denseflow)

    #Updating the previous frame to be the current frame
    previousGray = frameGray
    
    #If q is pressed it breaks out of the process, else it waits 10 milliseconds to proceed to the next image
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        

#release the video object and clean any windows left.
cap.release()
cv2.destroyAllWindows()
    
