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


#Parameters for Shi-Tomasi corner detection, maxCorners being the max number of features, quality level being our 
#thresholding metric, minDist being how far each corner can be from each other and blockSize being the window of corner detection.
feature_params = dict(maxCorners = 1000, qualityLevel = 0.3, minDistance = 15, blockSize = 7)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#How long to track each pixel for, larger values result in longer tracking
trackLen = 25
#How often to resample for our features
detectInterval = 5
#Storage of the track lines following each feature
tracks = []
#Index to keep track of which frame we are
frameIndex = 0


#read first frame of the video
_, frameStart = cap.read() 

#resizing the input video to increase frame rate by working with a smaller input
resizeDim = 600
maxDim = max(frameStart.shape)
scale = resizeDim/maxDim
frameStart = cv2.resize(frameStart,None,fx=scale,fy=scale)

#converting the first resized frame to grayscale
previousGray = cv2.cvtColor(frameStart,cv2.COLOR_BGR2GRAY)

#Find the features/corners using the Shi-Tomasi method to run on Lucas-Kanade algorithm
previous = cv2.goodFeaturesToTrack(previousGray, mask = None, **feature_params)

#max image saturation
mask = np.zeros_like(frameStart)

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
    
    #resizing the output frame
    frame = cv2.resize(frame,None,fx=scale,fy=scale)

    #Check if we have features
    if(len(tracks) > 0):
        
        #points/features from our tracks list, -1 obtains the last element of the list
        initialFeatures = np.float32([tr[-1] for tr in tracks]).reshape(-1,1,2)
        #the next position of the features
        flowForward, status1, err1 = cv2.calcOpticalFlowPyrLK(previousGray, frameGray, initialFeatures, None, **lk_params)
        #the calculated old position of the features based on the newly calculated positions to assure that
        #there is no large leaps or random displacements occuring
        flowBackward, status2, err2 = cv2.calcOpticalFlowPyrLK(frameGray, previousGray, flowForward, None, **lk_params)
        
        #Compare the initial features to the calculated initial positions
        displacement = abs(initialFeatures-flowBackward).reshape(-1,2).max(-1)
        #If the aboslute total replacement is less than 1 it is considered accurate.
        displacementCheck = displacement < 1
        
        #A new tracks list
        newTracks = []
        
        #For loop over the tracks list, (x,y) for each feature in the new positions, and our displacement check
        for tr, (x,y), goodCheck in zip(tracks, flowForward.reshape(-1,2), displacementCheck):
            
            #If the displacement check is bad we have errors, we skip this iteration and go to the next one
            if(not goodCheck):
                continue
                
            #We append this position to our tracks list
            tr.append((x,y))
            
            #If how long a features being tracked for is longer than the allowable track length we delete the first position
            #in the track allowing more
            if(len(tr) > trackLen):
                del tr[0]
                
            #We append our tracks
            newTracks.append(tr)
            
            #Draw a circle of thickness 2 with color green at position x,y
            cv2.circle(frame, (x,y), 2, (0, 255, 0), -1)
            
        #Assign our new tracks to our tracks list
        tracks = newTracks
        
        #For every position for each track we draw a line between them.
        cv2.polylines(frame, [np.int32(tr) for tr in tracks], False, (0,255,0))
        
    #This checks what the index is compared to how often we want to resample our features, this also occurs the first frame
    if(frameIndex % detectInterval == 0):
        
        #Create a mask of pixel values being 255
        mask = np.zeros_like(frameGray)
        mask[:] = 255
        
        #for every position in track, draw a circle there on our mask, this allows us to resample based on previous features
        for x,y in [np.int32(tr[-1]) for tr in tracks]:
            cv2.circle(mask, (x,y), 5,0,-1)
            
        #Using the marked positions on our mask, we run corner detection for updated features, often the same as previous
        #If there is no features, the mask is blank so we calculate features from scratch
        features = cv2.goodFeaturesToTrack(frameGray, mask=mask, **feature_params)
        
        #If the features are calculated, we append these to our tracks array to use in the above if loop
        if(features is not None):
            for x, y in np.float32(features).reshape(-1,2):
                tracks.append([(x,y)])
            
    #Increment index and update the previous frame to be the current frame
    frameIndex = frameIndex +1
    previousGray = frameGray
    
    #Show the results
    cv2.imshow('1k_tracks', frame)
      
    #If q is pressed it breaks out of the process, else it waits 10 milliseconds to proceed to the next image
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
#release the video object and clean any windows left.        
cap.release()
cv2.destroyAllWindows()
    

