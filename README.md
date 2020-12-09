# OpticalFlowTracking
This repository contains two programs with vehicular tracking based the sparse and dense optical flow algorithms of Lucas-Kanade and Gunnar-Farneback. Five test videos are also provided along with a paper written about these programs.

# Requirements
- Python
- OpenCV
- Numpy Library

# How to run
To run each file, simple download and run the python file. This will ouput a window with the tracking drawn onto the initial video.

To run different provided test files, please edit the variable "cap" near the top of the file. This can be done by commenting out the current running video with #. Proceed to uncomment the test file you wish to run by removing the # infront of its cap variable.

To run files outside of the provided ones, you can edit the filename inside the quotation marks and include your video. Please note this video must be in the same folder as the python file.
  An example of this is changing "cap = cv2.VideoCapture("test5.mp4")" to "cap = cv2.VideoCapture("carVideo.mp4")" to run a video named carVideo.mp4.
  
  
 Please note that video 4 and 5 have been compressed to allow upload to GitHub as it was limited to 25 MB. 
  
