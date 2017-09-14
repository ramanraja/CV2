# pedestrian detection in video with HOG descriptors- with nonmax suppression implemented
 
# http://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
# install imutils first: $ pip install imutils
# or upgrade it: $ pip install --upgrade imutils

from __future__ import print_function
import my_imutils
from my_imutils import non_max_suppression
from my_imutils import resize
import numpy as np
import argparse
import cv2

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# loop over the video frames
cap = cv2.VideoCapture ('pedestrian.mp4') #('768x576.avi')  #('pedestrian-s.mp4')
#cap = cv2.VideoCapture(0)  # camera
print ("cap.isOpened= ", cap.isOpened())

while cap.isOpened():
    flag, frame = cap.read()
    # load the image and resize it to reduce detection time
    # and improve detection accuracy
    #frame = my_imutils.resize(frame, 400)
    orig = frame.copy()
 
    # detect people in the image
    #(rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),padding=(8, 8), scale=1.05)
    #(rects, weights) = hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)
    (rects, weights) = hog.detectMultiScale(frame, winStride=(8,8), padding=(16,16), scale=1.05)
    
    # draw the original bounding boxes [useful for determining the parameters]
    #for (x, y, w, h) in rects:
        #cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # apply non-maximal suppression to the bounding boxes using a fairly large  
    # overlap threshold, to try to maintain overlapping boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    #pick = non_max_suppression(rects, probs=None, overlapThresh=0.2)   
    #pick = non_max_suppression(rects, probs=None, overlapThresh=0.4)   
    #pick = non_max_suppression(rects, probs=None, overlapThresh=0.7)   
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.9)   
    
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
      cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
    
    # show the number of bounding boxes
    print("Count: ", len(pick))
    
    # show the output images
    cv2.imshow("Pedestrians", frame)
    k = cv2.waitKey(30)
    if (k==27): break
	
cap.release()	
cv2.destroyAllWindows()	
	

 
 