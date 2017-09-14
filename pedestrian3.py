# pedestrian detection in static images, with my_imutils  
 
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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to images directory")
args = vars(ap.parse_args())

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# loop over the image paths
for imagePath in my_imutils.list_files(args["images"]):
	# load the image and resize it to reduce detection time
	# and also improve detection accuracy
	image = cv2.imread(imagePath)
	image = my_imutils.resize(image,400)
	orig = image.copy()
 
	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
		padding=(8, 8), scale=1.05)
 
	# draw the original bounding boxes
	#for (x, y, w, h) in rects:
		#cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
 
	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
 
	# show some information on the number of bounding boxes
	filename = imagePath[imagePath.rfind("/") + 1:]
	print("[INFO] {}: {} original boxes, {} after suppression".format(
		filename, len(rects), len(pick)))
 
	# show the output images
	cv2.imshow("Original", orig)
	cv2.imshow("Pedestrians", image)
	k = cv2.waitKey(0)
	if (k==27): break
	
cv2.destroyAllWindows()	
	

 
 