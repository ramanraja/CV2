# Human detection in videos with HOG descriptors- without nonmax suppression
# https://stackoverflow.com/questions/34871294/full-body-detection-and-tracking-using-opencvpython-2-7
# It is a headache to work with Video Capture mostly due to wrong installation of ffmpeg/gstreamer.
# Go to C:\opencv\build\x86\vc12\bin and take the file opencv_ffmpeg300_64.dll. Copy it into your 
# python root folder, for eg: C:\python27  or C:\Users\Raja\Anaconda
# https://stackoverflow.com/questions/35242735/can-not-read-or-play-a-video-in-opencvpython-using-videocapture

import numpy as np
import cv2


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


if __name__ == '__main__':

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
    cap=cv2.VideoCapture ('pedestrian.mp4') #('768x576.avi')
    while True:
        flag,frame=cap.read()
        #print flag
        found,w=hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)
        draw_detections(frame,found)
        cv2.imshow('feed',frame)
        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
    cv2.destroyAllWindows()
    