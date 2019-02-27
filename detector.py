import cv2
from matplotlib import pyplot as plt
import time
import numpy as np

class Stages(object):

    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.timer = False
        self.process_time = 0

    def timeCalculator(self, matches):

        if len(matches)>20:
            match=True
        
            if self.timer:
                self.end_time = time.time()
                self.timer = False
            print('match found')

        else:
            match=False

            if (self.timer==False):
                self.start_time = time.time()
                self.timer = True

            print len(matches)

        if (match):
            if (self.end_time - self.start_time)>10:
                self.process_time = self.end_time - self.start_time
                
    def getTime(self, stage):
        print ('time'+stage+' :%d'%(self.process_time))

    def findMatches(self, des1, des2):
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.5*n.distance:
                good.append(m)
        return good


    

        
            
#stages

stage1=Stages()
stage2=Stages()
stage3=Stages()

# trainImage
img2 = cv2.imread('1.jpg',0)          
img3 = cv2.imread('2.jpg',0)
img4 = cv2.imread('3.jpg',0)           

# Initiate SIFT detector
sift = cv2.SURF()

# find the keypoints and descriptors with SIFT

kp2, des2 = sift.detectAndCompute(img2,None)
kp3, des3 = sift.detectAndCompute(img3,None)
kp4, des4 = sift.detectAndCompute(img4,None)


cv2.namedWindow('image')
cap = cv2.VideoCapture(1)

ret = True

while ret:
    ret, img = cap.read()
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(img1,None)

  
    stage1.timeCalculator(stage1.findMatches(des1, des2))
    stage2.timeCalculator(stage2.findMatches(des1, des3))
    stage3.timeCalculator(stage3.findMatches(des1, des4))

    stage1.getTime('1')
    stage2.getTime('2')
    stage3.getTime('3')

    cv2.imshow('image',img)
    k = cv2.waitKey(500) & 0xFF
    if k == 27:
        break
        



