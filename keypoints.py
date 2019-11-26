# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:45:42 2019
特征点检测
包括0:AKAZE,1:BRISK,2:FAST,3:KAZE,4:ORB
Nonefree : 5:SIFT,6:SURF
@author: TOMATO
"""

import cv2
    


def detect_keypoints(argv):
    #0:AKAZE,1:BRISK,2:FAST,3:KAZE,4:ORB
    #5:SIFT,6:SURF
    #method
    method_index=cv2.getTrackbarPos('method','Feature keypoints')

    method=method_dict[method_index]()
    print(method)
    keypoints = method.detect(img, None) #第二个参数是mask，必须为8bit integer矩阵
    img2 = img.copy()
    img2 = cv2.drawKeypoints(img, keypoints, img2, color=(0,255,0))
    cv2.imshow('Feature keypoints', img2)


if __name__ == '__main__':

#    img=cv2.imread('test_images/lena512.bmp',0)
    img=cv2.imread('D:/python_projects/aliproject/imgs/6.jpg',0)    
#    img=cv2.imread('adaptiva_threshold.png',0)
    cv2.namedWindow('img',0)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    method_dict={
        0 : lambda:cv2.AKAZE_create(),
        1 : lambda:cv2.BRISK_create(),
        2 : lambda:cv2.FastFeatureDetector_create(),
        3 : lambda:cv2.KAZE_create(),
        4 : lambda:cv2.ORB_create(),
#        5 : lambda:cv2.xfeatures2d.SIFT_create(),
#        6 : lambda:cv2.xfeatures2d.SURF_create()
        }

    cv2.namedWindow('Feature keypoints',0)
    cv2.createTrackbar('method','Feature keypoints',0,4,detect_keypoints)
    cv2.waitKey(0)

    cv2.destroyAllWindows()







