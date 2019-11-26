# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:21:40 2019

@author: TOMATO
"""

#import numpy as np
import cv2



def threshold_segment(argv):
#    #阈值分割 

    #Threshold_type:cv2.THRESH_BINARY=0,cv2.THRESH_BINARY_INV=1
    #               cv2.THRESH_TRUNC=2,cv2.THRESH_TOZERO=3
    #               cv2.THRESH_TOZERO_INV=4,
    #               cv2.THRESH_MASK=7,
    #               cv2.THRESH_OTSU=8,cv2.THRESH_TRIANGLE=16
    threshold_type=cv2.getTrackbarPos('threshold_type','threshold_segmentation')    
    #邻域块大小
    threshold=cv2.getTrackbarPos('threshold','threshold_segmentation')
    algorithm=cv2.getTrackbarPos('algorithm','threshold_segmentation')*8

    thr,binary_img = cv2.threshold(img,threshold,255,threshold_type+algorithm)
    cv2.imshow('threshold_segmentation',binary_img)
#    para=[0 for i in range(4)]#保存参数
    para=[]
    para[:]=threshold_type,thr,algorithm
    return para


if __name__ == '__main__':

#    img=cv2.imread('D:/python_projects/aliproject/imgs/1.jpg',0)
    img=cv2.imread('D:/python_projects/aliproject/imgs/sobelX.png',0)
#    img=cv2.imread('sobelX.png',0)
    
    cv2.namedWindow('img',0)
    cv2.imshow('img',img)
    cv2.waitKey(0)

    cv2.namedWindow('threshold_segmentation',0)

    cv2.createTrackbar('threshold_type','threshold_segmentation',0,4,threshold_segment)
    cv2.createTrackbar('threshold','threshold_segmentation',0,255,threshold_segment)
    cv2.createTrackbar('algorithm','threshold_segmentation',0,2,threshold_segment)

    cv2.waitKey(0)
    
    para=threshold_segment(0)
    print(para)
    

    
    
    
    
    
    
    
    cv2.destroyAllWindows()
    
    
    
    
    