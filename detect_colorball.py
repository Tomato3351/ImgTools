# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 11:07:56 2018

@author: tomat
"""

import numpy as np
import cv2
#import os
#import sys

def do_open(argv):
#    执行开运算。参数为结构半径，迭代次数
    r_open=cv2.getTrackbarPos('r_open','mask_open')
    i_open=cv2.getTrackbarPos('i_open','mask_open')    
    s_open=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r_open,r_open))
    mask_open=cv2.morphologyEx(mask_p,cv2.MORPH_OPEN,s_open,iterations=i_open)
    cv2.imshow('mask_open',mask_open)   
    return mask_open

def do_close(argv):
#    执行闭运算。参数为结构半径，迭代次数
    r_close=cv2.getTrackbarPos('r_close','mask_close')
    i_close=cv2.getTrackbarPos('i_close','mask_close')    
    s_close=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r_close,r_close))
    mask_close=cv2.morphologyEx(mask_open,cv2.MORPH_CLOSE,s_close,iterations=i_close)
    cv2.imshow('mask_close',mask_close)    
    return mask_close


def HSV_detect(argv):
    """
    通过H,S,V参数调整，获取相应的mask
    """
    
    lower_Hue=cv2.getTrackbarPos('Hue','mask')
    Hue_range=cv2.getTrackbarPos('Hue_range','mask')
    
    lower_Saturation=cv2.getTrackbarPos('Saturation','mask')
    Saturation_range=cv2.getTrackbarPos('Saturation_range','mask')
    lower_Value=cv2.getTrackbarPos('Value','mask')
    Value_range=cv2.getTrackbarPos('Value_range','mask')   
    lower=np.array([lower_Hue,lower_Saturation,lower_Value])
    upper=np.array([lower_Hue+Hue_range,lower_Saturation+Saturation_range,lower_Value+Value_range])
    
    mask=cv2.inRange(HSV,lower,upper)
#    while(1):
#        cv2.imshow('mask',mask)
#        k=cv2.waitKey(1)&0xff
#        if k==27:
#            break   
    cv2.imshow('mask',mask)  
    HSV_p=[0 for i in range(6)]
    HSV_p[:]=lower_Hue,Hue_range,lower_Saturation,Saturation_range,lower_Value,Value_range
    
    return mask,HSV_p


if __name__ == '__main__':
#    img=cv2.imread("D:/Projects/camera control/camera control/capture_image/Side/39Left.jpg",1)
#    img=cv2.imread('ROI.jpg',-1)
    img=cv2.imread('D:/python_projects/longlianproject/imgs/10.jpg',-1)
    cv2.namedWindow('img',0)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    h,w,c=img.shape

#    shrink = cv2.resize(img, (13,13), interpolation=cv2.INTER_AREA)
    b=img[:,:,0]
    g=img[:,:,1]
    r=img[:,:,2]
#cv2.namedWindow('shrink',0)
#cv2.imshow('shrink',shrink)
#cv2.waitKey(0)
    cv2.namedWindow('b',0)  
    cv2.imshow('b',b)
#cv2.waitKey(0)
    cv2.namedWindow('g',0)
    cv2.imshow('g',g)
#cv2.waitKey(0)
    cv2.namedWindow('r',0)
    cv2.imshow('r',r)
    cv2.waitKey(0)

#r_g=r-g
#cv2.namedWindow('r_g',0)
#cv2.imshow('r_g',r_g)
#cv2.waitKey(0)
#    HSV=img
#    HSV=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    HSV=cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
    
    cv2.namedWindow('HSV',0)
    cv2.imshow('HSV',HSV)
    cv2.waitKey(0)
    

    cv2.namedWindow('mask',0)
    cv2.createTrackbar('Hue','mask',24,180,HSV_detect)
    cv2.createTrackbar('Hue_range','mask',34,180,HSV_detect)
    cv2.createTrackbar('Saturation','mask',5,255,HSV_detect)
    cv2.createTrackbar('Saturation_range','mask',183,255,HSV_detect)
    cv2.createTrackbar('Value','mask',215,255,HSV_detect)
    cv2.createTrackbar('Value_range','mask',43,255,HSV_detect)
    cv2.waitKey(0)
   
    mask_p,HSV_p=HSV_detect(0)
    print(HSV_p)    #HSV参数

    cv2.namedWindow('mask_open',0)
    cv2.createTrackbar('r_open','mask_open',6,50,do_open)
    cv2.createTrackbar('i_open','mask_open',1,30,do_open)
    cv2.waitKey(0)
    mask_open=do_open(0)
    
    cv2.namedWindow('mask_close',0)
    cv2.createTrackbar('r_close','mask_close',6,50,do_close)
    cv2.createTrackbar('i_close','mask_close',1,30,do_close)
    cv2.waitKey(0)
    mask_close=do_close(0)
    gray = cv2.cvtColor(mask_close,cv2.COLOR_BAYER_BG2GRAY)
    
    
    contours,hierarchy_L= cv2.findContours(mask_close,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    blank_contours=np.zeros((h,w))
    
    HTcircles_L=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,200,200,param2=15,minRadius=17,maxRadius=88)
    
    
    
    

    
    
    print (type(HTcircles_L))
    n=HTcircles_L.shape[1]
    print(n)
    if len(HTcircles_L.shape)==3:
        for i in range(n):
            center=(int(HTcircles_L[0,i,0]),int(HTcircles_L[0,i,1]))
            radius=HTcircles_L[0,i,2]
            cv2.circle(img,center,radius,(0,255,0),2)
    else:
        print('No ball detected')
        

            
#    
#    img_b,contours,hierarchy=cv2.findContours(mask_close,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    
    
    
#    (x,y),radius = cv2.minEnclosingCircle(cnt)
#    center = (int(x),int(y))
#    radius = int(radius)
#    cv2.circle(img,center,radius,(0,255,0),2)
#    
#    
#    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
#    
#    
#    
#    print('center_x=',x,',center_y=',y)

    cv2.namedWindow('result',0)
    cv2.imshow('result',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    








