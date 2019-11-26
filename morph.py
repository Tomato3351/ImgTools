# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:54:11 2019
Morph
    method_dict={
            0:cv2.MORPH_ERODE,
            1:cv2.MORPH_DILATE,
            2:cv2.MORPH_OPEN,
            3:cv2.MORPH_CLOSE,
            4:cv2.MORPH_GRADIENT,
            5:cv2.MORPH_TOPHAT,
            6:cv2.MORPH_BLACKHAT
            7:cv2.MORPH_HITMISS
            }
    kshape_dict={
            0:cv2.MORPH_RECT,
            1:cv2.MORPH_CROSS,
            2:cv2.MORPH_ELLIPSE,
            }    
@author: TOMATO
"""

import cv2

def morph_fun(argv):
    method=cv2.getTrackbarPos('method','morph')
    #核形状
    kshape=cv2.getTrackbarPos('kshape','morph')
    #核大小
    ksize=cv2.getTrackbarPos('ksize','morph')*2+1
    #迭代次数
    iteration=cv2.getTrackbarPos('iteration','morph')

    s=cv2.getStructuringElement(kshape,(ksize,ksize))
    morph_img=cv2.morphologyEx(img,method,s,0,iterations=iteration)
    cv2.imshow('morph',morph_img)
    para=[]
    para[:]=method,kshape,iteration
    return para


if __name__ == '__main__':

#    img=cv2.imread('D:/python_projects/aliproject/imgs/1.jpg',0)
#    img=cv2.imread('D:/python_projects/aliproject/imgs/sobelX.png',0)
    img=cv2.imread('adaptiva_threshold.png',0)
    
    cv2.namedWindow('img',0)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    
    s=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
    mask_open=cv2.morphologyEx(img,cv2.MORPH_HITMISS,s,1)

    cv2.namedWindow('morph',0)
    
    cv2.createTrackbar('method','morph',0,7,morph_fun)
    cv2.createTrackbar('kshape','morph',0,2,morph_fun)
    cv2.createTrackbar('ksize','morph',0,100,morph_fun)    
    cv2.createTrackbar('iteration','morph',1,100,morph_fun)

    cv2.waitKey(0)
    
    para=morph_fun(0)
    print(para)
#    

    
    
    
    
    
    
    
    cv2.destroyAllWindows()












