# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:12:42 2019

@author: TOMATO
"""

import cv2
import numpy as np


def canny(argv):
#    #阈值分割 

    #Threshold_type:cv2.THRESH_BINARY=0,cv2.THRESH_BINARY_INV=1
    #               cv2.THRESH_TRUNC=2,cv2.THRESH_TOZERO=3
    #               cv2.THRESH_TOZERO_INV=4
    #               cv2.THRESH_MASK=7,cv2.THRESH_TRIANGLE=16,
    thresh_low=cv2.getTrackbarPos('thresh_low','canny')
    #邻域块大小
    thresh_high=cv2.getTrackbarPos('thresh_high','canny')


    canny_img = cv2.Canny(img,thresh_low,thresh_high)
    cv2.imshow('canny',canny_img)
#    para=[0 for i in range(4)]#保存参数
    para=[]
    para[:]=thresh_low,thresh_high
    return para

def show_sobel(sobel_img,win_name):
#    sobel_img=(sobel_img+np.abs(np.min(sobel_img))).astype(np.uint8)#平移到0-255内，直方图无拉伸    
    sobel_img=np.abs(sobel_img).astype(np.uint8)    #负梯度折成正梯度（求绝对值）
    cv2.namedWindow(win_name,0)
    cv2.imshow(win_name,sobel_img)
   
    
    
if __name__ == '__main__':
#    img=cv2.imread('test_images/lena512.bmp',0)
    img=cv2.imread('D:/python_projects/aliproject/imgs/6.jpg',0)
    #裁剪
    img=img[:,400:900]
    #降采样
#    img=cv2.pyrDown(img)
#    img=cv2.pyrDown(img)
    
    cv2.namedWindow('img',0)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    img=cv2.GaussianBlur(img,(3,3),0)#sigmaX=1,sigmaY=1
    
    
    
    #laplacian算子
    lap=cv2.Laplacian(img,cv2.CV_64F)
    lap=np.uint8(np.absolute(lap))
    cv2.namedWindow('laplacian',0)
    cv2.imshow('laplacian',lap)
    cv2.waitKey(0)
    #sobel算子
    sobelX=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=1)#X方向
    sobelY=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=1)#Y方向
    sobelXY=cv2.Sobel(img,cv2.CV_64F,1,1,ksize=1)#XY方向
    sobel_comb=cv2.bitwise_or(sobelX,sobelY)
    
    show_sobel(sobelX,'sobelX')
    show_sobel(sobelY,'sobelY')
    show_sobel(sobelXY,'sobelXY')
    show_sobel(sobel_comb,'sobel_comb')
    cv2.waitKey(0)
    
    #canny算子
    
    cv2.namedWindow('canny',0)

    cv2.createTrackbar('thresh_low','canny',1,500,canny)
    cv2.createTrackbar('thresh_high','canny',1,500,canny)


    cv2.waitKey(0)
    
    para=canny(0)
    print(para)
    

    
    
    
    
    
    
    
    
    cv2.destroyAllWindows()