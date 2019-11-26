# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 16:45:30 2019

@author: TOMATO
"""


import cv2



def adaptive_threshold(argv):
#    #自适应阈值分割 
    #method:cv2.ADAPTIVE_THRESH_MEAN_C=0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C=1
    method=cv2.getTrackbarPos('Method','adaptiva_threshold')
    #Threshold_type:cv2.THRESH_BINARY=0,cv2.THRESH_BINARY_INV=1
    threshold_type=cv2.getTrackbarPos('threshold_type','adaptiva_threshold')    
    #邻域块大小
    block_size=cv2.getTrackbarPos('block_size','adaptiva_threshold')
    block_size=block_size*2+3
    #偏移值调整量C，阈值由块内加权均值减去常量C得到
    c=cv2.getTrackbarPos('C','adaptiva_threshold')
    c=c-20       
    binary_img = cv2.adaptiveThreshold(img,255,method,
                                        threshold_type,block_size,c)    
    cv2.imshow('adaptiva_threshold',binary_img)
#    para=[0 for i in range(4)]#保存参数
    para=[]
    para[:]=method,threshold_type,block_size,c
    return para


if __name__ == '__main__':
#    bin_imgs=np.load('d:/python_projects/cashbox/bin_imgs.npy')
#    img=bin_imgs[6,:,:]

#    img=cv2.imread('D:/python_projects/aliproject/imgs/1.jpg',0)
    img=cv2.imread('D:/python_projects/aliproject/imgs/sobelX.png',0)    
    cv2.namedWindow('img',0)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    
    
    

    cv2.namedWindow('adaptiva_threshold',0)
    cv2.createTrackbar('Method','adaptiva_threshold',0,1,adaptive_threshold)
    cv2.createTrackbar('threshold_type','adaptiva_threshold',0,1,adaptive_threshold) 
    cv2.createTrackbar('block_size','adaptiva_threshold',0,99,adaptive_threshold)
    cv2.createTrackbar('C','adaptiva_threshold',0,40,adaptive_threshold)

    
    
    cv2.waitKey(0)     
    para=adaptive_threshold(0)
    print(para)
    

    
    
    
    
    
    
    cv2.destroyAllWindows()