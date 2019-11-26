# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:40:29 2019
HoughLinesP(image, rho, theta, threshold[, lines[, minLineLength[, maxLineGap]]]) -> lines
@author: TOMATO
"""

import cv2
import numpy as np
def houghP(argv):
    rho_get=cv2.getTrackbarPos('rho','line_img')
    #邻域块大小
    theta_get=cv2.getTrackbarPos('theta','line_img')
    threshold_get=cv2.getTrackbarPos('threshold','line_img')
    minLineLength_get=cv2.getTrackbarPos('minLineLength','line_img')
    maxLineGap_get=cv2.getTrackbarPos('maxLineGap','line_img')
    lines=cv2.HoughLinesP(binary_img,rho=rho_get+1,theta=(np.pi/180)*(theta_get/2+0.5),threshold=threshold_get,
                      minLineLength=minLineLength_get,maxLineGap=maxLineGap_get)
    line_img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)#用于画检测到的直线
    if type(lines)==np.ndarray:
        print('lines:',lines.shape[0])
        for x1,y1,x2,y2 in lines[:,0,:]:
            cv2.line(line_img,(x1,y1),(x2,y2),(0,0,255),1)#画直线       
    cv2.imshow('line_img',line_img)
#    para=[0 for i in range(4)]#保存参数
    para=[]
    para[:]=rho_get+1,theta_get/2+0.5,threshold_get,minLineLength_get,maxLineGap_get
    return para

if __name__ == '__main__':
    
    img=cv2.imread('D:/python_projects/aliproject/imgs/6.jpg',0)
    #裁剪
    img=img[:,400:900]
    #降采样
#    img=cv2.pyrDown(img)
#    img=cv2.pyrDown(img)

    cv2.namedWindow('img',0)
    cv2.imshow('img',img)
    cv2.waitKey(0)

    img=cv2.GaussianBlur(img,(5,5),0)#sigmaX=1,sigmaY=1
    
    #canny算子求边缘
    canny_img = cv2.Canny(img,33,80)
    cv2.namedWindow('canny_img',0)
    cv2.imshow('canny_img',canny_img)
    cv2.waitKey(0)
    
    #sobel求梯度
    sobelX=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=1)#X方向
    sobelX=np.abs(sobelX).astype(np.uint8)    #负梯度折成正梯度（求绝对值）
    
#    binary_img=canny_img
    binary_img = cv2.adaptiveThreshold(sobelX,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,19,-8)#自适应阈值
    #阈值分割法cv2.THRESH_OTSU,cv2.THRESH_TRIANGLE
#    thr,binary_img= cv2.threshold(sobelX,16,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    cv2.namedWindow('binary_img',0)
    cv2.imshow('binary_img',binary_img)
    cv2.waitKey(0)

    cv2.namedWindow('line_img',0)
    cv2.createTrackbar('rho','line_img',0,50,houghP)
    cv2.createTrackbar('theta','line_img',1,180,houghP)
    cv2.createTrackbar('threshold','line_img',0,1000,houghP)
    cv2.createTrackbar('minLineLength','line_img',0,500,houghP)
    cv2.createTrackbar('maxLineGap','line_img',0,200,houghP)
    cv2.waitKey(0)
    para=houghP(0)
    print(para)
    
    
    
    
    
    
    
    
    
    
    cv2.destroyAllWindows()
    
    