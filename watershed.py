# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:16:58 2020

@author: TOMATO
"""

import numpy as np
import cv2


EPSILON = 4



def show_img(win_name,img,waitingtime):
    cv2.namedWindow(win_name,0)
    cv2.imshow(win_name,img)
    cv2.waitKey(waitingtime)

if __name__=="__main__":
    
    
#    img=cv2.imread('test_images/coins1.jpg',3)
#    img=cv2.imread("D:/projects/regiongrow/regiongrow/Images/error/3-1.png",3)
    img=cv2.imread('test_images/7-4.png',3)    
    binary=cv2.imread('test_images/luquan2.png',0)    
    h,w,c=img.shape
    show_img("img",img,0)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    show_img("thresh",thresh,0)
    
    s=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    morph_img=cv2.morphologyEx(binary,cv2.MORPH_OPEN,s,0,iterations=1)
#    show_img('morph_img',morph_img,0)
    morph_img=cv2.morphologyEx(morph_img,cv2.MORPH_CLOSE,s,0,iterations=1)
    show_img('morph_img',morph_img,0)
    
    sure_bg = cv2.dilate(morph_img,s,iterations=3)
    show_img('sure_bg',sure_bg,0)
    
    dist_transform = cv2.distanceTransform(morph_img,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
    cv2.normalize(dist_transform,dst = dist_transform, alpha = 0,beta = 1.0,norm_type=cv2.NORM_MINMAX)
    show_img('dist_transform',dist_transform,0)

    sure_fg = np.uint8(sure_fg)
    show_img('sure_fg',sure_fg,0)        
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    show_img('unknown',unknown,0)
    
    ret,markers=cv2.connectedComponents(sure_fg)
    markers +=1
    markers[unknown==255] = 0
    
    markers = cv2.watershed(img,markers)
    
    img[markers==-1]=[0,0,255]
    
    show_img('result',img,0)
    
    
    cv2.destroyAllWindows()