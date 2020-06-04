# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:22:41 2020

形状特征点匹配

@author: TOMATO
"""

import numpy as np
import cv2


EPSILON = 3


def show_img(win_name,img,waitingtime):
    cv2.namedWindow(win_name,0)
    cv2.imshow(win_name,img)
    cv2.waitKey(waitingtime)
    #####寻找构成多边形的关键点
def find_polygon_points(morph_img,epsilon):
    contours,hierarchy= cv2.findContours(morph_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contour_img=np.zeros((h,w,3),np.uint8)
    cv2.drawContours(contour_img,contours, -1, (0,255, 0), 2)
    points_list=[]
    points_img=np.zeros((h,w,3),np.uint8)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        for i in range(len(approx)):
            start = tuple(approx[i-1,0,:])
            end = tuple(approx[i,0,:])
            cv2.line(contour_img,start,end,(255,0,255),2)
            cv2.circle(contour_img,end,3,(0,0,255),-1)
            cv2.circle(points_img,end,2,(255,255,255),-1)
            points_list.append(end)
            text=str(i)
            cv2.putText(contour_img,text,end,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)
#    show_img('approx_poly',contour_img,0)
    sort_points_list=sorted(points_list, key=lambda x:x[1])
#    print(points_component_list)
    return sort_points_list,contour_img,points_img
###画出关键点
def draw_polygon_points(template_img,tmplt_points_mat,anchor_index_arr):
#    template_img=np.zeros((h,w,3),np.uint8)
    for i in range(len(tmplt_points_mat)):
        if i in anchor_index_arr:
            color=(0,0,255)
        else:
            color=(255,255,255)
        cv2.circle(template_img,(tmplt_points_mat[i,0],tmplt_points_mat[i,1]),2,color,-1)
#    show_img('template_img',template_img,0)
    return template_img

##创建模板
def create_template(morph_img,epsilon,filename="template_BD.xml"):
            #    find contours 
    anchors_list,contour_img,points_img = find_polygon_points(img,epsilon=99)
#    show_img('contour_img',contour_img,0)
#    show_img('points_img',points_img,0)
    points_list,contour_img,points_img = find_polygon_points(morph_img,epsilon)
#    show_img('contour_img',contour_img,0)
#    show_img('points_img',points_img,0)
    anchor_index_list = []
    for anchor in anchors_list:
        inpl = anchor in points_list
        if inpl:
            anchor_index = points_list.index(anchor)
            anchor_index_list.append(anchor_index)
#    ####模板点阵数据写入xml文件
    print("key points num:",len(points_list))
    print("anchor num:",len(anchors_list))
    print("valid anchor num:",len(anchor_index_list))
    points_arr=np.asarray(points_list)
    fs=cv2.FileStorage(filename,cv2.FILE_STORAGE_WRITE)
    fs.write('points_mat',points_arr)
    fs.write('anchor_index_mat',np.asarray(anchor_index_list))
    fs.release()
    print("create template_BD sucess! See file in ->"+filename)    
    
#######计算距离和角度
def calculate_dist_angle(points_arr,anchor_index):  
    points_sub_anchor=points_arr-points_arr[anchor_index]
    points_sq=points_sub_anchor**2
    dist_arr = (points_sq[:,0]+points_sq[:,1])**0.5 #各点到锚点的距离
    angles_arr_rad=np.arctan2(points_sub_anchor[:,1],points_sub_anchor[:,0])
    angles_arr_degree=np.rad2deg(angles_arr_rad)
    dist_angle_arr=np.concatenate(([dist_arr],[angles_arr_degree]),axis=0).T
    dist_angle_arr=np.array(sorted(dist_angle_arr, key=lambda x:x[1]))
#    print(dist_angle_arr)
    return dist_angle_arr

#
def normalize_ratio(src,range_min,range_max,alpha=1):
    return (src/(range_max-range_min))*alpha
    
###找两个数组中的匹配项，并返回待匹配数组中的索引
def find_match(template_arr,arr,error):    
    match_list_template=[]
    match_list=[]
    for i in range(len(template_arr)):
        diff_arr=abs(arr-template_arr[i])
        if np.min(diff_arr)<error:
            match_list_template.append(i)
            match_list.append(np.argmin(diff_arr))    
    return match_list_template, match_list


if __name__=="__main__":
#    img=cv2.imread('test_images/template_luquan.png',0)
    img=cv2.imread('test_images/luquan2.png',0)
    h,w=img.shape
#    show_img("img",img,0)
    s=cv2.getStructuringElement(cv2.MORPH_RECT,(21,21))
    morph_img=cv2.morphologyEx(img,cv2.MORPH_CLOSE,s,0,iterations=1)
#    show_img('img',morph_img,0)
    #######创建BD算法关键点模板
#    create_template(morph_img,epsilon=3)
#    #########读取模板
    fs=cv2.FileStorage("template_BD.xml",cv2.FILE_STORAGE_READ)
    tmplt_points_arr=fs.getNode('points_mat').mat()
    anchor_indices=fs.getNode('anchor_index_mat').mat()
    fs.release()
    ###画模板关键点
    template_img=np.zeros((h,w,3),np.uint8)
    template_img=draw_polygon_points(template_img,tmplt_points_arr,anchor_indices)
    show_img('template_img',template_img,0)
    ####找待匹配图像的多边形关键点
    points_list,contour_img,points_img = find_polygon_points(morph_img,epsilon=EPSILON)
    show_img('contour_img',contour_img,0)
    show_img('points_img',points_img,0)
    
    
    ###模板: 计算点阵中点到锚点的距离、角度
    ind= 4
    template_DB_arr=calculate_dist_angle(tmplt_points_arr,anchor_indices[ind])
#    diff_arr = np.diff(template_DB_arr,axis=0)
    template_dist_arr=template_DB_arr[:,0]
    template_angle_arr=(template_DB_arr-template_DB_arr[0])[:,1]
    
#    print(template_dist_arr)
    
#    #待匹配图像: 计算点阵中点到锚点的距离、角度    
    points_arr=np.asarray(points_list)
    
    max_match_num=0
    max_match_dist_num=0
    for i in range(len(points_arr)):
    ######以第i个点为锚点        
        DB_arr=calculate_dist_angle(points_arr,i)
        for j in range(len(points_arr)):
            ######以第j个点到锚点的角度为0度
            dist_arr=DB_arr[:,0]
            angle_arr=DB_arr[:,1]
            current_angle2zero=angle_arr-angle_arr[j]
            current_angle2zero[0:j]+=360
            sort_current_angle2zero=np.concatenate((current_angle2zero[j:],current_angle2zero[0:j]))
            #########比较sort_current_angle2zero和template_angle_arr中的相同元素个数及索引位置
            match_temp_list,match_list=find_match(template_angle_arr,sort_current_angle2zero,0.4)
            max_match_num=max(max_match_num,len(match_list))
            
            if (len(match_list)>len(template_angle_arr)*0.25):
#                print("i = ",i)
#                print("j =",j)
#                print(match_temp_list)
#                print(match_list)
#                print(sort_current_angle2zero[match_list])
#                print(len(match_list))
                current_dist=np.concatenate((dist_arr[j:],dist_arr[0:j]))
                current_dist_match=current_dist[match_list]
                #模板中的匹配项
                template_dist_match=template_dist_arr[match_temp_list]
#                print("current_dist_match=\n",current_dist_match)
#                print("template_dist_match=\n",template_dist_match)                
#                template_dist_match[template_dist_match==0]=1
#                current_dist_match[current_dist_match==0]=1
                
#                ratio_p=1
#                ratio=current_dist_match/template_dist_match
#                ratio=ratio[ratio<ratio_p+0.05]
#                ratio=ratio[ratio>ratio_p-0.05]
#                if len(ratio)>max_match_dist_num:
#                    max_match_dist_num = len(ratio)
#                    match_i=i
#                    match_j=j
                
                match_dist_temp_list,match_dist_list=find_match(template_dist_match,current_dist_match,4)
                if len(match_dist_list)>max_match_dist_num:    
                    max_match_dist_num=len(match_dist_list)
                    match_i=i
                    match_j=j
                    j_angle=angle_arr[j]
            
            
    
            
#    norm_dist_arr=np.zeros((DB_arr[:,0]).shape)
#    cv2.normalize(DB_arr[:,0],norm_dist_arr,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX)
    
#    norm_min=0
#    norm_max=(h**2+w**2)**0.5
#    norm_ratio=normalize_ratio(DB_arr[:,0],norm_min,norm_max)
    
    
    match_anchor_img=np.zeros((h,w,3),np.uint8)
    cv2.circle(match_anchor_img,(points_arr[match_i,0],points_arr[match_i,1]),2,(0,0,255),-1)
    show_img("match_anchor_img",match_anchor_img,0)    
    anchor_templt = tmplt_points_arr[anchor_indices[ind][0]]
    anchor_match = points_arr[match_i]
    
    tmplt_points_new=tmplt_points_arr+(anchor_match-anchor_templt)
    
    
    print(anchor_templt)
    print(anchor_match)    
#    print(template_DB_arr[0,1])
#    print(j_angle)    
    angle_new = -j_angle+template_DB_arr[0,1]
    
    
    
    M = cv2.getRotationMatrix2D((anchor_match[0],anchor_match[1]), angle_new, 1)
    M_homo = np.concatenate((M,np.array([[0,0,1]],np.float64)),axis=0)
    ######点阵旋转
    tmplt_points_new_homo=np.concatenate((tmplt_points_new,np.ones((len(tmplt_points_new),1),np.float64)),axis=1).T
    points_rotated=np.dot(M_homo,tmplt_points_new_homo).T.astype(np.int32)[:,0:2]
    
    
    
    template_new_img=np.zeros((h,w,3),np.uint8)
    template_img=draw_polygon_points(template_new_img,points_rotated,[])
    
    show_img('template_new_img',template_new_img,0)
    
    
    rect=cv2.minAreaRect(points_rotated)
    vertices=cv2.boxPoints(rect)
    ##画rect
    result=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    for i in range(4):
        p1=vertices[i,:]
        j=(i+1)%4
        p2=vertices[j,:]
        cv2.line(result,(p1[0],p1[1]),(p2[0],p2[1]),(255,0,255), 1)
    
    
    show_img('result',result,0)
    
    

#    template_new_rotated = cv2.warpAffine(template_new_img, M, (w, h))
#    
#    show_img('template_new_rotated',template_new_rotated,0)
    
    

    
    
    
    
    
    
    
    
    
    
    
    cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    