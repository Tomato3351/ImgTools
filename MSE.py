# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:37:25 2019

@author: TOMATO
"""
import numpy as np

REAL=10

#计算一维数据集的方差、均方差、均方误差、均方根误差

#方差variance


#均方差standard deviation


#均方误差mean squared error (MSE)


#均方根误差(RMSE)


if __name__ == '__main__':
    
#    a=np.array([10,9,11,10,11,9,9,11])
    a=np.array([10,9,8,12,9,10,10,9])
    
    
    mean=np.mean(a)
    var=np.var(a)
    std_dev=np.std(a)
    
    real=np.ones(len(a))*REAL
    mse=np.sum((a-real)**2)/len(a)   
    rmse=mse**0.5

    
    
    
    
    
    