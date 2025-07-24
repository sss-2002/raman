# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 20:21:06 2019

@author: Administrator
"""
import numpy as np


def LPnorm(arr, ord):
    row = arr.shape[0]
    col = arr.shape[1]
    Lpdata = np.zeros((row, col))
    for i in range(row):
        Lp = np.linalg.norm(arr[i,:], ord)
        if Lp !=0 :
            Lpdata[i,:] = arr[i,:] / Lp
        else:
            Lpdata[i,:] = arr[i,:]
    return Lpdata
