# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 17:11:34 2019

@author: Administrator
"""
import numpy as np
from sklearn.linear_model import LinearRegression


def MSC(sdata):
    n = sdata.shape[0]  # 样本数量
    k = np.zeros(sdata.shape[0])
    b = np.zeros(sdata.shape[0])
 
    M = np.mean(sdata, axis=0)
 
    for i in range(n):
        y = sdata[i, :]
        y = y.reshape(-1, 1)
        M = M.reshape(-1, 1)
        model = LinearRegression()
        model.fit(M, y)
        k[i] = model.coef_
        b[i] = model.intercept_
#global spec_msc
    spec_msc = np.zeros_like(sdata)
    for i in range(n):
        bb = np.repeat(b[i], sdata.shape[1])
        kk = np.repeat(k[i], sdata.shape[1])
        temp = (sdata[i, :] - bb)/kk
        spec_msc[i, :] = temp
    return spec_msc
