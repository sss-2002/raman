# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 16:01:44 2019

@author: Administrator
"""
import numpy as np
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False
def D1(sdata):
    "一阶差分"
    row = sdata.shape[0]
    col = sdata.shape[1]
    D1 = np.zeros((row , col))
    for i in range (row):
        tem = np.diff(sdata[i] ,1)
        temp = tem.tolist()
        temp.append(temp[-1])
        D1[i] = temp
    return D1
"""
wavenumbers = np.loadtxt("E:\\数据\\细胞\\wavenumbers.txt")
MCF = np.loadtxt("E:\\数据\\细胞\\MCF-7Raw.txt")
D1data = D1(MCF.T)
plt.figure()
plt.plot(D1data , label = u'一阶微分')
plt.xlim(600 ,2000)
plt.legend()
plt.show()
"""