# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:52:28 2018

@author: Administrator
  """

'''
  将数据标准化，均值为0 ， 方差为1 ， 可直接使用sklearn中的transform
  
'''
import numpy as np


def standardization(Datamat):
    mu = np.average(Datamat)
    sigma = np.std(Datamat)
    if sigma != 0:
        normDatamat = (Datamat - mu) / sigma
    else:
        normDatamat = Datamat - mu
    return normDatamat


def plotst(Data):
    row = Data.shape[0]
    col = Data.shape[1]
    st_Data = np.zeros((row, col))
    for i in range(row):
        st_Data[i] = standardization(Data[i])
    return st_Data
