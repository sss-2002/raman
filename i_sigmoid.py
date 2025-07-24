# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 19:36:53 2019

@author: Administrator
"""

import numpy as np


def i_sigmoid(X, maxn=10):
    row = X.shape[0]
    col = X.shape[1]
    s = np.zeros((row, col))
    for i in range(row):
        mi = np.min(X[i])
        diff = (np.max(X[i]) - mi) / maxn
        for j in range(col):
            t = (X[i, j] - mi) / diff - maxn / 2
            m = 1 + np.exp(-float(t))
            t = 1.0 / m
            s[i, j] = t * diff * maxn + mi
    return s
