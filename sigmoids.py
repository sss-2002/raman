# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 19:36:53 2019

@author: Administrator
"""

import numpy as np


def sigmoid(X):
    row = X.shape[0]
    col = X.shape[1]
    s = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            m = 1 + np.exp(-float(X[i, j]))
            s[i, j] = (1.0 / m)
    return s
