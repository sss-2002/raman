# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 23:25:01 2019

@author: Administrator
"""
import numpy as np
import scipy.signal as signal


def MWM(arr, n=7, it=1):
    row = arr.shape[0]
    col = arr.shape[1]
    median = np.zeros((row, col))
    ns = []
    for _ in range(it):
        ns.append(n)
        n -= 2
    for i in range(row):
        median[i] = arr[i].copy()
        nn = ns.copy()
        for _ in range(it):
            n = nn.pop()
            if n > 1:
                tmp = signal.medfilt(median[i], n)
                median[i] = tmp
    return median
