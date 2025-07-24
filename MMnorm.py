# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:44:35 2019

@author: Administrator
"""

import numpy as np
def MaMinorm(Oarr):
    row = Oarr.shape[0]
    col = Oarr.shape[1]
    MMarr = np.zeros((row, col))
    permax = np.ones((1, col))
    for i in range(row):
        diff = np.max(Oarr[i]) - np.min(Oarr[i])
        if diff != 0:
            MMarr[i] = ((Oarr[i] - permax*np.min(Oarr[i]))/ diff)*10-5
        else:
            MMarr[i] = Oarr[i] - permax * np.min(Oarr[i])
    return MMarr