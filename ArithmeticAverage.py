# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 23:44:20 2019

@author: Administrator
"""

import numpy as np


def MWA(arr, n=6, it=1, mode="full"):
    row = arr.shape[0]
    col = arr.shape[1]
    average = np.zeros((row, col))
    ns = []
    for _ in range(it):
        ns.append(n)
        n -= 2
    for i in range(row):
        average[i] = arr[i].copy()
        nn = ns.copy()
        for _ in range(it):
            n = nn.pop()
            if n > 1:
                tmp = np.convolve(average[i], np.ones((n,)) / n, mode=mode)
                for j in range(1, n):
                    tmp[j - 1] = tmp[j - 1] * n / j
                    tmp[-j] = tmp[-j] * n / j
                j = int(n / 2)
                k = n - j - 1
                average[i] = tmp[j:-k]
    return average
