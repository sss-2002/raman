# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 10:06:02 2018

@author: Administrator
"""

'"sgolayfilt滤波器"对光谱进行滤波处理，并保存结果'
"**********************************************"
"scipy.signal.savgol_filter滤波（参数设置）"
from scipy.signal import savgol_filter
import numpy as np


def SGfilter(Intensity, point, degree):  # 输入均为行
    Row = Intensity.shape[0]
    col = Intensity.shape[1]
    sgsmooth = np.zeros((Row, col))
    for i in range(Row):
        sgsmooth[i] = savgol_filter(Intensity[i], point, degree)
    return sgsmooth
