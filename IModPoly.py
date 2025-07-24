# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 11:56:40 2019

@author: Administrator
"""

"I-ModPoly: improved modified multi-polynomial fit method"
import numpy as np
import copy
import matplotlib.pyplot as plt


def IModPoly(wavenumbers, originalRaman, polyorder, max_iter=100, tolerance=0.005):
    """
    改进的多项式拟合基线校正

    参数:
        wavenumbers: 拉曼位移(cm^-1)的一维数组
        originalRaman: 原始拉曼光谱，形状为(n_samples, n_points)
        polyorder: 多项式阶数
        max_iter: 最大迭代次数 (默认100)
        tolerance: 收敛容差 (默认0.005)

    返回:
        校正后的光谱，形状与originalRaman相同
    """
    row, col = originalRaman.shape
    corrected = np.zeros((row, col))

    for j in range(row):
        prev_spectrum = originalRaman[j]
        curr_spectrum = prev_spectrum.copy()
        prev_std = 0
        converged = False
        iteration = 1

        while not converged and iteration <= max_iter:
            # 多项式拟合
            coeffs = np.polyfit(wavenumbers, curr_spectrum, polyorder)
            fitted = np.polyval(coeffs, wavenumbers)
            residual = curr_spectrum - fitted
            curr_std = np.std(residual)

            # 光谱修正
            if iteration == 1:
                # 首次迭代：去除明显峰
                mask = prev_spectrum > (fitted + curr_std)
                curr_spectrum[mask] = fitted[mask] + curr_std
            else:
                # 后续迭代：重建模型
                mask = prev_spectrum < (fitted + curr_std)
                curr_spectrum = np.where(mask, prev_spectrum, fitted + curr_std)

            # 检查收敛条件
            relative_change = abs((curr_std - prev_std) / curr_std)
            converged = relative_change < tolerance

            prev_spectrum = curr_spectrum
            prev_std = curr_std
            iteration += 1

        corrected[j] = originalRaman[j] - fitted

    return corrected