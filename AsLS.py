# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 21:04:10 2019

@author: Administrator
"""
"""非对称加权惩罚最小二乘基线校准"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
#import matplotlib.pyplot as plt


def baseline_als(y, lam, p, niter=10, tol=1e-6):
    """
    改进的AsLS算法

    参数:
        y: 输入光谱 (n_samples, n_points)
        lam: 平滑系数 (典型值1e5-1e12)
        p: 非对称系数 (0-1, 典型值0.001-0.1)
        niter: 最大迭代次数
        tol: 收敛阈值

    返回:
        基线校正后的光谱
    """
    if np.any(np.isnan(y)):
        raise ValueError("输入数据包含NaN值")

    y = np.asarray(y, dtype=np.float64)
    L = y.shape[1]
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    result = np.zeros_like(y)

    for j in range(y.shape[0]):
        w = np.ones(L)
        y_curr = y[j].copy()

        for _ in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y_curr)

            # 检查收敛
            if np.max(np.abs(z - y_curr)) < tol:
                break

            w = p * (y[j] > z) + (1 - p) * (y[j] < z)
            y_curr = z

        result[j] = y[j] - z

    return result
