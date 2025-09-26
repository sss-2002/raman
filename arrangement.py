import streamlit as st
import numpy as np
import pandas as pd
import re
import itertools
import matplotlib.pyplot as plt
import math
import os
from io import BytesIO
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import seaborn as sns
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter, medfilt
from scipy.fft import fft, ifft
from scipy.fftpack import fft as fftpack_fft, ifft as fftpack_ifft
import copy
from statsmodels.nonparametric.smoothers_lowess import lowess
import pywt
from sklearn.linear_model import LinearRegression
import scipy.signal as signal


# 基于余弦的挤压变换函数
def squashing(Data):
    row = Data.shape[0]
    col = Data.shape[1]
    sqData = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            sqData[i][j] = (1 - math.cos(Data[i][j] * math.pi)) / 2
    return sqData


# 小波线性阈值去噪函数
def waveletlinear(arr, threshold=0.3):
    row = arr.shape[0]
    col = arr.shape[1]
    datarec = np.zeros((row, col))
    w = pywt.Wavelet('db8')
    for i in range(row):
        maxlev = pywt.dwt_max_level(col, w.dec_len)
        coeffs = pywt.wavedec(arr[i], 'db8', level=maxlev)
        for j in range(0, len(coeffs)):
            coeffs[j] = pywt.threshold(coeffs[j], threshold * max(coeffs[j]))
        datarec[i] = pywt.waverec(coeffs, 'db8')
    return datarec


# 移动窗口中值滤波(MWM)函数
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


# Sigmoid函数
def sigmoid(X):
    row = X.shape[0]
    col = X.shape[1]
    s = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            m = 1 + np.exp(-float(X[i, j]))
            s[i, j] = (1.0 / m)
    return s


# 改进的i_sigmoid挤压函数
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


# 改进的i_squashing挤压函数
def i_squashing(Data):
    row = Data.shape[0]
    col = Data.shape[1]
    sqData = np.zeros((row, col))
    for i in range(row):
        mi = np.min(Data[i])
        diff = np.max(Data[i]) - mi
        for j in range(col):
            t = (Data[i, j] - mi) / diff if diff != 0 else 0
            m = (1 - math.cos(t * math.pi)) / 2
            sqData[i][j] = m * diff + mi
    return sqData


# 二阶差分(D2)函数
def D2(sdata):
    row = sdata.shape[0]
    col = sdata.shape[1]
    D2_result = np.zeros((row, col))
    for i in range(row):
        tem = np.diff(sdata[i], 2)
        temp = tem.tolist()
        temp.append(temp[-1])
        temp.append(temp[-1])
        D2_result[i] = temp
    return D2_result


# LP范数归一化函数
def LPnorm(arr, ord):
    row = arr.shape[0]
    col = arr.shape[1]
    Lpdata = np.zeros((row, col))
    for i in range(row):
        Lp = np.linalg.norm(arr[i, :], ord)
        if Lp != 0:
            Lpdata[i, :] = arr[i, :] / Lp
        else:
            Lpdata[i, :] = arr[i, :]
    return Lpdata


# MaMinorm归一化函数
def MaMinorm(Oarr):
    row = Oarr.shape[0]
    col = Oarr.shape[1]
    MMarr = np.zeros((row, col))
    permax = np.ones((1, col))
    for i in range(row):
        diff = np.max(Oarr[i]) - np.min(Oarr[i])
        if diff != 0:
            MMarr[i] = ((Oarr[i] - permax * np.min(Oarr[i])) / diff) * 10 - 5
        else:
            MMarr[i] = Oarr[i] - permax * np.min(Oarr[i])
    return MMarr


# 标准化函数（均值为0，方差为1）
def standardization(Datamat):
    mu = np.average(Datamat)
    sigma = np.std(Datamat)
    if sigma != 0:
        normDatamat = (Datamat - mu) / sigma
    else:
        normDatamat = Datamat - mu
    return normDatamat


# 逐行标准化函数
def plotst(Data):
    row = Data.shape[0]
    col = Data.shape[1]
    st_Data = np.zeros((row, col))
    for i in range(row):
        st_Data[i] = standardization(Data[i])
    return st_Data


# Smfft傅里叶滤波函数
def Smfft(arr, row_e=51):
    row = arr.shape[0]
    col = arr.shape[1]
    fftresult = np.zeros((row, col))
    for i in range(row):
        sfft = fftpack_fft(arr[i])
        row_s = len(arr[i])
        sfftn = copy.deepcopy(sfft)
        sfftn[row_e:row_s - row_e] = 0
        result = fftpack_ifft(sfftn)
        real_r = np.real(result)
        fftresult[i] = real_r
    return fftresult


# 多元散射校正(MSC)函数
def MSC(sdata):
    n = sdata.shape[0]
    k = np.zeros(sdata.shape[0])
    b = np.zeros(sdata.shape[0])
    M = np.mean(sdata, axis=0)

    for i in range(n):
        y = sdata[i, :].reshape(-1, 1)
        M_reshaped = M.reshape(-1, 1)
        model = LinearRegression()
        model.fit(M_reshaped, y)
        k[i] = model.coef_
        b[i] = model.intercept_

    spec_msc = np.zeros_like(sdata)
    for i in range(n):
        bb = np.repeat(b[i], sdata.shape[1])
        kk = np.repeat(k[i], sdata.shape[1])
        spec_msc[i, :] = (sdata[i, :] - bb) / kk

    return spec_msc


# 卡尔曼滤波算法
def Kalman(z, R):
    n_iter = len(z)
    sz = (n_iter,)
    Q = 1e-5

    xhat = np.zeros(sz)
    P = np.zeros(sz)
    xhatminus = np.zeros(sz)
    Pminus = np.zeros(sz)
    K = np.zeros(sz)

    xhat[0] = 0.0
    P[0] = 1.0

    for k in range(1, n_iter):
        xhatminus[k] = xhat[k - 1]
        Pminus[k] = P[k - 1] + Q

        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]

    return xhat


# 多维数据卡尔曼滤波
def KalmanF(xd, R):
    row = xd.shape[0]
    col = xd.shape[1]
    Fxd = np.zeros((row, col))
    for i in range(row):
        Fxd[i] = Kalman(xd[i], R)
    return Fxd


# 改进的多项式拟合基线校正(IModPoly)
def IModPoly(wavenumbers, originalRaman, polyorder, max_iter=100, tolerance=0.005):
    row, col = originalRaman.shape
    corrected = np.zeros((row, col))

    for j in range(row):
        prev_spectrum = originalRaman[j]
        curr_spectrum = prev_spectrum.copy()
        prev_std = 0
        converged = False
        iteration = 1

        while not converged and iteration <= max_iter:
            coeffs = np.polyfit(wavenumbers, curr_spectrum, polyorder)
            fitted = np.polyval(coeffs, wavenumbers)
            residual = curr_spectrum - fitted
            curr_std = np.std(residual)

            if iteration == 1:
                mask = prev_spectrum > (fitted + curr_std)
                curr_spectrum[mask] = fitted[mask] + curr_std
            else:
                mask = prev_spectrum < (fitted + curr_std)
                curr_spectrum = np.where(mask, prev_spectrum, fitted + curr_std)

            relative_change = abs((curr_std - prev_std) / curr_std) if curr_std != 0 else 0
            converged = relative_change < tolerance

            prev_spectrum = curr_spectrum
            prev_std = curr_std
            iteration += 1

        corrected[j] = originalRaman[j] - fitted

    return corrected


# 移动窗口平均（MWA）滤波
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


# 改进的非对称加权惩罚最小二乘基线校准(AsLS)
def baseline_als(y, lam, p, niter=10, tol=1e-6):
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

            if np.max(np.abs(z - y_curr)) < tol:
                break

            w = p * (y[j] > z) + (1 - p) * (y[j] < z)
            y_curr = z

        result[j] = y[j] - z

    return result


# 动态时间规整(DTW)算法
class DTW:
    def __init__(self, dist_method='euclidean'):
        self.dist_method = dist_method

    def distance(self, x, y):
        if self.dist_method == 'euclidean':
            return np.linalg.norm(x - y)
        elif self.dist_method == 'manhattan':
            return np.sum(np.abs(x - y))
        else:
            return np.linalg.norm(x - y)

    def __call__(self, reference, query):
        n = len(reference)
        m = len(query)
        dtw_matrix = np.zeros((n + 1, m + 1))
        dtw_matrix[:, :] = np.inf
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = self.distance(reference[i - 1], query[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],
                    dtw_matrix[i, j - 1],
                    dtw_matrix[i - 1, j - 1]
                )

        i, j = n, m
        path = []
        while i > 0 or j > 0:
            path.append((i - 1, j - 1))
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                min_val = min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
                if min_val == dtw_matrix[i - 1, j - 1]:
                    i -= 1
                    j -= 1
                elif min_val == dtw_matrix[i - 1, j]:
                    i -= 1
                else:
                    j -= 1

        return path[::-1], dtw_matrix[n, m]


# 传统逻辑函数（用于对比）
def squashing_legacy(x):
    return 1 / (1 + np.exp(-x))


# Savitzky-Golay滤波器实现
def SGfilter(Intensity, point, degree):
    Row = Intensity.shape[0]
    col = Intensity.shape[1]
    sgsmooth = np.zeros((Row, col))
    for i in range(Row):
        sgsmooth[i] = savgol_filter(Intensity[i], point, degree)
    return sgsmooth


# 多项式拟合基线校正
def polynomial_fit(wavenumbers, spectra, polyorder):
    baseline = np.zeros_like(spectra)
    for i in range(spectra.shape[1]):
        coeffs = np.polyfit(wavenumbers, spectra[:, i], deg=polyorder)
        baseline[:, i] = np.polyval(coeffs, wavenumbers)
    return spectra - baseline


# ModPoly基线校正
def modpoly(wavenumbers, spectra, k):
    baseline = np.zeros_like(spectra)
    n_points = len(wavenumbers)
    for i in range(spectra.shape[1]):
        y = spectra[:, i].copy()
        for _ in range(k):
            coeffs = np.polyfit(wavenumbers, y, deg=5)
            fitted = np.polyval(coeffs, wavenumbers)
            mask = y < fitted
            y[~mask] = fitted[~mask]
        baseline[:, i] = y
    return spectra - baseline


# PLS基线校正
def pls(spectra, lam):
    n_points = spectra.shape[0]
    baseline = np.zeros_like(spectra)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n_points, n_points - 2))
    D = lam * D.dot(D.transpose())
    for i in range(spectra.shape[1]):
        y = spectra[:, i]
        A = sparse.eye(n_points) + D
        baseline[:, i] = spsolve(A, y)
    return spectra - baseline


# airPLS基线校正
def airpls(spectra, lam, max_iter=15, threshold=0.001):
    n_points = spectra.shape[0]
    baseline = np.zeros_like(spectra)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n_points, n_points - 2))
    D = lam * D.dot(D.transpose())
    for i in range(spectra.shape[1]):
        y = spectra[:, i]
        w = np.ones(n_points)
        baseline_i = np.zeros(n_points)
        for j in range(max_iter):
            W = sparse.diags(w, 0)
            Z = W + D
            b = spsolve(Z, W * y)
            d = y - b
            neg_mask = d < 0
            w[neg_mask] = np.exp(j * np.abs(d[neg_mask]) / np.std(d[neg_mask]))
            w[~neg_mask] = 0
            if j > 0:
                diff = np.sum(np.abs(b - baseline_i)) / np.sum(np.abs(baseline_i)) if np.sum(np.abs(baseline_i)) > 0 else 0
                if diff < threshold:
                    break
            baseline_i = b
        baseline[:, i] = baseline_i
    return spectra - baseline


# DTW挤压算法
def dtw_squashing(x, l, k1, k2):
    n_samples, n_features = x.shape
    result = np.zeros_like(x)
    reference = np.mean(x, axis=1)
    dtw = DTW()

    for i in range(n_features):
        spectrum = x[:, i]
        path, cost = dtw(reference, spectrum)
        squashed = np.zeros_like(spectrum)
        for ref_idx, spec_idx in path:
            squashed[ref_idx] += spectrum[spec_idx]
        unique_ref_indices = np.unique([p[0] for p in path])
        for idx in unique_ref_indices:
            count = sum(1 for p in path if p[0] == idx)
            squashed[idx] /= count
        if k1 == "T":
            max_slope = l
            for j in range(1, len(path)):
                ref_diff = path[j][0] - path[j - 1][0]
                spec_diff = path[j][1] - path[j - 1][1]
                if ref_diff != 0:
                    slope = abs(spec_diff / ref_diff)
                    if slope > max_slope:
                        squashed[path[j][0]] = (squashed[path[j][0]] + squashed[path[j - 1][0]]) / 2
        if k2 == "T":
            ref_map_count = {}
            for ref_idx, _ in path:
                ref_map_count[ref_idx] = ref_map_count.get(ref_idx, 0) + 1
            for ref_idx, count in ref_map_count.items():
                if count > l:
                    window = min(l, len(spectrum))
                    start = max(0, ref_idx - window // 2)
                    end = min(n_samples, ref_idx + window // 2 + 1)
                    squashed[ref_idx] = np.mean(spectrum[start:end])
        if l > 1:
            for j in range(n_samples):
                start = max(0, j - l)
                end = min(n_samples, j + l + 1)
                squashed[j] = np.mean(squashed[start:end])
        result[:, i] = squashed
    return result


# 生成算法排列组合
def generate_permutations(algorithms):
    algorithm_list = [
        (1, "基线校准", algorithms['baseline']),
        (2, "缩放", algorithms['scaling']),
        (3, "滤波", algorithms['filtering']),
        (4, "挤压", algorithms['squashing'])
    ]

    all_permutations = []
    all_permutations.append([])

    # 1种算法排列
    for algo in algorithm_list:
        if algo[2] != "无":
            all_permutations.append([algo])

    # 2种算法排列
    for perm in itertools.permutations(algorithm_list, 2):
        if perm[0][2] != "无" and perm[1][2] != "无":
            all_permutations.append(list(perm))

    # 3种算法排列
    for perm in itertools.permutations(algorithm_list, 3):
        if perm[0][2] != "无" and perm[1][2] != "无" and perm[2][2] != "无":
            all_permutations.append(list(perm))

    # 4种算法排列
    for perm in itertools.permutations(algorithm_list, 4):
        if (perm[0][2] != "无" and perm[1][2] != "无" and
                perm[2][2] != "无" and perm[3][2] != "无"):
            all_permutations.append(list(perm))

    formatted_perms = []
    for i, perm in enumerate(all_permutations):
        perm_dict = {
            "name": "",
            "order": [],
            "details": perm,
            "count": len(perm),
            "first_step_type": "未知"
        }

        if not perm:
            perm_dict["name"] = "无预处理（原始光谱）"
            perm_dict["first_step_type"] = "无预处理"
        else:
            first_step_type = perm[0][1] if perm and len(perm) > 0 else "未知"
            perm_dict["first_step_type"] = first_step_type

            perm_details = []
            for step in perm:
                perm_details.append(f"{step[0]}.{step[1]}({step[2]})")
            perm_dict["name"] = " → ".join(perm_details)
            perm_dict["order"] = [step[0] for step in perm]

        formatted_perms.append(perm_dict)

    return formatted_perms


# K近邻分类算法
def knn_classify(train_data, train_labels, test_data, k=5):
    train_data = train_data.T
    test_data = test_data.T

    predictions = []
    for test_sample in test_data:
        distances = np.sqrt(np.sum((train_data - test_sample) **2, axis=1))
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [train_labels[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        predictions.append(most_common)
    return np.array(predictions)


# 预处理类
class Preprocessor:
    def __init__(self):
        self.BASELINE_ALGORITHMS = {
            "SD": self._sd_baseline,
            "FD": self._fd_baseline,
            "多项式拟合": polynomial_fit,
            "ModPoly": modpoly,
            "I-ModPoly": IModPoly,
            "PLS": pls,
            "AsLS": baseline_als,
            "airPLS": airpls,
            "二阶差分(D2)": self.d2
        }
        self.FILTERING_ALGORITHMS = {
            "Savitzky-Golay": self.savitzky_golay,
            "sgolayfilt滤波器": self.sgolay_filter_custom,
            "中值滤波(MF)": self.median_filter,
            "移动平均(MAF)": self.moving_average,
            "MWA（移动窗口平均）": self.mwa_filter,
            "MWM（移动窗口中值）": self.mwm_filter,
            "卡尔曼滤波": self.kalman_filter,
            "Lowess": self.lowess_filter,
            "FFT": self.fft_filter,
            "Smfft傅里叶滤波": self.smfft_filter,
            "小波变换(DWT)": self.wavelet_filter,
            "小波线性阈值去噪": self.wavelet_linear
        }

        self.SCALING_ALGORITHMS = {
            "Peak-Norm": self.peak_norm,
            "SNV": self.snv,
            "MSC": self.msc,
            "M-M-Norm": self.mm_norm,
            "L-范数": self.l_norm,
            "Ma-Minorm": self.ma_minorm,
            "标准化(均值0，方差1)": self.standardize
        }

        self.SQUASHING_ALGORITHMS = {
            "Sigmoid挤压": sigmoid,
            "改进的Sigmoid挤压": i_sigmoid,
            "逻辑函数": squashing_legacy,
            "余弦挤压(squashing)": squashing,
            "改进的逻辑函数": i_squashing,
            "DTW挤压": dtw_squashing
        }

    def process(self, wavenumbers, data,
                baseline_method="无", baseline_params=None,
                squashing_method="无", squashing_params=None,
                filtering_method="无", filtering_params=None,
                scaling_method="无", scaling_params=None,
                algorithm_order=None):
        if baseline_params is None:
            baseline_params = {}
        if squashing_params is None:
            squashing_params = {}
        if filtering_params is None:
            filtering_params = {}
        if scaling_params is None:
            scaling_params = {}

        if algorithm_order is not None and len(algorithm_order) == 0:
            return data.copy(), ["无预处理（原始光谱）"]

        y_processed = data.copy()
        method_name = []

        if algorithm_order is not None and len(algorithm_order) > 0:
            step_mapping = {
                1: ("baseline", baseline_method, baseline_params),
                2: ("scaling", scaling_method, scaling_params),
                3: ("filtering", filtering_method, filtering_params),
                4: ("squashing", squashing_method, squashing_params)
            }
            steps = [step_mapping[order] for order in algorithm_order]
        else:
            steps = []
            if baseline_method != "无":
                steps.append(("baseline", baseline_method, baseline_params))
            if squashing_method != "无":
                steps.append(("squashing", squashing_method, squashing_params))
            if filtering_method != "无":
                steps.append(("filtering", filtering_method, filtering_params))
            if scaling_method != "无":
                steps.append(("scaling", scaling_method, scaling_params))

        for step_type, method, params in steps:
            if method == "无":
                continue

            try:
                if step_type == "baseline":
                    algorithm_func = self.BASELINE_ALGORITHMS[method]
                    if method in ["多项式拟合", "ModPoly", "I-ModPoly"]:
                        y_processed = algorithm_func(wavenumbers, y_processed,** params)
                    elif method in ["PLS"]:
                        y_processed = algorithm_func(y_processed, **params)
                    elif method == "AsLS":
                        y_processed = algorithm_func(y_processed,** params)
                    elif method == "airPLS":
                        y_processed = algorithm_func(y_processed, **params)
                    elif method == "二阶差分(D2)":
                        y_processed = algorithm_func(y_processed)
                    else:
                        y_processed = algorithm_func(y_processed)
                    method_name.append(f"{method}({', '.join([f'{k}={v}' for k, v in params.items()])})")

                elif step_type == "squashing":
                    algorithm_func = self.SQUASHING_ALGORITHMS[method]
                    if method == "改进的Sigmoid挤压":
                        maxn = params.get("maxn", 10)
                        y_processed = algorithm_func(y_processed, maxn=maxn)
                        method_name.append(f"{method}(maxn={maxn})")
                    elif method == "改进的逻辑函数":
                        y_processed = algorithm_func(y_processed)
                        method_name.append(f"{method}")
                    elif method == "DTW挤压":
                        l = params.get("l", 1)
                        k1 = params.get("k1", "T")
                        k2 = params.get("k2", "T")
                        y_processed = algorithm_func(y_processed, l=l, k1=k1, k2=k2)
                        method_name.append(f"DTW挤压(l={l}, k1={k1}, k2={k2})")
                    elif method == "Sigmoid挤压":
                        y_processed = algorithm_func(y_processed)
                        method_name.append(f"{method}")
                    elif method == "余弦挤压(squashing)":
                        y_processed = algorithm_func(y_processed)
                        method_name.append(f"{method}")
                    else:
                        y_processed = algorithm_func(y_processed)
                        method_name.append(method)

                elif step_type == "filtering":
                    algorithm_func = self.FILTERING_ALGORITHMS[method]
                    y_processed = algorithm_func(y_processed,** params)
                    params_str = ', '.join([f'{k}={v}' for k, v in params.items()])
                    method_name.append(f"{method}({params_str})")

                    if method == "小波线性阈值去噪":
                        threshold = params.get("threshold", 0.3)
                        method_name[-1] = f"{method}(threshold={threshold})"

                elif step_type == "scaling":
                    algorithm_func = self.SCALING_ALGORITHMS[method]
                    y_processed = algorithm_func(y_processed, **params)
                    params_str = ', '.join([f'{k}={v}' for k, v in params.items()])
                    method_name.append(f"{method}({params_str})")

            except Exception as e:
                raise ValueError(f"{step_type}处理失败: {str(e)}")

        return y_processed, method_name

    def _sd_baseline(self, spectra):
        return spectra - np.min(spectra, axis=0)

    def _fd_baseline(self, spectra):
        return spectra - np.percentile(spectra, 5, axis=0)

    # 滤波算法实现
    def savitzky_golay(self, spectra, k, w):
        return savgol_filter(spectra, w, k, axis=0)

    def sgolay_filter_custom(self, spectra, point, degree):
        if spectra.shape[0] < spectra.shape[1]:
            filtered = SGfilter(spectra.T, point, degree)
            return filtered.T
        else:
            return SGfilter(spectra, point, degree)

    def median_filter(self, spectra, k, w):
        return medfilt(spectra, kernel_size=(w, 1))

    def moving_average(self, spectra, k, w):
        kernel = np.ones(w) / w
        return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, spectra)

    def mwa_filter(self, spectra, n=6, it=1, mode="full"):
        return MWA(spectra, n=n, it=it, mode=mode)

    def mwm_filter(self, spectra, n=7, it=1):
        if spectra.shape[0] < spectra.shape[1]:
            filtered = MWM(spectra.T, n=n, it=it)
            return filtered.T
        else:
            return MWM(spectra, n=n, it=it)

    def kalman_filter(self, spectra, R=0.1):
        return KalmanF(spectra, R)

    def lowess_filter(self, spectra, frac):
        result = np.zeros_like(spectra)
        for i in range(spectra.shape[1]):
            smoothed = lowess(spectra[:, i], np.arange(len(spectra)), frac=frac, it=0)
            result[:, i] = smoothed[:, 1]
        return result

    def fft_filter(self, spectra, cutoff):
        fft_result = fft(spectra, axis=0)
        frequencies = np.fft.fftfreq(spectra.shape[0])
        filter_mask = np.abs(frequencies) < cutoff
        fft_result[~filter_mask, :] = 0
        return np.real(ifft(fft_result, axis=0))

    def smfft_filter(self, spectra, row_e=51):
        if spectra.shape[0] < spectra.shape[1]:
            filtered = Smfft(spectra.T, row_e=row_e)
            return filtered.T
        else:
            return Smfft(spectra, row_e=row_e)

    def wavelet_filter(self, spectra, threshold):
        coeffs = pywt.wavedec(spectra, 'db4', axis=0)
        coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
        return pywt.waverec(coeffs, 'db4', axis=0)

    def wavelet_linear(self, spectra, threshold=0.3):
        if spectra.shape[0] < spectra.shape[1]:
            filtered = waveletlinear(spectra.T, threshold=threshold)
            return filtered.T
        else:
            return waveletlinear(spectra, threshold=threshold)

    # 缩放算法实现
    def peak_norm(self, spectra):
        return spectra / np.max(spectra, axis=0)

    def snv(self, spectra):
        mean = np.mean(spectra, axis=0)
        std = np.std(spectra, axis=0)
        return (spectra - mean) / std

    def msc(self, spectra):
        if spectra.shape[0] < spectra.shape[1]:
            corrected = MSC(spectra.T)
            return corrected.T
        else:
            return MSC(spectra)

    def mm_norm(self, spectra):
        min_vals = np.min(spectra, axis=0)
        max_vals = np.max(spectra, axis=0)
        return (spectra - min_vals) / (max_vals - min_vals)

    def l_norm(self, spectra, p):
        if p == "无穷大":
            return LPnorm(spectra, np.inf)
        else:
            p_val = float(p)
            return LPnorm(spectra, p_val)

    def ma_minorm(self, spectra):
        return MaMinorm(spectra)

    def standardize(self, spectra):
        if spectra.shape[0] < spectra.shape[1]:
            standardized = plotst(spectra.T)
            return standardized.T
        else:
            return plotst(spectra)

    # 二阶差分方法封装
    def d2(self, spectra):
        if spectra.shape[0] < spectra.shape[1]:
            diff_result = D2(spectra.T)
            return diff_result.T
        else:
            return D2(spectra)


# 文件处理类（支持文件夹上传）
class FileHandler:
    def load_data_from_folder(self, uploaded_files, wavenumber_filename="wavenumbers.txt"):
        """从上传的文件夹中加载数据，假设文件夹包含波数文件和多个光谱文件"""
        try:
            # 分离波数文件和光谱文件
            wavenumber_file = None
            spectrum_files = []
            
            for file in uploaded_files:
                if file.name == wavenumber_filename:
                    wavenumber_file = file
                elif file.name.lower().endswith('.txt'):
                    spectrum_files.append(file)
            
            if not wavenumber_file:
                raise ValueError(f"未找到波数文件: {wavenumber_filename}")
                
            if not spectrum_files:
                raise ValueError("未找到光谱数据文件")
            
            # 读取波数文件
            wavenumber_content = wavenumber_file.getvalue().decode("utf-8", errors="ignore")
            wavenumbers = np.array([float(num) for num in re.findall(r"-?\d+(?:\.\d+)?", wavenumber_content)]).ravel()
            
            # 读取所有光谱文件
            spectra = []
            for file in spectrum_files:
                content = file.getvalue().decode("utf-8", errors="ignore")
                numbers = list(map(float, re.findall(r"-?\d+(?:\.\d+)?", content)))
                spectra.append(numbers)
            
            # 确保所有光谱长度一致
            min_length = min(len(s) for s in spectra)
            wavenumbers = wavenumbers[:min_length]
            
            # 调整所有光谱长度
            adjusted_spectra = []
            for s in spectra:
                if len(s) > min_length:
                    adjusted_spectra.append(s[:min_length])
                else:
                    # 如果光谱太短，用0填充
                    padded = np.pad(s, (0, min_length - len(s)), mode='constant')
                    adjusted_spectra.append(padded.tolist())
            
            # 转换为numpy数组并转置为 (数据点, 光谱数) 格式
            data = np.array(adjusted_spectra).T
            return wavenumbers, data, len(spectrum_files)
            
        except Exception as e:
            raise ValueError(f"文件夹解析错误: {str(e)}")

    def export_data(self, filename, data):
        """导出预处理后的数据"""
        with open(filename, "w", encoding="utf-8") as f:
            for line in data.T:
                f.write("\t".join(map(str, line)) + "\n")


# 主函数
def main():
    # 初始化Session State
    if 'show_arrangements' not in st.session_state:
        st.session_state.show_arrangements = False

    test_states = {
        'k_value': 5,
        'test_results': None,
        'labels': None,
        'train_indices': None,
        'test_indices': None,
        'uploaded_folder_files': None
    }

    other_states = {
        'raw_data': None,
        'processed_data': None,
        'peaks': None,
        'train_test_split_ratio': 0.8,
        'arrangement_results': [],
        'selected_arrangement': None,
        'arrangement_details': {},
        'algorithm_permutations': [],
        'current_algorithms': {},
        'filtered_perms': [],
        'selected_perm_idx': 0
    }

    all_states = {** test_states, **other_states}
    for key, value in all_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # 页面配置
    st.set_page_config(layout="wide", page_icon="🔬", page_title="排列预处理模型")
    st.markdown("""
        <style>
        body {font-size: 0.85rem !important;}
        .css-1v0mbdj {padding: 0.5rem 1rem !important;}
        .css-1d391kg {padding: 0.3rem 0 !important;}
        .css-1x8cf1d {line-height: 1.2 !important;}
        .css-12ttj6m {margin-bottom: 0.5rem !important;}
        .css-1n543e5 {height: 220px !important;}
        .css-1b3298e {gap: 0.5rem !important;}
        .css-16huue1 {padding: 0.3rem 0.8rem !important;}
        </style>
    """, unsafe_allow_html=True)

    st.title("🌌 排列预处理模型")

    # 创建处理器实例
    file_handler = FileHandler()
    preprocessor = Preprocessor()

    # 三列布局
    col_left, col_mid, col_right = st.columns([1.2, 2.8, 1.1])

    # 左侧：数据管理（文件夹上传）
    with col_left:
        with st.expander("📁 数据管理", expanded=True):
            # 文件夹上传组件
            st.subheader("光谱数据文件夹", divider="gray")
            uploaded_files = st.file_uploader(
                "上传包含光谱数据的文件夹（请先压缩为ZIP）",
                type="zip",
                accept_multiple_files=False,
                key="folder_upload",
                help="包含波数文件(wavenumbers.txt)和多个光谱数据文件的ZIP压缩包"
            )

            # 波数文件名设置
            wavenumber_filename = st.text_input(
                "波数文件名",
                "wavenumbers.txt",
                key="wavenumber_filename",
                help="文件夹中包含波数数据的文件名"
            )

            # 保存上传的文件引用
            if uploaded_files is not None:
                st.session_state.uploaded_folder_files = uploaded_files
                st.success(f"✅ 已上传文件夹: {uploaded_files.name}")

            # 样本标签输入
            st.subheader("样本标签", divider="gray")
            num_classes = st.number_input("类别数量", min_value=1, value=2, step=1, key="num_cls")
            labels_input = st.text_input(
                "标签（逗号分隔，与光谱文件顺序一致）",
                placeholder="例：0,0,1,1",
                key="labels_in"
            )

            # 训练测试比例
            train_test_ratio = st.slider(
                "训练集比例",
                min_value=0.1,
                max_value=0.9,
                value=0.8,
                step=0.1,
                format="%.1f",
                key="train_ratio"
            )
            st.session_state.train_test_split_ratio = train_test_ratio

            # 加载数据按钮
            if st.button("📥 加载数据", type="primary", key="load_data_btn"):
                if st.session_state.uploaded_folder_files is None:
                    st.warning("⚠️ 请先上传包含光谱数据的ZIP文件夹")
                    return

                try:
                    # 解压并加载数据
                    import zipfile
                    
                    # 创建临时文件夹
                    import tempfile
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # 解压ZIP文件
                        with zipfile.ZipFile(st.session_state.uploaded_folder_files, 'r') as zip_ref:
                            zip_ref.extractall(temp_dir)
                            
                        # 获取所有文件
                        extracted_files = []
                        for root, dirs, files in os.walk(temp_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                # 读取文件内容
                                with open(file_path, 'rb') as f:
                                    file_content = BytesIO(f.read())
                                    file_content.name = file
                                    extracted_files.append(file_content)
                        
                        # 加载数据
                        wavenumbers, y, num_spectra = file_handler.load_data_from_folder(
                            extracted_files,
                            wavenumber_filename=wavenumber_filename
                        )
                        
                        st.session_state.raw_data = (wavenumbers, y)

                        # 标签处理
                        if labels_input:
                            try:
                                labels = np.array([int(l.strip()) for l in labels_input.split(',')])
                                n_spectra = y.shape[1]
                                if len(labels) == n_spectra:
                                    st.session_state.labels = labels
                                    n_samples = len(labels)
                                    train_size = int(n_samples * train_test_ratio)
                                    indices = np.random.permutation(n_samples)
                                    st.session_state.train_indices = indices[:train_size]
                                    st.session_state.test_indices = indices[train_size:]
                                    st.success(f"✅ 数据加载成功：{n_spectra}条光谱，{len(np.unique(labels))}类")
                                else:
                                    st.warning(f"⚠️ 标签数({len(labels)})≠光谱数({n_spectra})")
                                    st.session_state.labels = None
                            except Exception as e:
                                st.warning(f"⚠️ 标签格式错误: {str(e)}")
                                st.session_state.labels = None
                        else:
                            n_spectra = y.shape[1]
                            st.success(f"✅ 数据加载成功：{n_spectra}条光谱，{len(wavenumbers)}个数据点")
                            st.warning("⚠️ 请输入样本标签以进行分类测试")
                except Exception as e:
                    st.error(f"❌ 文件加载失败: {str(e)}")

        # 系统信息显示
        if st.session_state.get('raw_data'):
            wavenumbers, y = st.session_state.raw_data
            n_spectra = y.shape[1]
            n_points = y.shape[0]
            st.info(f"📊 数据维度: {n_spectra}条光谱 × {n_points}个数据点")
            st.info(f"🔢 训练集:{train_test_ratio:.1f} | 测试集:{1-train_test_ratio:.1f}")
            if st.session_state.get('labels') is not None:
                class_counts = np.bincount(st.session_state.labels)
                st.info(f"🏷️ 类别分布: {', '.join([f'类{i}:{count}个' for i, count in enumerate(class_counts) if count>0])}")
            if st.session_state.get('process_method'):
                st.success(f"🛠️ 处理流程: {st.session_state.process_method}")

        # 使用指南
        with st.expander("ℹ️ 使用指南", expanded=False):
            st.markdown("""
            1. 将光谱数据文件夹压缩为ZIP格式，需包含：
               - 波数文件（默认名为wavenumbers.txt）
               - 多个光谱数据文件（TXT格式）
            2. 上传ZIP文件并点击"加载数据"按钮
            3. 设置样本标签（与光谱文件顺序一致）
            4. 右侧选择预处理方法并应用排列方案
            5. 选择k值后点击"测试"
            6. 中间查看结果并导出
            """)

    # 中间：光谱可视化与结果导出
    with col_mid:
        st.subheader("📈 光谱可视化", divider="gray")

        # 原始光谱显示
        st.subheader("原始光谱", divider="gray")
        spec_cols = st.columns(2)
        with spec_cols[0]:
            if st.session_state.get('raw_data'):
                wavenumbers, y = st.session_state.raw_data
                idx1 = 0 if y.shape[1] > 0 else 0
                raw_data1 = pd.DataFrame({"原始光谱1": y[:, idx1]}, index=wavenumbers)
                st.line_chart(raw_data1, height=200)
            else:
                st.markdown('<div style="border:1px dashed #ccc; height:200px; display:flex; align-items:center; justify-content:center;">等待加载原始数据</div>', unsafe_allow_html=True)

        with spec_cols[1]:
            if st.session_state.get('raw_data') and y.shape[1] > 1:
                idx2 = 1
                raw_data2 = pd.DataFrame({"原始光谱2": y[:, idx2]}, index=wavenumbers)
                st.line_chart(raw_data2, height=200)
            elif st.session_state.get('raw_data'):
                st.markdown('<div style="border:1px dashed #ccc; height:200px; display:flex; align-items:center; justify-content:center;">仅1条原始光谱</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="border:1px dashed #ccc; height:200px; display:flex; align-items:center; justify-content:center;">等待加载原始数据</div>', unsafe_allow_html=True)

        # 更多原始光谱
        if st.session_state.get('raw_data') and y.shape[1] > 2:
            with st.expander("查看更多原始光谱", expanded=False):
                more_spec = st.columns(2)
                for i in range(2, min(y.shape[1], 6), 2):
                    with more_spec[0]:
                        if i < y.shape[1]:
                            data = pd.DataFrame({f"原始光谱{i+1}": y[:, i]}, index=wavenumbers)
                            st.line_chart(data, height=150)
                    with more_spec[1]:
                        if i+1 < y.shape[1]:
                            data = pd.DataFrame({f"原始光谱{i+2}": y[:, i+1]}, index=wavenumbers)
                            st.line_chart(data, height=150)

        # 预处理结果展示
        if st.session_state.get('selected_arrangement'):
            st.subheader("🔍 预处理结果", divider="gray")
            selected_arr = st.session_state.selected_arrangement
            arr_data = st.session_state.arrangement_details[selected_arr]['data']
            arr_method = st.session_state.arrangement_details[selected_arr]['method']
            arr_order = st.session_state.arrangement_details[selected_arr].get('order', [])

            st.caption(f"处理方法: {arr_method} | 执行顺序: {arr_order if arr_order else '无预处理'}")

            # 预处理后光谱
            st.subheader("预处理后光谱", divider="gray")
            proc_cols = st.columns(2)
            with proc_cols[0]:
                idx1 = 0 if arr_data.shape[1] > 0 else 0
                proc_data1 = pd.DataFrame({"预处理后1": arr_data[:, idx1]}, index=wavenumbers)
                st.line_chart(proc_data1, height=200)
            with proc_cols[1]:
                if arr_data.shape[1] > 1:
                    idx2 = 1
                    proc_data2 = pd.DataFrame({"预处理后2": arr_data[:, idx2]}, index=wavenumbers)
                    st.line_chart(proc_data2, height=200)
                else:
                    st.markdown('<div style="border:1px dashed #ccc; height:200px; display:flex; align-items:center; justify-content:center;">仅1条预处理光谱</div>', unsafe_allow_html=True)

            # k值曲线
            if arr_order:
                st.subheader("k值曲线", divider="gray")
                k_cols = st.columns(2)
                with k_cols[0]:
                    k_vals1 = np.abs(arr_data[:, 0] / (y[:, 0] + 1e-8)) if y.shape[1] > 0 else np.array([])
                    k_data1 = pd.DataFrame({"k值1": k_vals1}, index=wavenumbers)
                    st.line_chart(k_data1, height=200)
                with k_cols[1]:
                    if y.shape[1] > 1:
                        k_vals2 = np.abs(arr_data[:, 1] / (y[:, 1] + 1e-8))
                        k_data2 = pd.DataFrame({"k值2": k_vals2}, index=wavenumbers)
                        st.line_chart(k_data2, height=200)
                    else:
                        st.markdown('<div style="border:1px dashed #ccc; height:200px; display:flex; align-items:center; justify-content:center;">仅1条k值曲线</div>', unsafe_allow_html=True)
            else:
                st.info("ℹ️ 无预处理（原始光谱），不显示k值曲线")

            # 原始与预处理对比
            st.subheader("原始vs预处理对比", divider="gray")
            comp_cols = st.columns(2)
            with comp_cols[0]:
                if y.shape[1] > 0:
                    comp_data1 = pd.DataFrame({
                        "原始": y[:, 0],
                        "预处理后": arr_data[:, 0]
                    }, index=wavenumbers)
                    st.line_chart(comp_data1, height=200)
            with comp_cols[1]:
                if y.shape[1] > 1:
                    comp_data2 = pd.DataFrame({
                        "原始": y[:, 1],
                        "预处理后": arr_data[:, 1]
                    }, index=wavenumbers)
                    st.line_chart(comp_data2, height=200)
                else:
                    st.markdown('<div style="border:1px dashed #ccc; height:200px; display:flex; align-items:center; justify-content:center;">仅1条对比曲线</div>', unsafe_allow_html=True)

            # 测试结果
            if st.session_state.get('test_results') is not None:
                st.subheader("📊 分类测试结果", divider="gray")
                results = st.session_state.test_results

                # 指标显示
                metrics_cols = st.columns(2)
                with metrics_cols[0]:
                    st.metric("准确率", f"{results['accuracy']:.4f}", delta=None)
                with metrics_cols[1]:
                    st.metric("卡帕系数", f"{results['kappa']:.4f}", delta=None)

                # 混淆矩阵
                st.subheader("混淆矩阵", divider="gray")
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={"size": 8})
                ax.set_xlabel('预测标签', fontsize=8)
                ax.set_ylabel('真实标签', fontsize=8)
                ax.set_title('混淆矩阵', fontsize=10)
                plt.xticks(fontsize=7)
                plt.yticks(fontsize=7)
                st.pyplot(fig, use_container_width=True)
        else:
            st.info("ℹ️ 请在右侧选择预处理方法并应用排列方案")

        # 结果导出
        if st.session_state.arrangement_results or st.session_state.get('processed_data'):
            st.subheader("💾 结果导出", divider="gray")
            export_cols = st.columns([3, 1])
            with export_cols[0]:
                export_name = st.text_input("导出文件名", "processed_spectra.txt", key="export_name")
            with export_cols[1]:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("导出", type="secondary", key="export_btn"):
                    try:
                        if st.session_state.selected_arrangement:
                            arr_data = st.session_state.arrangement_details[st.session_state.selected_arrangement]['data']
                            file_handler.export_data(export_name, arr_data)
                        else:
                            wavenumbers, y_processed = st.session_state.processed_data
                            file_handler.export_data(export_name, y_processed)
                        st.success(f"✅ 已导出到 {export_name}")
                    except Exception as e:
                        st.error(f"❌ 导出失败: {str(e)}")
        else:
            st.markdown('<div style="border:1px dashed #ccc; height:80px; display:flex; align-items:center; justify-content:center;">处理完成后可导出结果</div>', unsafe_allow_html=True)

    # 右侧：预处理设置 + 排列方案 + 测试功能
    with col_right:
        with st.expander("⚙️ 预处理设置", expanded=True):
            # 基线校准设置
            st.subheader("基线校准", divider="gray")
            baseline_method = st.selectbox(
                "方法",
                ["无", "SD", "FD", "多项式拟合", "ModPoly", "I-ModPoly", "PLS", "AsLS", "airPLS", "二阶差分(D2)"],
                key="baseline_method",
                label_visibility="collapsed"
            )

            baseline_params = {}
            if baseline_method != "无":
                if baseline_method == "多项式拟合":
                    polyorder = st.slider("阶数k", 3, 6, 5, key="polyorder", label_visibility="collapsed")
                    baseline_params["polyorder"] = polyorder
                    st.caption(f"阶数: {polyorder}")
                elif baseline_method == "ModPoly":
                    k = st.slider("参数k", 4, 10, 10, key="k_mod", label_visibility="collapsed")
                    baseline_params["k"] = k
                    st.caption(f"k: {k}")
                elif baseline_method == "I-ModPoly":
                    polyorder = st.slider("多项式阶数", 3, 7, 5, key="imod_polyorder", label_visibility="collapsed")
                    max_iter = st.slider("最大迭代次数", 50, 200, 100, key="imod_maxiter", label_visibility="collapsed")
                    tolerance = st.slider("收敛容差", 0.001, 0.01, 0.005, key="imod_tol", label_visibility="collapsed")
                    baseline_params["polyorder"] = polyorder
                    baseline_params["max_iter"] = max_iter
                    baseline_params["tolerance"] = tolerance
                    st.caption(f"阶数: {polyorder}, 迭代: {max_iter}, 容差: {tolerance}")
                elif baseline_method == "PLS":
                    lam = st.selectbox("λ", [10**10, 10**8, 10**7], key="lam_pls", label_visibility="collapsed")
                    baseline_params["lam"] = lam
                    st.caption(f"λ: {lam}")
                elif baseline_method == "AsLS":
                    asls_cols = st.columns(2)
                    with asls_cols[0]:
                        p = st.selectbox("非对称系数p", [0.001, 0.01, 0.1], key="p_asls", label_visibility="collapsed")
                    with asls_cols[1]:
                        lam = st.selectbox("平滑系数λ", [10**5, 10**7, 10**9], key="lam_asls", label_visibility="collapsed")
                    niter = st.selectbox("迭代次数", [5, 10, 15], key="niter_asls", label_visibility="collapsed")
                    baseline_params["lam"] = lam
                    baseline_params["p"] = p
                    baseline_params["niter"] = niter
                    st.caption(f"p: {p}, λ: {lam}, 迭代次数: {niter}")
                elif baseline_method == "airPLS":
                    airpls_cols = st.columns(2)
                    with airpls_cols[0]:
                        lam = st.selectbox("λ", [10**7, 10**4, 10**2], key="lam_air", label_visibility="collapsed")
                    baseline_params["lam"] = lam
                    st.caption(f"λ: {lam}")
                elif baseline_method == "二阶差分(D2)":
                    st.caption("二阶差分可增强光谱特征，抑制基线漂移")

            # 缩放设置
            st.subheader("📏 缩放", divider="gray")
            scaling_method = st.selectbox(
                "方法",
                ["无", "Peak-Norm", "SNV", "MSC", "M-M-Norm", "L-范数", "Ma-Minorm", "标准化(均值0，方差1)"],
                key="scaling_method",
                label_visibility="collapsed"
            )

            scaling_params = {}
            if scaling_method == "L-范数":
                p = st.selectbox("p", ["无穷大", "4", "10"], key="p_scale", label_visibility="collapsed")
                scaling_params["p"] = p
                st.caption(f"p: {p}")
            elif scaling_method == "标准化(均值0，方差1)":
                st.caption("将数据标准化到均值为0，方差为1")

            # 滤波设置
            st.subheader("📶 滤波", divider="gray")
            filtering_method = st.selectbox(
                "方法",
                ["无", "Savitzky-Golay", "sgolayfilt滤波器", "中值滤波(MF)", "移动平均(MAF)",
                 "MWA（移动窗口平均）", "MWM（移动窗口中值）", "卡尔曼滤波", "Lowess", "FFT",
                 "Smfft傅里叶滤波", "小波变换(DWT)", "小波线性阈值去噪"],
                key="filtering_method",
                label_visibility="collapsed"
            )

            filtering_params = {}
            if filtering_method != "无":
                if filtering_method in ["Savitzky-Golay", "sgolayfilt滤波器"]:
                    sg_cols = st.columns(2)
                    with sg_cols[0]:
                        k = st.selectbox("多项式阶数", [3, 7], key="k_sg", label_visibility="collapsed")
                    with sg_cols[1]:
                        w = st.selectbox("窗口大小", [11, 31, 51], key="w_sg", label_visibility="collapsed")
                    filtering_params["point"] = w
                    filtering_params["degree"] = k
                    st.caption(f"阶数: {k}, 窗口: {w}")
                elif filtering_method in ["中值滤波(MF)", "移动平均(MAF)"]:
                    mf_cols = st.columns(2)
                    with mf_cols[0]:
                        k = st.selectbox("k", [1, 3], key="k_mf", label_visibility="collapsed")
                    with mf_cols[1]:
                        w = st.selectbox("w", [7, 11], key="w_mf", label_visibility="collapsed")
                    filtering_params["k"] = k
                    filtering_params["w"] = w
                    st.caption(f"k: {k}, w: {w}")
                elif filtering_method == "MWA（移动窗口平均）":
                    mwa_cols = st.columns(2)
                    with mwa_cols[0]:
                        n = st.selectbox("窗口大小n", [4, 6, 8], key="n_mwa", label_visibility="collapsed")
                    with mwa_cols[1]:
                        it = st.selectbox("迭代次数it", [1, 2, 3], key="it_mwa", label_visibility="collapsed")
                    filtering_params["n"] = n
                    filtering_params["it"] = it
                    filtering_params["mode"] = "full"
                    st.caption(f"窗口大小: {n}, 迭代次数: {it}")
                elif filtering_method == "MWM（移动窗口中值）":
                    mwm_cols = st.columns(2)
                    with mwm_cols[0]:
                        n = st.selectbox("窗口大小n", [5, 7, 9], key="n_mwm", label_visibility="collapsed")
                    with mwm_cols[1]:
                        it = st.selectbox("迭代次数it", [1, 2, 3], key="it_mwm", label_visibility="collapsed")
                    filtering_params["n"] = n
                    filtering_params["it"] = it
                    st.caption(f"窗口大小: {n}, 迭代次数: {it}")
                elif filtering_method == "卡尔曼滤波":
                    R = st.selectbox("测量噪声方差R", [0.01, 0.1, 0.5], key="r_kalman", label_visibility="collapsed")
                    filtering_params["R"] = R
                    st.caption(f"测量噪声方差: {R}")
                elif filtering_method == "Lowess":
                    frac = st.selectbox("系数", [0.01, 0.03], key="frac_low", label_visibility="collapsed")
                    filtering_params["frac"] = frac
                    st.caption(f"系数: {frac}")
                elif filtering_method == "FFT":
                    cutoff = st.selectbox("频率", [30, 50, 90], key="cutoff_fft", label_visibility="collapsed")
                    filtering_params["cutoff"] = cutoff
                    st.caption(f"频率: {cutoff}")
                elif filtering_method == "Smfft傅里叶滤波":
                    row_e = st.selectbox("保留低频分量数", [31, 51, 71], key="row_e_smfft", label_visibility="collapsed")
                    filtering_params["row_e"] = row_e
                    st.caption(f"保留低频分量数: {row_e}")
                elif filtering_method == "小波变换(DWT)":
                    threshold = st.selectbox("阈值", [0.1, 0.3, 0.5], key="thresh_dwt", label_visibility="collapsed")
                    filtering_params["threshold"] = threshold
                    st.caption(f"阈值: {threshold}")
                elif filtering_method == "小波线性阈值去噪":
                    threshold = st.selectbox("阈值", [0.1, 0.3, 0.5], key="thresh_wavelet_linear", label_visibility="collapsed")
                    filtering_params["threshold"] = threshold
                    st.caption(f"阈值: {threshold}")

            # 挤压设置
            st.subheader("🧪 挤压", divider="gray")
            squashing_method = st.selectbox(
                "方法",
                ["无", "Sigmoid挤压", "改进的Sigmoid挤压", "逻辑函数", "余弦挤压(squashing)", "改进的逻辑函数", "DTW挤压"],
                key="squashing_method",
                label_visibility="collapsed"
            )

            squashing_params = {}
            if squashing_method != "无":
                if squashing_method == "改进的逻辑函数":
                    st.caption("基于余弦的挤压变换，无额外参数")
                elif squashing_method == "改进的Sigmoid挤压":
                    maxn = st.selectbox("maxn", [5, 10, 15], key="maxn_isigmoid", label_visibility="collapsed")
                    squashing_params["maxn"] = maxn
                    st.caption(f"maxn: {maxn}")
                elif squashing_method == "DTW挤压":
                    dtw_row1 = st.columns(2)
                    with dtw_row1[0]:
                        l = st.selectbox("l", [1, 5], key="l_dtw", label_visibility="collapsed")
                    with dtw_row1[1]:
                        k1 = st.selectbox("k1", ["T", "F"], key="k1_dtw", label_visibility="collapsed")
                    k2 = st.selectbox("k2", ["T", "F"], key="k2_dtw", label_visibility="collapsed")
                    squashing_params["l"] = l
                    squashing_params["k1"] = k1
                    squashing_params["k2"] = k2
                    st.caption(f"l: {l}, k1: {k1}, k2: {k2}")
                elif squashing_method == "Sigmoid挤压":
                    st.caption("使用标准Sigmoid函数，无额外参数")
                elif squashing_method == "余弦挤压(squashing)":
                    st.caption("使用基于余弦的挤压变换，无额外参数")
                elif squashing_method == "逻辑函数":
                    st.caption("无额外参数")

            # 保存当前算法选择
            current_algorithms = {
                'baseline': baseline_method,
                'baseline_params': baseline_params,
                'scaling': scaling_method,
                'scaling_params': scaling_params,
                'filtering': filtering_method,
                'filtering_params': filtering_params,
                'squashing': squashing_method,
                'squashing_params': squashing_params
            }
            st.session_state.current_algorithms = current_algorithms

            # 操作按钮
            st.subheader("操作", divider="gray")
            btn_cols = st.columns(2)
            with btn_cols[0]:
                if st.button("🚀 应用处理", type="primary", use_container_width=True, key="apply_btn"):
                    if st.session_state.raw_data is None:
                        st.warning("⚠️ 请先上传数据")
                        return

                    try:
                        wavenumbers, y = st.session_state.raw_data
                        y_processed, method_name = preprocessor.process(
                            wavenumbers, y,
                            baseline_method=baseline_method,
                            baseline_params=baseline_params,
                            squashing_method=squashing_method,
                            squashing_params=squashing_params,
                            filtering_method=filtering_method,
                            filtering_params=filtering_params,
                            scaling_method=scaling_method,
                            scaling_params=scaling_params
                        )

                        st.session_state.processed_data = (wavenumbers, y_processed)
                        st.session_state.process_method = " → ".join(method_name) if method_name else "无预处理"
                        st.success("✅ 预处理完成")
                    except Exception as e:
                        st.error(f"❌ 预处理失败: {str(e)}")

            with btn_cols[1]:
                if st.button("🔄 生成排列方案", type="secondary", use_container_width=True, key="generate_btn"):
                    if st.session_state.raw_data is None:
                        st.warning("⚠️ 请先上传数据")
                        return

                    try:
                        perms = generate_permutations(current_algorithms)
                        st.session_state.algorithm_permutations = perms
                        st.session_state.filtered_perms = perms
                        st.session_state.show_arrangements = True
                        st.success(f"✅ 生成{len(perms)}种排列方案")
                    except Exception as e:
                        st.error(f"❌ 生成排列方案失败: {str(e)}")

        # 排列方案展示
        if st.session_state.show_arrangements and st.session_state.algorithm_permutations:
            with st.expander(f"📋 排列方案 ({len(st.session_state.filtered_perms)})", expanded=True):
                # 筛选选项
                first_step_filter = st.selectbox(
                    "按第一步筛选",
                    ["全部", "无预处理", "基线校准", "缩放", "滤波", "挤压"],
                    key="step_filter",
                    label_visibility="collapsed"
                )

                # 应用筛选
                if first_step_filter != "全部":
                    st.session_state.filtered_perms = [
                        p for p in st.session_state.algorithm_permutations
                        if p["first_step_type"] == first_step_filter
                    ]
                else:
                    st.session_state.filtered_perms = st.session_state.algorithm_permutations

                # 显示数量
                st.caption(f"显示 {len(st.session_state.filtered_perms)} 种方案")

                # 方案选择
                perm_names = [p["name"] for p in st.session_state.filtered_perms]
                selected_idx = st.selectbox(
                    "选择方案",
                    range(len(perm_names)),
                    format_func=lambda x: perm_names[x],
                    key="perm_select",
                    label_visibility="collapsed",
                    index=st.session_state.selected_perm_idx
                )
                st.session_state.selected_perm_idx = selected_idx

                # 应用选中的排列方案
                if st.button("▶️ 应用选中方案", use_container_width=True, key="apply_perm_btn"):
                    try:
                        selected_perm = st.session_state.filtered_perms[selected_idx]
                        wavenumbers, y = st.session_state.raw_data

                        y_processed, method_name = preprocessor.process(
                            wavenumbers, y,
                            baseline_method=current_algorithms['baseline'],
                            baseline_params=current_algorithms['baseline_params'],
                            squashing_method=current_algorithms['squashing'],
                            squashing_params=current_algorithms['squashing_params'],
                            filtering_method=current_algorithms['filtering'],
                            filtering_params=current_algorithms['filtering_params'],
                            scaling_method=current_algorithms['scaling'],
                            scaling_params=current_algorithms['scaling_params'],
                            algorithm_order=selected_perm["order"]
                        )

                        method_str = " → ".join(method_name) if method_name else "无预处理（原始光谱）"
                        st.session_state.arrangement_details[selected_perm["name"]] = {
                            "data": y_processed,
                            "method": method_str,
                            "order": selected_perm["order"]
                        }
                        st.session_state.selected_arrangement = selected_perm["name"]
                        st.success(f"✅ 已应用: {selected_perm['name']}")
                    except Exception as e:
                        st.error(f"❌ 应用排列方案失败: {str(e)}")

        # 分类测试功能
        if st.session_state.get('selected_arrangement') and st.session_state.get('labels') is not None:
            with st.expander("🧪 分类测试", expanded=True):
                st.subheader("KNN参数", divider="gray")
                k_value = st.slider(
                    "k值",
                    min_value=1,
                    max_value=15,
                    value=5,
                    step=2,
                    key="knn_k",
                    label_visibility="collapsed"
                )
                st.session_state.k_value = k_value

                if st.button("▶️ 开始测试", type="primary", use_container_width=True, key="test_btn"):
                    try:
                        # 获取数据
                        wavenumbers, y = st.session_state.raw_data
                        selected_arr = st.session_state.selected_arrangement
                        processed_data = st.session_state.arrangement_details[selected_arr]['data']
                        
                        # 划分训练集和测试集
                        train_idx = st.session_state.train_indices
                        test_idx = st.session_state.test_indices
                        labels = st.session_state.labels

                        # 确保索引有效
                        if len(train_idx) == 0 or len(test_idx) == 0:
                            st.warning("⚠️ 训练集或测试集为空，请调整训练集比例")
                            return

                        # 提取训练和测试数据
                        train_data = processed_data[:, train_idx]
                        test_data = processed_data[:, test_idx]
                        train_labels = labels[train_idx]
                        test_labels = labels[test_idx]

                        # 执行KNN分类
                        predictions = knn_classify(train_data, train_labels, test_data, k=k_value)

                        # 计算评估指标
                        accuracy = accuracy_score(test_labels, predictions)
                        kappa = cohen_kappa_score(test_labels, predictions)
                        cm = confusion_matrix(test_labels, predictions)

                        # 保存结果
                        st.session_state.test_results = {
                            'accuracy': accuracy,
                            'kappa': kappa,
                            'confusion_matrix': cm,
                            'predictions': predictions,
                            'test_labels': test_labels
                        }

                        st.success(f"✅ 测试完成 | 准确率: {accuracy:.4f} | 卡帕系数: {kappa:.4f}")
                    except Exception as e:
                        st.error(f"❌ 测试失败: {str(e)}")


if __name__ == "__main__":
    main()
    
