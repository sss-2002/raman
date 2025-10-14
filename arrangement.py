import streamlit as st
import numpy as np
import pandas as pd
import re
import itertools
import matplotlib.pyplot as plt
import math
import zipfile
import os
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
from sklearn.linear_model import LinearRegression  # 用于MSC
import scipy.signal as signal

# ===== 算法实现 =====
def polynomial_fit(wavenumbers, spectra, polyorder):
    """多项式拟合基线校正"""
    baseline = np.zeros_like(spectra)
    for i in range(spectra.shape[1]):
        coeffs = np.polyfit(wavenumbers, spectra[:, i], deg=polyorder)
        baseline[:, i] = np.polyval(coeffs, wavenumbers)
    return spectra - baseline  # 扣除基线

def modpoly(wavenumbers, spectra, k):
    """Modified Polynomial (ModPoly) 基线校正"""
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

def pls(spectra, lam):
    """Penalized Least Squares (PLS) 基线校正"""
    n_points = spectra.shape[0]
    baseline = np.zeros_like(spectra)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n_points, n_points - 2))
    D = lam * D.dot(D.transpose())
    for i in range(spectra.shape[1]):
        y = spectra[:, i]
        A = sparse.eye(n_points) + D
        baseline[:, i] = spsolve(A, y)
    return spectra - baseline

def airpls(spectra, lam, max_iter=15, threshold=0.001):
    """Adaptive Iteratively Reweighted Penalized Least Squares (airPLS) 基线校正"""
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
                diff = np.sum(np.abs(b - baseline_i)) / np.sum(np.abs(baseline_i)) if np.sum(
                    np.abs(baseline_i)) > 0 else 0
                if diff < threshold:
                    break
            baseline_i = b
        baseline[:, i] = baseline_i
    return spectra - baseline

def dtw_squashing(x, l, k1, k2):
    """动态时间规整(DTW)挤压算法"""
    n_samples, n_features = x.shape
    result = np.zeros_like(x)
    reference = np.mean(x, axis=1)  # 使用平均光谱作为参考
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

# ===== 分类算法实现 =====
def knn_classify(train_data, train_labels, test_data, k=5):
    """K近邻分类算法实现"""
    # 转置数据以适应样本数×特征数的格式
    train_data = train_data.T
    test_data = test_data.T

    predictions = []
    for test_sample in test_data:
        # 计算与所有训练样本的欧氏距离
        distances = np.sqrt(np.sum((train_data - test_sample) **2, axis=1))
        # 获取最近的k个样本的索引
        k_indices = np.argsort(distances)[:k]
        # 获取这些样本的标签
        k_nearest_labels = [train_labels[i] for i in k_indices]
        # 多数投票决定预测标签
        most_common = np.bincount(k_nearest_labels).argmax()
        predictions.append(most_common)
    return np.array(predictions)  

# ===== 预处理类 =====
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
        """执行预处理流程，支持指定算法顺序"""
        if baseline_params is None: baseline_params = {}
        if squashing_params is None: squashing_params = {}
        if filtering_params is None: filtering_params = {}
        if scaling_params is None: scaling_params = {}

        # 如果算法顺序为空，返回原始数据
        if algorithm_order is not None and len(algorithm_order) == 0:
            return data.copy(), ["无预处理（原始光谱）"]

        y_processed = data.copy()
        method_name = []

        # 如果指定了算法顺序，则按顺序执行
        if algorithm_order is not None and len(algorithm_order) > 0:
            # 根据算法编号映射到对应的处理步骤
            step_mapping = {
                1: ("baseline", baseline_method, baseline_params),
                2: ("scaling", scaling_method, scaling_params),
                3: ("filtering", filtering_method, filtering_params),
                4: ("squashing", squashing_method, squashing_params)
            }
            # 按指定顺序创建步骤列表
            steps = [step_mapping[order] for order in algorithm_order]
        else:
            # 默认顺序：基线 → 挤压 → 滤波 → 缩放
            steps = []
            if baseline_method != "无":
                steps.append(("baseline", baseline_method, baseline_params))
            if squashing_method != "无":
                steps.append(("squashing", squashing_method, squashing_params))
            if filtering_method != "无":
                steps.append(("filtering", filtering_method, filtering_params))
            if scaling_method != "无":
                steps.append(("scaling", scaling_method, scaling_params))

        # 按顺序执行预处理步骤
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
                    else:  # SD、FD 无额外参数
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

    # ===== 滤波算法实现 =====
    def savitzky_golay(self, spectra, window_length, polyorder):
        return savgol_filter(spectra, window_length, polyorder, axis=0)

    def sgolay_filter_custom(self, spectra, window_length, polyorder):
        if spectra.shape[0] < spectra.shape[1]:
            filtered = savgol_filter(spectra.T, window_length, polyorder, axis=0)
            return filtered.T
        else:
            return savgol_filter(spectra, window_length, polyorder, axis=0)

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

    # ===== 缩放算法实现 =====
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

    def d2(self, spectra):
        if spectra.shape[0] < spectra.shape[1]:
            diff_result = D2(spectra.T)
            return diff_result.T
        else:
            return D2(spectra)

# ===== 文件处理类 =====
class FileHandler:
    def load_data_from_zip(self, zip_file):
        with zipfile.ZipFile(zip_file, 'r') as zf:
            file_list = zf.namelist()

            wavenumber_files = [f for f in file_list if 'wave' in f.lower() or 'wn' in f.lower() or '波数' in f]
            data_files = [f for f in file_list if 'spec' in f.lower() or 'data' in f.lower() or '光谱' in f]

            if not wavenumber_files:
                raise ValueError("压缩包中未找到波数文件（通常包含'wave'、'wn'或'波数'）")
            if not data_files:
                raise ValueError("压缩包中未找到光谱数据文件（通常包含'spec'、'data'或'光谱'）")

            wn_file = wavenumber_files[0]
            data_file = data_files[0]

            with zf.open(wn_file) as f:
                wavenumbers = np.loadtxt(f).ravel()

            with zf.open(data_file) as f:
                content = f.read().decode("utf-8")
                data = self._parse_data(content)

            return wavenumbers, data.T

    def _parse_data(self, content):
        numb = re.compile(r"-?\d+(?:\.\d+)?")
        lines_list = content.splitlines()

        all_numbers = []
        for line in lines_list:
            all_numbers.extend(numb.findall(line))

        data = np.array([float(num) for num in all_numbers])

        n_rows = len(lines_list)
        n_cols = len(data) // n_rows if n_rows > 0 else 0

        if n_cols * n_rows != len(data):
            n_cols = len(data) // n_rows + 1
            data = data[:n_rows * n_cols]

        return data.reshape(n_rows, n_cols)

    def export_data(self, filename, data):
        with open(filename, "w") as f:
            for line in data.T:
                f.write("\t".join(map(str, line)) + "\n")

# 辅助算法函数
def squashing(Data):
    row = Data.shape[0]
    col = Data.shape[1]
    sqData = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            sqData[i][j] = (1 - math.cos(Data[i][j] * math.pi)) / 2
    return sqData

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

def sigmoid(X):
    row = X.shape[0]
    col = X.shape[1]
    s = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            m = 1 + np.exp(-float(X[i, j]))
            s[i, j] = (1.0 / m)
    return s

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

def KalmanF(xd, R):
    row = xd.shape[0]
    col = xd.shape[1]
    Fxd = np.zeros((row, col))
    for i in range(row):
        Fxd[i] = Kalman(xd[i], R)
    return Fxd

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

def squashing_legacy(x):
    return 1 / (1 + np.exp(-x))

def SGfilter(Intensity, window_length,polyorder):
    Row = Intensity.shape[0]
    col = Intensity.shape[1]
    sgsmooth = np.zeros((Row, col))
    for i in range(Row):
        sgsmooth[i] = savgol_filter(Intensity[i], window_length, polyorder)
    return sgsmooth

def generate_permutations(algorithms):
    algorithm_list = [
        (1, "基线校准", algorithms['baseline']),
        (2, "缩放", algorithms['scaling']),
        (3, "滤波", algorithms['filtering']),
        (4, "挤压", algorithms['squashing'])
    ]

    all_permutations = []
    all_permutations.append([])

    for algo in algorithm_list:
        if algo[2] != "无":
            all_permutations.append([algo])

    for perm in itertools.permutations(algorithm_list, 2):
        if perm[0][2] != "无" and perm[1][2] != "无":
            all_permutations.append(list(perm))

    for perm in itertools.permutations(algorithm_list, 3):
        if perm[0][2] != "无" and perm[1][2] != "无" and perm[2][2] != "无":
            all_permutations.append(list(perm))

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

def main():
    # 初始化session state
    if 'show_arrangements' not in st.session_state:
        st.session_state.show_arrangements = False

    test_states = {
        'test_results': None,
        'labels': None,
        'train_indices': None,
        'test_indices': None
    }
    
    file_handler = FileHandler()
    preprocessor = Preprocessor()
    
    current_algorithms = {
        'baseline': '无',
        'baseline_params': {},
        'scaling': '无',
        'scaling_params': {},
        'filtering': '无',
        'filtering_params': {},
        'squashing': '无',
        'squashing_params': {}
    }

    st.session_state['current_algorithms'] = current_algorithms
    
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

    all_states = {**test_states, **other_states}
    for key, value in all_states.items():
        if key not in st.session_state:
            st.session_state[key] = value
    st.session_state['current_algorithms'] = current_algorithms

    # 页面设置
    st.set_page_config(layout="wide", page_icon="🔬", page_title="排列预处理模型")
    
    # 紧凑布局样式调整
    st.markdown("""
        <style>
        .css-1v0mbdj {padding: 0.2rem 0.4rem !important;}
        .css-1d391kg {padding: 0.1rem 0 !important;}
        .css-12ttj6m {margin-bottom: 0.2rem !important;}
        .css-16huue1 {padding: 0.1rem 0.4rem !important; font-size: 0.8rem !important;}
        h3 {font-size: 1rem !important; margin: 0.2rem 0 !important;}
        .css-1b3298e {gap: 0.2rem !important;}
        .stSlider, .stSelectbox, .stTextInput {margin-bottom: 0.2rem !important;}
        .stCaption {font-size: 0.7rem !important; margin-top: -0.1rem !important;}
        .css-1544g2n {padding: 0.1rem 0.4rem !important;}
        </style>
    """, unsafe_allow_html=True)

    st.title("🌌 排列预处理模型")

    # 页面布局：左侧数据管理，右侧主要内容区
    col_left, col_right = st.columns([1.2, 3.9])

    # ===== 左侧：数据管理模块 =====
    with col_left:
        with st.expander("📁 数据管理", expanded=True):
            zip_file = st.file_uploader("上传包含波数和光谱数据的压缩包", type=['zip'], key="zip_file")
            st.caption("压缩包(.zip)需包含波数和光谱数据文件")

            st.subheader("样本标签")
            num_classes = st.number_input("类别数量", min_value=1, value=2, step=1, key="num_cls")
            labels_input = st.text_input(
                "标签（逗号分隔，与光谱顺序一致）",
                placeholder="例：0,0,1,1",
                key="labels_in"
            )

            st.subheader("训练测试划分")
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

            if zip_file:
                try:
                    st.session_state.raw_data = file_handler.load_data_from_zip(
                        zip_file
                    )

                    if labels_input:
                        try:
                            labels = np.array([int(l.strip()) for l in labels_input.split(',')])
                            if len(labels) == st.session_state.raw_data[1].shape[1]:
                                st.session_state.labels = labels
                                n_samples = len(labels)
                                train_size = int(n_samples * train_test_ratio)
                                
                                np.random.seed(42)
                                indices = np.random.permutation(n_samples)
                                st.session_state.train_indices = indices[:train_size]
                                st.session_state.test_indices = indices[train_size:]
                                st.success(f"数据加载成功！样本数: {n_samples}, 训练集: {train_size}, 测试集: {n_samples - train_size}")
                            else:
                                st.error(f"标签数量与样本数量不符！标签数: {len(labels)}, 样本数: {st.session_state.raw_data[1].shape[1]}")
                        except ValueError:
                            st.error("标签格式错误，请输入逗号分隔的整数")

                except Exception as e:
                    st.error(f"数据加载失败: {str(e)}")

        # 预处理参数设置
        with st.expander("🔧 预处理参数", expanded=True):
            st.subheader("基线校正")
            baseline_method = st.selectbox(
                "选择基线校正方法",
                ["无"] + list(preprocessor.BASELINE_ALGORITHMS.keys()),
                key="baseline_method"
            )
            baseline_params = {}
            
            if baseline_method == "多项式拟合":
                polyorder = st.slider("多项式阶数", 1, 10, 3, key="polyorder")
                baseline_params["polyorder"] = polyorder
            elif baseline_method == "ModPoly":
                k = st.slider("迭代次数", 1, 20, 5, key="modpoly_k")
                baseline_params["k"] = k
            elif baseline_method == "I-ModPoly":
                polyorder = st.slider("多项式阶数", 1, 10, 3, key="imodpoly_order")
                max_iter = st.slider("最大迭代次数", 10, 500, 100, key="imodpoly_iter")
                tolerance = st.slider("收敛容差", 0.001, 0.01, 0.005, 0.001, key="imodpoly_tol")
                baseline_params["polyorder"] = polyorder
                baseline_params["max_iter"] = max_iter
                baseline_params["tolerance"] = tolerance
            elif baseline_method == "PLS":
                lam = st.slider("平滑系数 (1e4-1e9)", 1e4, 1e9, 1e5, key="pls_lam")
                baseline_params["lam"] = lam
            elif baseline_method == "AsLS":
                lam = st.slider("平滑系数 (1e5-1e12)", 1e5, 1e12, 1e5, key="asls_lam")
                p = st.slider("非对称系数 (0.001-0.1)", 0.001, 0.1, 0.001, key="asls_p")
                baseline_params["lam"] = lam
                baseline_params["p"] = p
            elif baseline_method == "airPLS":
                lam = st.slider("平滑系数 (1e5-1e12)", 1e5, 1e12, 1e5, key="airpls_lam")
                max_iter = st.slider("最大迭代次数", 5, 50, 15, key="airpls_iter")
                baseline_params["lam"] = lam
                baseline_params["max_iter"] = max_iter

            st.subheader("挤压算法")
            squashing_method = st.selectbox(
                "选择挤压方法",
                ["无"] + list(preprocessor.SQUASHING_ALGORITHMS.keys()),
                key="squashing_method"
            )
            squashing_params = {}
            
            if squashing_method == "改进的Sigmoid挤压":
                maxn = st.slider("分段数", 5, 20, 10, key="sigmoid_maxn")
                squashing_params["maxn"] = maxn
            elif squashing_method == "DTW挤压":
                l = st.slider("窗口大小", 1, 10, 1, key="dtw_l")
                k1 = st.radio("启用斜率限制", ["T", "F"], key="dtw_k1")
                k2 = st.radio("启用窗口平滑", ["T", "F"], key="dtw_k2")
                squashing_params["l"] = l
                squashing_params["k1"] = k1
                squashing_params["k2"] = k2

            st.subheader("滤波算法")
            filtering_method = st.selectbox(
                "选择滤波方法",
                ["无"] + list(preprocessor.FILTERING_ALGORITHMS.keys()),
                key="filtering_method"
            )
            filtering_params = {}
            
            if filtering_method in ["Savitzky-Golay", "sgolayfilt滤波器"]:
                window_length = st.slider("窗口长度（奇数）", 3, 21, 5, 2, key="sg_window")
                polyorder = st.slider("多项式阶数", 1, 5, 2, key="sg_order")
                filtering_params["window_length"] = window_length
                filtering_params["polyorder"] = polyorder
            elif filtering_method == "中值滤波(MF)":
                k = st.slider("参数k", 1, 10, 2, key="mf_k")
                w = st.slider("窗口大小（奇数）", 3, 21, 5, 2, key="mf_window")
                filtering_params["k"] = k
                filtering_params["w"] = w
            elif filtering_method in ["移动平均(MAF)", "MWA（移动窗口平均）"]:
                k = st.slider("参数k", 1, 10, 2, key="ma_k")
                w = st.slider("窗口大小", 3, 21, 5, key="ma_window")
                filtering_params["k"] = k
                filtering_params["w"] = w
                if filtering_method == "MWA（移动窗口平均）":
                    it = st.slider("迭代次数", 1, 5, 1, key="mwa_it")
                    filtering_params["it"] = it
            elif filtering_method == "MWM（移动窗口中值）":
                n = st.slider("窗口大小（奇数）", 3, 21, 7, 2, key="mwm_n")
                it = st.slider("迭代次数", 1, 5, 1, key="mwm_it")
                filtering_params["n"] = n
                filtering_params["it"] = it
            elif filtering_method == "卡尔曼滤波":
                R = st.slider("测量噪声方差", 0.01, 1.0, 0.1, key="kalman_R")
                filtering_params["R"] = R
            elif filtering_method == "Lowess":
                frac = st.slider("平滑系数", 0.01, 0.5, 0.1, key="lowess_frac")
                filtering_params["frac"] = frac
            elif filtering_method == "FFT":
                cutoff = st.slider("截止频率", 0.01, 0.5, 0.1, key="fft_cutoff")
                filtering_params["cutoff"] = cutoff
            elif filtering_method == "Smfft傅里叶滤波":
                row_e = st.slider("保留低频分量数", 10, 100, 51, key="smfft_row_e")
                filtering_params["row_e"] = row_e
            elif filtering_method == "小波变换(DWT)":
                threshold = st.slider("阈值", 0.01, 1.0, 0.3, key="dwt_threshold")
                filtering_params["threshold"] = threshold
            elif filtering_method == "小波线性阈值去噪":
                threshold = st.slider("阈值", 0.01, 1.0, 0.3, key="wavelet_threshold")
                filtering_params["threshold"] = threshold

            st.subheader("缩放算法")
            scaling_method = st.selectbox(
                "选择缩放方法",
                ["无"] + list(preprocessor.SCALING_ALGORITHMS.keys()),
                key="scaling_method"
            )
            scaling_params = {}
            
            if scaling_method == "L-范数":
                p = st.selectbox(
                    "范数阶数",
                    ["1", "2", "无穷大"],
                    key="lp_p"
                )
                scaling_params["p"] = p

            st.session_state.current_algorithms = {
                'baseline': baseline_method,
                'baseline_params': baseline_params,
                'scaling': scaling_method,
                'scaling_params': scaling_params,
                'filtering': filtering_method,
                'filtering_params': filtering_params,
                'squashing': squashing_method,
                'squashing_params': squashing_params
            }

            st.subheader("算法顺序")
            algorithm_order = st.multiselect(
                "选择预处理步骤顺序（最多4步）",
                [1, 2, 3, 4],
                format_func=lambda x: {
                    1: "1. 基线校准",
                    2: "2. 缩放",
                    3: "3. 滤波",
                    4: "4. 挤压"
                }[x],
                key="algorithm_order"
            )

            if st.button("生成所有可能的预处理排列", key="generate_perms"):
                try:
                    perms = generate_permutations(st.session_state.current_algorithms)
                    st.session_state.algorithm_permutations = perms
                    st.session_state.filtered_perms = perms
                    st.success(f"已生成 {len(perms)} 种预处理排列方案")
                    st.session_state.show_arrangements = True
                except Exception as e:
                    st.error(f"生成排列失败: {str(e)}")

            if st.session_state.algorithm_permutations:
                first_step_filter = st.selectbox(
                    "按第一步筛选",
                    ["全部"] + list(set(p.get("first_step_type", "未知") for p in st.session_state.algorithm_permutations)),
                    key="first_step_filter"
                )
                
                if first_step_filter != "全部":
                    st.session_state.filtered_perms = [
                        p for p in st.session_state.algorithm_permutations 
                        if p.get("first_step_type", "未知") == first_step_filter
                    ]
                else:
                    st.session_state.filtered_perms = st.session_state.algorithm_permutations
                
                st.write(f"筛选后: {len(st.session_state.filtered_perms)} 种方案")

        # KNN分类参数
        with st.expander("🔍 KNN分类参数", expanded=True):
            st.slider("K值（近邻数量）", 1, 20, 5, key="k_value")
            
            if st.button("运行KNN分类", key="run_knn") and st.session_state.raw_data is not None:
                if st.session_state.labels is None:
                    st.error("请输入样本标签")
                else:
                    try:
                        wavenumbers, raw_spectra = st.session_state.raw_data
                        current_algos = st.session_state.current_algorithms
                        
                        processed_spectra, _ = preprocessor.process(
                            wavenumbers,
                            raw_spectra,
                            baseline_method=current_algos['baseline'],
                            baseline_params=current_algos['baseline_params'],
                            squashing_method=current_algos['squashing'],
                            squashing_params=current_algos['squashing_params'],
                            filtering_method=current_algos['filtering'],
                            filtering_params=current_algos['filtering_params'],
                            scaling_method=current_algos['scaling'],
                            scaling_params=current_algos['scaling_params'],
                            algorithm_order=algorithm_order
                        )
                        
                        train_data = processed_spectra[:, st.session_state.train_indices]
                        test_data = processed_spectra[:, st.session_state.test_indices]
                        train_labels = st.session_state.labels[st.session_state.train_indices]
                        test_labels = st.session_state.labels[st.session_state.test_indices]
                        
                        predictions = knn_classify(train_data, train_labels, test_data, k=st.session_state.k_value)
                        
                        accuracy = accuracy_score(test_labels, predictions)
                        kappa = cohen_kappa_score(test_labels, predictions)
                        cm = confusion_matrix(test_labels, predictions)
                        
                        st.session_state.test_results = {
                            'predictions': predictions,
                            'test_labels': test_labels,
                            'accuracy': accuracy,
                            'kappa': kappa,
                            'cm': cm
                        }
                        
                        st.success(f"KNN分类完成！准确率: {accuracy:.4f}, Kappa系数: {kappa:.4f}")
                    except Exception as e:
                        st.error(f"分类失败: {str(e)}")

    # ===== 右侧：主要内容区 =====
    with col_right:
        st.subheader("📊 光谱可视化")
        
        # 紧凑的四图布局
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        
        has_data = st.session_state.raw_data is not None
        has_processed_data = st.session_state.processed_data is not None
        has_test_results = st.session_state.test_results is not None
        
        # 1. 原始光谱区域
        with col1:
            st.markdown("**1. 原始光谱**")
            # 更紧凑的图表尺寸
            fig1, ax1 = plt.subplots(figsize=(4.5, 2.5))
            
            if has_data:
                wavenumbers, raw_spectra = st.session_state.raw_data
                num_to_plot = min(10, raw_spectra.shape[1])
                ax1.plot(wavenumbers, raw_spectra[:, :num_to_plot])
                ax1.set_xlabel('波数', fontsize=8)
                ax1.set_ylabel('强度', fontsize=8)
                ax1.set_title(f'原始光谱（前{num_to_plot}条）', fontsize=9)
                ax1.tick_params(axis='both', which='major', labelsize=7)
                plt.tight_layout()
            else:
                ax1.text(0.5, 0.5, '未导入数据', ha='center', va='center', transform=ax1.transAxes, color='gray')
                ax1.set_xticks([])
                ax1.set_yticks([])
            
            st.pyplot(fig1)
        
        # 2. 预处理后光谱区域
        with col2:
            st.markdown("**2. 预处理后光谱**")
            fig2, ax2 = plt.subplots(figsize=(4.5, 2.5))
            
            if has_processed_data:
                wavenumbers, _ = st.session_state.raw_data
                processed_spectra = st.session_state.processed_data
                num_to_plot = min(10, processed_spectra.shape[1])
                ax2.plot(wavenumbers, processed_spectra[:, :num_to_plot])
                ax2.set_xlabel('波数', fontsize=8)
                ax2.set_ylabel('强度', fontsize=8)
                ax2.set_title(f'预处理后光谱（前{num_to_plot}条）', fontsize=9)
                ax2.tick_params(axis='both', which='major', labelsize=7)
                plt.tight_layout()
            else:
                ax2.text(0.5, 0.5, '未进行预处理', ha='center', va='center', transform=ax2.transAxes, color='gray')
                ax2.set_xticks([])
                ax2.set_yticks([])
            
            st.pyplot(fig2)
        
        # 3. K值曲线区域
        with col3:
            st.markdown("**3. K值曲线**")
            fig3, ax3 = plt.subplots(figsize=(4.5, 2.5))
            
            if has_test_results and has_data:
                k_values = range(1, 21)
                accuracies = []
                
                wavenumbers, raw_spectra = st.session_state.raw_data
                current_algos = st.session_state.current_algorithms
                
                processed_spectra, _ = preprocessor.process(
                    wavenumbers,
                    raw_spectra,
                    baseline_method=current_algos['baseline'],
                    baseline_params=current_algos['baseline_params'],
                    squashing_method=current_algos['squashing'],
                    squashing_params=current_algos['squashing_params'],
                    filtering_method=current_algos['filtering'],
                    filtering_params=current_algos['filtering_params'],
                    scaling_method=current_algos['scaling'],
                    scaling_params=current_algos['scaling_params'],
                    algorithm_order=algorithm_order
                )
                
                train_data = processed_spectra[:, st.session_state.train_indices]
                test_data = processed_spectra[:, st.session_state.test_indices]
                train_labels = st.session_state.labels[st.session_state.train_indices]
                test_labels = st.session_state.labels[st.session_state.test_indices]
                
                for k in k_values:
                    predictions = knn_classify(train_data, train_labels, test_data, k=k)
                    accuracies.append(accuracy_score(test_labels, predictions))
                
                ax3.plot(k_values, accuracies, 'o-', markersize=3, linewidth=1)
                ax3.set_xlabel('K值', fontsize=8)
                ax3.set_ylabel('准确率', fontsize=8)
                ax3.set_title('K值与准确率关系', fontsize=9)
                ax3.grid(True, linestyle='--', alpha=0.7)
                ax3.tick_params(axis='both', which='major', labelsize=7)
                plt.tight_layout()
            else:
                ax3.text(0.5, 0.5, '未进行分类测试', ha='center', va='center', transform=ax3.transAxes, color='gray')
                ax3.set_xticks([])
                ax3.set_yticks([])
            
            st.pyplot(fig3)
        
        # 4. 混淆矩阵区域
        with col4:
            st.markdown("**4. 混淆矩阵**")
            fig4, ax4 = plt.subplots(figsize=(4.5, 2.5))
            
            if has_test_results:
                cm = st.session_state.test_results['cm']
                classes = np.unique(st.session_state.test_results['test_labels'])
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4, 
                           xticklabels=classes, yticklabels=classes, annot_kws={"size": 8})
                ax4.set_xlabel('预测标签', fontsize=8)
                ax4.set_ylabel('真实标签', fontsize=8)
                ax4.set_title('混淆矩阵', fontsize=9)
                ax4.tick_params(axis='both', which='major', labelsize=7)
                plt.tight_layout()
            else:
                ax4.text(0.5, 0.5, '未进行分类测试', ha='center', va='center', transform=ax4.transAxes, color='gray')
                ax4.set_xticks([])
                ax4.set_yticks([])
            
            st.pyplot(fig4)

        # 预处理执行区域
        with st.expander("▶️ 执行预处理", expanded=True):
            if st.button("运行当前预处理", key="run_preprocessing") and st.session_state.raw_data is not None:
                try:
                    wavenumbers, raw_spectra = st.session_state.raw_data
                    current_algos = st.session_state.current_algorithms
                    
                    processed_spectra, method_names = preprocessor.process(
                        wavenumbers,
                        raw_spectra,
                        baseline_method=current_algos['baseline'],
                        baseline_params=current_algos['baseline_params'],
                        squashing_method=current_algos['squashing'],
                        squashing_params=current_algos['squashing_params'],
                        filtering_method=current_algos['filtering'],
                        filtering_params=current_algos['filtering_params'],
                        scaling_method=current_algos['scaling'],
                        scaling_params=current_algos['scaling_params'],
                        algorithm_order=algorithm_order
                    )
                    
                    st.session_state.processed_data = processed_spectra
                    st.success(f"预处理完成！步骤: {', '.join(method_names)}")
                except Exception as e:
                    st.error(f"预处理失败: {str(e)}")

        # 排列结果展示区域
        if st.session_state.show_arrangements and st.session_state.filtered_perms:
            with st.expander("📋 预处理排列结果", expanded=True):
                perm_names = [p["name"] for p in st.session_state.filtered_perms]
                selected_perm_idx = st.selectbox(
                    "选择预处理排列方案",
                    range(len(perm_names)),
                    format_func=lambda i: f"{i+1}. {perm_names[i]}",
                    key="selected_perm"
                )
                st.session_state.selected_perm_idx = selected_perm_idx
                
                selected_perm = st.session_state.filtered_perms[selected_perm_idx]
                st.write(f"**选中方案**: {selected_perm['name']}")
                st.write(f"**步骤数量**: {selected_perm['count']}")
                
                if st.button("执行选中的排列方案", key="run_selected_perm") and st.session_state.raw_data is not None:
                    try:
                        wavenumbers, raw_spectra = st.session_state.raw_data
                        current_algos = st.session_state.current_algorithms
                        
                        processed_spectra, method_names = preprocessor.process(
                            wavenumbers,
                            raw_spectra,
                            baseline_method=current_algos['baseline'],
                            baseline_params=current_algos['baseline_params'],
                            squashing_method=current_algos['squashing'],
                            squashing_params=current_algos['squashing_params'],
                            filtering_method=current_algos['filtering'],
                            filtering_params=current_algos['filtering_params'],
                            scaling_method=current_algos['scaling'],
                            scaling_params=current_algos['scaling_params'],
                            algorithm_order=selected_perm["order"]
                        )
                        
                        st.session_state.processed_data = processed_spectra
                        st.session_state.selected_arrangement = selected_perm
                        st.success(f"排列预处理完成！步骤: {', '.join(method_names)}")
                    except Exception as e:
                        st.error(f"排列预处理失败: {str(e)}")

        # 结果分析区域
        with st.expander("📈 结果分析", expanded=True):
            if st.session_state.test_results:
                results = st.session_state.test_results
                st.write(f"**准确率**: {results['accuracy']:.4f}")
                st.write(f"**Kappa系数**: {results['kappa']:.4f}")
                
                comparison = pd.DataFrame({
                    '真实标签': results['test_labels'],
                    '预测标签': results['predictions']
                })
                st.write("**预测结果对比**:")
                st.dataframe(comparison)
            else:
                st.info("请先运行KNN分类以查看结果分析")

if __name__ == "__main__":
    main()
    
