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
import scipy.signal as signal  # 导入scipy.signal用于MWM函数
import csv
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import itertools

cloud_storage_dir = "/mnt/data/processed_spectra"  # 临时目录，用于存储文件


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


    # 确保 spectra 是二维数组
    if spectra.ndim == 1:
        spectra = spectra.reshape(1, -1)  # 将其变为 1 × 20 的数组

    baseline = np.zeros_like(spectra)
    n_points = len(wavenumbers)

    # 输出调试信息
    # st.write(f"[CHECK] n_points: {n_points}, spectra shape: {spectra.shape}")

    # 遍历每个样本的光谱
    for i in range(spectra.shape[0]):  # spectra.shape[0] 为样本数量
        y = spectra[i, :].copy()  # 每个样本的光谱

        # 输出当前光谱信息
        # st.write(f"[CHECK] Processing spectrum {i+1}, shape: {y.shape}")

        # 对每个光谱应用多项式拟合
        for _ in range(k):
            coeffs = np.polyfit(wavenumbers, y, deg=5)
            fitted = np.polyval(coeffs, wavenumbers)

            # 输出拟合结果
            # st.write(f"[CHECK] Fitted values: {fitted[:2]}")  # 只显示前5个值
            mask = y < fitted
            y[~mask] = fitted[~mask]

        # 将修正后的光谱添加到基线矩阵
        baseline[i, :] = y

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
        distances = np.sqrt(np.sum((train_data - test_sample) ** 2, axis=1))
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
            "I-ModPoly": IModPoly,  # 集成IModPoly算法
            "PLS": pls,
            "AsLS": baseline_als,  # 使用改进的AsLS算法
            "airPLS": airpls,
            "二阶差分(D2)": self.d2  # 将二阶差分归类到基线校准中
        }
        self.FILTERING_ALGORITHMS = {
            "Savitzky-Golay": self.savitzky_golay,
            "sgolayfilt滤波器": self.sgolay_filter_custom,  # 添加自定义SG滤波器
            "中值滤波(MF)": self.median_filter,
            "移动平均(MAF)": self.moving_average,
            "MWA（移动窗口平均）": self.mwa_filter,  # 添加MWA算法
            "MWM（移动窗口中值）": self.mwm_filter,  # MWM滤波算法
            "卡尔曼滤波": self.kalman_filter,  # 添加卡尔曼滤波算法
            "Lowess": self.lowess_filter,
            "FFT": self.fft_filter,
            "Smfft傅里叶滤波": self.smfft_filter,  # 添加Smfft傅里叶滤波
            "小波变换(DWT)": self.wavelet_filter,
            "小波线性阈值去噪": self.wavelet_linear  # 新增：小波线性阈值去噪
        }

        self.SCALING_ALGORITHMS = {
            "Peak-Norm": self.peak_norm,
            "SNV": self.snv,
            "MSC": self.msc,  # 使用新的MSC实现
            "M-M-Norm": self.mm_norm,
            "L-范数": self.l_norm,  # 使用LPnorm函数实现
            "Ma-Minorm": self.ma_minorm,  # 添加Ma-Minorm归一化
            "标准化(均值0，方差1)": self.standardize  # 添加标准化算法
        }

        self.SQUASHING_ALGORITHMS = {
            "Sigmoid挤压": sigmoid,  # 使用sigmoid函数
            "改进的Sigmoid挤压": i_sigmoid,  # 使用改进的i_sigmoid函数
            "逻辑函数": squashing_legacy,  # 保留原逻辑函数以便对比
            "余弦挤压(squashing)": squashing,  # 新增：基于余弦的挤压变换
            "改进的逻辑函数": i_squashing,  # 使用i_squashing函数
            "DTW挤压": dtw_squashing
        }

    def process(self, wavenumbers, data,
                baseline_method="无", baseline_params=None,
                squashing_method="无", squashing_params=None,
                filtering_method="无", filtering_params=None,
                scaling_method="无", scaling_params=None,
                algorithm_order=None):
        """执行预处理流程，支持指定算法顺序，空顺序表示返回原始数据"""

        if baseline_params is None: baseline_params = {}
        if squashing_params is None: squashing_params = {}
        if filtering_params is None: filtering_params = {}
        if scaling_params is None: scaling_params = {}

        # 如果算法顺序为空（无预处理），直接返回原始数据
        if algorithm_order is not None and len(algorithm_order) == 0:
            return data.copy(), ["无预处理（原始光谱）"]

        y_processed = data.copy()
        method_name = []

        # 确保每个算法顺序都是有效的
        valid_orders = [1, 2, 3, 4]  # 仅支持 1-4
        if algorithm_order is not None:
            invalid_orders = [order for order in algorithm_order if order not in valid_orders]
            if invalid_orders:
                raise ValueError(f"无效的算法步骤编号: {invalid_orders}")

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
            # for step in steps:
            #     st.write(f"[CHECK] step: {step}")  # 使用 st.write 输出每个步骤的详细信息
        else:
            # 默认顺序：基线 → 挤压 → 滤波 → 缩放（只执行已选择的方法）
            steps = []
            if baseline_method != "无":
                steps.append(("baseline", baseline_method, baseline_params))
            if squashing_method != "无":
                steps.append(("squashing", squashing_method, squashing_params))
            if filtering_method != "无":
                steps.append(("filtering", filtering_method, filtering_params))
            if scaling_method != "无":
                steps.append(("scaling", scaling_method, scaling_params))

        # 调试输出参数
        for step_type, method, params in steps:
            print(f"[CHECK] {step_type} 方法: {method}, 参数: {params}")

        # 按顺序执行预处理步骤
        for step_type, method, params in steps:
            if method == "无":
                continue

            try:
                if step_type == "baseline":
                    algorithm_func = self.BASELINE_ALGORITHMS[method]
                    print(f"[CHECK] 执行基线校正方法: {method}, 参数: {params}")  # 输出调试信息

                    if method in ["多项式拟合", "ModPoly", "I-ModPoly"]:
                        # st.write(f"[CHECK] params for {method}: {params}")  # 输出params内容
                        y_processed = algorithm_func(wavenumbers, y_processed, **params)
                    elif method in ["PLS"]:
                        y_processed = algorithm_func(y_processed, **params)
                    elif method == "AsLS":
                        # 适配改进的AsLS算法参数
                        y_processed = algorithm_func(y_processed, **params)
                    elif method == "airPLS":
                        y_processed = algorithm_func(y_processed, **params)
                    elif method == "二阶差分(D2)":  # 处理二阶差分
                        y_processed = algorithm_func(y_processed)
                    else:  # SD、FD 无额外参数
                        y_processed = algorithm_func(y_processed)
                    method_name.append(f"{method}({', '.join([f'{k}={v}' for k, v in params.items()])})")

                elif step_type == "squashing":
                    algorithm_func = self.SQUASHING_ALGORITHMS[method]
                    print(f"[CHECK] 执行挤压方法: {method}, 参数: {params}")  # 输出调试信息

                    if method == "改进的Sigmoid挤压":
                        # 使用改进的i_sigmoid函数，支持maxn参数
                        maxn = params.get("maxn", 10)
                        y_processed = algorithm_func(y_processed, maxn=maxn)
                        method_name.append(f"{method}(maxn={maxn})")
                    elif method == "改进的逻辑函数":
                        # i_squashing函数不需要额外参数
                        y_processed = algorithm_func(y_processed)
                        method_name.append(f"{method}")
                    elif method == "DTW挤压":
                        l = params.get("l", 1)
                        k1 = params.get("k1", "T")
                        k2 = params.get("k2", "T")
                        y_processed = algorithm_func(y_processed, l=l, k1=k1, k2=k2)
                        method_name.append(f"DTW挤压(l={l}, k1={k1}, k2={k2})")
                    elif method == "Sigmoid挤压":
                        # 使用sigmoid函数
                        y_processed = algorithm_func(y_processed)
                        method_name.append(f"{method}")
                    elif method == "余弦挤压(squashing)":
                        # 使用新添加的squashing函数
                        y_processed = algorithm_func(y_processed)
                        method_name.append(f"{method}")
                    else:
                        y_processed = algorithm_func(y_processed)
                        method_name.append(method)

                elif step_type == "filtering":
                    algorithm_func = self.FILTERING_ALGORITHMS[method]
                    print(f"[CHECK] 执行滤波方法: {method}, 参数: {params}")  # 输出调试信息

                    y_processed = algorithm_func(y_processed, **params)
                    params_str = ', '.join([f'{k}={v}' for k, v in params.items()])
                    method_name.append(f"{method}({params_str})")

                    # 特殊处理小波线性阈值去噪的参数
                    if method == "小波线性阈值去噪":
                        threshold = params.get("threshold", 0.3)
                        method_name[-1] = f"{method}(threshold={threshold})"

                elif step_type == "scaling":
                    algorithm_func = self.SCALING_ALGORITHMS[method]
                    print(f"[CHECK] 执行缩放方法: {method}, 参数: {params}")  # 输出调试信息

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

    # 自定义sgolayfilt滤波器的封装
    def sgolay_filter_custom(self, spectra, window_length, polyorder):
        # 确保输入数据形状与SGfilter要求一致
        if spectra.shape[0] < spectra.shape[1]:  # 特征数 < 样本数，需要转置
            filtered = savgol_filter(spectra.T, window_length, polyorder, axis=0)
            return filtered.T  # 转回原始形状
        else:
            return savgol_filter(spectra, window_length, polyorder, axis=0)

    def median_filter(self, spectra, k, w):
        return medfilt(spectra, kernel_size=(w, 1))

    def moving_average(self, spectra, k, w):
        kernel = np.ones(w) / w
        return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, spectra)

    # 添加MWA滤波方法的封装
    def mwa_filter(self, spectra, n=6, it=1, mode="full"):
        return MWA(spectra, n=n, it=it, mode=mode)

    # MWM滤波方法的封装
    def mwm_filter(self, spectra, n=7, it=1):
        """使用MWM函数进行移动窗口中值滤波"""
        # 确保输入数据形状与MWM要求一致
        if spectra.shape[0] < spectra.shape[1]:  # 特征数 < 样本数，需要转置
            filtered = MWM(spectra.T, n=n, it=it)
            return filtered.T  # 转回原始形状
        else:
            return MWM(spectra, n=n, it=it)

    # 添加卡尔曼滤波方法的封装
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

    # 添加Smfft傅里叶滤波方法的封装
    def smfft_filter(self, spectra, row_e=51):
        """使用Smfft函数进行傅里叶滤波"""
        # 确保输入数据形状与Smfft要求一致
        if spectra.shape[0] < spectra.shape[1]:  # 特征数 < 样本数，需要转置
            filtered = Smfft(spectra.T, row_e=row_e)
            return filtered.T  # 转回原始形状
        else:
            return Smfft(spectra, row_e=row_e)

    def wavelet_filter(self, spectra, threshold):
        coeffs = pywt.wavedec(spectra, 'db4', axis=0)
        coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
        return pywt.waverec(coeffs, 'db4', axis=0)

    # 新增：小波线性阈值去噪方法的封装
    def wavelet_linear(self, spectra, threshold=0.3):
        """使用新添加的waveletlinear函数进行小波线性阈值去噪"""
        # 确保输入数据形状与waveletlinear要求一致
        if spectra.shape[0] < spectra.shape[1]:  # 特征数 < 样本数，需要转置
            filtered = waveletlinear(spectra.T, threshold=threshold)
            return filtered.T  # 转回原始形状
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
        """使用新的MSC函数实现多元散射校正"""
        # 注意：输入数据形状需要与MSC函数要求一致 (n_samples, n_features)
        # 如果当前数据形状为(n_features, n_samples)，需要先转置
        if spectra.shape[0] < spectra.shape[1]:  # 特征数 < 样本数，说明需要转置
            corrected = MSC(spectra.T)  # 转置后处理
            return corrected.T  # 转回原始形状
        else:
            return MSC(spectra)

    def mm_norm(self, spectra):
        min_vals = np.min(spectra, axis=0)
        max_vals = np.max(spectra, axis=0)
        return (spectra - min_vals) / (max_vals - min_vals)

    def l_norm(self, spectra, p):
        """使用LPnorm函数实现L-范数归一化"""
        if p == "无穷大":
            return LPnorm(spectra, np.inf)
        else:
            p_val = float(p)
            return LPnorm(spectra, p_val)

    def ma_minorm(self, spectra):
        """使用MaMinorm函数实现归一化"""
        return MaMinorm(spectra)

    # 标准化算法实现（均值为0，方差为1）
    def standardize(self, spectra):
        """使用plotst函数实现标准化处理"""
        # 处理数据形状适配
        if spectra.shape[0] < spectra.shape[1]:  # 特征数 < 样本数，需要转置
            standardized = plotst(spectra.T)  # 转置后处理
            return standardized.T  # 转回原始形状
        else:
            return plotst(spectra)

    # 二阶差分方法的封装（归类到基线校准）
    def d2(self, spectra):
        """使用D2函数实现二阶差分计算"""
        # 处理数据形状适配
        if spectra.shape[0] < spectra.shape[1]:  # 特征数 < 样本数，需要转置
            diff_result = D2(spectra.T)  # 转置后处理
            return diff_result.T  # 转回原始形状
        else:
            return D2(spectra)


# ===== 文件处理类 =====
class FileHandler:
    def load_data_from_zip(self, zip_file):
        """从压缩包中加载波数和光谱数据，自动识别数据维度"""
        with zipfile.ZipFile(zip_file, 'r') as zf:
            # 列出压缩包中的所有文件
            file_list = zf.namelist()

            # 尝试识别波数文件和光谱数据文件
            wavenumber_files = [f for f in file_list if 'wave' in f.lower() or 'wn' in f.lower() or '波数' in f]
            data_files = [f for f in file_list if 'spec' in f.lower() or 'data' in f.lower() or '光谱' in f]

            if not wavenumber_files:
                raise ValueError("压缩包中未找到波数文件（通常包含'wave'、'wn'或'波数'）")
            if not data_files:
                raise ValueError("压缩包中未找到光谱数据文件（通常包含'spec'、'data'或'光谱'）")

            # 取第一个符合条件的文件
            wn_file = wavenumber_files[0]
            data_file = data_files[0]

            # 读取波数文件
            with zf.open(wn_file) as f:
                wavenumbers = np.loadtxt(f).ravel()

            # 读取光谱数据文件
            with zf.open(data_file) as f:
                content = f.read().decode("utf-8")
                data = self._parse_data(content)

            return wavenumbers, data.T

    def _parse_data(self, content):
        """解析光谱数据内容，自动识别数据维度"""
        numb = re.compile(r"-?\d+(?:\.\d+)?")
        lines_list = content.splitlines()

        # 提取所有数字
        all_numbers = []
        for line in lines_list:
            all_numbers.extend(numb.findall(line))

        # 尝试确定数据形状
        # 假设波数长度为数据点数
        # 光谱条数 = 总数据点 / 数据点数
        # 这里先简单处理为二维数组
        data = np.array([float(num) for num in all_numbers])

        # 尝试合理的形状（假设每行数据点大致相等）
        # 先按行数划分
        n_rows = len(lines_list)
        n_cols = len(data) // n_rows if n_rows > 0 else 0

        if n_cols * n_rows != len(data):
            # 如果无法完美划分，调整最后一行
            n_cols = len(data) // n_rows + 1
            data = data[:n_rows * n_cols]  # 截断多余数据

        return data.reshape(n_rows, n_cols)

    def export_data(self, filename, data):
        with open(filename, "w") as f:
            for line in data.T:  # 转置回原始格式
                f.write("\t".join(map(str, line)) + "\n")


# 新增：squashing函数（基于余弦的挤压变换）
def squashing(Data):
    row = Data.shape[0]
    col = Data.shape[1]
    sqData = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            sqData[i][j] = (1 - math.cos(Data[i][j] * math.pi)) / 2
    return sqData


# 新增：小波线性阈值去噪函数
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


# sigmoid函数
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


# i_squashing挤压函数（基于余弦的挤压变换，保留原实现以便对比）
def i_squashing(Data):
    row = Data.shape[0]
    col = Data.shape[1]
    sqData = np.zeros((row, col))
    for i in range(row):
        mi = np.min(Data[i])  # 每行的最小值
        diff = np.max(Data[i]) - mi  # 每行的最大值与最小值之差
        for j in range(col):
            # 将数据归一化到[0, 1]范围
            t = (Data[i, j] - mi) / diff if diff != 0 else 0
            # 应用基于余弦的挤压变换：(1 - cos(t * π)) / 2
            m = (1 - math.cos(t * math.pi)) / 2
            # 将结果映射回原始数据范围
            sqData[i][j] = m * diff + mi
    return sqData


# 二阶差分(D2)函数
def D2(sdata):
    """
    计算二阶差分，保持输出尺寸与输入相同
    参数:
        sdata: 输入光谱数据 (n_samples, n_features)
    返回:
        二阶差分结果，形状与输入相同
    """
    row = sdata.shape[0]
    col = sdata.shape[1]
    D2_result = np.zeros((row, col))
    for i in range(row):
        tem = np.diff(sdata[i], 2)
        temp = tem.tolist()
        # 填充最后两个元素以保持与输入相同的尺寸
        temp.append(temp[-1])
        temp.append(temp[-1])
        D2_result[i] = temp
    return D2_result


# LP范数归一化函数
def LPnorm(arr, ord):
    """
    对数组进行Lp范数归一化

    参数:
        arr: 输入数组，形状为(row, col)
        ord: 范数阶数

    返回:
        归一化后的数组，形状与输入相同
    """
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
    """
    对数组进行Ma-Minorm归一化处理
    将数据标准化到[-5, 5]范围

    参数:
        Oarr: 输入数组，形状为(row, col)

    返回:
        归一化后的数组，形状与输入相同
    """
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
    """
    将数据标准化，均值为0，方差为1

    参数:
        Datamat: 输入数据

    返回:
        标准化后的数据
    """
    mu = np.average(Datamat)
    sigma = np.std(Datamat)
    if sigma != 0:
        normDatamat = (Datamat - mu) / sigma
    else:
        normDatamat = Datamat - mu
    return normDatamat


def plotst(Data):
    """
    对数据的每一行进行标准化处理

    参数:
        Data: 输入数据，形状为(row, col)

    返回:
        标准化后的数据，形状与输入相同
    """
    row = Data.shape[0]
    col = Data.shape[1]
    st_Data = np.zeros((row, col))
    for i in range(row):
        st_Data[i] = standardization(Data[i])
    return st_Data


# Smfft傅里叶滤波函数
def Smfft(arr, row_e=51):
    """
    信号进行傅里叶变换，使高频信号的系数为零，再进行傅里叶逆变换
    转换回时域上的信号便是滤波后的效果。

    参数:
        arr: 输入数组，形状为(row, col)
        row_e: 保留的低频分量数量，默认51

    返回:
        滤波后的数组，形状与输入相同
    """
    row = arr.shape[0]
    col = arr.shape[1]
    fftresult = np.zeros((row, col))
    for i in range(row):
        sfft = fftpack_fft(arr[i])  # 使用scipy.fftpack的fft
        row_s = len(arr[i])  # 自适应输入信号长度
        sfftn = copy.deepcopy(sfft)
        # 将高频分量设为零
        sfftn[row_e:row_s - row_e] = 0
        result = fftpack_ifft(sfftn)  # 使用scipy.fftpack的ifft
        real_r = np.real(result)  # 取实部
        fftresult[i] = real_r
    return fftresult


# MSC（多元散射校正）函数
def MSC(sdata):
    """
    多元散射校正(MSC)算法实现

    参数:
        sdata: 输入光谱数据，形状为(n_samples, n_features)

    返回:
        校正后的光谱数据，形状与输入相同
    """
    n = sdata.shape[0]  # 样本数量
    k = np.zeros(sdata.shape[0])  # 斜率
    b = np.zeros(sdata.shape[0])  # 截距

    # 计算平均光谱作为参考
    M = np.mean(sdata, axis=0)

    # 对每个样本进行线性回归，计算斜率和截距
    for i in range(n):
        y = sdata[i, :].reshape(-1, 1)  # 当前样本光谱
        M_reshaped = M.reshape(-1, 1)  # 平均光谱，重塑为二维数组
        model = LinearRegression()
        model.fit(M_reshaped, y)
        k[i] = model.coef_  # 斜率
        b[i] = model.intercept_  # 截距

    # 应用MSC校正
    spec_msc = np.zeros_like(sdata)
    for i in range(n):
        # 将斜率和截距扩展到与光谱长度匹配
        bb = np.repeat(b[i], sdata.shape[1])
        kk = np.repeat(k[i], sdata.shape[1])
        # 应用校正公式：(原始光谱 - 截距) / 斜率
        spec_msc[i, :] = (sdata[i, :] - bb) / kk

    return spec_msc


# 卡尔曼滤波算法实现
def Kalman(z, R):
    """
    单变量卡尔曼滤波

    参数:
        z: 输入信号
        R: 测量噪声方差

    返回:
        滤波后的信号
    """
    n_iter = len(z)
    sz = (n_iter,)  # 数组大小

    Q = 1e-5  # 过程方差

    # 分配数组空间
    xhat = np.zeros(sz)  # 后验估计
    P = np.zeros(sz)  # 后验误差估计
    xhatminus = np.zeros(sz)  # 先验估计
    Pminus = np.zeros(sz)  # 先验误差估计
    K = np.zeros(sz)  # 卡尔曼增益

    # 初始猜测
    xhat[0] = 0.0
    P[0] = 1.0

    for k in range(1, n_iter):
        # 时间更新
        xhatminus[k] = xhat[k - 1]  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k), A=1, BU(k)=0
        Pminus[k] = P[k - 1] + Q  # P(k|k-1) = AP(k-1|k-1)A' + Q(k), A=1

        # 测量更新
        K[k] = Pminus[k] / (Pminus[k] + R)  # Kg(k) = P(k|k-1)H'/[HP(k|k-1)H' + R], H=1
        xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])  # X(k|k)更新
        P[k] = (1 - K[k]) * Pminus[k]  # P(k|k)更新

    return xhat


def KalmanF(xd, R):
    """
    对多维数据应用卡尔曼滤波

    参数:
        xd: 输入数据，形状为(n_samples, n_points)
        R: 测量噪声方差

    返回:
        滤波后的数据，形状与输入相同
    """
    row = xd.shape[0]
    col = xd.shape[1]
    Fxd = np.zeros((row, col))
    for i in range(row):
        Fxd[i] = Kalman(xd[i], R)
    return Fxd


# IModPoly: improved modified multi-polynomial fit method
def IModPoly(wavenumbers, originalRaman, polyorder, max_iter=100, tolerance=0.005):
    """
    改进的多项式拟合基线校正（自适配输入形状）
    入参:
        wavenumbers: 一维波数数组，长度 = n_points
        originalRaman: 光谱矩阵 (n_points, n_samples) 或 (n_samples, n_points)
        polyorder: 多项式阶数
        max_iter, tolerance: 迭代与收敛阈值
    返回:
        与 originalRaman 形状一致的校正结果
    """
    import numpy as np

    # --- 规范化输入 ---
    x = np.asarray(wavenumbers).ravel()
    Y = np.asarray(originalRaman, dtype=float)

    if Y.ndim != 2:
        raise ValueError(f"originalRaman 必须是二维矩阵，当前 ndim={Y.ndim}")

    # 识别并统一到内部形状 (n_samples, n_points)
    transposed_back = False
    if Y.shape[0] == x.size and Y.shape[1] != x.size:  # (n_points, n_samples)
        Y = Y.T
        transposed_back = True
    elif Y.shape[1] == x.size:
        pass  # 已是 (n_samples, n_points)
    else:
        raise ValueError(
            f"输入维度与波数不匹配：wavenumbers(len={x.size}), originalRaman{originalRaman.shape}"
        )

    n_samples, n_points = Y.shape

    # 去 NaN/Inf 并按 x 排序（x 与每条 y 用同一掩码/顺序）
    finite_mask = np.isfinite(x)
    if not np.all(finite_mask):
        x = x[finite_mask]
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]

    # 预先为每条谱准备同样的掩码/排序
    corrected = np.zeros_like(Y)

    for j in range(n_samples):
        y_full = Y[j]
        # 与 x 同步清理
        if not np.all(finite_mask):
            y = y_full[finite_mask]
        else:
            y = y_full

        # 若 y 与 x 仍不等长，说明该条谱自身有 NaN/Inf；再做一次行内掩码
        row_mask = np.isfinite(y)
        x_row = x_sorted if row_mask.all() else x_sorted[row_mask]
        y_row = y[sort_idx] if row_mask.all() else y[sort_idx][row_mask]

        if x_row.size != y_row.size:
            raise ValueError(
                f"样本 {j}：清理后 x 与 y 长度仍不等 (len(x)={x_row.size}, len(y)={y_row.size})"
            )

        # 迭代拟合
        prev_spectrum = y_row.copy()
        curr_spectrum = prev_spectrum.copy()
        prev_std = 0.0
        converged = False
        iteration = 1

        while not converged and iteration <= max_iter:
            # 多项式拟合与残差
            coeffs = np.polyfit(x_row, curr_spectrum, polyorder)
            fitted = np.polyval(coeffs, x_row)
            residual = curr_spectrum - fitted
            curr_std = np.std(residual)

            # 光谱修正
            if iteration == 1:
                mask = prev_spectrum > (fitted + curr_std)
                curr_spectrum[mask] = fitted[mask] + curr_std
            else:
                mask = prev_spectrum < (fitted + curr_std)
                curr_spectrum = np.where(mask, prev_spectrum, fitted + curr_std)

            # 收敛判定
            relative_change = abs((curr_std - prev_std) / curr_std) if curr_std != 0 else 0.0
            converged = (relative_change < tolerance)
            prev_spectrum = curr_spectrum
            prev_std = curr_std
            iteration += 1

        # 将校正结果还原到与 y_full 同长度（插回被掩掉的点）
        # 目标：original - baseline(fitted)
        baseline_row = np.polyval(np.polyfit(x_row, curr_spectrum, polyorder), x_row)

        # 构造完整长度 baseline_full
        baseline_full = np.zeros_like(y_full)
        if not np.all(finite_mask):
            # 先准备 x_full 的排序映射
            x_full = np.asarray(wavenumbers).ravel()
            order_full = np.argsort(x_full[finite_mask])
            # 放回有限位置
            tmp = np.zeros_like(x_row)
            tmp[order_full] = baseline_row  # baseline_row 已与 x_row 同排序
            baseline_full[finite_mask] = tmp
            # 对于非有限位置，保留原值的差为 0（等价于不校正）
            baseline_full[~finite_mask] = 0.0
        else:
            # x 全有限
            baseline_full[np.argsort(np.asarray(wavenumbers).ravel())] = baseline_row

        corrected[j] = y_full - baseline_full

    # 返回到原始形状
    return corrected.T if transposed_back else corrected


# 移动窗口平均（MWA）滤波算法
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


# 改进的非对称加权惩罚最小二乘基线校准算法
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
                    dtw_matrix[i - 1, j],  # 插入
                    dtw_matrix[i, j - 1],  # 删除
                    dtw_matrix[i - 1, j - 1]  # 匹配
                )

        # 回溯路径
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


# 挤压相关函数（已包含新的squashing函数）
def squashing_legacy(x):
    return 1 / (1 + np.exp(-x))


# sgolayfilt滤波器实现
def SGfilter(Intensity, window_length, polyorder):  # 输入均为行
    """
    Savitzky-Golay滤波器实现

    参数:
        Intensity: 输入光谱数据 (n_samples, n_features)
        point: 窗口大小
        degree: 多项式阶数

    返回:
        滤波后的光谱数据，形状与输入相同
    """
    Row = Intensity.shape[0]
    col = Intensity.shape[1]
    sgsmooth = np.zeros((Row, col))
    for i in range(Row):
        sgsmooth[i] = savgol_filter(Intensity[i], point, degree)
    return sgsmooth


# 生成排列时不包含编号 - 提前定义此函数以避免引用错误


def generate_permutations(algorithms):
    """生成完整的算法排列组合，排列名称不包含编号"""

    # 为四种预处理算法分配编号1-4
    algorithm_list = [
        (1, "基线校准", algorithms['baseline']['method'], algorithms['baseline']['params']),
        (2, "缩放", algorithms['scaling']['method'], algorithms['scaling']['params']),
        (3, "滤波", algorithms['filtering']['method'], algorithms['filtering']['params']),
        (4, "挤压", algorithms['squashing']['method'], algorithms['squashing']['params'])
    ]

    all_permutations = []

    # 1. 生成使用1种算法的排列（包括“无预处理”选项）
    for algo in algorithm_list:
        all_permutations.append([algo])  # 生成每种预处理的单独排列

    # 2. 生成使用2种算法的排列
    for comb in itertools.combinations(algorithm_list, 2):
        for perm in itertools.permutations(comb):  # 生成排列
            all_permutations.append(list(perm))

    # 3. 生成使用3种算法的排列
    for comb in itertools.combinations(algorithm_list, 3):
        for perm in itertools.permutations(comb):  # 生成排列
            all_permutations.append(list(perm))

    # 4. 生成使用4种算法的排列
    for comb in itertools.combinations(algorithm_list, 4):
        for perm in itertools.permutations(comb):  # 生成排列
            all_permutations.append(list(perm))

    # 5. 加入“无预处理（原始光谱）”选项
    all_permutations.append([])  # 空列表表示无预处理

    # 格式化排列组合
    formatted_perms = []
    for i, perm in enumerate(all_permutations):
        perm_dict = {
            "name": "",
            "order": [step[0] for step in perm],  # 正确生成order
            "details": perm,
            "count": len(perm),
            "first_step_type": "未知"
        }

        if not perm:  # 无预处理情况
            perm_dict["name"] = "无预处理（原始光谱）"
            perm_dict["first_step_type"] = "无预处理"
        else:
            first_step_type = perm[0][1] if perm and len(perm) > 0 else "未知"
            perm_dict["first_step_type"] = first_step_type

            perm_details = []
            for step in perm:
                perm_details.append(f"{step[1]}({step[2]})")
            perm_dict["name"] = " → ".join(perm_details)

        formatted_perms.append(perm_dict)

    # 输出排列数量以验证
    print(f"生成的排列数量: {len(formatted_perms)}")

    return formatted_perms
def main():
    # 最优先初始化session state
    if 'show_arrangements' not in st.session_state:
        st.session_state.show_arrangements = False

    # 初始化测试相关的session状态变量
    test_states = {
        'k_value': 5,  # 默认k值
        'test_results': None,  # 存储测试结果
        'labels': None,  # 存储样本标签
        'train_indices': None,  # 训练集索引
        'test_indices': None  # 测试集索引
    }
    file_handler = FileHandler()
    preprocessor = Preprocessor()
    # 初始化 current_algorithms 字典
    current_algorithms = {
        'baseline': '无',  # 默认基线校正方法
        'baseline_params': {},
        'scaling': '无',  # 默认缩放方法
        'scaling_params': {},
        'filtering': '无',  # 默认滤波方法
        'filtering_params': {},
        'squashing': '无',  # 默认挤压方法
        'squashing_params': {}
    }

    # 将 current_algorithms 存储到 session_state 中，以便全局访问
    st.session_state.setdefault('current_algorithms', current_algorithms)
    # 初始化其他必要的session状态变量
    other_states = {
        'raw_data': None,
        'processed_data': None,
        'peaks': None,
        'train_test_split_ratio': 0.8,
        'arrangement_results': [],
        'selected_arrangement': None,
        'arrangement_details': {},
        'algorithm_permutations': [],  # 存储算法排列组合
        'current_algorithms': {},  # 存储当前选择的算法
        'filtered_perms': [],  # 存储筛选后的排列方案
        'selected_perm_idx': 0  # 存储当前选中的排列索引
    }

    # 合并所有状态变量并初始化
    all_states = {**test_states, **other_states}
    for key, value in all_states.items():
        if key not in st.session_state:
            st.session_state[key] = value
    st.session_state['current_algorithms'] = current_algorithms
    # 设置页面：紧凑布局
    st.set_page_config(layout="wide", page_icon="🔬", page_title="排列预处理模型")
    # 全局样式调整：更紧凑的字体和间距，确保预处理设置在一行显示
    st.markdown("""
        <style>
        /* 全局字体缩小，确保预处理设置在一行显示 */
        body {font-size: 0.75rem !important;}
        .css-1v0mbdj {padding: 0.3rem 0.5rem !important;} /* 容器内边距 */
        .css-1d391kg {padding: 0.2rem 0 !important;} /* 标题间距 */
        .css-1x8cf1d {line-height: 1.1 !important;} /* 文本行高 */
        .css-12ttj6m {margin-bottom: 0.3rem !important;} /* 组件底部间距 */
        .css-16huue1 {padding: 0.2rem 0.5rem !important; font-size: 0.7rem !important;} /* 按钮内边距和字体 */
        h3 {font-size: 1rem !important; margin: 0.3rem 0 !important;} /* 子标题 */
        .css-1b3298e {gap: 0.3rem !important;} /* 列间距 */
        .stSlider, .stSelectbox, .stTextInput {margin-bottom: 0.3rem !important;} /* 输入组件间距 */
        .stCaption {font-size: 0.65rem !important; margin-top: -0.2rem !important;} /* 说明文字 */
        .css-1544g2n {padding: 0.2rem 0.5rem !important;} /* 展开面板内边距 */
        </style>
    """, unsafe_allow_html=True)

    st.title("🌌 排列预处理模型")

    # 页面整体布局：左侧数据管理，右侧主要内容区
    col_left, col_right = st.columns([1.2, 3.9])

    # ===== 左侧：数据管理模块（移除光谱条数和数据点数）=====
    with col_left:
        with st.expander("📁 数据管理", expanded=True):
            # 1. 提前初始化train_test_ratio
            train_test_ratio = st.session_state.get('train_test_split_ratio', 0.8)

            # 2. 上传压缩包
            zip_file = st.file_uploader("上传包含波数和光谱数据的压缩包", type=['zip'], key="zip_file")
            st.caption("压缩包(.zip)需包含波数和光谱数据文件")

            # 数据加载逻辑（压缩包上传后、样本标签前）
            if zip_file:
                try:
                    st.session_state.raw_data = file_handler.load_data_from_zip(zip_file)

                    if st.session_state.get('labels') is not None:
                        st.success(
                            f"✅ 数据加载成功：{st.session_state.raw_data[1].shape[1]}条光谱，{len(np.unique(st.session_state.labels))}类")
                    else:
                        st.success(
                            f"✅ 数据加载成功：{st.session_state.raw_data[1].shape[1]}条光谱，{st.session_state.raw_data[1].shape[0]}个点")
                        st.warning("⚠️ 请输入样本标签以进行分类测试")

                    # 【调整1：数据维度提示移至数据加载成功提示下方】
                    if st.session_state.get('raw_data'):
                        wavenumbers, y = st.session_state.raw_data
                        st.info(f"📊 数据维度: {y.shape[1]}条 × {y.shape[0]}点")

                except Exception as e:
                    st.error(f"❌ 文件加载失败: {str(e)}")

            # 3. 样本标签区域（数据维度提示下方）
            st.subheader("样本标签")
            num_classes = st.number_input("类别数量", min_value=1, value=2, step=1, key="num_cls")

            # 定义标签输入
            labels_input = st.text_input(
                "标签（逗号分隔，与光谱顺序一致）",
                placeholder="例：0,0,1,1",
                key="labels_in"
            )

            # 标签验证逻辑
            if labels_input and st.session_state.get('raw_data'):
                try:
                    labels = np.array([int(l.strip()) for l in labels_input.split(',')])
                    if len(labels) == st.session_state.raw_data[1].shape[1]:
                        st.session_state.labels = labels
                        n_samples = len(labels)
                        train_size = int(n_samples * train_test_ratio)
                        indices = np.random.permutation(n_samples)
                        st.session_state.train_indices = indices[:train_size]
                        st.session_state.test_indices = indices[train_size:]
                    else:
                        st.warning(f"⚠️ 标签数({len(labels)})≠光谱数({st.session_state.raw_data[1].shape[1]})")
                        st.session_state.labels = None
                except Exception as e:
                    st.warning(f"⚠️ 标签格式错误: {str(e)}")
                    st.session_state.labels = None

            # 【调整2：类别分布提示移至原数据维度位置（样本标签区域末尾）】
            if st.session_state.get('raw_data') and st.session_state.get('labels') is not None:
                class_counts = np.bincount(st.session_state.labels)
                st.info(
                    f"🏷️ 类别分布: {', '.join([f'类{i}:{count}个' for i, count in enumerate(class_counts) if count > 0])}")

            # 4. 训练测试划分区域
            st.subheader("训练测试划分")
            train_test_ratio = st.slider(
                "训练集比例",
                min_value=0.1,
                max_value=0.9,
                value=train_test_ratio,
                step=0.1,
                format="%.1f",
                key="train_ratio"
            )
            st.session_state.train_test_split_ratio = train_test_ratio

            # 训练集:测试集提示
            st.info(f"🔢 训练集:{train_test_ratio:.1f} | 测试集:{1 - train_test_ratio:.1f}")

        # 处理流程提示
        if st.session_state.get('process_method'):
            st.success(f"🛠️ 处理流程: {st.session_state.process_method}")

        # 使用说明
        with st.expander("ℹ️ 使用指南", expanded=False):
            st.markdown("""
            1. 上传包含波数和光谱数据的压缩包  
            2. 设置标签和训练测试比例  
            3. 右侧上方选择预处理方法  
            4. 点击"显示排列"生成方案  
            5. 选择k值后点击"测试"  
            6. 查看结果并导出
            """)

    # ===== 右侧：预处理设置和光谱可视化 =====
    with col_right:
        # ===== 预处理设置（横向排列在光谱可视化上方，与四种算法在同一行）=====
        st.subheader("⚙️ 预处理设置", divider="gray")

        # 布局列数保持10列
        preprocess_cols = st.columns([1, 1, 1, 1, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2], gap="small")

        # 1. 基线校准（第一列，不变）
        with preprocess_cols[0]:
            st.subheader("基线校准")
            baseline_method = st.selectbox(
                "方法",
                ["无", "SD", "FD", "多项式拟合", "ModPoly", "I-ModPoly", "PLS", "AsLS", "airPLS", "二阶差分(D2)"],
                key="baseline_method",
                label_visibility="collapsed"
            )

            # 基线参数（不变）
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
                elif baseline_method == "I-ModPoly":  # IModPoly参数设置
                    polyorder = st.slider("多项式阶数", 3, 7, 5, key="imod_polyorder", label_visibility="collapsed")
                    max_iter = st.slider("最大迭代次数", 50, 200, 100, key="imod_maxiter", label_visibility="collapsed")
                    tolerance = st.slider("收敛容差", 0.001, 0.01, 0.005, key="imod_tol", label_visibility="collapsed")
                    baseline_params["polyorder"] = polyorder
                    baseline_params["max_iter"] = max_iter
                    baseline_params["tolerance"] = tolerance
                    st.caption(f"阶数: {polyorder}, 迭代: {max_iter}, 容差: {tolerance}")
                elif baseline_method == "PLS":
                    lam = st.selectbox("λ", [10 ** 10, 10 ** 8, 10 ** 7], key="lam_pls", label_visibility="collapsed")
                    baseline_params["lam"] = lam
                    st.caption(f"λ: {lam}")
                elif baseline_method == "AsLS":
                    p = st.selectbox("非对称系数p", [0.001, 0.01, 0.1], key="p_asls", label_visibility="collapsed")
                    lam = st.selectbox("平滑系数λ", [10 ** 5, 10 ** 7, 10 ** 9], key="lam_asls",
                                       label_visibility="collapsed")
                    niter = st.selectbox("迭代次数", [5, 10, 15], key="niter_asls", label_visibility="collapsed")
                    baseline_params["lam"] = lam
                    baseline_params["p"] = p
                    baseline_params["niter"] = niter
                    st.caption(f"p: {p}, λ: {lam}, 迭代次数: {niter}")
                elif baseline_method == "airPLS":
                    lam = st.selectbox("λ", [10 ** 7, 10 ** 4, 10 ** 2], key="lam_air", label_visibility="collapsed")
                    baseline_params["lam"] = lam
                    st.caption(f"λ: {lam}")
                elif baseline_method == "二阶差分(D2)":  # 二阶差分参数说明
                    st.caption("二阶差分可增强光谱特征，抑制基线漂移")

        # 2. 缩放处理（第二列，不变）
        with preprocess_cols[1]:
            st.subheader("📏 缩放")
            scaling_method = st.selectbox(
                "方法",
                ["无", "Peak-Norm", "SNV", "MSC", "M-M-Norm", "L-范数", "Ma-Minorm", "标准化(均值0，方差1)"],
                key="scaling_method",
                label_visibility="collapsed"
            )

            # 缩放参数（不变）
            scaling_params = {}
            if scaling_method == "L-范数":
                p = st.selectbox("p", ["无穷大", "4", "10"], key="p_scale", label_visibility="collapsed")
                scaling_params["p"] = p
                st.caption(f"p: {p}")
            elif scaling_method == "标准化(均值0，方差1)":
                st.caption("将数据标准化到均值为0，方差为1")

        # 3. 滤波处理（第三列，不变）
        with preprocess_cols[2]:
            st.subheader("📶 滤波")
            filtering_method = st.selectbox(
                "方法",
                ["无", "Savitzky-Golay", "sgolayfilt滤波器", "中值滤波(MF)", "移动平均(MAF)",
                 "MWA（移动窗口平均）", "MWM（移动窗口中值）", "卡尔曼滤波", "Lowess", "FFT",
                 "Smfft傅里叶滤波", "小波变换(DWT)", "小波线性阈值去噪"],
                key="filtering_method",
                label_visibility="collapsed"
            )

            # 滤波参数（不变）
            filtering_params = {}
            if filtering_method != "无":
                if filtering_method in ["Savitzky-Golay", "sgolayfilt滤波器"]:
                    k = st.selectbox("多项式阶数", [3, 7], key="k_sg", label_visibility="collapsed")
                    w = st.selectbox("窗口大小", [11, 31, 51], key="w_sg", label_visibility="collapsed")
                    filtering_params["window_length"] = w
                    filtering_params["polyorder"] = k
                    st.caption(f"阶数: {k}, 窗口: {w}")
                elif filtering_method in ["中值滤波(MF)", "移动平均(MAF)"]:
                    k = st.selectbox("k", [1, 3], key="k_mf", label_visibility="collapsed")
                    w = st.selectbox("w", [7, 11], key="w_mf", label_visibility="collapsed")
                    filtering_params["k"] = k
                    filtering_params["w"] = w
                    st.caption(f"k: {k}, w: {w}")
                elif filtering_method == "MWA（移动窗口平均）":
                    n = st.selectbox("窗口大小n", [4, 6, 8], key="n_mwa", label_visibility="collapsed")
                    it = st.selectbox("迭代次数it", [1, 2, 3], key="it_mwa", label_visibility="collapsed")
                    filtering_params["n"] = n
                    filtering_params["it"] = it
                    filtering_params["mode"] = "full"
                    st.caption(f"窗口大小: {n}, 迭代次数: {it}")
                elif filtering_method == "MWM（移动窗口中值）":
                    n = st.selectbox("窗口大小n", [5, 7, 9], key="n_mwm", label_visibility="collapsed")
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
                    row_e = st.selectbox("保留低频分量数", [31, 51, 71], key="row_e_smfft",
                                         label_visibility="collapsed")
                    filtering_params["row_e"] = row_e
                    st.caption(f"保留低频分量数: {row_e}")
                elif filtering_method == "小波变换(DWT)":
                    threshold = st.selectbox("阈值", [0.1, 0.3, 0.5], key="thresh_dwt", label_visibility="collapsed")
                    filtering_params["threshold"] = threshold
                    st.caption(f"阈值: {threshold}")
                elif filtering_method == "小波线性阈值去噪":
                    threshold = st.selectbox("阈值", [0.1, 0.3, 0.5], key="thresh_wavelet_linear",
                                             label_visibility="collapsed")
                    filtering_params["threshold"] = threshold
                    st.caption(f"阈值: {threshold}")

        # 4. 挤压处理（第四列，不变）
        with preprocess_cols[3]:
            st.subheader("🧪 挤压")
            squashing_method = st.selectbox(
                "方法",
                ["无", "Sigmoid挤压", "改进的Sigmoid挤压", "逻辑函数", "余弦挤压(squashing)", "改进的逻辑函数",
                 "DTW挤压"],
                key="squashing_method",
                label_visibility="collapsed"
            )

            # 挤压参数（不变）
            squashing_params = {}
            if squashing_method != "无":
                if squashing_method == "改进的逻辑函数":
                    st.caption("基于余弦的挤压变换，无额外参数")
                elif squashing_method == "改进的Sigmoid挤压":
                    maxn = st.selectbox("maxn", [5, 10, 15], key="maxn_isigmoid", label_visibility="collapsed")
                    squashing_params["maxn"] = maxn
                    st.caption(f"maxn: {maxn}")
                elif squashing_method == "DTW挤压":
                    l = st.selectbox("l", [1, 5], key="l_dtw", label_visibility="collapsed")
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

        # 5-10列：操作相关内容
        # 原操作2 → 新操作1：显示排列与筛选（第4列，不变）
        with preprocess_cols[4]:
            st.subheader("操作1")
            # 应用处理按钮（移除了推荐应用按钮）
            if st.button("🚀 应用处理", type="primary", use_container_width=True, key="apply_btn"):
                if st.session_state.raw_data is None:
                    st.warning("⚠️ 请先上传数据")
                else:
                    try:
                        wavenumbers, y = st.session_state.raw_data
                        processed_data, method_name = preprocessor.process(
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

                        arr_name = f"排列_{len(st.session_state.arrangement_results) + 1}"
                        st.session_state.arrangement_results.append(arr_name)
                        st.session_state.arrangement_details[arr_name] = {
                            'data': processed_data,
                            'method': " → ".join(method_name),
                            'params': current_algorithms
                        }
                        st.session_state.selected_arrangement = arr_name
                        st.session_state.processed_data = (wavenumbers, processed_data)
                        st.session_state.process_method = " → ".join(method_name)
                        st.success(f"✅ 处理完成")

                    except Exception as e:
                        st.error(f"❌ 处理失败: {str(e)}")

        with preprocess_cols[4]:
            st.subheader("操作1")

            # 显示排列按钮（原逻辑不变）
            if st.button("🔍 显示排列", type="secondary", use_container_width=True, key="show_perm_btn"):
                st.session_state.show_arrangements = not st.session_state.show_arrangements

                if st.session_state.show_arrangements:
                    # 动态构建 selected_algorithms 字典
                    selected_algorithms = {
                        'baseline': {
                            'method': baseline_method,  # 使用用户选择的算法
                            'params': baseline_params if baseline_method != '无' else {}  # 根据用户选择传递参数
                        },
                        'scaling': {
                            'method': scaling_method,  # 使用用户选择的算法
                            'params': scaling_params if scaling_method != '无' else {}
                        },
                        'filtering': {
                            'method': filtering_method,  # 使用用户选择的算法
                            'params': filtering_params if filtering_method != '无' else {}
                        },
                        'squashing': {
                            'method': squashing_method,  # 使用用户选择的算法
                            'params': squashing_params if squashing_method != '无' else {}
                        }
                    }

                    # st.write("selected_algorithms: ", selected_algorithms)
                    # 生成排列组合并存储（原逻辑不变）
                    st.session_state.algorithm_permutations = generate_permutations(selected_algorithms)

                    st.session_state.filtered_perms = st.session_state.algorithm_permutations
                    # st.success(f"✅ 生成了 {len(st.session_state.algorithm_permutations)} 种排列组合")
                    # st.write("生成的排列组合: ", st.session_state.algorithm_permutations)

                    # 获取用户输入的标签（原逻辑不变）
                    if 'labels' not in st.session_state or st.session_state.labels is None:
                        st.error("❌ 标签尚未设置！请先通过主函数获取并验证标签。")
                        return

                    labels = st.session_state.labels  # 获取已存储的标签
                    # st.write("标签已加载：", labels)

                    # 获取原始光谱数据并进行处理（原逻辑不变）
                    if st.session_state.get('raw_data'):
                        wavenumbers, y = st.session_state.raw_data
                        S = len(labels)
                        P = len(st.session_state.algorithm_permutations)
                        N = len(wavenumbers)
                        # st.write("[CHECK] S, P, N =", S, P, N)
                        # st.write("[CHECK] raw_data shapes -> y:", np.asarray(y).shape, "; wavenumbers:",len(wavenumbers))
                        # --- 1) 构建 (S, P, N) 的三维立方体 ---
                        processed_cube = np.empty((S, P, N), dtype=np.float32)

                        y_arr = np.asarray(y)

                        def get_spectrum_j(j_idx: int) -> np.ndarray:
                            if y_arr.ndim == 2:
                                if y_arr.shape[0] == N and y_arr.shape[1] == S:  # N×S
                                    return y_arr[:, j_idx]
                                elif y_arr.shape[0] == S and y_arr.shape[1] == N:  # S×N
                                    return y_arr[j_idx, :]
                                else:
                                    raise ValueError(
                                        f"原始光谱矩阵维度不匹配，期望含有 N={N} 与 S={S} 之一的维度，当前形状={y_arr.shape}")
                            elif y_arr.ndim == 1:
                                raise ValueError("原始光谱只有 1 条，无法构建 (S,P,N) 立方体。")
                            else:
                                raise ValueError(f"不支持的原始光谱维度：{y_arr.ndim}")

                        # st.write(f"[CHECK] algorithm_permutations:", st.session_state.algorithm_permutations)

                        #遍历填充立方体（原逻辑不变）
                        for j in range(S):
                            spec_j = get_spectrum_j(j).astype(np.float32)
                            # st.write(f"[CHECK] 第 {j + 1} 条光谱数据：", spec_j)  # 输出当前光谱数据
                            if spec_j.shape[0] != N:
                                raise ValueError(f"第 {j + 1} 条光谱长度 {spec_j.shape[0]} 与波数长度 N={N} 不一致。")

                            for i, perm in enumerate(st.session_state.algorithm_permutations):
                                # st.write(f"[CHECK] perm {i}: {perm}")
                                algorithm_order = perm.get('order', [])  # 获取顺序
                                # st.write(f"[CHECK] algorithm_order: {algorithm_order}")
                                # st.write(f"[CHECK] perm['details']: {perm['details']}")

                                # 从 details 中获取每个算法的参数
                                bm = next((step[2] for step in perm['details'] if step[1] == '基线校准'), '无')
                                sm = next((step[2] for step in perm['details'] if step[1] == '缩放'), '无')
                                fm = next((step[2] for step in perm['details'] if step[1] == '滤波'), '无')
                                qm = next((step[2] for step in perm['details'] if step[1] == '挤压'), '无')

                                # 获取参数，并确保它们是字典格式
                                baseline_params = next(
                                    (step[3] if isinstance(step[3], dict) else {'k': step[3]} for step in
                                     perm['details'] if step[1] == '基线校准'), {'k': 8})
                                scaling_params = next(
                                    (step[3] if isinstance(step[3], dict) else {} for step in perm['details'] if
                                     step[1] == '缩放'), {})
                                filtering_params = next(
                                    (step[3] if isinstance(step[3], dict) else {} for step in perm['details'] if
                                     step[1] == '滤波'), {})
                                squashing_params = next(
                                    (step[3] if isinstance(step[3], dict) else {} for step in perm['details'] if
                                     step[1] == '挤压'), {})


                                processed_data, _method_name = preprocessor.process(
                                    wavenumbers, spec_j,
                                    baseline_method=bm, baseline_params=baseline_params,
                                    squashing_method=qm, squashing_params=squashing_params,
                                    filtering_method=fm, filtering_params=filtering_params,
                                    scaling_method=sm, scaling_params=scaling_params,
                                    algorithm_order=algorithm_order
                                )

                                # 输出处理后的数据
                                st.write(f"[CHECK] 处理后的数据 (排列 {i + 1}): {processed_data}")

                                arr = np.asarray(processed_data, dtype=np.float32).reshape(-1)
                                #
                                # st.write(f"[CHECK] 存入 processed_cube[{j}, {i}, :] 的数据: {arr}")

                        # st.write("[CHECK] processed_cube.shape =", processed_cube.shape)
                        # st.write("[CHECK] processed_cube[0, 0, :5] =", processed_cube[0, 0, :5].tolist())
                        # # --- 2) 元信息写入 ---
                        # st.session_state.wavenumbers = np.asarray(wavenumbers)
                        # st.session_state.labels = np.asarray(labels, dtype=int)
                        # st.session_state.perm_info = [
                        #     {
                        #         "name": perm.get("name", f"方案{i + 1}"),
                        #         "order": perm.get("order", []),
                        #         "params": perm.get("params", {})
                        #     }
                        #     for i, perm in enumerate(st.session_state.algorithm_permutations)
                        # ]
                        # st.session_state.processed_cube = processed_cube
                        # #--- 3) PCA+LDA评估（原逻辑不变）
                        # from sklearn.decomposition import PCA
                        # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
                        #
                        # X_labels = st.session_state.labels
                        # pca_pred_matrix = np.empty((P, S), dtype=int)
                        # pca_acc = np.empty(P, dtype=np.float32)
                        #
                        # for p in range(P):
                        #     X_p = processed_cube[:, p, :]
                        #     n_components = min(max(1, S - 1), X_p.shape[1])
                        #     pca = PCA(n_components=n_components, svd_solver="auto", random_state=0)
                        #     Z = pca.fit_transform(X_p)
                        #
                        #     if np.unique(X_labels).size < 2:
                        #         y_hat = np.full(S, int(X_labels[0]), dtype=int)
                        #     else:
                        #         clf = LDA(solver="lsqr")
                        #         clf.fit(Z, X_labels)
                        #         y_hat = clf.predict(Z)
                        #
                        #     pca_pred_matrix[p, :] = y_hat
                        #     pca_acc[p] = (y_hat == X_labels).mean().astype(np.float32)
                        #
                        # #排序与投票（原逻辑不变）
                        # st.session_state.pca_pred_matrix = pca_pred_matrix
                        # st.session_state.pca_acc = pca_acc
                        #
                        # sorted_idx = np.argsort(-st.session_state.pca_acc, kind="mergesort")
                        # st.session_state.pca_sorted_perm_indices = sorted_idx
                        # st.session_state.pca_sorted_acc = st.session_state.pca_acc[sorted_idx]
                        # st.session_state.pca_sorted_pred_matrix = st.session_state.pca_pred_matrix[sorted_idx]
                        #
                        # st.write("[CHECK] pca_pred_matrix.shape =", st.session_state.pca_pred_matrix.shape)
                        # st.write("[CHECK] pca_acc.shape =", st.session_state.pca_acc.shape)
                        # st.write("[CHECK] top-5 acc =", st.session_state.pca_sorted_acc[:5].round(3).tolist())
                        # st.write("[CHECK] top-1 preds =", st.session_state.pca_sorted_pred_matrix[0].tolist())
                        #
                        # from scipy.stats import mode
                        # P2, S2 = st.session_state.pca_sorted_pred_matrix.shape
                        # vote_pred_matrix_by_k = np.empty((P2, S2), dtype=int)
                        #
                        # for k in range(1, P2 + 1):
                        #     topk = st.session_state.pca_sorted_pred_matrix[:k, :]
                        #     voted = mode(topk, axis=0, keepdims=False).mode
                        #     vote_pred_matrix_by_k[k - 1, :] = voted
                        #
                        # st.session_state.vote_pred_matrix_by_k = vote_pred_matrix_by_k
                        # vote_acc_by_k = (vote_pred_matrix_by_k == st.session_state.labels.reshape(1, S2)).mean(
                        #     axis=1).astype(np.float32)
                        # st.session_state.vote_acc_by_k = vote_acc_by_k
                        #
                        # st.write("[CHECK] vote_pred_matrix_by_k.shape =", st.session_state.vote_pred_matrix_by_k.shape)
                        # st.write("[CHECK] vote_acc_by_k[:5] =", st.session_state.vote_acc_by_k[:5].round(3).tolist())
                        # st.write("[CHECK] k=5 voted preds =",st.session_state.vote_pred_matrix_by_k[4].tolist() if P2 >= 5 else "P<5")
                        # k_vals = np.arange(1, st.session_state.vote_acc_by_k.shape[0] + 1)
                        # best_k = int(k_vals[np.argmax(st.session_state.vote_acc_by_k)])
                        # best_acc = float(st.session_state.vote_acc_by_k.max())
                        #
                        # import matplotlib.pyplot as plt
                        # fig, ax = plt.subplots()
                        # ax.plot(k_vals, st.session_state.vote_acc_by_k, marker='o')
                        # ax.set_xlabel('k（前k个方案投票）')
                        # ax.set_ylabel('Accuracy')
                        # ax.set_title(f'k值曲线（最佳k={best_k}, acc={best_acc:.3f}）')
                        # ax.set_xlim(1, k_vals[-1])
                        # ax.set_ylim(0, 1)
                        # ax.grid(True, linestyle='--', alpha=0.4)
                        # st.pyplot(fig)
                        #
                        # st.write("[CHECK] best k =", best_k, "; preds =",
                        #          st.session_state.vote_pred_matrix_by_k[best_k - 1].tolist())
                        # st.success(
                        #     f"✅ 已构建立方体 processed_cube 形状 = {processed_cube.shape}，并完成 {P} 个方案的 PCA 评估。")

                    else:
                        st.error("❌ 请先上传原始光谱数据")
            else:
                st.session_state.filtered_perms = []

        # 原操作3 → 新操作2：排列选择与应用（第5列，不变）
        with preprocess_cols[5]:
            st.subheader("操作2")
            # 排列下拉框（原逻辑不变）
            if st.session_state.show_arrangements and st.session_state.filtered_perms:
                st.session_state.selected_perm_idx = st.selectbox(
                    f"选择方案（共{len(st.session_state.filtered_perms)}种）",
                    range(len(st.session_state.filtered_perms)),
                    format_func=lambda x: st.session_state.filtered_perms[x].get("name", f"方案{x + 1}"),
                    key="perm_select",
                    label_visibility="collapsed",
                    help="选择预处理算法顺序"
                )

                # 应用排列按钮（原逻辑不变）
                try:
                    selected_perm = st.session_state.filtered_perms[st.session_state.selected_perm_idx]
                    st.caption(f"当前: {selected_perm.get('name', '未知')}")

                    if st.button("✅ 应用方案", type="primary", use_container_width=True, key="apply_perm_btn"):
                        if st.session_state.raw_data is None:
                            st.warning("⚠️ 请先上传数据")
                        else:
                            try:
                                algos = {
                                    'baseline': baseline_method,
                                    'baseline_params': baseline_params,
                                    'squashing': squashing_method,
                                    'squashing_params': squashing_params,
                                    'filtering': filtering_method,
                                    'filtering_params': filtering_params,
                                    'scaling': scaling_method,
                                    'scaling_params': scaling_params,
                                }
                                wavenumbers, y = st.session_state.raw_data
                                processed_data, method_name = preprocessor.process(
                                    wavenumbers, y,
                                    baseline_method=baseline_method,
                                    baseline_params=baseline_params,
                                    squashing_method=squashing_method,
                                    squashing_params=squashing_params,
                                    filtering_method=filtering_method,
                                    filtering_params=filtering_params,
                                    scaling_method=scaling_method,
                                    scaling_params=scaling_params,
                                    algorithm_order=selected_perm.get('order', [])
                                )

                                arr_name = f"排列_{len(st.session_state.arrangement_results) + 1}"
                                st.session_state.arrangement_results.append(arr_name)
                                st.session_state.arrangement_details[arr_name] = {
                                    'data': processed_data,
                                    'method': " → ".join(method_name),
                                    'order': selected_perm.get('order', []),
                                    'params': algos
                                }
                                st.session_state.selected_arrangement = arr_name
                                st.session_state.processed_data = (wavenumbers, processed_data)
                                st.session_state.process_method = " → ".join(method_name)
                                st.success(f"✅ 方案应用完成")
                            except Exception as e:
                                st.error(f"❌ 应用失败: {str(e)}")
                except Exception as e:
                    st.error(f"❌ 方案处理错误: {str(e)}")
            else:
                if st.session_state.show_arrangements:
                    st.info("ℹ️ 无符合条件的方案")

        # 【修改】操作3：仅保留计算k值按钮（第6列）
        with preprocess_cols[6]:
            st.subheader("操作3")
            # 只保留计算k值按钮，移除结果显示相关元素
            if st.button("🔢 计算k值", type="secondary", use_container_width=True, key="calc_k_btn"):
                # 暂存计算结果（逻辑后续补充）
                st.session_state.calc_k_result = 5  # 示例默认值
                st.success("✅ k值计算完成")

        # 【修改】操作4：显示k值结果（第7列）
        with preprocess_cols[7]:
            st.subheader("k值结果为")  # 文本改为"k值结果为"
            # 移除k值设置输入框和确定按钮，仅显示计算后的k值
            calc_k_result = st.session_state.get('calc_k_result', "未计算")
            st.info(f" {calc_k_result}")

        # 【新增】操作5：选择k值（第8列，不变）
        with preprocess_cols[8]:
            st.subheader("操作5")
            # 选择k值按钮
            if st.button("📌 选择k值", type="secondary", use_container_width=True, key="select_k_btn"):
                # 暂存选择状态
                st.session_state.selected_k = st.session_state.get('calc_k_result', "未选择")
                st.success(f"✅ 已选择k值：{st.session_state.selected_k}")
            # 显示当前选择的k值
            selected_k = st.session_state.get('selected_k', "未选择")
            st.caption(f"当前选择：{selected_k}")

        # 原操作5 → 新操作6：测试按钮（第9列，不变）
        with preprocess_cols[9]:
            st.subheader("操作6")
            # 测试按钮（原逻辑不变）
            if st.button("测试", type="primary", use_container_width=True, key="test_btn"):
                if st.session_state.raw_data is None:
                    st.warning("⚠️ 请先上传数据")
                elif st.session_state.selected_arrangement is None:
                    st.warning("⚠️ 请先应用排列方案")
                elif st.session_state.labels is None:
                    st.warning("⚠️ 请先输入标签")
                elif st.session_state.train_indices is None:
                    st.warning("⚠️ 无法划分训练集")
                else:
                    try:
                        selected_arr = st.session_state.selected_arrangement
                        processed_data = st.session_state.arrangement_details[selected_arr]['data']
                        train_idx = st.session_state.train_indices
                        test_idx = st.session_state.test_indices

                        train_data = processed_data[:, train_idx]
                        test_data = processed_data[:, test_idx]
                        train_labels = st.session_state.labels[train_idx]
                        test_labels = st.session_state.labels[test_idx]

                        with st.spinner("测试中..."):
                            # 使用选择的k值进行测试
                            predictions = knn_classify(
                                train_data,
                                train_labels,
                                test_data,
                                k=st.session_state.get('selected_k', 1)  # 优先使用选择的k值
                            )

                        accuracy = accuracy_score(test_labels, predictions)
                        kappa = cohen_kappa_score(test_labels, predictions)
                        cm = confusion_matrix(test_labels, predictions)

                        st.session_state.test_results = {
                            'accuracy': accuracy,
                            'kappa': kappa,
                            'confusion_matrix': cm,
                            'predictions': predictions,
                            'test_labels': test_labels
                        }

                        st.success("✅ 测试完成！结果在下方")

                    except Exception as e:
                        st.error(f"❌ 测试失败: {str(e)}")

        # 保存当前选择的算法（不变）
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

        # ===== 光谱可视化与结果导出（在预处理设置下方）=====
        st.subheader("📈 光谱可视化", divider="gray")

        # 创建四个固定区域的布局：原始光谱、预处理后光谱、k值曲线、混淆矩阵
        # 第一行：原始光谱和预处理后光谱
        viz_row1 = st.columns(2, gap="medium")

        # 第二行：k值曲线和混淆矩阵
        viz_row2 = st.columns(2, gap="medium")

        # 1. 原始光谱区域（第一行第一列）- 随机显示一个原始光谱
        with viz_row1[0]:
            st.subheader("原始光谱", divider="gray")
            if st.session_state.get('raw_data'):
                wavenumbers, y = st.session_state.raw_data
                # 确保y是二维数组（N×S），取随机列索引
                num_samples = y.shape[1] if y.ndim == 2 else 1
                random_idx = np.random.randint(0, num_samples)  # 随机选择一个光谱
                # 显示随机选择的原始光谱
                raw_data = pd.DataFrame({f"原始光谱（随机）": y[:, random_idx]}, index=wavenumbers)
                st.line_chart(raw_data, height=250)
                st.caption(f"随机展示第 {random_idx + 1}/{num_samples} 条原始光谱")
            else:
                st.markdown(
                    '<div style="border:1px dashed #ccc; height:250px; display:flex; align-items:center; justify-content:center;">等待加载原始数据</div>',
                    unsafe_allow_html=True)

        # 2. 预处理后光谱区域（第一行第二列）
        with viz_row1[1]:
            st.subheader("预处理后的光谱", divider="gray")
            if st.session_state.get('selected_arrangement'):
                selected_arr = st.session_state.selected_arrangement
                arr_data = st.session_state.arrangement_details[selected_arr]['data']
                arr_method = st.session_state.arrangement_details[selected_arr]['method']
                st.caption(f"处理方法: {arr_method}")

                idx1 = 0 if arr_data.shape[1] > 0 else 0
                proc_data1 = pd.DataFrame({"预处理后1": arr_data[:, idx1]}, index=wavenumbers)
                st.line_chart(proc_data1, height=250)

                # 显示更多预处理后光谱（不使用嵌套列）
                if arr_data.shape[1] > 1:
                    with st.expander("查看更多预处理后光谱", expanded=False):
                        for i in range(1, min(arr_data.shape[1], 5)):
                            st.subheader(f"预处理后{i + 1}", divider="gray")
                            data = pd.DataFrame({f"预处理后{i + 1}": arr_data[:, i]}, index=wavenumbers)
                            st.line_chart(data, height=150)
            else:
                st.markdown(
                    '<div style="border:1px dashed #ccc; height:250px; display:flex; align-items:center; justify-content:center;">请先应用预处理方案</div>',
                    unsafe_allow_html=True)

        # 3. k值曲线区域（第二行第一列）
        with viz_row2[0]:
            st.subheader("k值曲线", divider="gray")
            with st.container():
                if st.session_state.get('selected_arrangement'):
                    selected_arr = st.session_state.selected_arrangement
                    arr_data = st.session_state.arrangement_details[selected_arr]['data']
                    wavenumbers, y = st.session_state.raw_data
                    arr_order = st.session_state.arrangement_details[selected_arr].get('order', [])

                    if arr_order:  # 只有应用了预处理才有k值曲线
                        idx1 = 0 if arr_data.shape[1] > 0 else 0
                        k_vals1 = np.abs(arr_data[:, 0] / (y[:, 0] + 1e-8)) if y.shape[1] > 0 else np.array([])
                        k_data1 = pd.DataFrame({"k值1": k_vals1}, index=wavenumbers)
                        st.line_chart(k_data1)

                        # 显示更多k值曲线（折叠面板）
                        if y.shape[1] > 1:
                            with st.expander("查看更多k值曲线", expanded=False):
                                for i in range(1, min(y.shape[1], 5)):
                                    st.subheader(f"k值{i + 1}", divider="gray")
                                    k_vals = np.abs(arr_data[:, i] / (y[:, i] + 1e-8))
                                    data = pd.DataFrame({f"k值{i + 1}": k_vals}, index=wavenumbers)
                                    st.line_chart(data, height=150)
                    else:
                        st.info("ℹ️ 无预处理（原始光谱），不显示k值曲线")
                else:
                    st.markdown(
                        '<div style="border:1px dashed #ccc; height:200px; display:flex; align-items:center; justify-content:center;">请先应用预处理方案</div>',
                        unsafe_allow_html=True)

        # 4. 混淆矩阵区域（第二行第二列）
        with viz_row2[1]:
            st.subheader("混淆矩阵", divider="gray")
            st.markdown("""
                <style>
                [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlock"] {
                    height: 100% !important;
                }
                [data-testid="stMatplotlibChart"] {
                    margin: 0 !important;
                    padding: 0 !important;
                }
                </style>
            """, unsafe_allow_html=True)

            if st.session_state.get('test_results') is not None:
                results = st.session_state.test_results

                fig, ax = plt.subplots(figsize=(2.5, 1.5))
                sns.heatmap(
                    results['confusion_matrix'],
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    ax=ax,
                    annot_kws={"size": 4},
                    cbar_kws={"shrink": 0.9}
                )
                ax.set_xlabel('预测标签', fontsize=2, labelpad=2)
                ax.set_ylabel('真实标签', fontsize=2, labelpad=2)
                ax.set_title('混淆矩阵', fontsize=4, pad=4)
                plt.xticks(fontsize=3, rotation=0)
                plt.yticks(fontsize=3, rotation=0)
                plt.tight_layout(pad=0.1)
                st.pyplot(fig, use_container_width=True)
            else:
                st.markdown(
                    '<div style="border:1px dashed #ccc; height:200px; display:flex; align-items:center; justify-content:center;">请先进行分类测试</div>',
                    unsafe_allow_html=True)
        # 结果导出
        if st.session_state.arrangement_results or st.session_state.get('processed_data'):
            st.subheader("💾 结果导出", divider="gray")
            export_cols = st.columns([3, 1], gap="small")
            with export_cols[0]:
                export_name = st.text_input("导出文件名", "processed_spectra.txt", key="export_name")
            with export_cols[1]:
                st.markdown("<br>", unsafe_allow_html=True)  # 垂直对齐
                if st.button("导出", type="secondary", key="export_btn"):
                    try:
                        if st.session_state.selected_arrangement:
                            arr_data = st.session_state.arrangement_details[st.session_state.selected_arrangement][
                                'data']
                            file_handler.export_data(export_name, arr_data)
                        else:
                            wavenumbers, y_processed = st.session_state.processed_data
                            file_handler.export_data(export_name, y_processed)
                        st.success(f"✅ 已导出到 {export_name}")
                    except Exception as e:
                        st.error(f"❌ 导出失败: {str(e)}")
        else:
            st.markdown(
                '<div style="border:1px dashed #ccc; height:80px; display:flex; align-items:center; justify-content:center;">处理完成后可导出结果</div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
