import numpy as np
import math
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
                        # 适配改进的AsLS算法参数
                        y_processed = algorithm_func(y_processed,** params)
                    elif method == "airPLS":
                        y_processed = algorithm_func(y_processed, **params)
                    elif method == "二阶差分(D2)":  # 处理二阶差分
                        y_processed = algorithm_func(y_processed)
                    else:  # SD、FD 无额外参数
                        y_processed = algorithm_func(y_processed)
                    method_name.append(f"{method}({', '.join([f'{k}={v}' for k, v in params.items()])})")

                elif step_type == "squashing":
                    algorithm_func = self.SQUASHING_ALGORITHMS[method]
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
                    y_processed = algorithm_func(y_processed,** params)
                    params_str = ', '.join([f'{k}={v}' for k, v in params.items()])
                    method_name.append(f"{method}({params_str})")

                    # 特殊处理小波线性阈值去噪的参数
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
            relative_change = abs((curr_std - prev_std) / curr_std) if curr_std != 0 else 0
            converged = relative_change < tolerance

            prev_spectrum = curr_spectrum
            prev_std = curr_std
            iteration += 1

        corrected[j] = originalRaman[j] - fitted

    return corrected


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
def SGfilter(Intensity, window_length,polyorder):  # 输入均为行
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
