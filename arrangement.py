import streamlit as st
import numpy as np
import pandas as pd
import re
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import seaborn as sns
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter, medfilt
from scipy.fft import fft, ifft
from statsmodels.nonparametric.smoothers_lowess import lowess
import pywt
from sklearn.linear_model import LinearRegression  # 用于MSC


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
        Lp = np.linalg.norm(arr[i,:], ord)
        if Lp != 0:
            Lpdata[i,:] = arr[i,:] / Lp
        else:
            Lpdata[i,:] = arr[i,:]
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
            MMarr[i] = ((Oarr[i] - permax*np.min(Oarr[i]))/ diff)*10 - 5
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
        M_reshaped = M.reshape(-1, 1)   # 平均光谱，重塑为二维数组
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
    xhat = np.zeros(sz)      # 后验估计
    P = np.zeros(sz)         # 后验误差估计
    xhatminus = np.zeros(sz) # 先验估计
    Pminus = np.zeros(sz)    # 先验误差估计
    K = np.zeros(sz)         # 卡尔曼增益

    # 初始猜测
    xhat[0] = 0.0
    P[0] = 1.0

    for k in range(1, n_iter):
        # 时间更新
        xhatminus[k] = xhat[k-1]  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k), A=1, BU(k)=0
        Pminus[k] = P[k-1] + Q    # P(k|k-1) = AP(k-1|k-1)A' + Q(k), A=1

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
                cost = self.distance(reference[i-1], query[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],    # 插入
                    dtw_matrix[i, j-1],    # 删除
                    dtw_matrix[i-1, j-1]   # 匹配
                )
        
        # 回溯路径
        i, j = n, m
        path = []
        while i > 0 or j > 0:
            path.append((i-1, j-1))
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                min_val = min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
                if min_val == dtw_matrix[i-1, j-1]:
                    i -= 1
                    j -= 1
                elif min_val == dtw_matrix[i-1, j]:
                    i -= 1
                else:
                    j -= 1
        
        return path[::-1], dtw_matrix[n, m]


# 挤压和 sigmoid 相关函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def squashing(x):
    return 1 / (1 + np.exp(-x))

def i_squashing(x):
    return 1 / (1 + np.exp(-0.5 * x))

def i_sigmoid(x, maxn=10):
    return 1 / (1 + np.exp(-x / maxn))


# sgolayfilt滤波器实现
def SGfilter(Intensity, point, degree):  # 输入均为行
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


def main():
    # 最优先初始化session state
    if 'show_arrangements' not in st.session_state:
        st.session_state.show_arrangements = False
    
    # 初始化测试相关的session状态变量
    test_states = {
        'k_value': 5,               # 默认k值
        'test_results': None,       # 存储测试结果
        'labels': None,             # 存储样本标签
        'train_indices': None,      # 训练集索引
        'test_indices': None        # 测试集索引
    }
    
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
        'current_algorithms': {},       # 存储当前选择的算法
        'filtered_perms': [],           # 存储筛选后的排列方案
        'selected_perm_idx': 0          # 存储当前选中的排列索引
    }
    
    # 合并所有状态变量并初始化
    all_states = {**test_states,** other_states}
    for key, value in all_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # 设置页面：紧凑布局
    st.set_page_config(layout="wide", page_icon="🔬", page_title="排列预处理模型")
    # 全局样式调整：紧凑字体和间距
    st.markdown("""
        <style>
        /* 全局字体缩小，间距紧凑 */
        body {font-size: 0.85rem !important;}
        .css-1v0mbdj {padding: 0.5rem 1rem !important;} /* 容器内边距 */
        .css-1d391kg {padding: 0.3rem 0 !important;} /* 标题间距 */
        .css-1x8cf1d {line-height: 1.2 !important;} /* 文本行高 */
        .css-12ttj6m {margin-bottom: 0.5rem !important;} /* 组件底部间距 */
        .css-1n543e5 {height: 220px !important;} /* 图表高度缩小 */
        .css-1b3298e {gap: 0.5rem !important;} /* 列间距 */
        .css-16huue1 {padding: 0.3rem 0.8rem !important;} /* 按钮内边距 */
        </style>
    """, unsafe_allow_html=True)

    st.title("🌌 排列预处理模型")
     
     
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
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n_points, n_points-2))
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
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n_points, n_points-2))
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
                    ref_diff = path[j][0] - path[j-1][0]
                    spec_diff = path[j][1] - path[j-1][1]
                    if ref_diff != 0:
                        slope = abs(spec_diff / ref_diff)
                        if slope > max_slope:
                            squashed[path[j][0]] = (squashed[path[j][0]] + squashed[path[j-1][0]]) / 2
            if k2 == "T":
                ref_map_count = {}
                for ref_idx, _ in path:
                    ref_map_count[ref_idx] = ref_map_count.get(ref_idx, 0) + 1
                for ref_idx, count in ref_map_count.items():
                    if count > l:
                        window = min(l, len(spectrum))
                        start = max(0, ref_idx - window//2)
                        end = min(n_samples, ref_idx + window//2 + 1)
                        squashed[ref_idx] = np.mean(spectrum[start:end])
            if l > 1:
                for j in range(n_samples):
                    start = max(0, j - l)
                    end = min(n_samples, j + l + 1)
                    squashed[j] = np.mean(squashed[start:end])
            result[:, i] = squashed
        return result
    
    # 生成排列时不包含编号
    def generate_permutations(algorithms):
        """生成完整的算法排列组合，排列名称不包含编号"""
        # 为四种算法分配编号1-4（二阶差分归类到基线校准中）
        algorithm_list = [
            (1, "基线校准", algorithms['baseline']),
            (2, "缩放", algorithms['scaling']),
            (3, "滤波", algorithms['filtering']),
            (4, "挤压", algorithms['squashing'])
        ]
        
        all_permutations = []
        
        # 0. 添加"无预处理（原始光谱）"选项（1种）
        all_permutations.append([])  # 空列表表示不使用任何算法
        
        # 1. 生成使用1种算法的排列
        for algo in algorithm_list:
            if algo[2] != "无":  # 只包含已选择的算法
                all_permutations.append([algo])
        
        # 2. 生成使用2种算法的排列
        for perm in itertools.permutations(algorithm_list, 2):
            # 确保两种算法都已选择
            if perm[0][2] != "无" and perm[1][2] != "无":
                all_permutations.append(list(perm))
        
        # 3. 生成使用3种算法的排列
        for perm in itertools.permutations(algorithm_list, 3):
            # 确保三种算法都已选择
            if perm[0][2] != "无" and perm[1][2] != "无" and perm[2][2] != "无":
                all_permutations.append(list(perm))
        
        # 4. 生成使用4种算法的排列
        for perm in itertools.permutations(algorithm_list, 4):
            # 确保四种算法都已选择
            if (perm[0][2] != "无" and perm[1][2] != "无" and 
                perm[2][2] != "无" and perm[3][2] != "无"):
                all_permutations.append(list(perm))
        
        # 格式化排列结果，确保每种排列都有first_step_type，且名称不包含编号
        formatted_perms = []
        for i, perm in enumerate(all_permutations):
            # 初始化默认值，确保属性存在
            perm_dict = {
                "name": "",
                "order": [],
                "details": perm,
                "count": len(perm),
                "first_step_type": "未知"  # 默认值，确保属性存在
            }
            
            if not perm:  # 无预处理情况
                perm_dict["name"] = "无预处理（原始光谱）"
                perm_dict["first_step_type"] = "无预处理"
            else:
                # 获取第一步算法的类型名称
                first_step_type = perm[0][1] if perm and len(perm) > 0 else "未知"
                perm_dict["first_step_type"] = first_step_type
                
                # 生成排列名称，不包含编号
                perm_details = []
                for step in perm:
                    perm_details.append(f"{step[0]}.{step[1]}({step[2]})")
                perm_dict["name"] = " → ".join(perm_details)
                perm_dict["order"] = [step[0] for step in perm]
            
            formatted_perms.append(perm_dict)
        
        return formatted_perms
    
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
                "卡尔曼滤波": self.kalman_filter,  # 添加卡尔曼滤波算法
                "Lowess": self.lowess_filter,
                "FFT": self.fft_filter,
                "小波变换(DWT)": self.wavelet_filter
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
                "Sigmoid挤压": sigmoid,
                "改进的Sigmoid挤压": i_sigmoid,
                "逻辑函数": squashing,
                "改进的逻辑函数": i_squashing,
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
                            y_processed = algorithm_func(y_processed, maxn=10)
                            method_name.append(f"{method}(maxn=10)")
                        elif method == "改进的逻辑函数":
                            m = params.get("m", 10)
                            y_processed = algorithm_func(y_processed)
                            method_name.append(f"{method}(m={m})")
                        elif method == "DTW挤压":
                            l = params.get("l", 1)
                            k1 = params.get("k1", "T")
                            k2 = params.get("k2", "T")
                            y_processed = algorithm_func(y_processed, l=l, k1=k1, k2=k2)
                            method_name.append(f"DTW挤压(l={l}, k1={k1}, k2={k2})")
                        else:
                            y_processed = algorithm_func(y_processed)
                            method_name.append(method)
                            
                    elif step_type == "filtering":
                        algorithm_func = self.FILTERING_ALGORITHMS[method]
                        y_processed = algorithm_func(y_processed,** params)
                        params_str = ', '.join([f'{k}={v}' for k, v in params.items()])
                        method_name.append(f"{method}({params_str})")
                        
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
        def savitzky_golay(self, spectra, k, w):
            return savgol_filter(spectra, w, k, axis=0)
        
        # 自定义sgolayfilt滤波器的封装
        def sgolay_filter_custom(self, spectra, point, degree):
            """使用自定义的SGfilter函数进行滤波"""
            # 确保输入数据形状与SGfilter要求一致
            if spectra.shape[0] < spectra.shape[1]:  # 特征数 < 样本数，需要转置
                filtered = SGfilter(spectra.T, point, degree)
                return filtered.T  # 转回原始形状
            else:
                return SGfilter(spectra, point, degree)
        
        def median_filter(self, spectra, k, w):
            return medfilt(spectra, kernel_size=(w, 1))
        
        def moving_average(self, spectra, k, w):
            kernel = np.ones(w) / w
            return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, spectra)
        
        # 添加MWA滤波方法的封装
        def mwa_filter(self, spectra, n=6, it=1, mode="full"):
            return MWA(spectra, n=n, it=it, mode=mode)
        
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
        
        def wavelet_filter(self, spectra, threshold):
            coeffs = pywt.wavedec(spectra, 'db4', axis=0)
            coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
            return pywt.waverec(coeffs, 'db4', axis=0)
        
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
        def load_data(self, wavenumber_file, data_file, lines, much):
            wavenumbers = np.loadtxt(wavenumber_file).ravel()
            return wavenumbers, self._getfromone(data_file, lines, much).T 
        
        def _getfromone(self, file, lines, much):
            numb = re.compile(r"-?\d+(?:\.\d+)?")
            ret = np.zeros((lines, much), dtype=float)
            content = file.getvalue().decode("utf-8")
            lines_list = content.splitlines()
            con = 0
            
            for line in lines_list:
                if con >= much:
                    break
                    
                li = numb.findall(line)
                for i in range(min(lines, len(li))):
                    ret[i][con] = float(li[i])
                con += 1
                
            return ret
        
        def export_data(self, filename, data):
            with open(filename, "w") as f:
                for line in data.T:  # 转置回原始格式
                    f.write("\t".join(map(str, line)) + "\n")
    
    # 创建处理器实例
    file_handler = FileHandler()
    preprocessor = Preprocessor()
    
    # 创建三列主布局
    col_left, col_mid, col_right = st.columns([1.2, 2.8, 1.1])
    
    # ===== 左侧：数据管理 =====
    with col_left:
        with st.expander("📁 数据管理", expanded=True):
            # 紧凑排列上传组件
            wavenumber_file = st.file_uploader("上传波数文件", type=['txt'], label_visibility="collapsed", key="wn_file")
            st.caption("波数文件(.txt)")
            uploaded_file = st.file_uploader("上传光谱数据文件", type=['txt'], label_visibility="collapsed", key="spec_file")
            st.caption("光谱数据文件(.txt)")
            
            # 紧凑标签输入
            st.subheader("样本标签", divider="gray")
            num_classes = st.number_input("类别数量", min_value=1, value=2, step=1, key="num_cls")
            labels_input = st.text_input(
                "标签（逗号分隔，与光谱顺序一致）", 
                placeholder="例：0,0,1,1",
                key="labels_in"
            )
            
            # 数据参数（横向排列更紧凑）
            param_col1, param_col2 = st.columns(2)
            with param_col1:
                lines = st.number_input("光谱条数", min_value=1, value=1, key="spec_lines")
            with param_col2:
                much = st.number_input("数据点数", min_value=1, value=2000, key="data_pts")

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
    
            # 数据加载逻辑
            if uploaded_file and wavenumber_file:
                try:
                    st.session_state.raw_data = file_handler.load_data(
                        wavenumber_file, uploaded_file, lines, much
                    )
                    
                    # 处理标签
                    if labels_input:
                        try:
                            labels = np.array([int(l.strip()) for l in labels_input.split(',')])
                            if len(labels) == st.session_state.raw_data[1].shape[1]:
                                st.session_state.labels = labels
                                n_samples = len(labels)
                                train_size = int(n_samples * train_test_ratio)
                                indices = np.random.permutation(n_samples)
                                st.session_state.train_indices = indices[:train_size]
                                st.session_state.test_indices = indices[train_size:]
                                st.success(f"✅ 数据加载成功：{lines}条光谱，{len(np.unique(labels))}类")
                            else:
                                st.warning(f"⚠️ 标签数({len(labels)})≠光谱数({st.session_state.raw_data[1].shape[1]})")
                                st.session_state.labels = None
                        except Exception as e:
                            st.warning(f"⚠️ 标签格式错误: {str(e)}")
                            st.session_state.labels = None
                    else:
                        st.success(f"✅ 数据加载成功：{lines}条光谱，{much}个点")
                        st.warning("⚠️ 请输入样本标签以进行分类测试")
                except Exception as e:
                    st.error(f"❌ 文件加载失败: {str(e)}")
        
        # 系统信息（紧凑显示）
        if st.session_state.get('raw_data'):
            wavenumbers, y = st.session_state.raw_data
            st.info(f"📊 数据维度: {y.shape[1]}条 × {y.shape[0]}点")
            st.info(f"🔢 训练集:{train_test_ratio:.1f} | 测试集:{1-train_test_ratio:.1f}")
            if st.session_state.get('labels') is not None:
                class_counts = np.bincount(st.session_state.labels)
                st.info(f"🏷️ 类别分布: {', '.join([f'类{i}:{count}个' for i, count in enumerate(class_counts) if count>0])}")
            if st.session_state.get('process_method'):
                st.success(f"🛠️ 处理流程: {st.session_state.process_method}")
        
        # 使用说明（精简）
        with st.expander("ℹ️ 使用指南", expanded=False):
            st.markdown("""
            1. 上传波数+光谱文件  
            2. 设置标签和数据参数  
            3. 右侧选择预处理方法  
            4. 点击"显示排列"生成方案  
            5. 选择k值后点击"测试"  
            6. 中间查看结果并导出
            """)
     
    # ===== 中间：光谱可视化与结果导出 =====
    with col_mid:
        st.subheader("📈 光谱可视化", divider="gray")
        
        # 1. 原始光谱区域
        st.subheader("原始光谱", divider="gray")
        # 第一层列（允许）
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
            
            # 可选：显示更多原始光谱（不使用深层嵌套）
            if st.session_state.get('raw_data') and y.shape[1] > 2:
                with st.expander("查看更多原始光谱", expanded=False):
                    # 仅使用一层列
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
            
        # 2. 处理结果展示
        if st.session_state.get('selected_arrangement'):
            st.subheader("🔍 预处理结果", divider="gray")
            selected_arr = st.session_state.selected_arrangement
            arr_data = st.session_state.arrangement_details[selected_arr]['data']
            arr_method = st.session_state.arrangement_details[selected_arr]['method']
            arr_order = st.session_state.arrangement_details[selected_arr].get('order', [])
            
            # 处理信息（紧凑显示）
            st.caption(f"处理方法: {arr_method} | 执行顺序: {arr_order if arr_order else '无预处理'}")
            
            # 预处理后光谱（仅一层列）
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
            
            # k值曲线（仅一层列）
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
            
            # 原始与处理后对比（仅一层列）
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
                
                # 指标（仅一层列）
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
            # 未选择排列时的提示
            st.info("ℹ️ 请在右侧选择预处理方法并应用排列方案")
            
        # 结果导出
        if st.session_state.arrangement_results or st.session_state.get('processed_data'):
            st.subheader("💾 结果导出", divider="gray")
            # 仅一层列
            export_cols = st.columns([3, 1])
            with export_cols[0]:
                export_name = st.text_input("导出文件名", "processed_spectra.txt", key="export_name")
            with export_cols[1]:
                st.markdown("<br>", unsafe_allow_html=True)  # 垂直对齐
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

    
    # ===== 右侧：预处理设置 + 排列方案选择 + 测试功能 =====
    with col_right:
        with st.expander("⚙️ 预处理设置", expanded=True):
            # 1. 基线校准（包含二阶差分）
            st.subheader("基线校准", divider="gray")
            baseline_method = st.selectbox(
                "方法",
                ["无", "SD", "FD", "多项式拟合", "ModPoly", "I-ModPoly", "PLS", "AsLS", "airPLS", "二阶差分(D2)"],
                key="baseline_method",
                label_visibility="collapsed"
            )
    
            # 基线参数
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
                    lam = st.selectbox("λ", [10**10, 10**8, 10**7], key="lam_pls", label_visibility="collapsed")
                    baseline_params["lam"] = lam
                    st.caption(f"λ: {lam}")
                elif baseline_method == "AsLS":
                    # 为改进的AsLS算法配置参数
                    asls_cols = st.columns(2)
                    with asls_cols[0]:
                        p = st.selectbox("非对称系数p", [0.001, 0.01, 0.1], key="p_asls", label_visibility="collapsed")
                    with asls_cols[1]:
                        lam = st.selectbox("平滑系数λ", [10**5, 10**7, 10**9], key="lam_asls", label_visibility="collapsed")
                    # 添加迭代次数和收敛阈值参数
                    niter = st.selectbox("迭代次数", [5, 10, 15], key="niter_asls", label_visibility="collapsed")
                    baseline_params["lam"] = lam
                    baseline_params["p"] = p
                    baseline_params["niter"] = niter
                    st.caption(f"p: {p}, λ: {lam}, 迭代次数: {niter}")
                elif baseline_method == "airPLS":
                    # 仅一层列
                    airpls_cols = st.columns(2)
                    with airpls_cols[0]:
                        lam = st.selectbox("λ", [10**7, 10**4, 10**2], key="lam_air", label_visibility="collapsed")
                    baseline_params["lam"] = lam
                    st.caption(f"λ: {lam}")
                elif baseline_method == "二阶差分(D2)":  # 二阶差分参数说明
                    st.caption("二阶差分可增强光谱特征，抑制基线漂移")
    
            # 2. 缩放处理
            st.subheader("📏 缩放", divider="gray")
            scaling_method = st.selectbox(
                "方法",
                ["无", "Peak-Norm", "SNV", "MSC", "M-M-Norm", "L-范数", "Ma-Minorm", "标准化(均值0，方差1)"],
                key="scaling_method",
                label_visibility="collapsed"
            )
    
            # 缩放参数
            scaling_params = {}
            if scaling_method == "L-范数":
                p = st.selectbox("p", ["无穷大", "4", "10"], key="p_scale", label_visibility="collapsed")
                scaling_params["p"] = p
                st.caption(f"p: {p}")
            # 标准化算法不需要额外参数，但添加说明
            elif scaling_method == "标准化(均值0，方差1)":
                st.caption("将数据标准化到均值为0，方差为1")
    
            # 3. 滤波处理
            st.subheader("📶 滤波", divider="gray")
            filtering_method = st.selectbox(
                "方法",
                ["无", "Savitzky-Golay", "sgolayfilt滤波器", "中值滤波(MF)", "移动平均(MAF)", 
                 "MWA（移动窗口平均）", "卡尔曼滤波", "Lowess", "FFT", "小波变换(DWT)"],
                key="filtering_method",
                label_visibility="collapsed"
            )
    
            # 滤波参数
            filtering_params = {}
            if filtering_method != "无":
                if filtering_method in ["Savitzky-Golay", "sgolayfilt滤波器"]:
                    # 仅一层列
                    sg_cols = st.columns(2)
                    with sg_cols[0]:
                        k = st.selectbox("多项式阶数", [3, 7], key="k_sg", label_visibility="collapsed")
                    with sg_cols[1]:
                        w = st.selectbox("窗口大小", [11, 31, 51], key="w_sg", label_visibility="collapsed")
                    filtering_params["point"] = w  # 对于sgolayfilt滤波器使用point参数名
                    filtering_params["degree"] = k  # 对于sgolayfilt滤波器使用degree参数名
                    st.caption(f"阶数: {k}, 窗口: {w}")
                elif filtering_method in ["中值滤波(MF)", "移动平均(MAF)"]:
                    # 仅一层列
                    mf_cols = st.columns(2)
                    with mf_cols[0]:
                        k = st.selectbox("k", [1, 3], key="k_mf", label_visibility="collapsed")
                    with mf_cols[1]:
                        w = st.selectbox("w", [7, 11], key="w_mf", label_visibility="collapsed")
                    filtering_params["k"] = k
                    filtering_params["w"] = w
                    st.caption(f"k: {k}, w: {w}")
                # MWA（移动窗口平均）参数配置
                elif filtering_method == "MWA（移动窗口平均）":
                    mwa_cols = st.columns(2)
                    with mwa_cols[0]:
                        n = st.selectbox("窗口大小n", [4, 6, 8], key="n_mwa", label_visibility="collapsed")
                    with mwa_cols[1]:
                        it = st.selectbox("迭代次数it", [1, 2, 3], key="it_mwa", label_visibility="collapsed")
                    filtering_params["n"] = n
                    filtering_params["it"] = it
                    filtering_params["mode"] = "full"  # 默认模式
                    st.caption(f"窗口大小: {n}, 迭代次数: {it}")
                # 卡尔曼滤波参数配置
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
                elif filtering_method == "小波变换(DWT)":
                    threshold = st.selectbox("阈值", [0.1, 0.3, 0.5], key="thresh_dwt", label_visibility="collapsed")
                    filtering_params["threshold"] = threshold
                    st.caption(f"阈值: {threshold}")

            # 4. 挤压处理（彻底避免三层嵌套）
            st.subheader("🧪 挤压", divider="gray")
            squashing_method = st.selectbox(
                "方法",
                ["无", "Sigmoid挤压", "改进的Sigmoid挤压", "逻辑函数", "改进的逻辑函数", "DTW挤压"],
                key="squashing_method",
                label_visibility="collapsed"
            )
    
            # 挤压参数（使用分组而非多层列）
            squashing_params = {}
            if squashing_method != "无":
                if squashing_method == "改进的逻辑函数":
                    m = st.selectbox("m", [10, 20], key="m_squash", label_visibility="collapsed")
                    squashing_params["m"] = m
                    st.caption(f"m: {m}")
                elif squashing_method == "DTW挤压":
                    # 使用两行组件而非三层列
                    dtw_row1 = st.columns(2)
                    with dtw_row1[0]:
                        l = st.selectbox("l", [1, 5], key="l_dtw", label_visibility="collapsed")
                    with dtw_row1[1]:
                        k1 = st.selectbox("k1", ["T", "F"], key="k1_dtw", label_visibility="collapsed")
                    
                    # 第二行单独放置
                    k2 = st.selectbox("k2", ["T", "F"], key="k2_dtw", label_visibility="collapsed")
                    
                    squashing_params["l"] = l
                    squashing_params["k1"] = k1
                    squashing_params["k2"] = k2
                    st.caption(f"l: {l}, k1: {k1}, k2: {k2}")
                elif squashing_method == "改进的Sigmoid挤压":
                    st.caption("默认: maxn=10")
    
            
            # 保存当前选择的算法
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
            
            # 应用处理与推荐应用按钮
            st.subheader("操作", divider="gray")
            btn_cols = st.columns(2)
            with btn_cols[0]:
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
        
            with btn_cols[1]:
                if st.button("🌟 推荐应用", type="primary", use_container_width=True, key="recommend_btn"):
                    if st.session_state.raw_data is None:
                        st.warning("⚠️ 请先上传数据")
                    else:
                        try:
                            wavenumbers, y = st.session_state.raw_data
                            recommended_params = {
                                'baseline_method': "二阶差分(D2)",  # 推荐使用二阶差分作为基线校正方法
                                'baseline_params': {},
                                'scaling_method': "标准化(均值0，方差1)",  # 推荐使用新添加的标准化方法
                                'scaling_params': {},
                                'filtering_method': "sgolayfilt滤波器",  # 推荐使用sgolayfilt滤波器
                                'filtering_params': {'point': 11, 'degree': 3},
                                'squashing_method': "改进的Sigmoid挤压",
                                'squashing_params': {}
                            }
                            
                            processed_data, method_name = preprocessor.process(
                                wavenumbers, y,** recommended_params
                            )
                            
                            arr_name = f"推荐排列_{len(st.session_state.arrangement_results) + 1}"
                            st.session_state.arrangement_results.append(arr_name)
                            st.session_state.arrangement_details[arr_name] = {
                                'data': processed_data,
                                'method': " → ".join(method_name),
                                'params': recommended_params
                            }
                            st.session_state.selected_arrangement = arr_name
                            st.session_state.processed_data = (wavenumbers, processed_data)
                            st.session_state.process_method = " → ".join(method_name)
                            st.success(f"✅ 推荐处理完成")
                        except Exception as e:
                            st.error(f"❌ 推荐失败: {str(e)}")
        
            # 显示排列按钮
            if st.button("🔍 显示排列", type="secondary", use_container_width=True, key="show_perm_btn"):
                st.session_state.show_arrangements = not st.session_state.show_arrangements
                
                if st.session_state.show_arrangements:
                    selected_algorithms = {
                        'baseline': baseline_method,
                        'scaling': scaling_method,
                        'filtering': filtering_method,
                        'squashing': squashing_method
                    }
                    st.session_state.algorithm_permutations = generate_permutations(selected_algorithms)
                    st.session_state.filtered_perms = st.session_state.algorithm_permutations
                    st.success(f"✅ 生成{len(st.session_state.algorithm_permutations)}种方案")
                else:
                    st.session_state.filtered_perms = []
                
                st.rerun()
            
            # 排列方案选择
            if st.session_state.show_arrangements and st.session_state.algorithm_permutations:
                st.subheader("🔄 排列方案", divider="gray")
                
                # 第一步类型筛选
                try:
                    all_first_step_types = list({
                        perm.get("first_step_type", "未知") 
                        for perm in st.session_state.algorithm_permutations
                    })
                    all_first_step_types.sort()
                except Exception as e:
                    st.error(f"❌ 筛选错误: {str(e)}")
                    all_first_step_types = ["全部", "无预处理", "基线校准", "缩放", "滤波", "挤压"]
                
                selected_first_step = st.selectbox(
                    "第一步类型",
                    ["全部"] + all_first_step_types,
                    key="first_step_filter",
                    label_visibility="collapsed"
                )
                
                # 筛选排列
                if selected_first_step == "全部":
                    st.session_state.filtered_perms = st.session_state.algorithm_permutations
                else:
                    st.session_state.filtered_perms = [
                        p for p in st.session_state.algorithm_permutations 
                        if p.get("first_step_type") == selected_first_step
                    ]
                
                # 排列下拉框
                if st.session_state.filtered_perms:
                    st.session_state.selected_perm_idx = st.selectbox(
                        f"选择方案（共{len(st.session_state.filtered_perms)}种）",
                        range(len(st.session_state.filtered_perms)),
                        format_func=lambda x: st.session_state.filtered_perms[x].get("name", f"方案{x+1}"),
                        key="perm_select",
                        label_visibility="collapsed",
                        help="选择预处理算法顺序"
                    )
                    
                    # 应用排列按钮
                    try:
                        selected_perm = st.session_state.filtered_perms[st.session_state.selected_perm_idx]
                        st.caption(f"当前: {selected_perm.get('name', '未知')}")
                        
                        if st.button("✅ 应用方案", type="primary", use_container_width=True, key="apply_perm_btn"):
                            if st.session_state.raw_data is None:
                                st.warning("⚠️ 请先上传数据")
                            else:
                                try:
                                    wavenumbers, y = st.session_state.raw_data
                                    algos = st.session_state.current_algorithms
                                    
                                    processed_data, method_name = preprocessor.process(
                                        wavenumbers, y, 
                                        baseline_method=algos['baseline'],
                                        baseline_params=algos['baseline_params'],
                                        squashing_method=algos['squashing'],
                                        squashing_params=algos['squashing_params'],
                                        filtering_method=algos['filtering'],
                                        filtering_params=algos['filtering_params'],
                                        scaling_method=algos['scaling'],
                                        scaling_params=algos['scaling_params'],
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
                    st.info("ℹ️ 无符合条件的方案")
                
                # 分类测试
                st.subheader("📝 分类测试", divider="gray")
                # k值输入和确定按钮（仅一层列）
                k_cols = st.columns([2, 1])
                with k_cols[0]:
                    k_value = st.number_input(
                        "k值", 
                        min_value=1, 
                        value=st.session_state.k_value,
                        step=1,
                        key="k_input",
                        label_visibility="collapsed"
                    )
                with k_cols[1]:
                    if st.button("确定", type="secondary", use_container_width=True, key="k_confirm_btn"):
                        st.session_state.k_value = k_value
                        st.success(f"k={k_value}")
                
                # 测试按钮
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
                                predictions = knn_classify(
                                    train_data, 
                                    train_labels, 
                                    test_data, 
                                    k=st.session_state.k_value
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
                            
                            st.success("✅ 测试完成！结果在中间面板")
                            
                        except Exception as e:
                            st.error(f"❌ 测试失败: {str(e)}")

if __name__ == "__main__":
    main()
