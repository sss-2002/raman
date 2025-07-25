import streamlit as st
import numpy as np
import pandas as pd
import re
from SD import D2
from FD import D1
from sigmoids import sigmoid
from squashing import squashing  
from i_squashing import i_squashing 
from i_sigmoid import i_sigmoid
from IModPoly import IModPoly
from AsLS import baseline_als
from LPnorm import LPnorm
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter, medfilt
from scipy.fft import fft, ifft
 

# 设置页面
st.set_page_config(layout="wide", page_title="光谱预处理系统")
st.title("🌌 光谱预处理系统")

# 初始化session状态
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'peaks' not in st.session_state:
    st.session_state.peaks = None

# ===== 算法实现 =====
def polynomial_fit(wavenumbers, spectra, polyorder):
    """多项式拟合基线校正"""
    baseline = np.zeros_like(spectra)
    
    for i in range(spectra.shape[1]):
        # 使用 np.polyfit 拟合多项式基线
        coeffs = np.polyfit(wavenumbers, spectra[:, i], deg=polyorder)
        baseline[:, i] = np.polyval(coeffs, wavenumbers)
    
    return spectra - baseline  # 扣除基线

def modpoly(wavenumbers, spectra, k):
    """Modified Polynomial (ModPoly) 基线校正"""
    baseline = np.zeros_like(spectra)
    n_points = len(wavenumbers)
    
    for i in range(spectra.shape[1]):
        y = spectra[:, i].copy()
        
        # 迭代k次
        for _ in range(k):
            # 拟合多项式
            coeffs = np.polyfit(wavenumbers, y, deg=5)
            fitted = np.polyval(coeffs, wavenumbers)
            
            # 更新数据：只保留低于当前拟合线的点
            mask = y < fitted
            y[~mask] = fitted[~mask]
        
        baseline[:, i] = y
    
    return spectra - baseline

def imodpoly(wavenumbers, spectra, k):
    """Improved ModPoly (I-ModPoly) 基线校正"""
    baseline = np.zeros_like(spectra)
    n_points = len(wavenumbers)
    
    for i in range(spectra.shape[1]):
        y = spectra[:, i].copy()
        
        # 初始多项式拟合
        coeffs = np.polyfit(wavenumbers, y, deg=5)
        fitted = np.polyval(coeffs, wavenumbers)
        
        # 迭代k次
        for _ in range(k):
            # 找到低于当前拟合线的点
            mask = y < fitted
            
            # 仅使用这些点重新拟合
            coeffs = np.polyfit(wavenumbers[mask], y[mask], deg=5)
            fitted = np.polyval(coeffs, wavenumbers)
        
        baseline[:, i] = fitted
    
    return spectra - baseline

def pls(spectra, lam):
    """Penalized Least Squares (PLS) 基线校正"""
    n_points = spectra.shape[0]
    baseline = np.zeros_like(spectra)
    
    # 构建差分矩阵
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n_points, n_points-2))
    D = lam * D.dot(D.transpose())
    
    for i in range(spectra.shape[1]):
        y = spectra[:, i]
        # 求解 (I + D)c = y
        A = sparse.eye(n_points) + D
        baseline[:, i] = spsolve(A, y)
    
    return spectra - baseline

def asls(spectra, lam, p, max_iter=10):
    """Asymmetric Least Squares (AsLS) 基线校正"""
    n_points = spectra.shape[0]
    baseline = np.zeros_like(spectra)
    
    # 构建差分矩阵
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n_points, n_points-2))
    D = lam * D.dot(D.transpose())
    
    for i in range(spectra.shape[1]):
        y = spectra[:, i]
        w = np.ones(n_points)
        
        for _ in range(max_iter):
            # 加权最小二乘求解
            W = sparse.diags(w, 0)
            Z = W + D
            b = spsolve(Z, W * y)
            
            # 更新权重
            mask = y > b
            w[mask] = p
            w[~mask] = 1 - p
        
        baseline[:, i] = b
    
    return spectra - baseline

def airpls(spectra, lam, max_iter=15, threshold=0.001):
    """Adaptive Iteratively Reweighted Penalized Least Squares (airPLS) 基线校正"""
    n_points = spectra.shape[0]
    baseline = np.zeros_like(spectra)
    
    # 构建差分矩阵
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n_points, n_points-2))
    D = lam * D.dot(D.transpose())
    
    for i in range(spectra.shape[1]):
        y = spectra[:, i]
        w = np.ones(n_points)
        baseline_i = np.zeros(n_points)
        
        for j in range(max_iter):
            # 加权最小二乘求解
            W = sparse.diags(w, 0)
            Z = W + D
            b = spsolve(Z, W * y)
            
            # 计算残差
            d = y - b
            
            # 更新权重
            neg_mask = d < 0
            w[neg_mask] = np.exp(j * np.abs(d[neg_mask]) / np.std(d[neg_mask]))
            w[~neg_mask] = 0
            
            # 检查收敛
            if j > 0:
                diff = np.sum(np.abs(b - baseline_i)) / np.sum(np.abs(baseline_i))
                if diff < threshold:
                    break
            
            baseline_i = b
        
        baseline[:, i] = baseline_i
    
    return spectra - baseline

# ===== 数据变换函数 =====
def sigmoid(x):
    """原始Sigmoid函数"""
    return 1 / (1 + np.exp(-x))

def squashing(x):
    """原始挤压函数"""
    return x / np.sqrt(1 + x**2)

def i_sigmoid(x, maxn=10):
    """归一化版Sigmoid函数"""
    x_norm = x / maxn
    return sigmoid(x_norm)

def i_squashing(x):
    """归一化版挤压函数"""
    # 先归一化到[-1,1]
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
    return squashing(x_norm)

# ===== LP范数归一化 =====
def LPnorm(x, p):
    """计算Lp范数归一化"""
    norm = np.linalg.norm(x, ord=p, axis=0)
    norm[norm == 0] = 1  # 避免除零错误
    return x / norm

# ===== 预处理类 =====
class Preprocessor:
    def __init__(self):
        # 算法名称到处理函数的映射
        self.BASELINE_ALGORITHMS = {
            "SD": self._sd_baseline,  # 示例实现
            "FD": self._fd_baseline,  # 示例实现
            "多项式拟合": polynomial_fit,
            "ModPoly": modpoly,
            "I-ModPoly": imodpoly,
            "PLS": pls,
            "AsLS": asls,
            "airPLS": airpls,
        }
    # 滤波算法映射
        self.FILTERING_ALGORITHMS = {
            "Savitzky-Golay": self.savitzky_golay,
            "中值滤波(MF)": self.median_filter,
            "移动平均(MAF)": self.moving_average,
            "Lowess": self.lowess_filter,
            "FFT": self.fft_filter,
            "小波变换(DWT)": self.wavelet_filter
        }
        
        # 缩放算法映射
        self.SCALING_ALGORITHMS = {
            "Peak-Norm": self.peak_norm,
            "SNV": self.snv,
            "MSC": self.msc,
            "M-M-Norm": self.mm_norm,
            "L-范数": self.l_norm
        }

    def process(self, wavenumbers, data, 
                baseline_method="无", baseline_params=None,
                squashing_method="无", squashing_params=None,
                filtering_method="无", filtering_params=None,
                scaling_method="无", scaling_params=None):
        """执行完整的预处理流程"""
        if baseline_params is None: baseline_params = {}
        if squashing_params is None: squashing_params = {}
        if filtering_params is None: filtering_params = {}
        if scaling_params is None: scaling_params = {}
            
        y_processed = data.copy()
        method_name = []

        # 基线处理
        if baseline_method != "无":
            try:
                # 获取对应的算法函数
                algorithm_func = self.BASELINE_ALGORITHMS[baseline_method]
                
                # 根据算法类型传递参数
                if baseline_method in ["多项式拟合", "ModPoly", "I-ModPoly"]:
                    y_processed = algorithm_func(wavenumbers, y_processed, **baseline_params)
                elif baseline_method in ["PLS", "AsLS", "airPLS"]:
                    y_processed = algorithm_func(y_processed, **baseline_params)
                else:  # SD、FD 无额外参数
                    y_processed = algorithm_func(y_processed)
                    
                method_name.append(f"{baseline_method}({', '.join([f'{k}={v}' for k, v in baseline_params.items()])})")
            except Exception as e:
                raise ValueError(f"基线校正失败: {str(e)}")

         #挤压处理
        if squashing_method != "无":
            try:
                algorithm_func = self.FILTERING_ALGORITHMS[squashing_method]
                y_processed = algorithm_func(y_processed, **squashing_params)
                params_str = ', '.join([f'{k}={v}' for k, v in squashing_params.items()])
                method_name.append(f"{squashing_method}({params_str})")
            except Exception as e:
                raise ValueError(f"挤压处理失败: {str(e)}")

        # 滤波处理
        if filtering_method != "无":
            try:
                algorithm_func = self.FILTERING_ALGORITHMS[filtering_method]
                y_processed = algorithm_func(y_processed, **filtering_params)
                params_str = ', '.join([f'{k}={v}' for k, v in filtering_params.items()])
                method_name.append(f"{filtering_method}({params_str})")
            except Exception as e:
                raise ValueError(f"滤波处理失败: {str(e)}")

        # 缩放处理
        if scaling_method != "无":
            try:
                algorithm_func = self.SCALING_ALGORITHMS[scaling_method]
                y_processed = algorithm_func(y_processed, **scaling_params)
                params_str = ', '.join([f'{k}={v}' for k, v in scaling_params.items()])
                method_name.append(f"{scaling_method}({params_str})")
            except Exception as e:
                raise ValueError(f"缩放处理失败: {str(e)}")

        return y_processed, method_name
    
    def _sd_baseline(self, spectra):
        """示例SD基线校正实现"""
        # 这里应该是实际的SD算法实现
        # 这里仅作示例
        return spectra - np.min(spectra, axis=0)
    
    def _fd_baseline(self, spectra):
        """示例FD基线校正实现"""
        # 这里应该是实际的FD算法实现
        # 这里仅作示例
        return spectra - np.percentile(spectra, 5, axis=0)

# ===== 滤波算法实现 =====
    def savitzky_golay(self, spectra, k, w):
        """Savitzky-Golay滤波"""
        window_length = w
        polyorder = k
        return savgol_filter(spectra, window_length, polyorder, axis=0)
    
    def median_filter(self, spectra, k, w):
        """中值滤波"""
        kernel_size = w
        return medfilt(spectra, kernel_size=(kernel_size, 1))
    
    def moving_average(self, spectra, k, w):
        """移动平均滤波"""
        window_size = w
        # 使用卷积实现移动平均
        kernel = np.ones(window_size) / window_size
        return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, spectra)
    
    def lowess_filter(self, spectra, frac):
        """Lowess平滑滤波"""
        # 由于Lowess计算较慢，这里使用简化实现
        # 实际应用中应使用statsmodels的lowess函数
        from statsmodels.nonparametric.smoothers_lowess import lowess
        result = np.zeros_like(spectra)
        for i in range(spectra.shape[1]):
            smoothed = lowess(spectra[:, i], np.arange(len(spectra)), frac=frac, it=0)
            result[:, i] = smoothed[:, 1]
        return result
    
    def fft_filter(self, spectra, cutoff):
        """FFT滤波"""
        fft_result = fft(spectra, axis=0)
        frequencies = np.fft.fftfreq(spectra.shape[0])
        
        # 创建滤波器
        filter_mask = np.abs(frequencies) < cutoff
        fft_result[~filter_mask, :] = 0
        
        return np.real(ifft(fft_result, axis=0))
    
    def wavelet_filter(self, spectra, threshold):
        """小波变换滤波"""
        import pywt
        coeffs = pywt.wavedec(spectra, 'db4', axis=0)
        # 阈值处理
        coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
        return pywt.waverec(coeffs, 'db4', axis=0)
    
    # ===== 缩放算法实现 =====
    def peak_norm(self, spectra):
        """Peak-Norm归一化"""
        return spectra / np.max(spectra, axis=0)
    
    def snv(self, spectra):
        """标准正态变量变换(SNV)"""
        mean = np.mean(spectra, axis=0)
        std = np.std(spectra, axis=0)
        return (spectra - mean) / std
    
    def msc(self, spectra):
        """多元散射校正(MSC)"""
        mean_spectrum = np.mean(spectra, axis=1)
        return np.apply_along_axis(lambda x: np.polyval(np.polyfit(mean_spectrum, x, 1), mean_spectrum), 0, spectra)
    
    def mm_norm(self, spectra):
        """M-M-Norm归一化"""
        min_vals = np.min(spectra, axis=0)
        max_vals = np.max(spectra, axis=0)
        return (spectra - min_vals) / (max_vals - min_vals)
    
    def l_norm(self, spectra, p):
        """L-范数归一化"""
        if p == "无穷大":
            return spectra / np.max(np.abs(spectra), axis=0)
        else:
            p_val = float(p)
            norm = np.linalg.norm(spectra, ord=p_val, axis=0)
            norm[norm == 0] = 1  # 避免除零错误
            return spectra / norm

# ===== 文件处理类 =====
class FileHandler:
    def load_data(self, wavenumber_file, data_file, lines, much):
        """加载波数和光谱数据"""
        # 读取波数数据
        wavenumbers = np.loadtxt(wavenumber_file).ravel()
        
        # 读取光谱数据
        ret = self._getfromone(data_file, lines, much)
        
        return wavenumbers, ret.T  # 转置为(点数, 光谱数)
    
    def _getfromone(self, file, lines, much):
        """从文件中解析光谱数据"""
        numb = re.compile(r"-?\d+(?:\.\d+)?")
        ret = np.zeros((lines, much), dtype=float)
        
        # 读取文件内容
        content = file.getvalue().decode("utf-8")
        
        # 解析数据
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
        """导出处理后的数据"""
        with open(filename, "w") as f:
            for line in data.T:  # 转置回原始格式
                f.write("\t".join(map(str, line)) + "\n")

# 创建处理器实例
file_handler = FileHandler()
preprocessor = Preprocessor()

# 创建两列布局
col1, col2 = st.columns([1.2, 3])

with col1:
    # ===== 数据管理 =====
    with st.expander("📁 数据管理", expanded=True):
        # 波数文件上传
        wavenumber_file = st.file_uploader("上传波数文件", type=['txt'])
        
        # 光谱数据上传
        uploaded_file = st.file_uploader("上传光谱数据文件", type=['txt'])
        
        # 参数设置
        lines = st.number_input("光谱条数", min_value=1, value=1)
        much = st.number_input("每条光谱数据点数", min_value=1, value=2000)

        if uploaded_file and wavenumber_file:
            try:
                # 读取数据
                st.session_state.raw_data = file_handler.load_data(
                    wavenumber_file, uploaded_file, lines, much
                )
                st.success(f"数据加载成功！{lines}条光谱，每条{much}个点")
                
            except Exception as e:
                st.error(f"文件加载失败: {str(e)}")

    # ===== 预处理设置 =====
    with st.expander("⚙️ 预处理设置", expanded=True):
        # 基线校准
        st.subheader("基线校准")
        baseline_method = st.selectbox(
            "基线校准方法",
            ["无", "SD", "FD", "多项式拟合", "ModPoly", "I-ModPoly", "PLS", "AsLS", "airPLS"],
            key="baseline_method"
        )

        # 收集基线校准参数
        baseline_params = {}
        if baseline_method != "无":
            if baseline_method == "多项式拟合":
                polyorder = st.slider("多项式阶数 k", 3, 6, 5, key="polyorder_polyfit")
                baseline_params["polyorder"] = polyorder
            elif baseline_method == "ModPoly":
                k = st.slider("参数 k", 4, 10, 10, key="k_modpoly")
                baseline_params["k"] = k
            elif baseline_method == "I-ModPoly":
                k = st.slider("参数 k", 5, 9, 6, key="k_imodpoly")
                baseline_params["k"] = k
            elif baseline_method == "PLS":
                lam = st.selectbox("λ(平滑度)", [10**10, 10**8, 10**7], key="lam_pls")
                baseline_params["lam"] = lam
            elif baseline_method == "AsLS":
                p = st.selectbox("p(不对称性)", [0.2, 0.1], key="p_asls")
                lam = st.selectbox("λ(平滑度)", [10**9, 10**6], key="lam_asls")
                baseline_params["p"] = p
                baseline_params["lam"] = lam
            elif baseline_method == "airPLS":
                lam = st.selectbox("λ(平滑度)", [10**7, 10**4, 10**2], key="lam_airpls")
                baseline_params["lam"] = lam


     
        # ===== 挤压处理 =====
        st.subheader("🧪 挤压")
        squashing_method = st.selectbox(
            "挤压方法",
            ["无", 
             "Sigmoid挤压(原始版)",  # 对应 from sigmoids import sigmoid
             "改进的Sigmoid挤压(归一化版)",  # 对应 from i_sigmoid import i_sigmoid
             "逻辑函数(原始版)",  # 可根据实际函数命名调整
             "改进的逻辑函数(归一化版)",
             "DTW挤压"
            ],
            key="squashing_method"
        )

    # 挤压参数（根据论文表2.4扩展）
        squashing_params = {}
        if squashing_method != "无":
            if squashing_method == "Sigmoid挤压":
                # Sigmoid挤压无额外参数，按论文表2.4
                squashing_params["params"] = "无额外参数"
            elif squashing_method == "改进的Sigmoid挤压":
                # 改进的Sigmoid挤压无额外参数，按论文表2.4
                squashing_params["params"] = "无额外参数"
            elif squashing_method == "逻辑函数":
                # 逻辑函数无额外参数，按论文表2.4
                squashing_params["params"] = "无额外参数"
            elif squashing_method == "改进的逻辑函数":
                m = st.selectbox(
                    "参数m", 
                    [10, 20], 
                    key="m_squashing"
                )
                squashing_params["m"] = m
            elif squashing_method == "DTW":
                l = st.selectbox(
                    "参数l", 
                    [1, 5],  
                    key="l_dtw"
                )
                k1 = st.selectbox(
                    "参数k1", 
                    ["T", "F"], 
                    key="k1_dtw"
                )
                k2 = st.selectbox(
                    "参数k2", 
                    ["T", "F"],  
                    key="k2_dtw"
                )
                squashing_params["l"] = l
                squashing_params["k1"] = k1
                squashing_params["k2"] = k2
        
            try:
                if squashing_method == "Sigmoid挤压":
                    from sigmoids import sigmoid
                    y_processed = sigmoid(y_processed)
                    method_name.append("sigmoid")
                elif squashing_method == "改进的Sigmoid挤压":
                    from i_sigmoid import i_sigmoid
                    y_processed = i_sigmoid(y_processed)
                    method_name.append("i_sigmoid")
                elif squashing_method == "逻辑函数":
                    from Squashing import squashing
                    y_processed = squashing(y_processed)
                    method_name.append("squashing")
                elif squashing_method == "改进的逻辑函数":
                    from i_squashing import i_squashing
                    y_processed = i_squashing(y_processed, squashing_params["m"])
                    method_name.append(f"i_squashing(m={squashing_params['m']})")
                elif squashing_method == "DTW":
                    from DTW import DTW
                    y_processed = DTW(y_processed, l=squashing_params["l"], k1=squashing_params["k1"], k2=squashing_params["k2"])
                    method_name.append(f"DTW(l={squashing_params['l']}, k1={squashing_params['k1']}, k2={squashing_params['k2']})")
            except Exception as e:
                raise ValueError(f"挤压处理失败: {str(e)}")
                    
         
                    
        

        # ===== 滤波处理 =====
        st.subheader("📶 滤波")
        filtering_method = st.selectbox(
            "滤波方法",
            ["无", "Savitzky-Golay", "中值滤波(MF)", "移动平均(MAF)", "Lowess", "FFT", "小波变换(DWT)", "卡尔曼滤波"],
            key="filtering_method"
        )

        # 滤波参数
        filtering_params = {}
        if filtering_method != "无":
            if filtering_method == "Savitzky-Golay":
                k = st.selectbox("阶数(k)", [3, 7], key="k_sg")
                w = st.selectbox("窗口大小(w)", [11, 31, 51], key="w_sg")
                filtering_params["k"] = k
                filtering_params["w"] = w
            elif filtering_method in ["中值滤波(MF)", "移动平均(MAF)"]:
                k = st.selectbox("核大小(k)", [1, 3], key="k_mf")
                w = st.selectbox("窗口大小(w)", [7, 11], key="w_mf")
                filtering_params["k"] = k
                filtering_params["w"] = w
            elif filtering_method == "Lowess":
                frac = st.selectbox("平滑系数(k)", [0.01, 0.03], key="frac_lowess")
                filtering_params["frac"] = frac
            elif filtering_method == "FFT":
                cutoff = st.selectbox("截止频率(l)", [90, 50, 30], key="cutoff_fft")
                filtering_params["cutoff"] = cutoff
            elif filtering_method == "小波变换(DWT)":
                threshold = st.selectbox("阈值(k)", [0.1, 0.3, 0.5], key="threshold_dwt")
                filtering_params["threshold"] = threshold
            elif filtering_method == "卡尔曼滤波":
                 # 论文参数：r∈[1e-5, 1e-3]（根据论文表格补充）
                 # 用字符串显示十进制形式的选项
                 options = ["0.00001", "0.0001", "0.001"]
                 # 让用户选择显示的字符串
                 selected_str = st.selectbox("过程噪声(r)", options, key="r_kalman")
                 # 将选择的字符串转换为对应的数值类型（float）
                 r = float(selected_str)
                 filtering_params["r"] = r

        # ===== 缩放处理 =====
        st.subheader("📏 缩放")
        scaling_method = st.selectbox(
            "缩放方法",
            ["无", "Peak-Norm", "SNV", "MSC", "M-M-Norm", "L-范数"],
            key="scaling_method"
        )

        # 缩放参数
        scaling_params = {}
        if scaling_method == "L-范数":
            p = st.selectbox("范数阶数(p)", ["无穷大", "10", "4"], key="p_scaling")
            scaling_params["p"] = p

        # 处理按钮
        if st.button("🚀 应用处理", type="primary", use_container_width=True):
            if st.session_state.raw_data is None:
                st.warning("请先上传数据文件")
            else:
                try:
                    wavenumbers, y = st.session_state.raw_data
                    
                    # 执行预处理
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
                    
                    st.session_state.processed_data = (wavenumbers, processed_data)
                    st.session_state.process_method = " → ".join(method_name)
                    st.success(f"处理完成: {st.session_state.process_method}")
                except Exception as e:
                    st.error(f"处理失败: {str(e)}")

        

with col2:
    # ===== 系统信息 =====
    if st.session_state.get('raw_data'):
        wavenumbers, y = st.session_state.raw_data
        cols = st.columns([1, 2])
        with cols[0]:
            st.info(f"📊 数据维度: {y.shape[1]}条光谱 × {y.shape[0]}点")
        with cols[1]:
            if st.session_state.get('process_method'):
                st.success(f"🛠️ 处理流程: {st.session_state.process_method}")
    
    st.divider()
    
    # ===== 光谱图 =====
    st.subheader("📈 光谱可视化")
    if st.session_state.get('raw_data'):
        wavenumbers, y = st.session_state.raw_data
        chart_data = pd.DataFrame(y, index=wavenumbers)
        
        if st.session_state.get('processed_data'):
            _, y_processed = st.session_state.processed_data
            chart_data = pd.DataFrame({
                "原始数据": y.mean(axis=1),
                "处理后数据": y_processed.mean(axis=1)
            }, index=wavenumbers)
        
        st.line_chart(chart_data)
    else:
        st.info("请先上传并处理数据")

    # ===== 结果导出 =====
    if st.session_state.get('processed_data'):
        st.subheader("💾 结果导出")
        export_name = st.text_input("导出文件名", "processed_spectra.txt")
        
        if st.button("导出处理结果", type="secondary"):
            try:
                wavenumbers, y_processed = st.session_state.processed_data
                file_handler.export_data(export_name, y_processed)
                st.success(f"结果已导出到 {export_name}")
            except Exception as e:
                st.error(f"导出失败: {str(e)}")

# 使用说明
with st.expander("ℹ️ 使用指南", expanded=False):
    st.markdown("""
    **标准操作流程:**
    1. 上传波数文件（单列文本）
    2. 上传光谱数据文件（多列文本）
    3. 设置光谱条数和数据点数
    4. 选择预处理方法
    5. 点击"应用处理"
    6. 导出结果

    **文件格式要求:**
    - 波数文件: 每行一个波数值
    - 光谱数据: 每列代表一条光谱，每行对应相同波数位置
    """)
