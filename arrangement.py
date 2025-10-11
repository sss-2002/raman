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
        
        # 将提取到的数字转换为浮动类型
        data = np.array([float(num) for num in all_numbers])
        
        # 假设每条光谱的点数为 `much`
        n_rows = len(lines_list)
        n_cols = len(data) // n_rows if n_rows > 0 else 0
        data = data[:n_rows * n_cols]  # 截取多余的数据
        return data.reshape(n_rows, n_cols)

    def export_data(self, filename, data):
        with open(filename, "w") as f:
            for line in data.T:  # 转置回原始格式
                f.write("\t".join(map(str, line)) + "\n")


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
        
        # 添加 MWA 作为过滤算法
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

    # 添加 MWA 滤波算法
    def mwa_filter(self, spectra, n=6, it=1, mode="full"):
        """
        MWA（移动窗口平均）滤波器
        参数：
        - spectra: 输入的光谱数据（二维数组）
        - n: 窗口大小（默认6）
        - it: 迭代次数（默认1）
        - mode: 计算模式，"full"表示完整窗口
        """
        # 确保数据的形状是适合处理的
        if spectra.shape[0] < spectra.shape[1]:
            spectra = spectra.T  # 转置，使每行是一个样本
        
        # 移动窗口平均处理
        smoothed_spectra = np.copy(spectra)
        
        for i in range(spectra.shape[1]):
            for j in range(it):
                smoothed_spectra[:, i] = self.moving_average_1d(spectra[:, i], n)

        return smoothed_spectra

    def moving_average_1d(self, data, window_size):
        """
        对一维数据进行移动平均
        """
        return np.convolve(data, np.ones(window_size) / window_size, mode='same')

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
        
        # 根据选择的步骤执行预处理
        if filtering_method == "MWA（移动窗口平均）":
            y_processed = self.mwa_filter(y_processed, **filtering_params)
            method_name.append(f"{filtering_method}({', '.join([f'{k}={v}' for k, v in filtering_params.items()])})")
        
        # 继续其他处理步骤...
        
        return y_processed, method_name
    
    def _sd_baseline(self, spectra):
        return spectra - np.min(spectra, axis=0)
    
    def _fd_baseline(self, spectra):
        return spectra - np.percentile(spectra, 5, axis=0)
    
    # 其他基线校正方法和滤波方法省略...

# Streamlit UI 部分
import streamlit as st

def main():
    # 初始化Preprocessor
    preprocessor = Preprocessor()
    
    st.title("🌌 排列预处理模型")
    
    # 选择滤波方法
    st.subheader("📶 滤波")
    filtering_method = st.selectbox(
        "方法",
        ["无", "Savitzky-Golay", "中值滤波(MF)", "移动平均(MAF)", "MWA（移动窗口平均）"],
        key="filtering_method"
    )
    
    filtering_params = {}
    if filtering_method == "MWA（移动窗口平均）":
        n = st.slider("窗口大小n", 4, 10, 6, key="mwa_n")
        it = st.slider("迭代次数it", 1, 5, 1, key="mwa_it")
        filtering_params["n"] = n
        filtering_params["it"] = it
        st.caption(f"窗口大小: {n}, 迭代次数: {it}")
    
    # 使用相应的预处理方法
    if st.button("应用处理", type="primary", use_container_width=True, key="apply_btn"):
        # 加载数据（假设已经上传）
        data = np.random.rand(100, 10)  # 模拟数据
        wavenumbers = np.linspace(400, 4000, 100)
        
        # 处理数据
        processed_data, method_name = preprocessor.process(
            wavenumbers, data, 
            filtering_method=filtering_method, 
            filtering_params=filtering_params
        )
        st.success(f"✅ 处理完成: {', '.join(method_name)}")
        st.write(processed_data)

if __name__ == "__main__":
    main()
