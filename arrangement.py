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


def main():
    # 最优先初始化session state
    if 'show_arrangements' not in st.session_state:
        st.session_state.show_arrangements = False

    # 初始化k_value和其他session状态
    if 'k_value' not in st.session_state:
        st.session_state.k_value = 5  # 设置k_value的默认值

    # 初始化测试相关的session状态变量
    test_states = {
        'k_value': st.session_state.k_value,  # 现在从session_state获取k_value
        'test_results': None,  # 存储测试结果
        'labels': None,  # 存储样本标签
        'train_indices': None,  # 训练集索引
        'test_indices': None  # 测试集索引
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
        'current_algorithms': {},  # 存储当前选择的算法
        'filtered_perms': [],  # 存储筛选后的排列方案
        'selected_perm_idx': 0  # 存储当前选中的排列索引
    }

    # 合并所有状态变量并初始化
    all_states = {**test_states, **other_states}
    for key, value in all_states.items():
        if key not in st.session_state:
            st.session_state[key] = value
    file_handler = FileHandler()

    # 设置页面：紧凑布局
    st.set_page_config(layout="wide", page_icon="🔬", page_title="排列预处理模型")

    # 全局样式调整：更紧凑的字体和间距，确保预处理设置在一行显示
    st.markdown("""
        <style>
        body {font-size: 0.75rem !important;}
        .css-1v0mbdj {padding: 0.3rem 0.5rem !important;} 
        .css-1d391kg {padding: 0.2rem 0 !important;} 
        .css-1x8cf1d {line-height: 1.1 !important;} 
        .css-12ttj6m {margin-bottom: 0.3rem !important;} 
        .css-16huue1 {padding: 0.2rem 0.5rem !important; font-size: 0.7rem !important;} 
        h3 {font-size: 1rem !important; margin: 0.3rem 0 !important;} 
        .css-1b3298e {gap: 0.3rem !important;} 
        .stSlider, .stSelectbox, .stTextInput {margin-bottom: 0.3rem !important;} 
        .stCaption {font-size: 0.65rem !important; margin-top: -0.2rem !important;} 
        .css-1544g2n {padding: 0.2rem 0.5rem !important;} 
        </style>
    """, unsafe_allow_html=True)

    st.title("🌌 排列预处理模型")

    # 页面整体布局：左侧数据管理，右侧主要内容区
    col_left, col_right = st.columns([1.2, 3.9])

    # ===== 左侧：数据管理模块 =====
    with col_left:
        with st.expander("📁 数据管理", expanded=True):
            zip_file = st.file_uploader("上传包含波数和光谱数据的压缩包", type=['zip'], key="zip_file")
            st.caption("压缩包(.zip)需包含波数和光谱数据文件")

            st.subheader("样本标签")
            num_classes = st.number_input("类别数量", min_value=1, value=2, step=1, key="num_cls")
            labels_input = st.text_input("标签（逗号分隔，与光谱顺序一致）", placeholder="例：0,0,1,1", key="labels_in")

            st.subheader("训练测试划分")
            train_test_ratio = st.slider("训练集比例", min_value=0.1, max_value=0.9, value=0.8, step=0.1, format="%.1f",
                                         key="train_ratio")
            st.session_state.train_test_split_ratio = train_test_ratio

            if zip_file:
                try:
                    st.session_state.raw_data = file_handler.load_data_from_zip(zip_file)

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
                                st.success(
                                    f"✅ 数据加载成功：{st.session_state.raw_data[1].shape[1]}条光谱，{len(np.unique(labels))}类")
                            else:
                                st.warning(f"⚠️ 标签数({len(labels)})≠光谱数({st.session_state.raw_data[1].shape[1]})")
                                st.session_state.labels = None
                        except Exception as e:
                            st.warning(f"⚠️ 标签格式错误: {str(e)}")
                            st.session_state.labels = None
                    else:
                        st.success(
                            f"✅ 数据加载成功：{st.session_state.raw_data[1].shape[1]}条光谱，{st.session_state.raw_data[1].shape[0]}个点")
                        st.warning("⚠️ 请输入样本标签以进行分类测试")
                except Exception as e:
                    st.error(f"❌ 文件加载失败: {str(e)}")

        if st.session_state.get('raw_data'):
            wavenumbers, y = st.session_state.raw_data
            st.info(f"📊 数据维度: {y.shape[1]}条 × {y.shape[0]}点")
            st.info(f"🔢 训练集:{train_test_ratio:.1f} | 测试集:{1 - train_test_ratio:.1f}")
            if st.session_state.get('labels') is not None:
                class_counts = np.bincount(st.session_state.labels)
                st.info(
                    f"🏷️ 类别分布: {', '.join([f'类{i}:{count}个' for i, count in enumerate(class_counts) if count > 0])}")
            if st.session_state.get('process_method'):
                st.success(f"🛠️ 处理流程: {st.session_state.process_method}")

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

            # 使用9列布局：4个算法列 + 5个操作相关列，确保所有内容横向排列
            preprocess_cols = st.columns([1, 1, 1, 1, 1.2, 1.2, 1.2, 1.2, 1.2], gap="small")

            # 1. 基线校准（第一列）
            st.subheader("基线校准")
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

            # 2. 缩放处理（第二列）
            st.subheader("📏 缩放")
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
            elif scaling_method == "标准化(均值0，方差1)":
                st.caption("将数据标准化到均值为0，方差为1")

            # 3. 滤波处理（第三列）
            st.subheader("📶 滤波")
            filtering_method = st.selectbox(
                "方法",
                ["无", "Savitzky-Golay", "sgolayfilt滤波器", "中值滤波(MF)", "移动平均(MAF)",
                 "MWA（移动窗口平均）", "MWM（移动窗口中值）", "卡尔曼滤波", "Lowess", "FFT",
                 "Smfft傅里叶滤波", "小波变换(DWT)", "小波线性阈值去噪"],
                key="filtering_method",
                label_visibility="collapsed"
            )

            # 滤波参数
            filtering_params = {}
            if filtering_method != "无":
                if filtering_method in ["Savitzky-Golay", "sgolayfilt滤波器"]:
                    k = st.selectbox("多项式阶数", [3, 7], key="k_sg", label_visibility="collapsed")
                    w = st.selectbox("窗口大小", [11, 31, 51], key="w_sg", label_visibility="collapsed")
                    filtering_params["point"] = w
                    filtering_params["degree"] = k
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

            # 4. 挤压处理（第四列）
            st.subheader("🧪 挤压")
            squashing_method = st.selectbox(
                "方法",
                ["无", "Sigmoid挤压", "改进的Sigmoid挤压", "逻辑函数", "余弦挤压(squashing)", "改进的逻辑函数",
                 "DTW挤压"],
                key="squashing_method",
                label_visibility="collapsed"
            )
    
    if __name__ == "__main__":
        main()
