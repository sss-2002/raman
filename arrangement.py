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
            train_test_ratio = st.slider("训练集比例", min_value=0.1, max_value=0.9, value=0.8, step=0.1, format="%.1f", key="train_ratio")
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
                                st.success(f"✅ 数据加载成功：{st.session_state.raw_data[1].shape[1]}条光谱，{len(np.unique(labels))}类")
                            else:
                                st.warning(f"⚠️ 标签数({len(labels)})≠光谱数({st.session_state.raw_data[1].shape[1]})")
                                st.session_state.labels = None
                        except Exception as e:
                            st.warning(f"⚠️ 标签格式错误: {str(e)}")
                            st.session_state.labels = None
                    else:
                        st.success(f"✅ 数据加载成功：{st.session_state.raw_data[1].shape[1]}条光谱，{st.session_state.raw_data[1].shape[0]}个点")
                        st.warning("⚠️ 请输入样本标签以进行分类测试")
                except Exception as e:
                    st.error(f"❌ 文件加载失败: {str(e)}")
        
        if st.session_state.get('raw_data'):
            wavenumbers, y = st.session_state.raw_data
            st.info(f"📊 数据维度: {y.shape[1]}条 × {y.shape[0]}点")
            st.info(f"🔢 训练集:{train_test_ratio:.1f} | 测试集:{1-train_test_ratio:.1f}")
            if st.session_state.get('labels') is not None:
                class_counts = np.bincount(st.session_state.labels)
                st.info(f"🏷️ 类别分布: {', '.join([f'类{i}:{count}个' for i, count in enumerate(class_counts) if count>0])}")
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
        
        preprocess_cols = st.columns([1, 1, 1, 1, 1.2, 1.2, 1.2, 1.2, 1.2], gap="small")
        
        # 1. 基线校准
         st.subheader("基线校准")
         baseline_method = st.selectbox("方法", ["无", "SD", "FD", "多项式拟合", "ModPoly", "I-ModPoly", "PLS", "AsLS", "airPLS", "二阶差分(D2)"], key="baseline_method", label_visibility="collapsed")
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
                    p = st.selectbox("非对称系数p", [0.001, 0.01, 0.1], key="p_asls", label_visibility="collapsed")
                    lam = st.selectbox("平滑系数λ", [10**5, 10**7, 10**9], key="lam_asls", label_visibility="collapsed")
                    niter = st.selectbox("迭代次数", [5, 10, 15], key="niter_asls", label_visibility="collapsed")
                    baseline_params["lam"] = lam
                    baseline_params["p"] = p
                    baseline_params["niter"] = niter
                    st.caption(f"p: {p}, λ: {lam}, 迭代次数: {niter}")
                elif baseline_method == "airPLS":
                    lam = st.selectbox("λ", [10**7, 10**4, 10**2], key="lam_air", label_visibility="collapsed")
                    baseline_params["lam"] = lam
                    st.caption(f"λ: {lam}")
                elif baseline_method == "二阶差分(D2)":  # 二阶差分参数说明
                    st.caption("二阶差分可增强光谱特征，抑制基线漂移")

        # 2. 缩放处理
        with preprocess_cols[1]:
            st.subheader("📏 缩放")
            scaling_method = st.selectbox("方法", ["无", "Peak-Norm", "SNV", "MSC", "M-M-Norm", "L-范数", "Ma-Minorm", "标准化(均值0，方差1)"], key="scaling_method", label_visibility="collapsed")
            scaling_params = {}
            if scaling_method == "L-范数":
                p = st.selectbox("p", ["无穷大", "4", "10"], key="p_scale", label_visibility="collapsed")
                scaling_params["p"] = p
                st.caption(f"p: {p}")
            elif scaling_method == "标准化(均值0，方差1)":
                st.caption("将数据标准化到均值为0，方差为1")

        # 3. 滤波处理
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

        # 4. 挤压处理（第四列）
        with preprocess_cols[3]:
            st.subheader("🧪 挤压")
            squashing_method = st.selectbox(
                "方法",
                ["无", "Sigmoid挤压", "改进的Sigmoid挤压", "逻辑函数", "余弦挤压(squashing)", "改进的逻辑函数", "DTW挤压"],
                key="squashing_method",
                label_visibility="collapsed"
            )

            # 挤压参数
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

        # 5-9列：操作相关内容（横向排列在四个预处理算法后面）
        # 5. 应用处理按钮
        with preprocess_cols[4]:
            st.subheader("操作1")
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
            
            if st.button("🌟 推荐应用", type="primary", use_container_width=True, key="recommend_btn"):
                if st.session_state.raw_data is None:
                    st.warning("⚠️ 请先上传数据")
                else:
                    try:
                        wavenumbers, y = st.session_state.raw_data
                        recommended_params = {
                            'baseline_method': "二阶差分(D2)",
                            'baseline_params': {},
                            'scaling_method': "标准化(均值0，方差1)",
                            'scaling_params': {},
                            'filtering_method': "小波线性阈值去噪",
                            'filtering_params': {'threshold': 0.3},
                            'squashing_method': "余弦挤压(squashing)",
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

        # 6. 显示排列与筛选
        with preprocess_cols[5]:
            st.subheader("操作2")
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

            if st.session_state.show_arrangements and st.session_state.algorithm_permutations:
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
                
                if selected_first_step == "全部":
                    st.session_state.filtered_perms = st.session_state.algorithm_permutations
                else:
                    st.session_state.filtered_perms = [
                        p for p in st.session_state.algorithm_permutations 
                        if p.get("first_step_type") == selected_first_step
                    ]

        # 7. 排列选择与应用
        with preprocess_cols[6]:
            st.subheader("操作3")
            if st.session_state.show_arrangements and st.session_state.filtered_perms:
                st.session_state.selected_perm_idx = st.selectbox(
                    f"选择方案（共{len(st.session_state.filtered_perms)}种）",
                    range(len(st.session_state.filtered_perms)),
                    format_func=lambda x: st.session_state.filtered_perms[x].get("name", f"方案{x+1}"),
                    key="perm_select",
                    label_visibility="collapsed",
                    help="选择预处理算法顺序"
                )
                
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
                if st.session_state.show_arrangements:
                    st.info("ℹ️ 无符合条件的方案")
        
        # 8. 分类测试参数
        with preprocess_cols[7]:
            st.subheader("操作4")
            st.subheader("📝 分类测试")
            k_value = st.number_input(
                "k值", 
                min_value=1, 
                value=st.session_state.k_value,
                step=1,
                key="k_input",
                label_visibility="collapsed"
            )
            
            if st.button("确定k值", type="secondary", use_container_width=True, key="k_confirm_btn"):
                st.session_state.k_value = k_value
                st.success(f"k={k_value}")

        # 9. 测试按钮
        with preprocess_cols[8]:
            st.subheader("操作5")
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
                        
                        st.success("✅ 测试完成！结果在下方")
                        
                    except Exception as e:
                        st.error(f"❌ 测试失败: {str(e)}")

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

        # ===== 光谱可视化与结果导出（在预处理设置下方）=====
        st.subheader("📈 光谱可视化", divider="gray")
        
        # 1. 原始光谱区域
        st.subheader("原始光谱", divider="gray")
        spec_cols = st.columns(2, gap="small")
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
            
            if st.session_state.get('raw_data') and y.shape[1] > 2:
                with st.expander("查看更多原始光谱", expanded=False):
                    more_spec = st.columns(2, gap="small")
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
            
            st.caption(f"处理方法: {arr_method} | 执行顺序: {arr_order if arr_order else '无预处理'}")
            
            st.subheader("预处理后光谱", divider="gray")
            proc_cols = st.columns(2, gap="small")
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
                    k_cols = st.columns(2, gap="small")
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
                
                # 原始与处理后对比
                st.subheader("原始vs预处理对比", divider="gray")
                comp_cols = st.columns(2, gap="small")
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
                    
                    # 指标
                    metrics_cols = st.columns(2, gap="small")
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
                    st.info("ℹ️ 请在上方选择预处理方法并应用排列方案")
                    
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
    
    # 预处理类
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
        
    if __name__ == "__main__":
        main()
