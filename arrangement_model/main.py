import streamlit as st
import numpy as np
# 正确：从当前文件夹（arrangement_model）导入子模块
from .algorithms.preprocessing import Preprocessor
from .utils.file_handler import FileHandler
from algorithms.classification import knn_classify
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import itertools

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
    st.session_state['current_algorithms'] = current_algorithms
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
    all_states = {** test_states, **other_states}
    for key, value in all_states.items():
        if key not in st.session_state:
            st.session_state[key] = value
    st.session_state['current_algorithms'] = current_algorithms
    # 设置页面：紧凑布局
    st.set_page_config(layout="wide", page_icon="🔬", page_title="排列预处理模型")
    # 全局样式调整
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
            # 上传文件夹压缩包
            zip_file = st.file_uploader("上传包含波数和光谱数据的压缩包", type=['zip'], key="zip_file")
            st.caption("压缩包(.zip)需包含波数和光谱数据文件")

            # 标签输入
            st.subheader("样本标签")
            num_classes = st.number_input("类别数量", min_value=1, value=2, step=1, key="num_cls")
            labels_input = st.text_input(
                "标签（逗号分隔，与光谱顺序一致）",
                placeholder="例：0,0,1,1",
                key="labels_in"
            )

            # 训练测试比例
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

            # 数据加载逻辑（从压缩包加载）
            if zip_file:
                try:
                    st.session_state.raw_data = file_handler.load_data_from_zip(
                        zip_file
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

        # 系统信息
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
        # ===== 预处理设置 =====
        st.subheader("⚙️ 预处理设置", divider="gray")

        # 使用9列布局
        preprocess_cols = st.columns([1, 1, 1, 1, 1.2, 1.2, 1.2, 1.2, 1.2], gap="small")

        # 1. 基线校准（第一列）
        with preprocess_cols[0]:
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
                    lam = st.selectbox("平滑系数λ", [10**5, 10**7, 10**9], key="lam_asls",
                                       label_visibility="collapsed")
                    niter = st.selectbox("迭代次数", [5, 10, 15], key="niter_asls", label_visibility="collapsed")
                    baseline_params["lam"] = lam
                    baseline_params["p"] = p
                    baseline_params["niter"] = niter
                    st.caption(f"p: {p}, λ: {lam}, 迭代次数: {niter}")
                elif baseline_method == "airPLS":
                    lam = st.selectbox("λ", [10**7, 10**4, 10**2], key="lam_air", label_visibility="collapsed")
                    baseline_params["lam"] = lam
                    st.caption(f"λ: {lam}")
                elif baseline_method == "二阶差分(D2)":
                    st.caption("二阶差分可增强光谱特征，抑制基线漂移")

        # 2. 缩放处理（第二列）
        with preprocess_cols[1]:
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

            # 滤波参数
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

        # 4. 挤压处理（第四列）
        with preprocess_cols[3]:
            st.subheader("🧪 挤压")
            squashing_method = st.selectbox(
                "方法",
                ["无", "Sigmoid挤压", "改进的Sigmoid挤压", "逻辑函数", "余弦挤压(squashing)", "改进的逻辑函数",
                 "DTW挤压"],
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

        # 5-9列：操作相关内容
        with preprocess_cols[4]:
            st.subheader("操作1")
            # 应用处理按钮
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

        with preprocess_cols[5]:
            st.subheader("操作2")
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

        with preprocess_cols[6]:
            st.subheader("操作3")
            # 排列下拉框
            if st.session_state.show_arrangements and st.session_state.filtered_perms:
                st.session_state.selected_perm_idx = st.selectbox(
                    f"选择方案（共{len(st.session_state.filtered_perms)}种）",
                    range(len(st.session_state.filtered_perms)),
                    format_func=lambda x: st.session_state.filtered_perms[x].get("name", f"方案{x + 1}"),
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
                if st.session_state.show_arrangements:
                    st.info("ℹ️ 无符合条件的方案")

        with preprocess_cols[7]:
            st.subheader("操作4")
            # k值设置
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

        with preprocess_cols[8]:
            st.subheader("操作5")
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

        # ===== 光谱可视化与结果导出 =====
        st.subheader("📈 光谱可视化", divider="gray")

        # 创建四个固定区域的布局
        viz_row1 = st.columns(2, gap="medium")
        viz_row2 = st.columns(2, gap="medium")

        # 1. 原始光谱区域
        with viz_row1[0]:
            st.subheader("原始光谱", divider="gray")
            if st.session_state.get('raw_data'):
                wavenumbers, y = st.session_state.raw_data
                idx1 = 0 if y.shape[1] > 0 else 0
                raw_data1 = {
                    "原始光谱1": y[:, idx1]
                }
                st.line_chart(raw_data1, x=wavenumbers, height=250)
                
                # 显示更多原始光谱
                if y.shape[1] > 1:
                    with st.expander("查看更多原始光谱", expanded=False):
                        for i in range(1, min(y.shape[1], 5)):
                            st.subheader(f"原始光谱{i + 1}", divider="gray")
                            data = {f"原始光谱{i + 1}": y[:, i]}
                            st.line_chart(data, x=wavenumbers, height=150)
            else:
                st.markdown(
                    '<div style="border:1px dashed #ccc; height:250px; display:flex; align-items:center; justify-content:center;">等待加载原始数据</div>',
                    unsafe_allow_html=True)

        # 2. 预处理后光谱区域
        with viz_row1[1]:
            st.subheader("预处理后的光谱", divider="gray")
            if st.session_state.get('selected_arrangement'):
                selected_arr = st.session_state.selected_arrangement
                arr_data = st.session_state.arrangement_details[selected_arr]['data']
                arr_method = st.session_state.arrangement_details[selected_arr]['method']
                st.caption(f"处理方法: {arr_method}")
                
                idx1 = 0 if arr_data.shape[1] > 0 else 0
                proc_data1 = {"预处理后1": arr_data[:, idx1]}
                st.line_chart(proc_data1, x=wavenumbers, height=250)
                
                # 显示更多预处理后光谱
                if arr_data.shape[1] > 1:
                    with st.expander("查看更多预处理后光谱", expanded=False):
                        for i in range(1, min(arr_data.shape[1], 5)):
                            st.subheader(f"预处理后{i + 1}", divider="gray")
                            data = {f"预处理后{i + 1}": arr_data[:, i]}
                            st.line_chart(data, x=wavenumbers, height=150)
            else:
                st.markdown(
                    '<div style="border:1px dashed #ccc; height:250px; display:flex; align-items:center; justify-content:center;">请先应用预处理方案</div>',
                    unsafe_allow_html=True)

        # 3. k值曲线区域
        with viz_row2[0]:
            st.subheader("k值曲线", divider="gray")
            if st.session_state.get('selected_arrangement'):
                selected_arr = st.session_state.selected_arrangement
                arr_data = st.session_state.arrangement_details[selected_arr]['data']
                wavenumbers, y = st.session_state.raw_data
                arr_order = st.session_state.arrangement_details[selected_arr].get('order', [])
                
                if arr_order:  # 只有应用了预处理才有k值曲线
                    idx1 = 0 if arr_data.shape[1] > 0 else 0
                    k_vals1 = np.abs(arr_data[:, 0] / (y[:, 0] + 1e-8)) if y.shape[1] > 0 else np.array([])
                    k_data1 = {"k值1": k_vals1}
                    st.line_chart(k_data1, x=wavenumbers, height=250)
                    
                    # 显示更多k值曲线
                    if y.shape[1] > 1:
                        with st.expander("查看更多k值曲线", expanded=False):
                            for i in range(1, min(y.shape[1], 5)):
                                st.subheader(f"k值{i + 1}", divider="gray")
                                k_vals = np.abs(arr_data[:, i] / (y[:, i] + 1e-8))
                                data = {f"k值{i + 1}": k_vals}
                                st.line_chart(data, x=wavenumbers, height=150)
                else:
                    st.info("ℹ️ 无预处理（原始光谱），不显示k值曲线")
            else:
                st.markdown(
                    '<div style="border:1px dashed #ccc; height:250px; display:flex; align-items:center; justify-content:center;">请先应用预处理方案</div>',
                    unsafe_allow_html=True)

        # 4. 混淆矩阵区域
        with viz_row2[1]:
            st.subheader("混淆矩阵", divider="gray")
            if st.session_state.get('test_results') is not None:
                results = st.session_state.test_results
                
                # 显示分类指标
                st.markdown("**分类指标**")
                st.text(f"准确率: {results['accuracy']:.4f}")
                st.text(f"卡帕系数: {results['kappa']:.4f}")
                
                # 显示混淆矩阵
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax,
                            annot_kws={"size": 8})
                ax.set_xlabel('预测标签', fontsize=8)
                ax.set_ylabel('真实标签', fontsize=8)
                ax.set_title('混淆矩阵', fontsize=10)
                plt.xticks(fontsize=7)
                plt.yticks(fontsize=7)
                st.pyplot(fig, use_container_width=True)
            else:
                st.markdown(
                    '<div style="border:1px dashed #ccc; height:250px; display:flex; align-items:center; justify-content:center;">请先进行分类测试</div>',
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
