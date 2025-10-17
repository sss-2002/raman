# arrangement/main.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .state import init_state
from .algorithms import Preprocessor
from .utils.file_handler import FileHandler
from .utils.permutations import generate_permutations
from .utils.classifier import knn_classify, evaluate_classification

def main():
    # 初始化状态
    init_state()
    
    # 初始化工具类
    file_handler = FileHandler()
    preprocessor = Preprocessor()

    # 设置页面
    st.set_page_config(layout="wide", page_icon="🔬", page_title="排列预处理模型")
    
    # 全局样式
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

    # 布局
    col_left, col_right = st.columns([1.2, 3.9])

    # 左侧：数据管理
    with col_left:
        with st.expander("📁 数据管理", expanded=True):
            zip_file = st.file_uploader("上传压缩包", type=['zip'], key="zip_file")
            st.caption("需包含波数和光谱数据文件")

            # 标签输入
            st.subheader("样本标签")
            num_classes = st.number_input("类别数量", min_value=1, value=2, step=1, key="num_cls")
            labels_input = st.text_input(
                "标签（逗号分隔）",
                placeholder="例：0,0,1,1",
                key="labels_in"
            )

            # 训练测试比例
            st.subheader("训练测试划分")
            train_test_ratio = st.slider(
                "训练集比例",
                0.1, 0.9, 0.8, 0.1,
                format="%.1f",
                key="train_ratio"
            )
            st.session_state.train_test_split_ratio = train_test_ratio

            # 数据加载
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
                                st.success(f"✅ 加载成功：{st.session_state.raw_data[1].shape[1]}条光谱，{len(np.unique(labels))}类")
                            else:
                                st.warning(f"⚠️ 标签数≠光谱数")
                        except Exception as e:
                            st.warning(f"⚠️ 标签格式错误: {str(e)}")
                    else:
                        st.success(f"✅ 加载成功：{st.session_state.raw_data[1].shape[1]}条光谱")
                except Exception as e:
                    st.error(f"❌ 加载失败: {str(e)}")

        # 系统信息
        if st.session_state.get('raw_data'):
            wavenumbers, y = st.session_state.raw_data
            st.info(f"📊 数据维度: {y.shape[1]}条 × {y.shape[0]}点")
            st.info(f"🔢 训练集:{train_test_ratio:.1f} | 测试集:{1 - train_test_ratio:.1f}")
            if st.session_state.get('labels') is not None:
                class_counts = np.bincount(st.session_state.labels)
                st.info(f"🏷️ 类别分布: {', '.join([f'类{i}:{count}个' for i, count in enumerate(class_counts) if count > 0])}")

        # 使用指南
        with st.expander("ℹ️ 使用指南", expanded=False):
            st.markdown("""
            1. 上传含波数和光谱的ZIP包  
            2. 设置标签和训练测试比例  
            3. 选择预处理方法  
            4. 点击"显示排列"生成方案  
            5. 选择k值后点击"测试"  
            6. 查看结果并导出
            """)

    # 右侧：预处理与可视化
    with col_right:
        st.subheader("⚙️ 预处理设置", divider="gray")
        preprocess_cols = st.columns([1, 1, 1, 1, 1.2, 1.2, 1.2, 1.2, 1.2], gap="small")

        # 1. 基线校准
        with preprocess_cols[0]:
            st.subheader("基线校准")
            baseline_method = st.selectbox(
                "方法",
                ["无", "SD", "FD", "多项式拟合", "ModPoly", "I-ModPoly", "PLS", "AsLS", "airPLS", "二阶差分(D2)"],
                key="baseline_method",
                label_visibility="collapsed"
            )
            baseline_params = {}
            if baseline_method == "多项式拟合":
                polyorder = st.slider("阶数k", 3, 6, 5, key="polyorder", label_visibility="collapsed")
                baseline_params["polyorder"] = polyorder
                st.caption(f"阶数: {polyorder}")
            elif baseline_method == "ModPoly":
                k = st.slider("参数k", 4, 10, 10, key="k_mod", label_visibility="collapsed")
                baseline_params["k"] = k
                st.caption(f"k: {k}")
            elif baseline_method == "I-ModPoly":
                polyorder = st.slider("阶数", 3, 7, 5, key="imod_polyorder", label_visibility="collapsed")
                max_iter = st.slider("迭代次数", 50, 200, 100, key="imod_maxiter", label_visibility="collapsed")
                tolerance = st.slider("容差", 0.001, 0.01, 0.005, key="imod_tol", label_visibility="collapsed")
                baseline_params = {"polyorder": polyorder, "max_iter": max_iter, "tolerance": tolerance}
                st.caption(f"阶数: {polyorder}, 迭代: {max_iter}")
            elif baseline_method == "PLS":
                lam = st.selectbox("λ", [10**10, 10**8, 10**7], key="lam_pls", label_visibility="collapsed")
                baseline_params["lam"] = lam
                st.caption(f"λ: {lam}")
            elif baseline_method == "AsLS":
                p = st.selectbox("p", [0.001, 0.01, 0.1], key="p_asls", label_visibility="collapsed")
                lam = st.selectbox("λ", [10**5, 10**7, 10**9], key="lam_asls", label_visibility="collapsed")
                niter = st.selectbox("迭代", [5, 10, 15], key="niter_asls", label_visibility="collapsed")
                baseline_params = {"lam": lam, "p": p, "niter": niter}
                st.caption(f"p: {p}, λ: {lam}")
            elif baseline_method == "airPLS":
                lam = st.selectbox("λ", [10**7, 10**4, 10**2], key="lam_air", label_visibility="collapsed")
                baseline_params["lam"] = lam
                st.caption(f"λ: {lam}")

        # 2. 缩放处理
        with preprocess_cols[1]:
            st.subheader("📏 缩放")
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
            if filtering_method in ["Savitzky-Golay", "sgolayfilt滤波器"]:
                k = st.selectbox("阶数", [3, 7], key="k_sg", label_visibility="collapsed")
                w = st.selectbox("窗口", [11, 31, 51], key="w_sg", label_visibility="collapsed")
                filtering_params = {"window_length": w, "polyorder": k}
                st.caption(f"阶数: {k}, 窗口: {w}")
            elif filtering_method in ["中值滤波(MF)", "移动平均(MAF)"]:
                k = st.selectbox("k", [1, 3], key="k_mf", label_visibility="collapsed")
                w = st.selectbox("w", [7, 11], key="w_mf", label_visibility="collapsed")
                filtering_params = {"k": k, "w": w}
                st.caption(f"k: {k}, w: {w}")
            elif filtering_method == "小波线性阈值去噪":
                threshold = st.selectbox("阈值", [0.1, 0.3, 0.5], key="thresh_wavelet", label_visibility="collapsed")
                filtering_params["threshold"] = threshold
                st.caption(f"阈值: {threshold}")

        # 4. 挤压处理
        with preprocess_cols[3]:
            st.subheader("🧪 挤压")
            squashing_method = st.selectbox(
                "方法",
                ["无", "Sigmoid挤压", "改进的Sigmoid挤压", "逻辑函数", "余弦挤压(squashing)", "改进的逻辑函数",
                 "DTW挤压"],
                key="squashing_method",
                label_visibility="collapsed"
            )
            squashing_params = {}
            if squashing_method == "改进的Sigmoid挤压":
                maxn = st.selectbox("maxn", [5, 10, 15], key="maxn_isigmoid", label_visibility="collapsed")
                squashing_params["maxn"] = maxn
                st.caption(f"maxn: {maxn}")
            elif squashing_method == "DTW挤压":
                l = st.selectbox("l", [1, 5], key="l_dtw", label_visibility="collapsed")
                k1 = st.selectbox("k1", ["T", "F"], key="k1_dtw", label_visibility="collapsed")
                k2 = st.selectbox("k2", ["T", "F"], key="k2_dtw", label_visibility="collapsed")
                squashing_params = {"l": l, "k1": k1, "k2": k2}
                st.caption(f"l: {l}, k1: {k1}")

        # 5. 应用处理
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
                            'params': st.session_state.current_algorithms
                        }
                        st.session_state.selected_arrangement = arr_name
                        st.session_state.processed_data = (wavenumbers, processed_data)
                        st.session_state.process_method = " → ".join(method_name)
                        st.success(f"✅ 处理完成")
                    except Exception as e:
                        st.error(f"❌ 处理失败: {str(e)}")

        # 6. 显示排列
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
                all_first_step_types = list({p.get("first_step_type", "未知") for p in st.session_state.algorithm_permutations})
                all_first_step_types.sort()
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
                    format_func=lambda x: st.session_state.filtered_perms[x].get("name", f"方案{x + 1}"),
                    key="perm_select",
                    label_visibility="collapsed"
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
                                train_data, train_labels, test_data, k=st.session_state.k_value
                            )
                        results = evaluate_classification(test_labels, predictions)
                        results['predictions'] = predictions
                        results['test_labels'] = test_labels
                        st.session_state.test_results = results
                        st.success("✅ 测试完成！结果在下方")
                    except Exception as e:
                        st.error(f"❌ 测试失败: {str(e)}")

        # 保存当前算法设置
        st.session_state.current_algorithms.update({
            'baseline': baseline_method,
            'baseline_params': baseline_params,
            'scaling': scaling_method,
            'scaling_params': scaling_params,
            'filtering': filtering_method,
            'filtering_params': filtering_params,
            'squashing': squashing_method,
            'squashing_params': squashing_params
        })

        # 光谱可视化
        st.subheader("📈 光谱可视化", divider="gray")
        viz_row1 = st.columns(2, gap="medium")
        viz_row2 = st.columns(2, gap="medium")

        # 原始光谱
        with viz_row1[0]:
            st.subheader("原始光谱", divider="gray")
            if st.session_state.get('raw_data'):
                wavenumbers, y = st.session_state.raw_data
                idx1 = 0 if y.shape[1] > 0 else 0
                raw_data1 = pd.DataFrame({"原始光谱1": y[:, idx1]}, index=wavenumbers)
                st.line_chart(raw_data1, height=250)
                if y.shape[1] > 1:
                    with st.expander("查看更多原始光谱", expanded=False):
                        for i in range(1, min(y.shape[1], 5)):
                            st.subheader(f"原始光谱{i + 1}", divider="gray")
                            data = pd.DataFrame({f"原始光谱{i + 1}": y[:, i]}, index=wavenumbers)
                            st.line_chart(data, height=150)
            else:
                st.markdown(
                    '<div style="border:1px dashed #ccc; height:250px; display:flex; align-items:center; justify-content:center;">等待加载原始数据</div>',
                    unsafe_allow_html=True)

        # 预处理后光谱
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
                if arr_data.shape[1] > 1:
                    with st.expander("查看更多预处理后光谱", expanded=False):
                        for i in range(1, min(arr_data.shape[1], 5)):
                            data = pd.DataFrame({f"预处理后{i + 1}": arr_data[:, i]}, index=wavenumbers)
                            st.line_chart(data, height=150)
            else:
                st.markdown(
                    '<div style="border:1px dashed #ccc; height:250px; display:flex; align-items:center; justify-content:center;">请先应用预处理方案</div>',
                    unsafe_allow_html=True)

        # k值曲线
        with viz_row2[0]:
            st.subheader("k值曲线", divider="gray")
            if st.session_state.get('selected_arrangement'):
                selected_arr = st.session_state.selected_arrangement
                arr_data = st.session_state.arrangement_details[selected_arr]['data']
                wavenumbers, y = st.session_state.raw_data
                arr_order = st.session_state.arrangement_details[selected_arr].get('order', [])
                if arr_order and y.shape[1] > 0:
                    idx1 = 0
                    k_vals1 = np.abs(arr_data[:, 0] / (y[:, 0] + 1e-8))
                    k_data1 = pd.DataFrame({"k值1": k_vals1}, index=wavenumbers)
                    st.line_chart(k_data1, height=250)
                    if y.shape[1] > 1:
                        with st.expander("查看更多k值曲线", expanded=False):
                            for i in range(1, min(y.shape[1], 5)):
                                k_vals = np.abs(arr_data[:, i] / (y[:, i] + 1e-8))
                                data = pd.DataFrame({f"k值{i + 1}": k_vals}, index=wavenumbers)
                                st.line_chart(data, height=150)
                else:
                    st.info("ℹ️ 无预处理，不显示k值曲线")
            else:
                st.markdown(
                    '<div style="border:1px dashed #ccc; height:250px; display:flex; align-items:center; justify-content:center;">请先应用预处理方案</div>',
                    unsafe_allow_html=True)

        # 混淆矩阵
        with viz_row2[1]:
            st.subheader("混淆矩阵", divider="gray")
            if st.session_state.get('test_results') is not None:
                results = st.session_state.test_results
                st.markdown("**分类指标**")
                st.text(f"准确率: {results['accuracy']:.4f}")
                st.text(f"卡帕系数: {results['kappa']:.4f}")
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
            st.markdown(
                '<div style="border:1px dashed #ccc; height:80px; display:flex; align-items:center; justify-content:center;">处理完成后可导出结果</div>',
                unsafe_allow_html=True)

if __name__ == "__main__":
    main()
