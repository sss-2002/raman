import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .state import init_state
from .algorithms import Preprocessor
from .utils.file_handler import FileHandler
from .utils.permutations import generate_permutations
from .utils.classifier import knn_classify, evaluate_classification

# 定义main()函数，作为arrangement模块的入口
def main():
    # 初始化会话状态（确保所有必要变量已创建）
    init_state()
    
    # 初始化工具类
    file_handler = FileHandler()  # 文件处理工具
    preprocessor = Preprocessor()  # 预处理算法控制器

    # 页面样式（优化为全屏显示，与主文件保持一致）
    st.markdown("""
        <style>
        /* 清除Streamlit默认根容器的边距和宽度限制 */
        .css-18e3th9 {
            padding: 0 !important;
            max-width: 100% !important;
        }
        .block-container {
            padding: 0 10px !important;
            max-width: 100% !important;
        }
        
        /* 原有页面和按钮样式 */
        .main {
            background-color: #f5f7fa;
            padding: 0px 10px;
        }
        .stButton > button {
            border-radius: 6px;
            background-color: #165DFF;
            color: white;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #0E42D2;
        }
        </style>
    """, unsafe_allow_html=True)

    # 页面标题
    st.title("🌌 排列预处理模型")

    # 布局：左侧数据管理，右侧预处理与可视化
    col_left, col_right = st.columns([1.2, 3.9])

    # 左侧：数据管理区域
    with col_left:
        with st.expander("📁 数据管理", expanded=True):
            # 上传ZIP文件
            zip_file = st.file_uploader("上传压缩包（含波数和光谱文件）", type=['zip'], key="zip_file")
            st.caption("支持格式：ZIP（内含波数文件+光谱数据文件）")

            # 样本标签输入
            st.subheader("样本标签")
            num_classes = st.number_input("类别数量", min_value=1, value=2, step=1, key="num_cls")
            labels_input = st.text_input(
                "标签（逗号分隔，如0,0,1,1）",
                placeholder="示例：0,0,1,1,2,2",
                key="labels_in"
            )

            # 训练/测试集划分比例
            st.subheader("训练测试划分")
            train_test_ratio = st.slider(
                "训练集占比",
                0.1, 0.9, 0.8, 0.1,
                format="%.1f",
                key="train_ratio"
            )
            st.session_state.train_test_split_ratio = train_test_ratio

            # 加载数据逻辑
            if zip_file:
                try:
                    # 从ZIP中加载波数和光谱数据
                    st.session_state.raw_data = file_handler.load_data_from_zip(zip_file)
                    wavenumbers, spectra = st.session_state.raw_data

                    # 处理标签
                    if labels_input:
                        try:
                            labels = np.array([int(l.strip()) for l in labels_input.split(',')])
                            if len(labels) == spectra.shape[1]:  # 标签数需与样本数一致
                                st.session_state.labels = labels
                                # 划分训练/测试索引
                                n_samples = len(labels)
                                train_size = int(n_samples * train_test_ratio)
                                indices = np.random.permutation(n_samples)
                                st.session_state.train_indices = indices[:train_size]
                                st.session_state.test_indices = indices[train_size:]
                                st.success(f"✅ 加载成功：{spectra.shape[1]}条光谱，{len(np.unique(labels))}类")
                            else:
                                st.warning(f"⚠️ 标签数（{len(labels)}）与光谱数（{spectra.shape[1]}）不匹配")
                        except Exception as e:
                            st.warning(f"⚠️ 标签格式错误：{str(e)}（请输入整数，用逗号分隔）")
                    else:
                        st.success(f"✅ 加载成功：{spectra.shape[1]}条光谱（未输入标签）")
                except Exception as e:
                    st.error(f"❌ 数据加载失败：{str(e)}")

        # 显示系统信息
        if st.session_state.get('raw_data'):
            wavenumbers, spectra = st.session_state.raw_data
            st.info(f"📊 数据维度：{spectra.shape[1]}条光谱 × {spectra.shape[0]}个波数点")
            st.info(f"🔄 划分比例：训练集{train_test_ratio:.1f} | 测试集{1-train_test_ratio:.1f}")
            if st.session_state.get('labels') is not None:
                class_counts = np.bincount(st.session_state.labels)
                st.info(f"🏷️ 类别分布：{', '.join([f'类{i}: {count}个' for i, count in enumerate(class_counts) if count > 0])}")

        # 使用指南
        with st.expander("ℹ️ 使用指南", expanded=False):
            st.markdown("""
            1. 上传包含波数文件（含"wave"/"wn"/"波数"）和光谱文件（含"spec"/"data"/"光谱"）的ZIP包  
            2. 输入样本标签（整数，逗号分隔）并设置训练/测试比例  
            3. 选择预处理算法（基线校正、缩放、滤波、挤压）及参数  
            4. 点击"显示排列"生成所有可能的算法执行顺序  
            5. 选择排列方案并点击"应用方案"  
            6. 设置KNN的k值，点击"测试"评估分类效果  
            7. 查看结果并导出预处理后的数据
            """)

    # 右侧：预处理与可视化区域
    with col_right:
        st.subheader("⚙️ 预处理算法设置", divider="gray")
        preprocess_cols = st.columns([1, 1, 1, 1, 1.2, 1.2, 1.2, 1.2, 1.2], gap="small")

        # 1. 基线校正算法选择
        with preprocess_cols[0]:
            st.subheader("基线校正")
            baseline_method = st.selectbox(
                "选择算法",
                ["无", "SD", "FD", "多项式拟合", "ModPoly", "I-ModPoly", "PLS", "AsLS", "airPLS", "二阶差分(D2)"],
                key="baseline_method",
                label_visibility="collapsed"
            )
            baseline_params = {}
            # 根据算法类型设置参数
            if baseline_method == "多项式拟合":
                polyorder = st.slider("阶数", 3, 6, 5, key="polyorder", label_visibility="collapsed")
                baseline_params["polyorder"] = polyorder
                st.caption(f"阶数: {polyorder}")
            elif baseline_method == "ModPoly":
                k = st.slider("迭代参数", 4, 10, 10, key="k_mod", label_visibility="collapsed")
                baseline_params["k"] = k
                st.caption(f"k: {k}")
            elif baseline_method == "I-ModPoly":
                polyorder = st.slider("阶数", 3, 7, 5, key="imod_poly", label_visibility="collapsed")
                max_iter = st.slider("迭代次数", 50, 200, 100, key="imod_iter", label_visibility="collapsed")
                baseline_params = {"polyorder": polyorder, "max_iter": max_iter}
                st.caption(f"阶数: {polyorder}, 迭代: {max_iter}")
            # 其他算法参数设置（省略部分重复代码，保持与之前一致）

        # 2. 缩放算法选择
        with preprocess_cols[1]:
            st.subheader("📏 缩放")
            scaling_method = st.selectbox(
                "选择算法",
                ["无", "Peak-Norm", "SNV", "MSC", "M-M-Norm", "L-范数", "Ma-Minorm", "标准化(均值0，方差1)"],
                key="scaling_method",
                label_visibility="collapsed"
            )
            scaling_params = {}
            if scaling_method == "L-范数":
                p = st.selectbox("范数阶数", ["无穷大", "4", "10"], key="p_scale", label_visibility="collapsed")
                scaling_params["p"] = p
                st.caption(f"p: {p}")

        # 3. 滤波算法选择
        with preprocess_cols[2]:
            st.subheader("📶 滤波")
            filtering_method = st.selectbox(
                "选择算法",
                ["无", "Savitzky-Golay", "中值滤波(MF)", "移动平均(MAF)", "小波线性阈值去噪"],
                key="filtering_method",
                label_visibility="collapsed"
            )
            filtering_params = {}
            if filtering_method in ["Savitzky-Golay"]:
                window = st.selectbox("窗口大小", [11, 31, 51], key="sg_window", label_visibility="collapsed")
                order = st.selectbox("多项式阶数", [3, 7], key="sg_order", label_visibility="collapsed")
                filtering_params = {"window_length": window, "polyorder": order}
                st.caption(f"窗口: {window}, 阶数: {order}")

        # 4. 挤压算法选择
        with preprocess_cols[3]:
            st.subheader("🧪 挤压")
            squashing_method = st.selectbox(
                "选择算法",
                ["无", "Sigmoid挤压", "改进的Sigmoid挤压", "余弦挤压(squashing)", "DTW挤压"],
                key="squashing_method",
                label_visibility="collapsed"
            )
            squashing_params = {}
            if squashing_method == "DTW挤压":
                l = st.selectbox("窗口参数", [1, 5], key="dtw_l", label_visibility="collapsed")
                squashing_params = {"l": l, "k1": "T", "k2": "T"}
                st.caption(f"窗口: {l}")

        # 5. 应用当前预处理设置
        with preprocess_cols[4]:
            st.subheader("操作1")
            if st.button("🚀 应用处理", type="primary", use_container_width=True, key="apply_btn"):
                if not st.session_state.raw_data:
                    st.warning("⚠️ 请先上传数据")
                else:
                    try:
                        wavenumbers, spectra = st.session_state.raw_data
                        # 执行预处理
                        processed_data, method_names = preprocessor.process(
                            wavenumbers, spectra,
                            baseline_method=baseline_method,
                            baseline_params=baseline_params,
                            scaling_method=scaling_method,
                            scaling_params=scaling_params,
                            filtering_method=filtering_method,
                            filtering_params=filtering_params,
                            squashing_method=squashing_method,
                            squashing_params=squashing_params
                        )
                        # 保存结果到会话状态
                        arr_name = f"排列_{len(st.session_state.arrangement_results) + 1}"
                        st.session_state.arrangement_results.append(arr_name)
                        st.session_state.arrangement_details[arr_name] = {
                            "data": processed_data,
                            "method": " → ".join(method_names),
                            "params": st.session_state.current_algorithms
                        }
                        st.session_state.selected_arrangement = arr_name
                        st.success(f"✅ 预处理完成：{arr_name}")
                    except Exception as e:
                        st.error(f"❌ 处理失败：{str(e)}")

        # 6. 生成算法排列组合
        with preprocess_cols[5]:
            st.subheader("操作2")
            if st.button("🔍 显示排列", type="secondary", use_container_width=True, key="show_perm_btn"):
                st.session_state.show_arrangements = not st.session_state.show_arrangements
                if st.session_state.show_arrangements:
                    # 生成所有可能的算法排列
                    selected_algos = {
                        "baseline": baseline_method,
                        "scaling": scaling_method,
                        "filtering": filtering_method,
                        "squashing": squashing_method
                    }
                    st.session_state.algorithm_permutations = generate_permutations(selected_algos)
                    st.session_state.filtered_perms = st.session_state.algorithm_permutations
                    st.success(f"✅ 生成{len(st.session_state.algorithm_permutations)}种排列方案")
                else:
                    st.session_state.filtered_perms = []
                st.experimental_rerun()

            # 筛选排列方案（按第一步类型）
            if st.session_state.show_arrangements and st.session_state.algorithm_permutations:
                first_steps = list({p["first_step_type"] for p in st.session_state.algorithm_permutations})
                selected_step = st.selectbox(
                    "筛选第一步类型",
                    ["全部"] + first_steps,
                    key="step_filter",
                    label_visibility="collapsed"
                )
                if selected_step != "全部":
                    st.session_state.filtered_perms = [
                        p for p in st.session_state.algorithm_permutations
                        if p["first_step_type"] == selected_step
                    ]

        # 7. 选择并应用排列方案
        with preprocess_cols[6]:
            st.subheader("操作3")
            if st.session_state.show_arrangements and st.session_state.filtered_perms:
                # 选择排列方案
                selected_idx = st.selectbox(
                    f"共{len(st.session_state.filtered_perms)}种方案",
                    range(len(st.session_state.filtered_perms)),
                    format_func=lambda x: st.session_state.filtered_perms[x]["name"],
                    key="perm_select",
                    label_visibility="collapsed"
                )
                selected_perm = st.session_state.filtered_perms[selected_idx]
                st.caption(f"当前方案：{selected_perm['name']}")

                # 应用选中的排列方案
                if st.button("✅ 应用方案", type="primary", use_container_width=True, key="apply_perm_btn"):
                    if not st.session_state.raw_data:
                        st.warning("⚠️ 请先上传数据")
                    else:
                        try:
                            wavenumbers, spectra = st.session_state.raw_data
                            # 执行排列好的预处理步骤
                            processed_data, method_names = preprocessor.process(
                                wavenumbers, spectra,
                                baseline_method=st.session_state.current_algorithms["baseline"],
                                baseline_params=st.session_state.current_algorithms["baseline_params"],
                                scaling_method=st.session_state.current_algorithms["scaling"],
                                scaling_params=st.session_state.current_algorithms["scaling_params"],
                                filtering_method=st.session_state.current_algorithms["filtering"],
                                filtering_params=st.session_state.current_algorithms["filtering_params"],
                                squashing_method=st.session_state.current_algorithms["squashing"],
                                squashing_params=st.session_state.current_algorithms["squashing_params"],
                                algorithm_order=selected_perm["order"]  # 排列顺序
                            )
                            # 保存结果
                            arr_name = f"排列_{len(st.session_state.arrangement_results) + 1}"
                            st.session_state.arrangement_results.append(arr_name)
                            st.session_state.arrangement_details[arr_name] = {
                                "data": processed_data,
                                "method": " → ".join(method_names),
                                "order": selected_perm["order"]
                            }
                            st.session_state.selected_arrangement = arr_name
                            st.success(f"✅ 方案应用完成：{arr_name}")
                        except Exception as e:
                            st.error(f"❌ 应用失败：{str(e)}")

        # 8. 设置KNN参数k
        with preprocess_cols[7]:
            st.subheader("操作4")
            k_value = st.number_input(
                "KNN的k值",
                min_value=1,
                value=st.session_state.k_value,
                step=1,
                key="k_input",
                label_visibility="collapsed"
            )
            if st.button("确定k值", use_container_width=True, key="k_confirm_btn"):
                st.session_state.k_value = k_value
                st.success(f"✅ k值已设置为：{k_value}")

        # 9. 执行分类测试
        with preprocess_cols[8]:
            st.subheader("操作5")
            if st.button("▶️ 执行测试", type="primary", use_container_width=True, key="test_btn"):
                if not st.session_state.raw_data:
                    st.warning("⚠️ 请先上传数据")
                elif not st.session_state.selected_arrangement:
                    st.warning("⚠️ 请先应用预处理方案")
                elif st.session_state.labels is None:
                    st.warning("⚠️ 请先输入样本标签")
                else:
                    try:
                        # 获取预处理后的数据和标签
                        arr_data = st.session_state.arrangement_details[st.session_state.selected_arrangement]["data"]
                        train_idx = st.session_state.train_indices
                        test_idx = st.session_state.test_indices
                        train_data = arr_data[:, train_idx]
                        test_data = arr_data[:, test_idx]
                        train_labels = st.session_state.labels[train_idx]
                        test_labels = st.session_state.labels[test_idx]

                        # KNN分类
                        with st.spinner("测试中..."):
                            predictions = knn_classify(train_data, train_labels, test_data, k=st.session_state.k_value)
                        # 评估结果
                        results = evaluate_classification(test_labels, predictions)
                        results["predictions"] = predictions
                        results["test_labels"] = test_labels
                        st.session_state.test_results = results
                        st.success("✅ 测试完成！结果如下：")
                    except Exception as e:
                        st.error(f"❌ 测试失败：{str(e)}")

        # 保存当前算法参数到会话状态
        st.session_state.current_algorithms.update({
            "baseline": baseline_method,
            "baseline_params": baseline_params,
            "scaling": scaling_method,
            "scaling_params": scaling_params,
            "filtering": filtering_method,
            "filtering_params": filtering_params,
            "squashing": squashing_method,
            "squashing_params": squashing_params
        })

        # 光谱可视化区域

        st.subheader("📊 结果可视化", divider="gray")

        # 创建2×2网格布局并固定高度
        vis_cols = st.columns(2)
        
        # 左侧第一行：原始光谱
        with vis_cols[0]:
            with st.container(border=True, height=350):  # 固定高度
                st.subheader("原始光谱")
                if ("raw_data" in st.session_state and 
                    st.session_state.raw_data is not None and 
                    len(st.session_state.raw_data) == 2):
                    
                    wavenumbers, spectra = st.session_state.raw_data
                    if hasattr(spectra, 'shape') and len(spectra.shape) >= 2:
                        total_samples = spectra.shape[1]
                        
                        # 添加样本选择器（默认显示第一条）
                        sample_idx = st.select_slider(
                            "选择样本",
                            options=range(1, total_samples + 1),
                            value=1,
                            key="raw_spectrum_selector"
                        )
                        
                        # 只显示选中的单条光谱
                        fig, ax = plt.subplots(figsize=(5, 3))
                        ax.plot(wavenumbers, spectra[:, sample_idx - 1], 
                                color='#1f77b4', linewidth=1.5)
                        ax.set_xlabel("波数")
                        ax.set_ylabel("强度")
                        ax.set_title(f"样本 {sample_idx}/{total_samples}")
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("原始光谱数据格式不正确")
                else:
                    st.info("👈 请上传数据以显示原始光谱")
        
            # 左侧第二行：k值曲线
            with st.container(border=True, height=350):  # 固定高度
                st.subheader("k值曲线")
                if ("test_results" in st.session_state and 
                    st.session_state.test_results is not None and 
                    "k_accuracies" in st.session_state.test_results):
                    
                    k_values = st.session_state.test_results.get('k_values', list(range(1, 11)))
                    accuracies = st.session_state.test_results['k_accuracies']
                    
                    fig, ax = plt.subplots(figsize=(5, 3))
                    ax.plot(k_values, accuracies, 'o-', color='#ff7f0e')
                    ax.set_xlabel("k值")
                    ax.set_ylabel("准确率")
                    ax.set_ylim(0, 1.05)
                    best_k = k_values[np.argmax(accuracies)]
                    ax.set_title(f"最佳k值：{best_k}")
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("📊 运行测试后显示k值曲线")
        
        # 右侧第一行：预处理后光谱
        with vis_cols[1]:
            with st.container(border=True, height=350):  # 固定高度
                st.subheader("预处理后光谱")
                if ("processed_data" in st.session_state and 
                    st.session_state.processed_data is not None and 
                    "raw_data" in st.session_state and 
                    st.session_state.raw_data is not None):
                    
                    wavenumbers, _ = st.session_state.raw_data
                    processed_spectra = st.session_state.processed_data
                    
                    if hasattr(processed_spectra, 'shape') and len(processed_spectra.shape) >= 2:
                        total_samples = processed_spectra.shape[1]
                        
                        # 样本选择器（与原始光谱保持同步）
                        sample_idx = st.select_slider(
                            "选择样本",
                            options=range(1, total_samples + 1),
                            value=1,
                            key="processed_spectrum_selector"
                        )
                        
                        # 只显示选中的单条光谱
                        fig, ax = plt.subplots(figsize=(5, 3))
                        ax.plot(wavenumbers, processed_spectra[:, sample_idx - 1], 
                                color='#2ca02c', linewidth=1.5)
                        ax.set_xlabel("波数")
                        ax.set_ylabel("预处理后强度")
                        ax.set_title(f"样本 {sample_idx}/{total_samples}")
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("预处理光谱数据格式不正确")
                else:
                    st.info("🚀 应用预处理后显示结果光谱")
        
            # 右侧第二行：混淆矩阵
            with st.container(border=True, height=350):  # 固定高度
                st.subheader("混淆矩阵")
                if ("test_results" in st.session_state and 
                    st.session_state.test_results is not None and 
                    'confusion_matrix' in st.session_state.test_results):
                    
                    cm = st.session_state.test_results['confusion_matrix']
                    fig, ax = plt.subplots(figsize=(5, 3))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                ax=ax, cbar=False, annot_kws={"size": 9})
                    ax.set_xlabel("预测标签")
                    ax.set_ylabel("真实标签")
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("✅ 运行测试后显示混淆矩阵")

        # 结果导出
        if st.session_state.arrangement_results:
            st.subheader("💾 结果导出", divider="gray")
            export_name = st.text_input("导出文件名", "preprocessed_spectra.txt", key="export_name")
            if st.button("导出预处理数据", key="export_btn"):
                try:
                    arr_data = st.session_state.arrangement_details[st.session_state.selected_arrangement]["data"]
                    file_handler.export_data(export_name, arr_data)
                    st.success(f"✅ 数据已导出至：{export_name}")
                except Exception as e:
                    st.error(f"❌ 导出失败：{str(e)}")
