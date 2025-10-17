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

def main():
    # 初始化状态
    init_state()
    file_handler = FileHandler()
    preprocessor = Preprocessor()

    # 页面样式统一
    st.markdown("""
        <style>
        .stButton > button {
            border-radius: 6px;
            background-color: #165DFF;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #0E42D2;
        }
        .stSelectbox, .stSlider, .stTextInput {
            margin-bottom: 0.8rem;
        }
        .card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            padding: 1rem;
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🌌 排列预处理模型")

    # 左右分栏布局
    col_left, col_right = st.columns([1.2, 3.9])

    # 左侧：数据管理与配置
    with col_left:
        with st.expander("📁 数据管理", expanded=True):
            # 上传文件
            zip_file = st.file_uploader(
                "上传ZIP压缩包（含波数文件+光谱数据文件）",
                type=['zip'],
                key="arr_zip",
                help="压缩包内需包含：波数文件（含'wave/wn/波数'）、光谱文件（含'spec/data/光谱'）"
            )

            # 样本标签
            st.subheader("样本标签")
            num_classes = st.number_input(
                "类别数量",
                min_value=1,
                value=2,
                step=1,
                key="arr_num_cls"
            )
            labels_input = st.text_input(
                "标签（逗号分隔，如0,0,1,1）",
                placeholder="示例：0,0,1,1,2,2",
                key="arr_labels"
            )

            # 训练测试划分
            st.subheader("训练测试划分")
            train_ratio = st.slider(
                "训练集占比",
                0.1, 0.9, 0.8, 0.1,
                format="%.1f",
                key="arr_train_ratio"
            )

            # 数据加载逻辑
            if zip_file:
                try:
                    wavenumbers, spectra = file_handler.load_data_from_zip(zip_file)
                    st.session_state.arr_raw_data = (wavenumbers, spectra)
                    st.success(f"✅ 数据加载成功：{spectra.shape[1]}条光谱 × {spectra.shape[0]}个波数点")

                    if labels_input:
                        try:
                            labels = np.array([int(l.strip()) for l in labels_input.split(',')])
                            if len(labels) == spectra.shape[1]:
                                st.session_state.arr_labels = labels
                                n = len(labels)
                                train_size = int(n * train_ratio)
                                indices = np.random.permutation(n)
                                st.session_state.arr_train_idx = indices[:train_size]
                                st.session_state.arr_test_idx = indices[train_size:]
                                class_counts = np.bincount(labels)
                                st.info(f"🏷️ 标签分布：{', '.join([f'类{i}:{c}个' for i, c in enumerate(class_counts) if c>0])}")
                            else:
                                st.warning(f"⚠️ 标签数（{len(labels)}）与光谱数（{spectra.shape[1]}）不匹配")
                        except Exception as e:
                            st.warning(f"⚠️ 标签格式错误：{str(e)}")
                except Exception as e:
                    st.error(f"❌ 数据加载失败：{str(e)}")

        # 使用指南
        with st.expander("ℹ️ 使用指南", expanded=False):
            st.markdown("""
            1. 上传ZIP压缩包  
            2. 输入样本标签（可选）并设置训练/测试比例  
            3. 选择预处理算法及参数  
            4. 点击「显示排列」生成所有算法执行顺序  
            5. 选择方案并点击「应用」  
            6. 设置K值后点击「执行测试」查看分类效果
            """)

        # 返回首页按钮
        if st.button("返回首页 🔙", use_container_width=True):
            st.session_state.current_page = "home"
            st.experimental_rerun()

    # 右侧：预处理功能与结果展示
    with col_right:
        st.subheader("⚙️ 预处理算法设置", divider="gray")
        algo_cols = st.columns(4, gap="small")

        # 1. 基线校正
        with algo_cols[0]:
            st.subheader("基线校正")
            baseline_method = st.selectbox(
                "选择算法",
                ["无", "SD", "FD", "多项式拟合", "airPLS"],
                key="arr_baseline"
            )
            baseline_params = {}
            if baseline_method == "多项式拟合":
                baseline_params["polyorder"] = st.slider(
                    "阶数", 3, 6, 5, key="arr_baseline_order"
                )

        # 2. 滤波去噪
        with algo_cols[1]:
            st.subheader("滤波去噪")
            filter_method = st.selectbox(
                "选择算法",
                ["无", "Savitzky-Golay", "中值滤波"],
                key="arr_filter"
            )
            filter_params = {}
            if filter_method == "Savitzky-Golay":
                filter_params["window_length"] = st.slider(
                    "窗口大小", 5, 31, 11, key="arr_filter_window"
                )
                filter_params["polyorder"] = st.slider(
                    "多项式阶数", 1, 5, 3, key="arr_filter_order"
                )

        # 3. 缩放归一化
        with algo_cols[2]:
            st.subheader("缩放归一化")
            scale_method = st.selectbox(
                "选择算法",
                ["无", "SNV", "Peak-Norm", "标准化"],
                key="arr_scale"
            )

        # 4. 挤压变换
        with algo_cols[3]:
            st.subheader("挤压变换")
            squash_method = st.selectbox(
                "选择算法",
                ["无", "Sigmoid", "余弦挤压"],
                key="arr_squash"
            )

        # 操作按钮区
        st.subheader("🔄 排列与测试", divider="gray")
        op_cols = st.columns(5, gap="small")

        # 生成排列
        with op_cols[0]:
            if st.button("📊 显示排列", key="arr_gen_perm"):
                if not st.session_state.get("arr_raw_data"):
                    st.warning("⚠️ 请先上传数据")
                else:
                    selected_algos = {
                        "baseline": baseline_method,
                        "filtering": filter_method,
                        "scaling": scale_method,
                        "squashing": squash_method
                    }
                    st.session_state.arr_permutations = generate_permutations(selected_algos)
                    st.success(f"✅ 生成{len(st.session_state.arr_permutations)}种方案")

        # 显示排列选择框
        if st.session_state.get("arr_permutations"):
            perm_names = [p["name"] for p in st.session_state.arr_permutations]
            selected_perm_idx = st.selectbox(
                "选择预处理方案",
                range(len(perm_names)),
                format_func=lambda x: perm_names[x],
                key="arr_selected_perm"
            )
            selected_perm = st.session_state.arr_permutations[selected_perm_idx]

            # 应用方案
            with op_cols[1]:
                if st.button("🚀 应用方案", key="arr_apply_perm"):
                    try:
                        wavenumbers, spectra = st.session_state.arr_raw_data
                        processed, method_log = preprocessor.process(
                            wavenumbers, spectra,
                            baseline_method=baseline_method,
                            baseline_params=baseline_params,
                            filtering_method=filter_method,
                            filtering_params=filter_params,
                            scaling_method=scale_method,
                            squashing_method=squash_method,
                            algorithm_order=selected_perm["order"]
                        )
                        st.session_state.arr_processed = processed
                        st.session_state.arr_method_log = method_log
                        st.success(f"✅ 预处理完成：{', '.join(method_log)}")
                    except Exception as e:
                        st.error(f"❌ 应用失败：{str(e)}")

        # 确定K值
        with op_cols[2]:
            k_value = st.number_input(
                "KNN的k值",
                min_value=1,
                value=5,
                step=1,
                key="arr_k"
            )

        # 执行测试
        with op_cols[3]:
            if st.button("▶️ 执行测试", key="arr_test"):
                try:
                    processed = st.session_state.arr_processed
                    labels = st.session_state.arr_labels
                    train_idx = st.session_state.arr_train_idx
                    test_idx = st.session_state.arr_test_idx

                    train_data = processed[:, train_idx]
                    test_data = processed[:, test_idx]
                    train_labels = labels[train_idx]
                    test_labels = labels[test_idx]

                    with st.spinner("测试中..."):
                        preds = knn_classify(train_data, train_labels, test_data, k=k_value)
                    results = evaluate_classification(test_labels, preds)
                    st.session_state.arr_test_results = results
                    st.success("✅ 测试完成！")
                except Exception as e:
                    st.error(f"❌ 测试失败：{str(e)}")

        # 光谱可视化
        st.subheader("📈 光谱可视化", divider="gray")
        vis_cols = st.columns(2)

        with vis_cols[0]:
            st.subheader("原始光谱")
            if st.session_state.get("arr_raw_data"):
                wavenumbers, spectra = st.session_state.arr_raw_data
                df = pd.DataFrame({"原始强度": spectra[:, 0]}, index=wavenumbers)
                st.line_chart(df, height=250)
                with st.expander("查看更多原始光谱", expanded=False):
                    for i in range(1, min(5, spectra.shape[1])):
                        df = pd.DataFrame({f"样本{i+1}": spectra[:, i]}, index=wavenumbers)
                        st.line_chart(df, height=150)
            else:
                st.info("⏳ 请上传数据以显示原始光谱")

        with vis_cols[1]:
            st.subheader("预处理后光谱")
            if st.session_state.get("arr_processed"):
                processed = st.session_state.arr_processed
                wavenumbers, _ = st.session_state.arr_raw_data
                df = pd.DataFrame({"处理后强度": processed[:, 0]}, index=wavenumbers)
                st.line_chart(df, height=250)
                with st.expander("查看更多预处理光谱", expanded=False):
                    for i in range(1, min(5, processed.shape[1])):
                        df = pd.DataFrame({f"样本{i+1}": processed[:, i]}, index=wavenumbers)
                        st.line_chart(df, height=150)
                st.caption(f"处理流程：{', '.join(st.session_state.arr_method_log)}")
            else:
                st.info("⏳ 请应用预处理方案以显示结果")

        # 分类结果展示
        if st.session_state.get("arr_test_results"):
            st.subheader("📊 分类测试结果", divider="gray")
            results = st.session_state.arr_test_results
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric("准确率", f"{results['accuracy']:.4f}")
            with metric_cols[1]:
                st.metric("卡帕系数", f"{results['kappa']:.4f}")

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(results["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("预测标签")
            ax.set_ylabel("真实标签")
            ax.set_title("混淆矩阵")
            st.pyplot(fig)

        # 结果导出
        if st.session_state.get("arr_processed"):
            st.subheader("💾 结果导出", divider="gray")
            export_name = st.text_input("导出文件名", "preprocessed_spectra.txt", key="arr_export_name")
            if st.button("导出预处理数据", key="arr_export"):
                try:
                    file_handler.export_data(export_name, st.session_state.arr_processed)
                    st.success(f"✅ 数据已导出至：{export_name}")
                except Exception as e:
                    st.error(f"❌ 导出失败：{str(e)}")
