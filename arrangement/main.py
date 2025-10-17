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

    # 页面样式：移除边距，实现铺满效果
    st.markdown("""
        <style>
        .reportview-container .main .block-container {
            padding-top: 0rem;
            padding-right: 0rem;
            padding-left: 0rem;
            padding-bottom: 0rem;
        }
        .stButton > button {
            border-radius: 6px;
            background-color: #165DFF;
            color: white;
            border: none;
            transition: all 0.3s ease;
            padding: 0.5rem 1rem;
        }
        .stButton > button:hover {
            background-color: #0E42D2;
        }
        .card {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .section-header {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🌌 排列预处理模型")

    # 整体布局：采用三栏结构，左侧数据管理，右侧功能区
    col_left, col_right = st.columns([1, 3], gap="medium")

    # 左侧：数据管理与配置
    with col_left:
        with st.container():
            st.markdown('<div class="card"><h3 class="section-header">📁 数据管理</h3>', unsafe_allow_html=True)
            # 上传文件
            zip_file = st.file_uploader(
                "上传ZIP压缩包（含波数文件+光谱数据文件）",
                type=['zip'],
                key="arr_zip",
                help="压缩包内需包含：波数文件（含'wave/wn/波数'）、光谱文件（含'spec/data/光谱'）"
            )
            st.caption("支持格式：ZIP（内含波数文件+光谱数据文件）")

            # 样本标签
            st.markdown('<h4 class="section-header">样本标签</h4>', unsafe_allow_html=True)
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
            st.markdown('<h4 class="section-header">训练测试划分</h4>', unsafe_allow_html=True)
            train_ratio = st.slider(
                "训练集占比",
                0.1, 0.9, 0.8, 0.1,
                format="%.1f",
                key="arr_train_ratio"
            )
            st.session_state.train_test_split_ratio = train_ratio

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
            st.markdown('</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="card"><h3 class="section-header">ℹ️ 系统信息</h3>', unsafe_allow_html=True)
            if st.session_state.get('arr_raw_data'):
                wavenumbers, spectra = st.session_state.arr_raw_data
                st.info(f"📊 数据维度：{spectra.shape[1]}条光谱 × {spectra.shape[0]}个波数点")
                st.info(f"🔄 划分比例：训练集{train_ratio:.1f} | 测试集{1-train_ratio:.1f}")
                if st.session_state.get('arr_labels') is not None:
                    class_counts = np.bincount(st.session_state.arr_labels)
                    st.info(f"🏷️ 类别分布：{', '.join([f'类{i}:{count}个' for i, count in enumerate(class_counts) if count > 0])}")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="card"><h3 class="section-header">ℹ️ 使用指南</h3>', unsafe_allow_html=True)
            st.markdown("""
            1. 上传ZIP压缩包  
            2. 输入样本标签（可选）并设置训练/测试比例  
            3. 选择预处理算法及参数  
            4. 点击「显示排列」生成所有算法执行顺序  
            5. 选择方案并点击「应用」  
            6. 设置K值后点击「执行测试」查看分类效果
            """)
            if st.button("返回首页 🔙", use_container_width=True):
                st.session_state.current_page = "home"
                st.experimental_rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # 右侧：预处理功能与结果展示
    with col_right:
        st.markdown('<div class="card"><h3 class="section-header">⚙️ 预处理算法设置</h3>', unsafe_allow_html=True)
        algo_cols = st.columns(4, gap="small")

        # 1. 基线校正
        with algo_cols[0]:
            st.markdown('<h4 class="section-header">基线校正</h4>', unsafe_allow_html=True)
            baseline_method = st.selectbox(
                "选择算法",
                ["无", "SD", "FD", "多项式拟合", "ModPoly", "I-ModPoly", "PLS", "AsLS", "airPLS", "二阶差分(D2)"],
                key="baseline_method"
            )
            baseline_params = {}
            if baseline_method == "多项式拟合":
                baseline_params["polyorder"] = st.slider(
                    "阶数", 3, 6, 5, key="baseline_polyorder"
                )
                st.caption(f"阶数: {baseline_params['polyorder']}")
            elif baseline_method == "ModPoly":
                k = st.slider("迭代参数", 4, 10, 10, key="modpoly_k")
                baseline_params["k"] = k
                st.caption(f"k: {k}")
            elif baseline_method == "I-ModPoly":
                polyorder = st.slider("阶数", 3, 7, 5, key="imodpoly_polyorder")
                max_iter = st.slider("迭代次数", 50, 200, 100, key="imodpoly_maxiter")
                baseline_params = {"polyorder": polyorder, "max_iter": max_iter}
                st.caption(f"阶数: {polyorder}, 迭代: {max_iter}")

        # 2. 缩放
        with algo_cols[1]:
            st.markdown('<h4 class="section-header">📏 缩放</h4>', unsafe_allow_html=True)
            scaling_method = st.selectbox(
                "选择算法",
                ["无", "Peak-Norm", "SNV", "MSC", "M-M-Norm", "L-范数", "Ma-Minorm", "标准化(均值0，方差1)"],
                key="scaling_method"
            )
            scaling_params = {}
            if scaling_method == "L-范数":
                p = st.selectbox("范数阶数", ["无穷大", "4", "10"], key="l_norm_p")
                scaling_params["p"] = p
                st.caption(f"p: {p}")

        # 3. 滤波
        with algo_cols[2]:
            st.markdown('<h4 class="section-header">📶 滤波</h4>', unsafe_allow_html=True)
            filtering_method = st.selectbox(
                "选择算法",
                ["无", "Savitzky-Golay", "中值滤波(MF)", "移动平均(MAF)", "小波线性阈值去噪"],
                key="filtering_method"
            )
            filtering_params = {}
            if filtering_method in ["Savitzky-Golay"]:
                window = st.selectbox("窗口大小", [11, 31, 51], key="sg_window")
                order = st.selectbox("多项式阶数", [3, 7], key="sg_order")
                filtering_params = {"window_length": window, "polyorder": order}
                st.caption(f"窗口: {window}, 阶数: {order}")

        # 4. 挤压
        with algo_cols[3]:
            st.markdown('<h4 class="section-header">🧪 挤压</h4>', unsafe_allow_html=True)
            squashing_method = st.selectbox(
                "选择算法",
                ["无", "Sigmoid挤压", "改进的Sigmoid挤压", "余弦挤压(squashing)", "DTW挤压"],
                key="squashing_method"
            )
            squashing_params = {}
            if squashing_method == "DTW挤压":
                l = st.selectbox("窗口参数", [1, 5], key="dtw_l")
                squashing_params = {"l": l, "k1": "T", "k2": "T"}
                st.caption(f"窗口: {l}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><h3 class="section-header">🔄 排列与测试</h3>', unsafe_allow_html=True)
        op_cols = st.columns(5, gap="small")

        # 生成排列
        with op_cols[0]:
            if st.button("📊 显示排列", key="gen_perm_btn"):
                if not st.session_state.get("arr_raw_data"):
                    st.warning("⚠️ 请先上传数据")
                else:
                    selected_algos = {
                        "baseline": baseline_method,
                        "scaling": scaling_method,
                        "filtering": filtering_method,
                        "squashing": squashing_method
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
                key="perm_select"
            )
            selected_perm = st.session_state.arr_permutations[selected_perm_idx]

            # 应用方案
            with op_cols[1]:
                if st.button("🚀 应用方案", key="apply_perm_btn"):
                    try:
                        wavenumbers, spectra = st.session_state.arr_raw_data
                        processed, method_log = preprocessor.process(
                            wavenumbers, spectra,
                            baseline_method=baseline_method,
                            baseline_params=baseline_params,
                            scaling_method=scaling_method,
                            scaling_params=scaling_params,
                            filtering_method=filtering_method,
                            filtering_params=filtering_params,
                            squashing_method=squashing_method,
                            squashing_params=squashing_params,
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
                key="knn_k"
            )

        # 执行测试
        with op_cols[3]:
            if st.button("▶️ 执行测试", key="run_test_btn"):
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

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><h3 class="section-header">📈 光谱可视化</h3>', unsafe_allow_html=True)
        vis_cols = st.columns(2)

        with vis_cols[0]:
            st.markdown('<h4 class="section-header">原始光谱</h4>', unsafe_allow_html=True)
            if st.session_state.get("arr_raw_data"):
                wavenumbers, spectra = st.session_state.arr_raw_data
                df = pd.DataFrame({"原始强度": spectra[:, 0]}, index=wavenumbers)
                st.line_chart(df, height=300)
                with st.expander("查看更多原始光谱", expanded=False):
                    for i in range(1, min(5, spectra.shape[1])):
                        df = pd.DataFrame({f"样本{i+1}": spectra[:, i]}, index=wavenumbers)
                        st.line_chart(df, height=200)
            else:
                st.info("⏳ 请上传数据以显示原始光谱")

        with vis_cols[1]:
            st.markdown('<h4 class="section-header">预处理后光谱</h4>', unsafe_allow_html=True)
            if st.session_state.get("arr_processed"):
                processed = st.session_state.arr_processed
                wavenumbers, _ = st.session_state.arr_raw_data
                df = pd.DataFrame({"处理后强度": processed[:, 0]}, index=wavenumbers)
                st.line_chart(df, height=300)
                with st.expander("查看更多预处理光谱", expanded=False):
                    for i in range(1, min(5, processed.shape[1])):
                        df = pd.DataFrame({f"样本{i+1}": processed[:, i]}, index=wavenumbers)
                        st.line_chart(df, height=200)
                st.caption(f"处理流程：{', '.join(st.session_state.arr_method_log)}")
            else:
                st.info("⏳ 请应用预处理方案以显示结果")
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get("arr_test_results"):
            st.markdown('<div class="card"><h3 class="section-header">📊 分类测试结果</h3>', unsafe_allow_html=True)
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
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.get("arr_processed"):
            st.markdown('<div class="card"><h3 class="section-header">💾 结果导出</h3>', unsafe_allow_html=True)
            export_name = st.text_input("导出文件名", "preprocessed_spectra.txt", key="export_name")
            if st.button("导出预处理数据", key="export_data_btn"):
                try:
                    file_handler.export_data(export_name, st.session_state.arr_processed)
                    st.success(f"✅ 数据已导出至：{export_name}")
                except Exception as e:
                    st.error(f"❌ 导出失败：{str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
