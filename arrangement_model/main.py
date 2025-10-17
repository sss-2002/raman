import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
import itertools
# 从主页面导入跳转函数
from ..main import navigate_to

# 导入本地模块
from .algorithms.preprocessing import Preprocessor
from .utils.file_handler import FileHandler
from .algorithms.classification import knn_classify

def generate_permutations(algorithms):
    """生成算法排列组合"""
    algorithm_list = [
        (1, "基线校准", algorithms['baseline']),
        (2, "缩放", algorithms['scaling']),
        (3, "滤波", algorithms['filtering']),
        (4, "挤压", algorithms['squashing'])
    ]

    all_permutations = []
    all_permutations.append([])  # 无预处理

    # 生成1-4种算法的排列
    for length in range(1, 5):
        for perm in itertools.permutations(algorithm_list, length):
            if all(algo[2] != "无" for algo in perm):
                all_permutations.append(list(perm))

    # 格式化排列结果
    formatted_perms = []
    for perm in all_permutations:
        perm_dict = {
            "name": "",
            "order": [],
            "details": perm,
            "count": len(perm),
            "first_step_type": "未知"
        }

        if not perm:
            perm_dict["name"] = "无预处理（原始光谱）"
            perm_dict["first_step_type"] = "无预处理"
        else:
            perm_dict["first_step_type"] = perm[0][1]
            perm_details = [f"{step[0]}.{step[1]}({step[2]})" for step in perm]
            perm_dict["name"] = " → ".join(perm_details)
            perm_dict["order"] = [step[0] for step in perm]

        formatted_perms.append(perm_dict)

    return formatted_perms


def main():
    # 初始化会话状态（仅在未定义时）
    if 'show_arrangements' not in st.session_state:
        st.session_state.show_arrangements = False
    if 'k_value' not in st.session_state:
        st.session_state.k_value = 5
    if 'test_results' not in st.session_state:
        st.session_state.test_results = None
    if 'train_test_split_ratio' not in st.session_state:
        st.session_state.train_test_split_ratio = 0.8
    if 'arrangement_results' not in st.session_state:
        st.session_state.arrangement_results = []
    if 'selected_arrangement' not in st.session_state:
        st.session_state.selected_arrangement = None
    if 'arrangement_details' not in st.session_state:
        st.session_state.arrangement_details = {}
    if 'algorithm_permutations' not in st.session_state:
        st.session_state.algorithm_permutations = []
    if 'filtered_perms' not in st.session_state:
        st.session_state.filtered_perms = []
    if 'selected_perm_idx' not in st.session_state:
        st.session_state.selected_perm_idx = 0

    # 初始化算法配置
    current_algorithms = {
        'baseline': '无',
        'baseline_params': {},
        'scaling': '无',
        'scaling_params': {},
        'filtering': '无',
        'filtering_params': {},
        'squashing': '无',
        'squashing_params': {}
    }
    st.session_state['current_algorithms'] = current_algorithms

    # 页面样式
    st.markdown("""
        <style>
        body {font-size: 0.75rem !important;}
        .css-1v0mbdj {padding: 0.3rem 0.5rem !important;}
        h3 {font-size: 1rem !important; margin: 0.3rem 0 !important;}
        </style>
    """, unsafe_allow_html=True)

    st.title("🌌 排列预处理模型")

    # 布局：左侧数据管理，右侧功能区
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
                key="train_ratio"
            )
            st.session_state.train_test_split_ratio = train_test_ratio

            # 加载数据（使用全局状态）
            if zip_file:
                try:
                    file_handler = FileHandler()
                    wavenumbers, y = file_handler.load_data_from_zip(zip_file)
                    st.session_state.raw_spectra = (wavenumbers, y)  # 存入全局状态

                    # 处理标签
                    if labels_input:
                        try:
                            labels = np.array([int(l.strip()) for l in labels_input.split(',')])
                            if len(labels) == y.shape[1]:
                                st.session_state.labels = labels
                                n_samples = len(labels)
                                train_size = int(n_samples * train_test_ratio)
                                indices = np.random.permutation(n_samples)
                                st.session_state.train_indices = indices[:train_size]
                                st.session_state.test_indices = indices[train_size:]
                                st.success(f"✅ 数据加载成功：{y.shape[1]}条光谱，{len(np.unique(labels))}类")
                            else:
                                st.warning(f"⚠️ 标签数({len(labels)})≠光谱数({y.shape[1]})")
                        except Exception as e:
                            st.warning(f"⚠️ 标签格式错误: {str(e)}")
                    else:
                        st.success(f"✅ 数据加载成功：{y.shape[1]}条光谱")
                except Exception as e:
                    st.error(f"❌ 文件加载失败: {str(e)}")

        # 系统信息
        if st.session_state.raw_spectra:
            wavenumbers, y = st.session_state.raw_spectra
            st.info(f"📊 数据维度: {y.shape[1]}条 × {y.shape[0]}点")
            st.info(f"🔢 训练集:{train_test_ratio:.1f} | 测试集:{1-train_test_ratio:.1f}")
            if st.session_state.labels is not None:
                class_counts = np.bincount(st.session_state.labels)
                st.info(f"🏷️ 类别分布: {', '.join([f'类{i}:{count}' for i, count in enumerate(class_counts) if count>0])}")

        # 使用指南
        with st.expander("ℹ️ 使用指南", expanded=False):
            st.markdown("""
            1. 上传光谱数据压缩包  
            2. 设置标签和训练测试比例  
            3. 选择预处理方法并生成排列  
            4. 应用方案并测试  
            5. 查看结果并导出
            """)

        # 返回主页按钮（关键）
        if st.button("返回主页", key="back_to_home_btn", use_container_width=True):
            navigate_to("home")

    # 右侧：预处理设置和可视化
    with col_right:
        st.subheader("⚙️ 预处理设置", divider="gray")
        preprocess_cols = st.columns([1,1,1,1,1.2,1.2,1.2,1.2,1.2], gap="small")

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
                polyorder = st.slider("阶数", 3, 6, 5, key="polyorder", label_visibility="collapsed")
                baseline_params["polyorder"] = polyorder

        # 2. 缩放处理
        with preprocess_cols[1]:
            st.subheader("📏 缩放")
            scaling_method = st.selectbox(
                "方法",
                ["无", "Peak-Norm", "SNV", "MSC", "M-M-Norm", "标准化(均值0，方差1)"],
                key="scaling_method",
                label_visibility="collapsed"
            )
            scaling_params = {}

        # 3. 滤波处理
        with preprocess_cols[2]:
            st.subheader("📶 滤波")
            filtering_method = st.selectbox(
                "方法",
                ["无", "Savitzky-Golay", "中值滤波(MF)", "移动平均(MAF)", "小波变换(DWT)"],
                key="filtering_method",
                label_visibility="collapsed"
            )
            filtering_params = {}
            if filtering_method in ["Savitzky-Golay"]:
                w = st.selectbox("窗口", [11, 31], key="w_sg", label_visibility="collapsed")
                filtering_params["window_length"] = w

        # 4. 挤压处理
        with preprocess_cols[3]:
            st.subheader("🧪 挤压")
            squashing_method = st.selectbox(
                "方法",
                ["无", "Sigmoid挤压", "余弦挤压", "DTW挤压"],
                key="squashing_method",
                label_visibility="collapsed"
            )
            squashing_params = {}

        # 5-9列：操作区
        with preprocess_cols[4]:
            st.subheader("操作1")
            if st.button("🚀 应用处理", type="primary", use_container_width=True):
                if not st.session_state.raw_spectra:
                    st.warning("⚠️ 请先上传数据")
                else:
                    try:
                        wavenumbers, y = st.session_state.raw_spectra
                        preprocessor = Preprocessor()
                        processed_data, method_name = preprocessor.process(
                            wavenumbers, y,
                            baseline_method=baseline_method,
                            baseline_params=baseline_params,
                            scaling_method=scaling_method,
                            scaling_params=scaling_params,
                            filtering_method=filtering_method,
                            filtering_params=filtering_params,
                            squashing_method=squashing_method,
                            squashing_params=squashing_params
                        )
                        st.session_state.processed_spectra = (wavenumbers, processed_data)
                        st.success(f"✅ 处理完成: {method_name}")
                    except Exception as e:
                        st.error(f"❌ 处理失败: {e}")

        with preprocess_cols[5]:
            st.subheader("操作2")
            if st.button("🔍 显示排列", type="secondary", use_container_width=True):
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
                st.experimental_rerun()

            # 第一步类型筛选
            if st.session_state.show_arrangements and st.session_state.algorithm_permutations:
                all_types = list({p.get("first_step_type") for p in st.session_state.algorithm_permutations})
                selected_type = st.selectbox(
                    "第一步类型", ["全部"] + all_types,
                    key="first_step_filter",
                    label_visibility="collapsed"
                )
                if selected_type != "全部":
                    st.session_state.filtered_perms = [
                        p for p in st.session_state.algorithm_permutations
                        if p.get("first_step_type") == selected_type
                    ]

        with preprocess_cols[6]:
            st.subheader("操作3")
            if st.session_state.show_arrangements and st.session_state.filtered_perms:
                st.session_state.selected_perm_idx = st.selectbox(
                    f"选择方案（共{len(st.session_state.filtered_perms)}种）",
                    range(len(st.session_state.filtered_perms)),
                    format_func=lambda x: st.session_state.filtered_perms[x]["name"],
                    key="perm_select",
                    label_visibility="collapsed"
                )
                selected_perm = st.session_state.filtered_perms[st.session_state.selected_perm_idx]
                st.caption(f"当前: {selected_perm['name']}")

                if st.button("✅ 应用方案", use_container_width=True):
                    if not st.session_state.raw_spectra:
                        st.warning("⚠️ 请先上传数据")
                    else:
                        try:
                            wavenumbers, y = st.session_state.raw_spectra
                            preprocessor = Preprocessor()
                            algos = st.session_state.current_algorithms
                            processed_data, method_name = preprocessor.process(
                                wavenumbers, y,
                                baseline_method=algos['baseline'],
                                baseline_params=algos['baseline_params'],
                                scaling_method=algos['scaling'],
                                scaling_params=algos['scaling_params'],
                                filtering_method=algos['filtering'],
                                filtering_params=algos['filtering_params'],
                                squashing_method=algos['squashing'],
                                squashing_params=algos['squashing_params'],
                                algorithm_order=selected_perm['order']
                            )
                            st.session_state.processed_spectra = (wavenumbers, processed_data)
                            st.session_state.selected_arrangement = f"方案_{selected_perm['name']}"
                            st.success(f"✅ 方案应用完成")
                        except Exception as e:
                            st.error(f"❌ 应用失败: {e}")

        with preprocess_cols[7]:
            st.subheader("操作4")
            k_value = st.number_input(
                "k值", 1, value=st.session_state.k_value,
                key="k_input", label_visibility="collapsed"
            )
            if st.button("确定k值", use_container_width=True):
                st.session_state.k_value = k_value
                st.success(f"k={k_value}")

        with preprocess_cols[8]:
            st.subheader("操作5")
            if st.button("测试", type="primary", use_container_width=True):
                if not st.session_state.raw_spectra:
                    st.warning("⚠️ 请先上传数据")
                elif not st.session_state.selected_arrangement:
                    st.warning("⚠️ 请先应用方案")
                elif st.session_state.labels is None:
                    st.warning("⚠️ 请输入标签")
                else:
                    try:
                        wavenumbers, processed_data = st.session_state.processed_spectra
                        train_idx = st.session_state.train_indices
                        test_idx = st.session_state.test_indices

                        train_data = processed_data[:, train_idx]
                        test_data = processed_data[:, test_idx]
                        train_labels = st.session_state.labels[train_idx]
                        test_labels = st.session_state.labels[test_idx]

                        predictions = knn_classify(
                            train_data, train_labels, test_data, st.session_state.k_value
                        )

                        st.session_state.test_results = {
                            'accuracy': accuracy_score(test_labels, predictions),
                            'kappa': cohen_kappa_score(test_labels, predictions),
                            'confusion_matrix': confusion_matrix(test_labels, predictions),
                            'predictions': predictions,
                            'test_labels': test_labels
                        }
                        st.success("✅ 测试完成！")
                    except Exception as e:
                        st.error(f"❌ 测试失败: {e}")

        # 保存当前算法配置
        st.session_state.current_algorithms = {
            'baseline': baseline_method,
            'baseline_params': baseline_params,
            'scaling': scaling_method,
            'scaling_params': scaling_params,
            'filtering': filtering_method,
            'filtering_params': filtering_params,
            'squashing': squashing_method,
            'squashing_params': squashing_params
        }

        # 光谱可视化
        st.subheader("📈 光谱可视化", divider="gray")
        viz_row1 = st.columns(2)
        viz_row2 = st.columns(2)

        # 原始光谱
        with viz_row1[0]:
            st.subheader("原始光谱", divider="gray")
            if st.session_state.raw_spectra:
                wavenumbers, y = st.session_state.raw_spectra
                st.line_chart({f"样本0": y[:, 0]}, x=wavenumbers, height=250)
            else:
                st.info("等待加载数据")

        # 预处理后光谱
        with viz_row1[1]:
            st.subheader("预处理后光谱", divider="gray")
            if st.session_state.processed_spectra:
                wavenumbers, processed_data = st.session_state.processed_spectra
                st.line_chart({f"样本0": processed_data[:, 0]}, x=wavenumbers, height=250)
            else:
                st.info("请先应用预处理")

        # k值曲线
        with viz_row2[0]:
            st.subheader("k值曲线", divider="gray")
            if st.session_state.raw_spectra and st.session_state.processed_spectra:
                wavenumbers, y = st.session_state.raw_spectra
                _, processed_data = st.session_state.processed_spectra
                k_vals = np.abs(processed_data[:, 0] / (y[:, 0] + 1e-8))
                st.line_chart({"k值": k_vals}, x=wavenumbers, height=250)
            else:
                st.info("预处理后显示k值曲线")

        # 混淆矩阵
        with viz_row2[1]:
            st.subheader("混淆矩阵", divider="gray")
            if st.session_state.test_results:
                results = st.session_state.test_results
                st.text(f"准确率: {results['accuracy']:.4f}")
                st.text(f"卡帕系数: {results['kappa']:.4f}")
                
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('预测标签')
                ax.set_ylabel('真实标签')
                st.pyplot(fig)
            else:
                st.info("测试后显示混淆矩阵")

        # 结果导出
        if st.session_state.processed_spectra:
            st.subheader("💾 结果导出", divider="gray")
            export_name = st.text_input("导出文件名", "processed_spectra.txt")
            if st.button("导出"):
                try:
                    _, processed_data = st.session_state.processed_spectra
                    FileHandler().export_data(export_name, processed_data)
                    st.success(f"✅ 导出成功: {export_name}")
                except Exception as e:
                    st.error(f"❌ 导出失败: {e}")

if __name__ == "__main__":
    main()
