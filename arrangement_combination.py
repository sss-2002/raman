import streamlit as st

# 使用 st.write 显示文本
st.write("这是在Streamlit上显示的一段话。")

# 或者使用 st.text 显示文本
st.text("这是在Streamlit上显示的一段话。")
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
    st.session_state.setdefault('current_algorithms', current_algorithms)
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
    st.session_state['current_algorithms'] = current_algorithms
    # 设置页面：紧凑布局
    st.set_page_config(layout="wide", page_icon="🔬", page_title="排列预处理模型")
    # 全局样式调整：更紧凑的字体和间距，确保预处理设置在一行显示
    st.markdown("""
        <style>
        /* 全局字体缩小，确保预处理设置在一行显示 */
        body {font-size: 0.75rem !important;}
        .css-1v0mbdj {padding: 0.3rem 0.5rem !important;} /* 容器内边距 */
        .css-1d391kg {padding: 0.2rem 0 !important;} /* 标题间距 */
        .css-1x8cf1d {line-height: 1.1 !important;} /* 文本行高 */
        .css-12ttj6m {margin-bottom: 0.3rem !important;} /* 组件底部间距 */
        .css-16huue1 {padding: 0.2rem 0.5rem !important; font-size: 0.7rem !important;} /* 按钮内边距和字体 */
        h3 {font-size: 1rem !important; margin: 0.3rem 0 !important;} /* 子标题 */
        .css-1b3298e {gap: 0.3rem !important;} /* 列间距 */
        .stSlider, .stSelectbox, .stTextInput {margin-bottom: 0.3rem !important;} /* 输入组件间距 */
        .stCaption {font-size: 0.65rem !important; margin-top: -0.2rem !important;} /* 说明文字 */
        .css-1544g2n {padding: 0.2rem 0.5rem !important;} /* 展开面板内边距 */
        </style>
    """, unsafe_allow_html=True)

    st.title("🌌 排列预处理模型")

    # 页面整体布局：左侧数据管理，右侧主要内容区
    col_left, col_right = st.columns([1.2, 3.9])

    # ===== 左侧：数据管理模块（移除光谱条数和数据点数）=====
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
