import streamlit as st
import random
import numpy as np

# 设置页面配置（标题、图标，必须放在所有 Streamlit 命令之前）
st.set_page_config(
    page_title="排列预处理工具",
    page_icon="🔢",
    layout="wide"  # 宽屏布局，适配更多内容
)

# ---------------------- 1. 页面标题与说明 ----------------------
st.title("🔢 排列预处理工具")
st.markdown("""
    支持序列的基础排序、去重排序、反向排序、打乱顺序等预处理操作。
    可手动输入序列或生成随机序列，实时查看处理结果与日志。
""")
st.divider()  # 分割线，优化视觉

# ---------------------- 2. 输入区域（手动输入 + 随机生成） ----------------------
with st.container(border=True):  # 带边框的容器，区分区域
    st.subheader("📥 输入数据")
    
    # 手动输入序列
    input_str = st.text_input(
        label="请输入需要处理的序列（用逗号分隔，例如：1,3,5,2）",
        value="1,3,5,2,4,6,8,7,9",  # 默认示例数据
        help="输入格式：数字之间用英文逗号分隔，无需空格"
    )
    
    # 随机生成序列（折叠面板，避免占用过多空间）
    with st.expander("🔀 生成随机序列（点击展开）"):
        col1, col2 = st.columns([1, 2])  # 分两列布局，优化排版
        with col1:
            random_count = st.number_input(
                label="元素数量",
                min_value=1,  # 最少1个元素
                max_value=100,  # 最多100个元素
                value=10,  # 默认10个元素
                step=1
            )
        with col2:
            if st.button("生成随机序列", use_container_width=True):
                # 生成 1-100 之间的随机整数序列
                random_seq = [random.randint(1, 100) for _ in range(random_count)]
                # 更新输入框内容（覆盖原有手动输入）
                input_str = ",".join(map(str, random_seq))
                # 用 Streamlit 的会话状态缓存随机序列，避免刷新后丢失
                st.session_state["input_str"] = input_str
                # 刷新页面，让输入框显示新生成的序列
                st.rerun()

# 从会话状态恢复输入（如果之前生成过随机序列）
if "input_str" in st.session_state:
    input_str = st.session_state["input_str"]

# ---------------------- 3. 预处理选项（单选按钮 + 复选框） ----------------------
with st.container(border=True):
    st.subheader("⚙️ 预处理选项")
    
    # 预处理类型（单选按钮）
    preprocess_type = st.radio(
        label="选择预处理类型",
        options=[
            ("basic", "基础排序"),
            ("unique", "去重后排序"),
            ("reverse", "反向排序"),
            ("shuffle", "打乱顺序")
        ],
        format_func=lambda x: x[1],  # 显示选项的中文名称
        index=0,  # 默认选择“基础排序”
        horizontal=True  # 水平排列，节省空间
    )[0]  # 取元组的第一个元素（实际值：basic/unique/reverse/shuffle）
    
    # 额外选项：升序/降序（仅基础排序时显示）
    sort_ascending = True
    if preprocess_type == "basic":
        sort_ascending = st.checkbox(
            label="升序排列",
            value=True,  # 默认升序
            help="取消勾选则为降序排列"
        )

# ---------------------- 4. 执行预处理与结果展示 ----------------------
with st.container(border=True):
    st.subheader("📊 处理结果")
    
    # 初始化结果变量
    original_seq = None
    processed_seq = None
    log = []
    
    # 执行预处理按钮（点击后触发逻辑）
    if st.button("▶️ 执行预处理", use_container_width=True, type="primary"):
        # 1. 验证并解析输入序列
        if not input_str.strip():
            st.error("❌ 请输入序列或生成随机序列后再执行！")
            st.stop()  # 终止后续逻辑
        
        try:
            # 分割字符串并转换为整数列表
            original_seq = list(map(int, input_str.split(',')))
            log.append(f"✅ 解析输入序列：{original_seq}")
        except ValueError:
            st.error("❌ 输入格式错误！请使用英文逗号分隔数字（例如：1,3,5,2）。")
            st.stop()
        
        # 2. 执行对应的预处理逻辑
        processed_seq = original_seq.copy()
        if preprocess_type == "basic":
            log.append("🔄 开始执行：基础排序")
            processed_seq.sort(reverse=not sort_ascending)
            order = "升序" if sort_ascending else "降序"
            log.append(f"✅ 完成 {order} 排序")
        
        elif preprocess_type == "unique":
            log.append("🔄 开始执行：去重后排序")
            processed_seq = list(np.unique(processed_seq))  # 去重
            processed_seq.sort(reverse=not sort_ascending)  # 排序
            order = "升序" if sort_ascending else "降序"
            log.append(f"✅ 完成去重 + {order} 排序")
        
        elif preprocess_type == "reverse":
            log.append("🔄 开始执行：反向排序")
            processed_seq = processed_seq[::-1]  # 反转列表
            log.append("✅ 完成反向排序")
        
        elif preprocess_type == "shuffle":
            log.append("🔄 开始执行：打乱顺序")
            random.shuffle(processed_seq)  # 打乱列表
            log.append("✅ 完成打乱顺序")
        
        log.append("🎉 预处理全部完成！")
    
    # 3. 展示结果（仅当处理完成后显示）
    if original_seq is not None and processed_seq is not None:
        # 分两列展示“原始序列”和“处理后序列”
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**原始序列**")
            st.code(", ".join(map(str, original_seq)))  # 用代码块展示，更清晰
        with col2:
            st.markdown("**处理后序列**")
            st.code(", ".join(map(str, processed_seq)))
        
        # 展示处理日志（折叠面板）
        with st.expander("📝 查看处理日志（点击展开）"):
            for line in log:
                st.write(line)
    
    # 清除结果按钮
    if st.button("🗑️ 清除结果", use_container_width=True):
        # 重置会话状态和输入框
        if "input_str" in st.session_state:
            del st.session_state["input_str"]
        # 刷新页面
        st.rerun()

# ---------------------- 5. 页脚说明 ----------------------
st.divider()
st.markdown("""
    <div style="text-align: center; color: #666;">
        部署说明：此工具基于 Streamlit 构建，可直接在 GitHub 配合 Streamlit Community Cloud 部署。
    </div>
""", unsafe_allow_html=True)
