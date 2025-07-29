import streamlit as st
import importlib

# 初始化会话状态
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

# 主页内容
def show_home_page():
    # 设置页面布局
    st.set_page_config(layout="wide")
    
    # 标题区域
    st.title("🔬 光谱分析系统")
    st.markdown("### 欢迎使用光谱预处理与分析平台")
    st.markdown("---")
    
    # 添加图表描述（根据图片内容）
    st.markdown("## 图表")
    st.markdown("""
    - 图表中关于光谱的描述
    - 图表中关于光谱的描述（laouloir Lua）是基于2015年9月，随着数据传输速率逐渐增加的趋势和变化，
      用最长周期进行测量结果的数据显示，致力于对光谱的解释和实现能力的影响。
    """)
    st.markdown("---")
    
    # 创建模块信息
    modules = [
        {
            "name": "生物光学实验室介绍",
            "description": "西安电子科技大学生物光学实验室（BIOLIGHT LAB）成立于2015年9月，是智能医学检测技术的创造者和实践者，用成长型思维打造勇往直前的生物态团队，致力于培养富有创新精神和实践能力的新时代人才。",
            "target_page": "main"
        },
        {
            "name": "光谱分析模块",
            "description": "提供专业的光谱数据处理与分析功能，包括数据预处理、特征提取、模型建立和结果可视化等完整流程，支持多种光谱仪器的数据格式导入。",
            "target_page": "main"
        },
    ]

    # 使用两列布局，设置列宽比例
    col1, col2 = st.columns([1, 1])  # 各占50%宽度
    
    # 左侧模块
    with col1:
        container = st.container(border=True, height=200)  # 固定高度容器
        with container:
            st.subheader(modules[0]["name"])
            st.write(modules[0]["description"])
            if st.button("进入模块", key="btn1", use_container_width=True):
                st.session_state.current_page = modules[0]["target_page"]
                st.experimental_rerun()
    
    # 右侧模块
    with col2:
        container = st.container(border=True, height=200)  # 固定高度容器
        with container:
            st.subheader(modules[1]["name"])
            st.write(modules[1]["description"])
            if st.button("进入模块", key="btn2", use_container_width=True):
                st.session_state.current_page = modules[1]["target_page"]
                st.experimental_rerun()

# 动态加载目标页面
def show_target_page(page_name):
    # 添加返回主页按钮
    if st.button("返回主页"):
        st.session_state.current_page = 'home'
        st.experimental_rerun()
    
    try:
        module = importlib.import_module(page_name)
        if hasattr(module, 'main'):
            module.main()  # 调用 main.py 中的 main() 函数
        else:
            st.write(f"请在 {page_name}.py 中定义 main() 函数")
    except Exception as e:
        st.error(f"加载页面失败: {e}")

# 根据状态显示内容
if st.session_state.current_page == 'home':
    show_home_page()
else:
    show_target_page(st.session_state.current_page)
