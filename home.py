import streamlit as st
import importlib

# 初始化会话状态
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

# 自定义 CSS 样式，设置按钮宽度和颜色
def set_button_style():
    st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;  /* 设置按钮圆角 */
        padding: 10px 20px;  /* 设置按钮内边距 */
    }
    </style>
    """, unsafe_allow_html=True)

# 主页内容
def show_home_page():
    set_button_style()  # 设置按钮样式
    st.title("🔬 光谱分析系统")
    st.markdown("### 欢迎使用光谱预处理与分析平台")

    modules = [
        {
            "name": "生物光学实验室介绍",
            "description": "西安电子科技大学生物光学实验室（BIOLIGHT LAB）成立于2015年9月，是智能医学检测技术的创造者和实践者，用成长型思维打造勇往直前的生物态团队，致力于培养富有创新精神和实践能力的新时代人才。",
            "target_page": "biolight"
        },
        {
            "name": "拉曼光谱预处理算法",
            "description": "拉曼光谱预处理的关键不是 “用哪种算法”，而是 **“针对干扰类型选算法”**：噪声强则优先小波或 SG 平滑，荧光背景强则侧重 airPLS 基线校正，样品差异大则需归一化。最终目标是让处理后的光谱 “峰位清晰、基线平坦、强度可对比”，为后续建模（如 PCA、PLS、机器学习）提供高质量输入。",
            "target_page": "main"
        },
    ]

    cols = st.columns(2)
    for idx, module in enumerate(modules):
        with cols[idx % 2]:
            if st.button(f"{module['name']}\n\n{module['description']}"):
                st.session_state.current_page = module['target_page']
                st.experimental_rerun()  # 刷新页面

# 动态加载目标页面
def show_target_page(page_name):
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
