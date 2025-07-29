import streamlit as st
import importlib

# 初始化会话状态
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

# 主页内容
def show_home_page():
    st.title("🔬 光谱分析系统")
    st.markdown("### 欢迎使用光谱预处理与分析平台")

    modules = [
        {
            "name": "生物光学实验室介绍",
            "description": "西安电子科技大学生物光学实验室（BIOLIGHT LAB）成立于2015年9月，是智能医学检测技术的创造者和实践者，用成长型思维打造勇往直前的生物态团队，致力于培养富有创新精神和实践能力的新时代人才。",
            "target_page": "main"
        },
        {
            "name": "2",
            "description": "222222",
            "target_page": "main"
        },
    ]

    # 使用 st.columns 创建两列，每列宽度相同
    cols = st.columns(len(modules))
    for idx, module in enumerate(modules):
        with cols[idx]:
            # 设置按钮的样式，使其大小一致
            button_text = f"{module['name']}\n\n{module['description']}"
            if st.button(button_text):
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
