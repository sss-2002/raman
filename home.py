import streamlit as st
import importlib

# 初始化会话状态
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

# 设置页面配置
st.set_page_config(
    page_title="光谱分析系统",
    page_icon="🔬",
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
<style>
    .clickable-card {
        width: 400px;
        height: 200px;
        border-radius: 16px;
        padding: 20px;
        margin: 10px;
        background: white;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .clickable-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(22, 93, 255, 0.15);
    }
    .card-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #165DFF;
    }
    .card-description {
        font-size: 16px;
        color: #4E5969;
    }
    .hidden {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# 页面跳转回调函数
def navigate_to(page):
    st.session_state.current_page = page
    st.experimental_rerun()

# 主页内容
def show_home_page():
    st.title("🔬 光谱分析系统")
    st.markdown("### 欢迎使用光谱预处理与分析平台")
    
    # 创建一行两列的布局
    col1, col2 = st.columns(2)
    
    # 模块1 - 生物光学实验室介绍
    with col1:
        st.markdown("""
        <div class="clickable-card" onclick="document.getElementById('btn-module-1').click()">
            <div class="card-title">生物光学实验室介绍</div>
            <div class="card-description">
                西安电子科技大学生物光学实验室（BIOLIGHT LAB）成立于2015年9月，
                是智能医学检测技术的创造者和实践者，用成长型思维打造勇往直前的生物态团队，
                致力于培养富有创新精神和实践能力的新时代人才。
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 创建隐藏按钮用于触发页面跳转
        st.button(
            "模块1跳转按钮",
            key="btn-module-1",
            on_click=navigate_to,
            args=("main",),
            kwargs=None,
            help=None,
            disabled=False,
            use_container_width=False
        )
    
    # 模块2 - 示例模块
    with col2:
        st.markdown("""
        <div class="clickable-card" onclick="document.getElementById('btn-module-2').click()">
            <div class="card-title">2</div>
            <div class="card-description">222222</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 创建隐藏按钮用于触发页面跳转
        st.button(
            "模块2跳转按钮",
            key="btn-module-2",
            on_click=navigate_to,
            args=("main",),
            kwargs=None,
            help=None,
            disabled=False,
            use_container_width=False
        )

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
