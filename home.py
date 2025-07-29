import streamlit as st

# 设置页面配置（必须在最开始且仅调用一次）
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
        border: 2px solid transparent;
    }
    .clickable-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(22, 93, 255, 0.15);
        border-color: #165DFF;
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
def set_page(page_name):
    st.session_state.current_page = page_name

# 主页内容
def home_page():
    st.title("🔬 光谱分析系统")
    st.markdown("### 欢迎使用光谱预处理与分析平台")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 显示可点击卡片
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
        
        # 创建隐藏按钮，用于触发回调
        st.button(
            "模块1跳转按钮",
            key="btn-module-1",
            on_click=set_page,
            args=("main",),
            use_container_width=False
        )
        # 通过CSS隐藏按钮
        st.markdown('<style>div[data-testid="stButton"]:nth-of-type(1) {display: none;}</style>', unsafe_allow_html=True)
    
    with col2:
        # 显示可点击卡片
        st.markdown("""
        <div class="clickable-card" onclick="document.getElementById('btn-module-2').click()">
            <div class="card-title">2</div>
            <div class="card-description">222222</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 创建隐藏按钮，用于触发回调
        st.button(
            "模块2跳转按钮",
            key="btn-module-2",
            on_click=set_page,
            args=("main",),
            use_container_width=False
        )
        # 通过CSS隐藏按钮
        st.markdown('<style>div[data-testid="stButton"]:nth-of-type(2) {display: none;}</style>', unsafe_allow_html=True)

# 主页面内容
def main_page():
    st.title("主页面")
    st.write("这是主页面内容")
    # 在这里添加你的主页面功能代码
    
    if st.button("返回主页"):
        st.session_state.current_page = "home"

# 初始化会话状态
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

# 页面注册表
pages = {
    "home": home_page,
    "main": main_page
}

# 运行当前选中的页面
pages[st.session_state.current_page]()
