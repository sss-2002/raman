import streamlit as st

# 设置页面配置
st.set_page_config(
    page_title="光谱分析系统",
    page_icon="🌌",
    layout="wide"
)

# 标题和简介
st.title("光谱分析系统")
st.markdown("### 欢迎使用光谱预处理与分析平台")

# 定义四个模块的链接和描述
modules = [
    {
        "name": "1",
        "description": "111111",
        "page": "main"
    },
    {
        "name": "2",
        "description": "222222",
        "page": "SpectraApp"
    },
    {
        "name": "3",
        "description": "3333333",
        "page": "SpectraApp"
    },
    {
        "name": "4",
        "description": "44444444",
        "page": "SpectraApp"
    }
]

# 创建网格布局展示模块
cols = st.columns(2)  # 两列布局

for i, module in enumerate(modules):
    col = cols[i % 2]  # 轮流使用左右列
    with col:
        # 创建卡片式链接
        st.markdown(f"""
        <a href="/{module['page']}" target="_self" style="text-decoration: none;">
            <div class="module-card" style="
                background-color: #f0f2f6;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                transition: transform 0.3s;
            ">
                <h3 style="color: #007BFF; margin-bottom: 10px;">{module['name']}</h3>
                <p style="color: #666;">{module['description']}</p>
            </div>
        </a>
        """, unsafe_allow_html=True)

# 添加自定义CSS使卡片更美观
st.markdown("""
<style>
.module-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 8px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)

# 页脚信息
st.markdown("---")
st.markdown("© 2025 光谱分析系统 | 版本 1.0.0")
