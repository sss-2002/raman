import streamlit as st

# 设置页面基础配置
st.set_page_config(
    page_title="光谱分析系统",
    layout="wide"
)

# 页面标题等内容
st.title("光谱分析系统")
st.markdown("### 欢迎使用光谱预处理与分析平台")

# 定义四个方块对应的信息，包括跳转目标页面（这里写的是目标文件的名称，不含.py后缀）
modules = [
    {
        "name": "1",
        "description": "111111",
        "target_page": "page1"  # 对应 page1.py 文件，点击跳转到该页面
    },
    {
        "name": "2",
        "description": "222222",
        "target_page": "page2"  # 对应 page2.py 文件
    },
    {
        "name": "3",
        "description": "3333333",
        "target_page": "page3"  # 对应 page3.py 文件
    },
    {
        "name": "4",
        "description": "44444444",
        "target_page": "page4"  # 对应 page4.py 文件
    }
]

# 创建两列布局来放方块
cols = st.columns(2)
for idx, module in enumerate(modules):
    with cols[idx % 2]:
        # 构建 HTML 形式的链接和卡片样式，实现点击方块跳转
        html_code = f"""
        <a href="/{module['target_page']}" target="_self" style="text-decoration: none;">
            <div style="border: 1px solid #eee; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: all 0.3s ease;">
                <h3 style="color: #007bff; margin-bottom: 10px;">{module['name']}</h3>
                <p style="color: #666;">{module['description']}</p>
            </div>
        </a>
        """
        st.markdown(html_code, unsafe_allow_html=True)

# 页脚信息
st.markdown("---")
st.markdown("© 2025 光谱分析系统 | 版本 1.0.0")
