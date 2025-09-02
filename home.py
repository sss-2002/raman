import streamlit as st
import importlib

# 初始化会话状态
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

# 自定义 CSS 样式
def set_custom_style():
    st.markdown(
        """
        <style>
        .stButton > button {
            width: 100%;
            border-radius: 5px;
            padding: 15px 25px;
            background-color: #007bff;
            color: white;
            border: none;
        }
        .stButton > button:hover {
            background-color: #0056b3;
        }
        .nav-link {
            color: white;
            text-decoration: none;
            margin: 0 10px;
            font-size: 16px;
        }
        .nav-link:hover {
            text-decoration: underline;
        }
        .title-text {
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
            margin-bottom: 20px;
        }
        .card {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# 顶部导航栏
def show_navbar():
    st.markdown(
        """
        <div style="background-color: #007bff; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
            <a href="#" class="nav-link" onclick="pageChange('home')">首页</a>
            <a href="#" class="nav-link" onclick="pageChange('about')">关于我们</a>
            <a href="#" class="nav-link" onclick="pageChange('contact')">联系我们</a>
            <a href="#" class="nav-link" onclick="pageChange('help')">帮助中心</a>
            <a href="#" class="nav-link" style="float: right;" onclick="pageChange('login')">登录</a>
        </div>
        <script>
        function pageChange(page) {
            sessionStorage.setItem('current_page', page);
            window.parent.location.reload();
        }
        </script>
        """,
        unsafe_allow_html=True,
    )

# 主页内容
def show_home_page():
    set_custom_style()
    show_navbar()
    st.markdown('<div class="title-text">光谱分析系统</div>', unsafe_allow_html=True)
    st.markdown("### 欢迎使用光谱预处理与分析平台")

    modules = [
        {
            "name": "生物光学实验室介绍",
            "description": "西安电子科技大学生物光学实验室（BIOLIGHT LAB）成立于2015年9月，是智能医学检测技术的创造者和实践者，用成长型思维打造勇往直前的生物态团队，致力于培养富有创新精神和实践能力的新时代人才。",
            "target_page": "biolight",
            "icon": "🏫",
        },
        {
            "name": "拉曼光谱预处理算法",
            "description": "拉曼光谱预处理的关键不是 “用哪种算法”，而是 **“针对干扰类型选算法”**：噪声强则优先小波或 SG 平滑，荧光背景强则侧重 airPLS 基线校正，样品差异大则需归一化。最终目标是让处理后的光谱 “峰位清晰、基线平坦、强度可对比”，为后续建模（如 PCA、PLS、机器学习）提供高质量输入。",
            "target_page": "main",
            "icon": "🔬",
        },
        {
            "name": "高值化合物分析",
            "description": "对各类高价值化合物进行光谱特征分析与研究，助力相关科研与应用。",
            "target_page": "compound",
            "icon": "🧪",
        },
        {
            "name": "个人中心",
            "description": "管理个人相关设置与信息。",
            "target_page": "personal",
            "icon": "👤",
        },
    ]

    cols = st.columns(2)
    for idx, module in enumerate(modules):
        with cols[idx % 2]:
            st.markdown(
                f"""
                <div class="card">
                    <h3>{module['icon']} {module['name']}</h3>
                    <p>{module['description']}</p>
                    <button onclick="pageChange('{module['target_page']}')" style="width: 100%; border-radius: 5px; padding: 10px; background-color: #007bff; color: white; border: none;">进入</button>
                </div>
                """,
                unsafe_allow_html=True,
            )

# 关于我们页面
def show_about_page():
    set_custom_style()
    show_navbar()
    st.title("关于我们")
    st.markdown("这里是关于我们的详细介绍...")

# 联系我们页面
def show_contact_page():
    set_custom_style()
    show_navbar()
    st.title("联系我们")
    st.markdown("联系方式：xxx@example.com")

# 帮助中心页面
def show_help_page():
    set_custom_style()
    show_navbar()
    st.title("帮助中心")
    st.markdown("常见问题解答...")

# 登录页面
def show_login_page():
    set_custom_style()
    show_navbar()
    st.title("登录")
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    if st.button("登录"):
        st.success("登录成功")

# 动态加载目标页面
def show_target_page(page_name):
    try:
        if page_name == "home":
            show_home_page()
        elif page_name == "about":
            show_about_page()
        elif page_name == "contact":
            show_contact_page()
        elif page_name == "help":
            show_help_page()
        elif page_name == "login":
            show_login_page()
        else:
            module = importlib.import_module(page_name)
            if hasattr(module, "main"):
                module.main()
            else:
                st.write(f"请在 {page_name}.py 中定义 main() 函数")
    except Exception as e:
        st.error(f"加载页面失败: {e}")

# 根据状态显示内容
current_page = st.session_state.get("current_page", "home")
show_target_page(current_page)
