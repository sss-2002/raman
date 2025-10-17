import streamlit as st
import importlib

# 关键：必须是脚本中第一个 Streamlit 命令，且只调用一次
st.set_page_config(
    layout="wide",
    page_title="光谱分析系统",
    page_icon="🔬"
)

# 初始化会话状态
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

# 页面跳转函数
def navigate_to(page):
    st.session_state.current_page = page
    st.experimental_rerun()

# 自定义CSS样式
def set_custom_style():
    st.markdown(
        """
        <style>
        .main {
            background-color: #f5f7fa;
            padding: 0px 10px;
            max-width: 100% !important;
            width: 100% !important;
        }
        .css-18e3th9 {
            padding-top: 0rem;
            padding-bottom: 1rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }
        .stButton > button {
            width: 100%;
            border-radius: 6px;
            padding: 10px 0;
            background-color: #165DFF;
            color: white;
            border: none;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #0E42D2;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(22, 93, 255, 0.2);
        }
        .title-text {
            font-size: 28px;
            font-weight: 700;
            color: #1D2939;
            margin: 0 0 15px 0;
            padding: 0;
        }
        .subtitle-text {
            color: #4B5563;
            margin: 0 0 30px 0;
        }
        .card {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            padding: 25px;
            min-height: 285px;
            display: flex;
            flex-direction: column;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
        }
        .card-icon {
            font-size: 28px;
            margin-bottom: 15px;
            color: #165DFF;
        }
        .card-title {
            font-size: 18px;
            font-weight: 600;
            color: #1D2939;
            margin: 0 0 15px 0;
        }
        .card-description {
            color: #4B5563;
            font-size: 14px;
            line-height: 1.6;
            flex-grow: 1;
            margin: 0 0 20px 0;
            overflow-y: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# 主页内容
def show_home_page():
    set_custom_style()
    st.markdown('<h1 class="title-text">🔬 光谱分析系统</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">欢迎使用专业的光谱预处理与分析平台</p>', unsafe_allow_html=True)

    modules = [
        {
            "name": "拉曼光谱预处理分析",
            "description": "提供一站式拉曼光谱预处理解决方案，支持噪声去除（SG平滑、小波去噪）、基线校正（airPLS、ALS）、归一化（Min-Max、标准化）等核心功能。",
            "target_page": "main",
            "icon": "📊",
        },
        {
            "name": "排列预处理模型",
            "description": "针对单一干扰类型的系统化预处理方案，按“干扰识别→算法匹配→参数优化”流程排列预处理步骤。",
            "target_page": "arrangement",
            "icon": "🔄",
        },
        {
            "name": "组合预处理模型",
            "description": "面向复杂干扰场景的多算法协同处理模型，支持自由组合2-4种预处理算法。",
            "target_page": "combination",
            "icon": "🧩",
        },
        {
            "name": "排列组合预处理模型",
            "description": "融合“步骤排列”与“算法组合”的高阶预处理模型，兼顾流程规范性与算法灵活性。",
            "target_page": "arrangement_combination",
            "icon": "🔀",
        },
    ]

    cols = st.columns(2)
    for idx, module in enumerate(modules):
        with cols[idx % 2]:
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-icon">{module['icon']}</div>
                    <h3 class="card-title">{module['name']}</h3>
                    <p class="card-description">{module['description']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"进入 {module['name']}", key=f"btn_{module['target_page']}"):
                navigate_to(module['target_page'])

# 其他页面内容
def show_about_page():
    set_custom_style()
    st.title("关于我们")
    st.markdown("""
    ### 西安电子科技大学生物光学实验室
    成立于2015年9月，专注于智能医学检测技术的研究与应用。
    """)
    if st.button("返回首页"):
        navigate_to("home")

def show_contact_page():
    set_custom_style()
    st.title("联系我们")
    st.markdown("""
    - 地址：陕西省西安市雁塔区西安电子科技大学
    - 邮箱：biolight@xidian.edu.cn
    """)
    if st.button("返回首页"):
        navigate_to("home")

def show_help_page():
    set_custom_style()
    st.title("帮助中心")
    st.markdown("""
    ### 常见问题
    1. **如何选择合适的光谱预处理算法？**
    答：根据您的光谱特点选择，噪声强则优先平滑算法。
    """)
    if st.button("返回首页"):
        navigate_to("home")

def show_login_page():
    set_custom_style()
    st.title("用户登录")
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    if st.button("登录"):
        st.success("登录成功！")
        navigate_to("home")
    if st.button("返回首页"):
        navigate_to("home")

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
            # 导入其他页面模块（确保这些模块中没有set_page_config）
            module = importlib.import_module(page_name)
            if hasattr(module, "main"):
                module.main()
                if st.button("返回首页"):
                    navigate_to("home")
            else:
                st.write(f"请在 {page_name} 模块中定义 main() 函数")
                if st.button("返回首页"):
                    navigate_to("home")
    except Exception as e:
        st.error(f"加载页面失败: {e}")
        if st.button("返回首页"):
            navigate_to("home")

# 显示当前页面
current_page = st.session_state.get("current_page", "home")
show_target_page(current_page)
