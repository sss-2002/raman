import streamlit as st
import importlib

# 初始化会话状态
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

# 页面跳转函数 - 使用Streamlit原生机制
def navigate_to(page):
    st.session_state.current_page = page
    st.experimental_rerun()

# 自定义CSS样式
def set_custom_style():
    st.markdown(
        """
        <style>
        /* 页面整体样式 */
        .main {
            background-color: #f5f7fa;
            padding: 0px 10px;
        }
        
        /* 按钮样式 */
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
        
        /* 导航栏样式 */
        .navbar {
            background-color: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin: 10px 0 25px 0;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            border: 1px solid #f0f0f0;
        }
        .nav-item {
            display: inline-block;
            margin: 0 15px;
        }
        .nav-link {
            color: #1D2939;
            text-decoration: none;
            font-size: 16px;
            font-weight: 500;
            padding: 5px 0;
            position: relative;
        }
        .nav-link:after {
            content: '';
            position: absolute;
            width: 0;
            height: 2px;
            bottom: 0;
            left: 0;
            background-color: #165DFF;
            transition: width 0.3s ease;
        }
        .nav-link:hover {
            color: #165DFF;
        }
        .nav-link:hover:after {
            width: 100%;
        }
        .login-container {
            float: right;
        }
        
        /* 标题样式 */
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
        
        /* 卡片样式 */
        .card {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            padding: 25px;
            height: 420px;
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

# 顶部导航栏 - 使用Streamlit按钮实现跳转
def show_navbar():
    # 创建导航容器
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    
    with col1:
        if st.button("首页", key="nav_home"):
            navigate_to("home")
    
    with col2:
        if st.button("关于我们", key="nav_about"):
            navigate_to("about")
    
    with col3:
        if st.button("联系我们", key="nav_contact"):
            navigate_to("contact")
    
    with col4:
        if st.button("帮助中心", key="nav_help"):
            navigate_to("help")
    
    with col5:
        if st.button("登录", key="nav_login"):
            navigate_to("login")

# 主页内容
def show_home_page():
    set_custom_style()
    show_navbar()
    
    # 页面标题
    st.markdown('<h1 class="title-text">🔬 光谱分析系统</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">欢迎使用专业的光谱预处理与分析平台</p>', unsafe_allow_html=True)

    # 功能模块
    modules = [
        {
            "name": "生物光学实验室介绍",
            "description": "西安电子科技大学生物光学实验室（BIOLIGHT LAB）成立于2015年9月，是智能医学检测技术的创造者和实践者。实验室用成长型思维打造勇往直前的生物态团队，致力于培养富有创新精神和实践能力的新时代人才，推动生物光学领域的前沿研究与应用。",
            "target_page": "biolight",
            "icon": "🏫",
        },
        {
            "name": "拉曼光谱预处理算法",
            "description": "拉曼光谱预处理的关键不是“用哪种算法”，而是“针对干扰类型选算法”：噪声强则优先小波或SG平滑，荧光背景强则侧重airPLS基线校正，样品差异大则需归一化。最终目标是让处理后的光谱“峰位清晰、基线平坦、强度可对比”，为后续建模提供高质量输入。",
            "target_page": "main",
            "icon": "🔬",
        },
        {
            "name": "高值化合物分析",
            "description": "对各类高价值化合物进行光谱特征分析与研究，通过先进算法提取特征峰，建立成分与光谱特征的关联模型。系统支持多种化合物的快速识别与定量分析，为新材料研发、药物分析等领域提供高效可靠的检测手段，助力相关科研与应用。",
            "target_page": "compound",
            "icon": "🧪",
        },
        {
            "name": "个人中心",
            "description": "管理个人实验数据、分析报告和系统设置，查看历史分析记录，保存常用分析参数，个性化定制您的分析工作流。支持数据备份与分享，多设备同步分析结果，设置个人偏好与通知，让光谱分析工作更加高效便捷。",
            "target_page": "personal",
            "icon": "👤",
        },
    ]

    # 创建2列布局
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
            # 使用Streamlit原生按钮实现跳转
            if st.button(f"进入 {module['name']}", key=f"btn_{module['target_page']}"):
                navigate_to(module['target_page'])

# 其他页面内容
def show_about_page():
    set_custom_style()
    show_navbar()
    st.title("关于我们")
    st.markdown("""
    ### 西安电子科技大学生物光学实验室
    成立于2015年9月，专注于智能医学检测技术的研究与应用。
    
    我们的使命是：用创新科技推动生物医学领域的发展，为人类健康事业贡献力量。
    """)
    # 返回首页按钮
    if st.button("返回首页"):
        navigate_to("home")

def show_contact_page():
    set_custom_style()
    show_navbar()
    st.title("联系我们")
    st.markdown("""
    - 地址：陕西省西安市雁塔区西安电子科技大学
    - 邮箱：biolight@xidian.edu.cn
    - 电话：029-XXXXXXXX
    """)
    if st.button("返回首页"):
        navigate_to("home")

def show_help_page():
    set_custom_style()
    show_navbar()
    st.title("帮助中心")
    st.markdown("""
    ### 常见问题
    
    1. **如何选择合适的光谱预处理算法？**
    答：根据您的光谱特点选择，噪声强则优先平滑算法，背景干扰强则选择基线校正算法。
    
    2. **分析结果如何导出？**
    答：在分析结果页面，点击右上角"导出"按钮，可选择导出格式。
    """)
    if st.button("返回首页"):
        navigate_to("home")

def show_login_page():
    set_custom_style()
    show_navbar()
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
            # 尝试导入外部模块
            module = importlib.import_module(page_name)
            if hasattr(module, "main"):
                module.main()
                # 外部页面添加返回首页按钮
                if st.button("返回首页"):
                    navigate_to("home")
            else:
                st.write(f"请在 {page_name}.py 中定义 main() 函数")
                if st.button("返回首页"):
                    navigate_to("home")
    except Exception as e:
        st.error(f"加载页面失败: {e}")
        if st.button("返回首页"):
            navigate_to("home")

# 根据状态显示内容
current_page = st.session_state.get("current_page", "home")
show_target_page(current_page)
    
