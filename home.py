import streamlit as st
import importlib

# 初始化会话状态
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

# 自定义 CSS 样式 - 导航栏改为白色
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
        
        /* 导航栏样式 - 改为白色背景 */
        .navbar {
            background-color: white;
            padding: 15px 20px;
            border-radius: 8px;
            margin: 10px 0 25px 0;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            border: 1px solid #f0f0f0; /* 轻微边框增加层次感 */
        }
        .nav-link {
            color: #1D2939; /* 深色文字 */
            text-decoration: none;
            margin: 0 15px;
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
            background-color: #165DFF; /* 蓝色下划线 */
            transition: width 0.3s ease;
        }
        .nav-link:hover {
            color: #165DFF; /* hover时文字变蓝 */
        }
        .nav-link:hover:after {
            width: 100%;
        }
        .nav-link.login {
            float: right;
            color: #165DFF; /* 登录按钮文字蓝色 */
            font-weight: 600;
        }
        .nav-link.login:hover {
            color: #0E42D2;
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
        
        /* 卡片样式 - 保持大小一致 */
        .card-container {
            height: 100%;
        }
        .card {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            padding: 25px;
            height: 320px; /* 固定卡片高度 */
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
            flex-grow: 1; /* 让描述部分填充空间 */
            margin: 0 0 20px 0;
            overflow-y: auto; /* 内容过多时可滚动 */
        }
        .card-button {
            margin-top: auto; /* 按钮始终在底部 */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# 顶部导航栏
def show_navbar():
    st.markdown(
        """
        <div class="navbar">
            <a href="#" class="nav-link" onclick="pageChange('home')">首页</a>
            <a href="#" class="nav-link" onclick="pageChange('about')">关于我们</a>
            <a href="#" class="nav-link" onclick="pageChange('contact')">联系我们</a>
            <a href="#" class="nav-link" onclick="pageChange('help')">帮助中心</a>
            <a href="#" class="nav-link login" onclick="pageChange('login')">登录</a>
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

    # 创建2列布局，确保卡片大小一致
    cols = st.columns(2)
    for idx, module in enumerate(modules):
        with cols[idx % 2]:
            st.markdown(
                f"""
                <div class="card-container">
                    <div class="card">
                        <div class="card-icon">{module['icon']}</div>
                        <h3 class="card-title">{module['name']}</h3>
                        <p class="card-description">{module['description']}</p>
                        <button class="card-button" onclick="pageChange('{module['target_page']}')">进入</button>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

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

def show_contact_page():
    set_custom_style()
    show_navbar()
    st.title("联系我们")
    st.markdown("""
    - 地址：陕西省西安市雁塔区西安电子科技大学
    - 邮箱：biolight@xidian.edu.cn
    - 电话：029-XXXXXXXX
    """)

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

def show_login_page():
    set_custom_style()
    show_navbar()
    st.title("用户登录")
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    if st.button("登录"):
        st.success("登录成功！")

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
    
