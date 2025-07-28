import streamlit as st
import importlib

# 初始化会话状态
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

# 主页内容
def show_home_page():
    st.title("光谱分析系统")
    st.markdown("### 欢迎使用光谱预处理与分析平台")

    # 定义四个方块对应的信息
    modules = [
        {
            "name": "1",
            "description": "111111",
            "target_page": "main"
        },
        {
            "name": "2",
            "description": "222222",
            "target_page": "page2"
        },
        {
            "name": "3",
            "description": "3333333",
            "target_page": "page3"
        },
        {
            "name": "4",
            "description": "44444444",
            "target_page": "page4"
        }
    ]

    # 创建两列布局来放方块
    cols = st.columns(2)
    for idx, module in enumerate(modules):
        with cols[idx % 2]:
            # 使用按钮而非HTML链接
            if st.button(f"{module['name']}\n\n{module['description']}"):
                st.session_state.current_page = module['target_page']
                st.experimental_rerun()  # 刷新页面以显示目标内容

# 动态加载并显示目标页面
def show_target_page(page_name):
    try:
        # 动态导入模块
        module = importlib.import_module(page_name)
        # 调用模块中的main函数（如果存在）
        if hasattr(module, 'main'):
            module.main()
        else:
            # 或者直接运行模块代码
            module.app()  # 假设模块中定义了app()函数
    except Exception as e:
        st.error(f"加载页面失败: {e}")
        st.write(f"请确保 {page_name}.py 文件存在并定义了可运行函数。")

# 根据当前页面状态显示内容
if st.session_state.current_page == 'home':
    show_home_page()
else:
    show_target_page(st.session_state.current_page)
