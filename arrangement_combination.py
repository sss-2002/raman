import streamlit as st

def main():
    # 页面配置
    st.set_page_config(page_title="排列预处理模型", layout="wide")

    # 页面标题
    st.title("排列预处理模型")
    
    # 页面左侧的列
    col1, col2 = st.columns([3, 2])
    
    # 第一个列（数据管理部分）
    with col1:
        st.sidebar.header("数据管理")
        uploaded_file = st.file_uploader("上传数据文件", type=["zip", "csv", "txt"])
        if uploaded_file is not None:
            st.sidebar.write(f"已上传文件: {uploaded_file.name}")
        
        # 配置部分
        st.sidebar.header("预处理设置")
        baseline = st.sidebar.selectbox("基线校正", ["无", "方法1", "方法2"])
        scaling = st.sidebar.selectbox("缩放", ["无", "方法1", "方法2"])
        filtering = st.sidebar.selectbox("滤波", ["无", "方法1", "方法2"])
        squashing = st.sidebar.selectbox("挤压", ["无", "方法1", "方法2"])

        # 训练/测试集比例
        st.sidebar.header("训练测试集划分")
        train_test_split = st.sidebar.slider("训练集比例", min_value=0.1, max_value=0.9, value=0.8)

        # 处理按钮
        st.sidebar.button("应用处理", use_container_width=True)

    # 第二个列（展示区域）
    with col2:
        # 结果展示
        st.subheader("预处理结果展示")
        st.write("这里显示预处理结果的预览")

        # 原始数据展示
        st.subheader("原始光谱数据")
        st.write("此处展示上传的数据或者光谱数据的相关内容")

        # 预测结果展示
        st.subheader("预测结果展示")
        st.write("预测结果或处理之后的数据")

        # 测试按钮
        st.button("测试", use_container_width=True)

    # 页面底部
    st.markdown("---")
    st.text("返回首页 | 退出")

if __name__ == "__main__":
    main()
