import streamlit as st
import numpy as np
import pandas as pd
from SD import D2
from FD import D1
from sigmoids import sigmoid
from squashing import squashing  
from i_squashing import i_squashing 
from i_sigmoid import i_sigmoid
from IModPoly import IModPoly
from AsLS import baseline_als
from LPnorm import LPnorm

# 设置页面
st.set_page_config(layout="wide", page_title="光谱预处理系统")
st.title("🌌 光谱预处理系统")

# 初始化session状态
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'peaks' not in st.session_state:
    st.session_state.peaks = None

# 文件读取函数 (从您原有代码提取)
def getfromone(path, lines, much):
    numb = re.compile(r"-?\d+(?:\.\d+)?")
    ret = np.zeros((lines, much), dtype=float)
    with open(path) as f:
        con = 0
        for line in f:
            li = numb.findall(line)
            for i in range(lines):
                ret[i][con] = float(li[i])
            con += 1
    return ret

# 创建两列布局
col1, col2 = st.columns([1.2, 3])

with col1:
    # ===== 数据管理 =====
    with st.expander("📁 数据管理", expanded=True):
        # 波数文件上传
        wavenumber_file = st.file_uploader("上传波数文件", type=['txt'])
        
        # 光谱数据上传
        uploaded_file = st.file_uploader("上传光谱数据文件", type=['txt'])
        
        # 参数设置
        lines = st.number_input("光谱条数", min_value=1, value=1)
        much = st.number_input("每条光谱数据点数", min_value=1, value=2000)

        if uploaded_file and wavenumber_file:
            try:
                # 读取波数数据
                wavenumbers = np.loadtxt(wavenumber_file).ravel()
                
                # 读取光谱数据
                ret = getfromone(uploaded_file, lines, much)
                
                st.session_state.raw_data = (wavenumbers, ret.T)  # 转置为(点数, 光谱数)
                st.success(f"数据加载成功！{lines}条光谱，每条{much}个点")
                
            except Exception as e:
                st.error(f"文件加载失败: {str(e)}")

    # ===== 预处理设置 =====
    with st.expander("⚙️ 预处理设置", expanded=True):
        # 基线校准
        st.subheader("基线校准")
        baseline_method = st.selectbox(
            "基线校准方法",
            ["无", "SD", "FD", "I-ModPoly", "AsLS"],
            key="baseline_method"
        )

        # 动态参数
        if baseline_method == "I-ModPoly":
            polyorder = st.slider("多项式阶数", 3, 10, 6, key="polyorder")
        elif baseline_method == "AsLS":
            lam = st.number_input("λ(平滑度)", value=1e7, format="%e", key="lam")
            p = st.slider("p(不对称性)", 0.01, 0.5, 0.1, key="p")

        # ===== 数据变换 =====
        st.subheader("🧩 数据测试变换")
        transform_method = st.selectbox(
            "变换方法",
            ["无", "挤压函数(归一化版)", "挤压函数(原始版)", 
             "Sigmoid(归一化版)", "Sigmoid(原始版)"],
            key="transform_method",
            help="选择要应用的数据变换方法"
        )

        # 动态参数
        if "Sigmoid(归一化版)" in transform_method:
            maxn = st.slider("归一化系数", 1, 20, 10, 
                           help="控制归一化程度，值越大归一化效果越强")
        
        if "挤压函数(归一化版)" in transform_method:
            st.info("此方法会自动对数据进行归一化处理")

        # 归一化
        st.subheader("归一化")
        norm_method = st.selectbox(
            "归一化方法",
            ["无", "无穷大范数", "L10范数", "L4范数"],
            key="norm_method"
        )

        # 处理按钮
        if st.button("🚀 应用处理", type="primary", use_container_width=True):
            if st.session_state.raw_data is None:
                st.warning("请先上传数据文件")
            else:
                wavenumbers, y = st.session_state.raw_data
                y_processed = y.copy()
                method_name = []

                # 基线处理
                if baseline_method == "SD":
                    y_processed = D2(y_processed)
                    method_name.append("SD基线校准")
                elif baseline_method == "FD":
                    y_processed = D1(y_processed)
                    method_name.append("FD基线校准")
                elif baseline_method == "I-ModPoly":
                    y_processed = IModPoly(wavenumbers, y_processed, polyorder)
                    method_name.append(f"I-ModPoly(阶数={polyorder})")
                elif baseline_method == "AsLS":
                    y_processed = baseline_als(y_processed, lam, p, 10)
                    method_name.append(f"AsLS(λ={lam:.1e},p={p})")

                # 数据变换处理
                if transform_method == "挤压函数(归一化版)":
                    y_processed = i_squashing(y_processed)
                    method_name.append("i_squashing")
                elif transform_method == "挤压函数(原始版)":
                    y_processed = squashing(y_processed)
                    method_name.append("squashing")
                elif transform_method == "Sigmoid(归一化版)":
                    y_processed = i_sigmoid(y_processed, maxn)
                    method_name.append(f"i_sigmoid(maxn={maxn})")
                elif transform_method == "Sigmoid(原始版)":
                    y_processed = sigmoid(y_processed)
                    method_name.append("sigmoid")

                # 归一化处理
                if norm_method == "无穷大范数":
                    y_processed = LPnorm(y_processed, np.inf)
                    method_name.append("无穷大范数")
                elif norm_method == "L10范数":
                    y_processed = LPnorm(y_processed, 10)
                    method_name.append("L10范数")
                elif norm_method == "L4范数":
                    y_processed = LPnorm(y_processed, 4)
                    method_name.append("L4范数")

                st.session_state.processed_data = (wavenumbers, y_processed)
                st.session_state.process_method = " → ".join(method_name)
                st.success(f"处理完成: {st.session_state.process_method}")

with col2:
    # ===== 系统信息 =====
    if st.session_state.get('raw_data'):
        wavenumbers, y = st.session_state.raw_data
        cols = st.columns([1, 2])
        with cols[0]:
            st.info(f"📊 数据维度: {y.shape[1]}条光谱 × {y.shape[0]}点")
        with cols[1]:
            if st.session_state.get('process_method'):
                st.success(f"🛠️ 处理流程: {st.session_state.process_method}")
    
    st.divider()
    
    # ===== 光谱图 =====
    st.subheader("📈 光谱可视化")
    if st.session_state.get('raw_data'):
        wavenumbers, y = st.session_state.raw_data
        chart_data = pd.DataFrame(y, index=wavenumbers)
        
        if st.session_state.get('processed_data'):
            _, y_processed = st.session_state.processed_data
            chart_data = pd.DataFrame({
                "原始数据": y.mean(axis=1),
                "处理后数据": y_processed.mean(axis=1)
            }, index=wavenumbers)
        
        st.line_chart(chart_data)
    else:
        st.info("请先上传并处理数据")

    # ===== 结果导出 =====
    if st.session_state.get('processed_data'):
        st.subheader("💾 结果导出")
        export_name = st.text_input("导出文件名", "processed_spectra.txt")
        
        if st.button("导出处理结果", type="secondary"):
            wavenumbers, y_processed = st.session_state.processed_data
            with open(export_name, "w") as f:
                for line in y_processed.T:  # 转置回原始格式
                    f.write("\t".join(map(str, line)) + "\n")
            st.success(f"结果已导出到 {export_name}")

# 使用说明
with st.expander("ℹ️ 使用指南", expanded=False):
    st.markdown("""
    **标准操作流程:**
    1. 上传波数文件（单列文本）
    2. 上传光谱数据文件（多列文本）
    3. 设置光谱条数和数据点数
    4. 选择预处理方法
    5. 点击"应用处理"
    6. 导出结果

    **文件格式要求:**
    - 波数文件: 每行一个波数值
    - 光谱数据: 每列代表一条光谱，每行对应相同波数位置
    """)
