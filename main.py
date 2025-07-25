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
from scipy import sparse
from scipy.sparse.linalg import spsolve

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

# ===== 算法实现 =====
def polynomial_fit(wavenumbers, spectra, polyorder):
    """多项式拟合基线校正"""
    baseline = np.zeros_like(spectra)
    
    for i in range(spectra.shape[1]):
        # 使用 np.polyfit 拟合多项式基线
        coeffs = np.polyfit(wavenumbers, spectra[:, i], deg=polyorder)
        baseline[:, i] = np.polyval(coeffs, wavenumbers)
    
    return spectra - baseline  # 扣除基线

def modpoly(wavenumbers, spectra, k):
    """Modified Polynomial (ModPoly) 基线校正"""
    baseline = np.zeros_like(spectra)
    n_points = len(wavenumbers)
    
    for i in range(spectra.shape[1]):
        y = spectra[:, i].copy()
        
        # 迭代k次
        for _ in range(k):
            # 拟合多项式
            coeffs = np.polyfit(wavenumbers, y, deg=5)
            fitted = np.polyval(coeffs, wavenumbers)
            
            # 更新数据：只保留低于当前拟合线的点
            mask = y < fitted
            y[~mask] = fitted[~mask]
        
        baseline[:, i] = y
    
    return spectra - baseline

def imodpoly(wavenumbers, spectra, k):
    """Improved ModPoly (I-ModPoly) 基线校正"""
    baseline = np.zeros_like(spectra)
    n_points = len(wavenumbers)
    
    for i in range(spectra.shape[1]):
        y = spectra[:, i].copy()
        
        # 初始多项式拟合
        coeffs = np.polyfit(wavenumbers, y, deg=5)
        fitted = np.polyval(coeffs, wavenumbers)
        
        # 迭代k次
        for _ in range(k):
            # 找到低于当前拟合线的点
            mask = y < fitted
            
            # 仅使用这些点重新拟合
            coeffs = np.polyfit(wavenumbers[mask], y[mask], deg=5)
            fitted = np.polyval(coeffs, wavenumbers)
        
        baseline[:, i] = fitted
    
    return spectra - baseline

def pls(spectra, lam):
    """Penalized Least Squares (PLS) 基线校正"""
    n_points = spectra.shape[0]
    baseline = np.zeros_like(spectra)
    
    # 构建差分矩阵
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n_points, n_points-2))
    D = lam * D.dot(D.transpose())
    
    for i in range(spectra.shape[1]):
        y = spectra[:, i]
        # 求解 (I + D)c = y
        A = sparse.eye(n_points) + D
        baseline[:, i] = spsolve(A, y)
    
    return spectra - baseline

def asls(spectra, lam, p, max_iter=10):
    """Asymmetric Least Squares (AsLS) 基线校正"""
    n_points = spectra.shape[0]
    baseline = np.zeros_like(spectra)
    
    # 构建差分矩阵
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n_points, n_points-2))
    D = lam * D.dot(D.transpose())
    
    for i in range(spectra.shape[1]):
        y = spectra[:, i]
        w = np.ones(n_points)
        
        for _ in range(max_iter):
            # 加权最小二乘求解
            W = sparse.diags(w, 0)
            Z = W + D
            b = spsolve(Z, W * y)
            
            # 更新权重
            mask = y > b
            w[mask] = p
            w[~mask] = 1 - p
        
        baseline[:, i] = b
    
    return spectra - baseline

def airpls(spectra, lam, max_iter=15, threshold=0.001):
    """Adaptive Iteratively Reweighted Penalized Least Squares (airPLS) 基线校正"""
    n_points = spectra.shape[0]
    baseline = np.zeros_like(spectra)
    
    # 构建差分矩阵
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n_points, n_points-2))
    D = lam * D.dot(D.transpose())
    
    for i in range(spectra.shape[1]):
        y = spectra[:, i]
        w = np.ones(n_points)
        baseline_i = np.zeros(n_points)
        
        for j in range(max_iter):
            # 加权最小二乘求解
            W = sparse.diags(w, 0)
            Z = W + D
            b = spsolve(Z, W * y)
            
            # 计算残差
            d = y - b
            
            # 更新权重
            neg_mask = d < 0
            w[neg_mask] = np.exp(j * np.abs(d[neg_mask]) / np.std(d[neg_mask]))
            w[~neg_mask] = 0
            
            # 检查收敛
            if j > 0:
                diff = np.sum(np.abs(b - baseline_i)) / np.sum(np.abs(baseline_i))
                if diff < threshold:
                    break
            
            baseline_i = b
        
        baseline[:, i] = baseline_i
    
    return spectra - baseline

# ===== 数据变换函数 =====
def sigmoid(x):
    """原始Sigmoid函数"""
    return 1 / (1 + np.exp(-x))

def squashing(x):
    """原始挤压函数"""
    return x / np.sqrt(1 + x**2)

def i_sigmoid(x, maxn=10):
    """归一化版Sigmoid函数"""
    x_norm = x / maxn
    return sigmoid(x_norm)

def i_squashing(x):
    """归一化版挤压函数"""
    # 先归一化到[-1,1]
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
    return squashing(x_norm)

# ===== LP范数归一化 =====
def LPnorm(x, p):
    """计算Lp范数归一化"""
    norm = np.linalg.norm(x, ord=p, axis=0)
    norm[norm == 0] = 1  # 避免除零错误
    return x / norm

# ===== 预处理类 =====
class Preprocessor:
    def __init__(self):
        # 算法名称到处理函数的映射
        self.BASELINE_ALGORITHMS = {
            "SD": self._sd_baseline,  # 示例实现
            "FD": self._fd_baseline,  # 示例实现
            "多项式拟合": polynomial_fit,
            "ModPoly": modpoly,
            "I-ModPoly": imodpoly,
            "PLS": pls,
            "AsLS": asls,
            "airPLS": airpls,
        }
    
    def process(self, wavenumbers, data, 
                baseline_method="无", baseline_params=None,
                transform_method="无", transform_params=None,
                norm_method="无"):
        """执行完整的预处理流程"""
        if baseline_params is None:
            baseline_params = {}
        if transform_params is None:
            transform_params = {}
            
        y_processed = data.copy()
        method_name = []

        # 基线处理
        if baseline_method != "无":
            try:
                # 获取对应的算法函数
                algorithm_func = self.BASELINE_ALGORITHMS[baseline_method]
                
                # 根据算法类型传递参数
                if baseline_method in ["多项式拟合", "ModPoly", "I-ModPoly"]:
                    y_processed = algorithm_func(wavenumbers, y_processed, **baseline_params)
                elif baseline_method in ["PLS", "AsLS", "airPLS"]:
                    y_processed = algorithm_func(y_processed, **baseline_params)
                else:  # SD、FD 无额外参数
                    y_processed = algorithm_func(y_processed)
                    
                method_name.append(f"{baseline_method}({', '.join([f'{k}={v}' for k, v in baseline_params.items()])})")
            except Exception as e:
                raise ValueError(f"基线校正失败: {str(e)}")

        # 数据变换处理
        if transform_method != "无":
            try:
                if transform_method == "挤压函数(归一化版)":
                    y_processed = i_squashing(y_processed)
                    method_name.append("i_squashing")
                elif transform_method == "挤压函数(原始版)":
                    y_processed = squashing(y_processed)
                    method_name.append("squashing")
                elif transform_method == "Sigmoid(归一化版)":
                    maxn = transform_params.get("maxn", 10)
                    y_processed = i_sigmoid(y_processed, maxn)
                    method_name.append(f"i_sigmoid(maxn={maxn})")
                elif transform_method == "Sigmoid(原始版)":
                    y_processed = sigmoid(y_processed)
                    method_name.append("sigmoid")
            except Exception as e:
                raise ValueError(f"数据变换失败: {str(e)}")

        # 归一化处理
        if norm_method != "无":
            try:
                if norm_method == "无穷大范数":
                    y_processed = LPnorm(y_processed, np.inf)
                    method_name.append("无穷大范数")
                elif norm_method == "L10范数":
                    y_processed = LPnorm(y_processed, 10)
                    method_name.append("L10范数")
                elif norm_method == "L4范数":
                    y_processed = LPnorm(y_processed, 4)
                    method_name.append("L4范数")
            except Exception as e:
                raise ValueError(f"归一化失败: {str(e)}")

        return y_processed, method_name
    
    def _sd_baseline(self, spectra):
        """示例SD基线校正实现"""
        # 这里应该是实际的SD算法实现
        # 这里仅作示例
        return spectra - np.min(spectra, axis=0)
    
    def _fd_baseline(self, spectra):
        """示例FD基线校正实现"""
        # 这里应该是实际的FD算法实现
        # 这里仅作示例
        return spectra - np.percentile(spectra, 5, axis=0)

# ===== 文件处理类 =====
class FileHandler:
    def load_data(self, wavenumber_file, data_file, lines, much):
        """加载波数和光谱数据"""
        # 读取波数数据
        wavenumbers = np.loadtxt(wavenumber_file).ravel()
        
        # 读取光谱数据
        ret = self._getfromone(data_file, lines, much)
        
        return wavenumbers, ret.T  # 转置为(点数, 光谱数)
    
    def _getfromone(self, file, lines, much):
        """从文件中解析光谱数据"""
        numb = re.compile(r"-?\d+(?:\.\d+)?")
        ret = np.zeros((lines, much), dtype=float)
        
        # 读取文件内容
        content = file.getvalue().decode("utf-8")
        
        # 解析数据
        lines_list = content.splitlines()
        con = 0
        
        for line in lines_list:
            if con >= much:
                break
                
            li = numb.findall(line)
            for i in range(min(lines, len(li))):
                ret[i][con] = float(li[i])
            con += 1
            
        return ret
    
    def export_data(self, filename, data):
        """导出处理后的数据"""
        with open(filename, "w") as f:
            for line in data.T:  # 转置回原始格式
                f.write("\t".join(map(str, line)) + "\n")

# 创建处理器实例
file_handler = FileHandler()
preprocessor = Preprocessor()

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
                # 读取数据
                st.session_state.raw_data = file_handler.load_data(
                    wavenumber_file, uploaded_file, lines, much
                )
                st.success(f"数据加载成功！{lines}条光谱，每条{much}个点")
                
            except Exception as e:
                st.error(f"文件加载失败: {str(e)}")

    # ===== 预处理设置 =====
    with st.expander("⚙️ 预处理设置", expanded=True):
        # 基线校准
        st.subheader("基线校准")
        baseline_method = st.selectbox(
            "基线校准方法",
            ["无", "SD", "FD", "多项式拟合", "ModPoly", "I-ModPoly", "PLS", "AsLS", "airPLS"],
            key="baseline_method"
        )

        # 收集基线校准参数
        baseline_params = {}
        if baseline_method != "无":
            if baseline_method == "多项式拟合":
                polyorder = st.slider("多项式阶数 k", 3, 6, 5, key="polyorder_polyfit")
                baseline_params["polyorder"] = polyorder
            elif baseline_method == "ModPoly":
                k = st.slider("参数 k", 4, 10, 10, key="k_modpoly")
                baseline_params["k"] = k
            elif baseline_method == "I-ModPoly":
                k = st.slider("参数 k", 5, 9, 6, key="k_imodpoly")
                baseline_params["k"] = k
            elif baseline_method == "PLS":
                lam = st.selectbox("λ(平滑度)", [10**10, 10**8, 10**7], key="lam_pls")
                baseline_params["lam"] = lam
            elif baseline_method == "AsLS":
                p = st.selectbox("p(不对称性)", [0.2, 0.2, 0.1, 0.1], key="p_asls")
                lam = st.selectbox("λ(平滑度)", [10**9, 10**9, 10**9, 10**6], key="lam_asls")
                baseline_params["p"] = p
                baseline_params["lam"] = lam
            elif baseline_method == "airPLS":
                lam = st.selectbox("λ(平滑度)", [10**7, 10**4, 10**2], key="lam_airpls")
                baseline_params["lam"] = lam

        # ===== 数据变换 =====
        st.subheader("🧩 数据测试变换")
        transform_method = st.selectbox(
            "变换方法",
            ["无", "挤压函数(归一化版)", "挤压函数(原始版)", 
             "Sigmoid(归一化版)", "Sigmoid(原始版)"],
            key="transform_method"
        )

        # 动态参数
        transform_params = {}
        if "Sigmoid(归一化版)" in transform_method:
            maxn = st.slider("归一化系数", 1, 20, 10)
            transform_params["maxn"] = maxn

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
                try:
                    wavenumbers, y = st.session_state.raw_data
                    
                    # 执行预处理
                    processed_data, method_name = preprocessor.process(
                        wavenumbers, y, 
                        baseline_method=baseline_method,
                        baseline_params=baseline_params,
                        transform_method=transform_method,
                        transform_params=transform_params,
                        norm_method=norm_method
                    )
                    
                    st.session_state.processed_data = (wavenumbers, processed_data)
                    st.session_state.process_method = " → ".join(method_name)
                    st.success(f"处理完成: {st.session_state.process_method}")
                except Exception as e:
                    st.error(f"处理失败: {str(e)}")

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
            try:
                wavenumbers, y_processed = st.session_state.processed_data
                file_handler.export_data(export_name, y_processed)
                st.success(f"结果已导出到 {export_name}")
            except Exception as e:
                st.error(f"导出失败: {str(e)}")

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
