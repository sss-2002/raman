import streamlit as st
import numpy as np
import pandas as pd
import re
import itertools
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
from scipy.signal import savgol_filter, medfilt
from scipy.fft import fft, ifft
from statsmodels.nonparametric.smoothers_lowess import lowess
import pywt
from DTW import DTW


def main():
    # 最优先初始化session state
    if 'show_arrangements' not in st.session_state:
        st.session_state.show_arrangements = False
    
    # 初始化其他必要的session状态变量
    other_states = {
        'raw_data': None,
        'processed_data': None,
        'peaks': None,
        'train_test_split_ratio': 0.8,
        'arrangement_results': [],
        'selected_arrangement': None,
        'arrangement_details': {},
        'algorithm_permutations': [],  # 存储65种算法排列组合
        'current_algorithms': {},       # 存储当前选择的算法
        'filtered_perms': [],           # 存储筛选后的排列方案
        'selected_perm_idx': 0          # 存储当前选中的排列索引
    }
    for key, value in other_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # 设置页面
    st.set_page_config(layout="wide", page_icon="🔬", page_title="排列预处理模型")
    st.title("🌌 排列预处理模型")
     
     
    # ===== 算法实现 =====
    def polynomial_fit(wavenumbers, spectra, polyorder):
        """多项式拟合基线校正"""
        baseline = np.zeros_like(spectra)
        for i in range(spectra.shape[1]):
            coeffs = np.polyfit(wavenumbers, spectra[:, i], deg=polyorder)
            baseline[:, i] = np.polyval(coeffs, wavenumbers)
        return spectra - baseline  # 扣除基线
    
    def modpoly(wavenumbers, spectra, k):
        """Modified Polynomial (ModPoly) 基线校正"""
        baseline = np.zeros_like(spectra)
        n_points = len(wavenumbers)
        for i in range(spectra.shape[1]):
            y = spectra[:, i].copy()
            for _ in range(k):
                coeffs = np.polyfit(wavenumbers, y, deg=5)
                fitted = np.polyval(coeffs, wavenumbers)
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
            coeffs = np.polyfit(wavenumbers, y, deg=5)
            fitted = np.polyval(coeffs, wavenumbers)
            for _ in range(k):
                mask = y < fitted
                coeffs = np.polyfit(wavenumbers[mask], y[mask], deg=5)
                fitted = np.polyval(coeffs, wavenumbers)
            baseline[:, i] = fitted
        return spectra - baseline
    
    def pls(spectra, lam):
        """Penalized Least Squares (PLS) 基线校正"""
        n_points = spectra.shape[0]
        baseline = np.zeros_like(spectra)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n_points, n_points-2))
        D = lam * D.dot(D.transpose())
        for i in range(spectra.shape[1]):
            y = spectra[:, i]
            A = sparse.eye(n_points) + D
            baseline[:, i] = spsolve(A, y)
        return spectra - baseline
    
    def asls(spectra, lam, p, max_iter=10):
        """Asymmetric Least Squares (AsLS) 基线校正"""
        n_points = spectra.shape[0]
        baseline = np.zeros_like(spectra)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n_points, n_points-2))
        D = lam * D.dot(D.transpose())
        for i in range(spectra.shape[1]):
            y = spectra[:, i]
            w = np.ones(n_points)
            for _ in range(max_iter):
                W = sparse.diags(w, 0)
                Z = W + D
                b = spsolve(Z, W * y)
                mask = y > b
                w[mask] = p
                w[~mask] = 1 - p
            baseline[:, i] = b
        return spectra - baseline
    
    def airpls(spectra, lam, max_iter=15, threshold=0.001):
        """Adaptive Iteratively Reweighted Penalized Least Squares (airPLS) 基线校正"""
        n_points = spectra.shape[0]
        baseline = np.zeros_like(spectra)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n_points, n_points-2))
        D = lam * D.dot(D.transpose())
        for i in range(spectra.shape[1]):
            y = spectra[:, i]
            w = np.ones(n_points)
            baseline_i = np.zeros(n_points)
            for j in range(max_iter):
                W = sparse.diags(w, 0)
                Z = W + D
                b = spsolve(Z, W * y)
                d = y - b
                neg_mask = d < 0
                w[neg_mask] = np.exp(j * np.abs(d[neg_mask]) / np.std(d[neg_mask]))
                w[~neg_mask] = 0
                if j > 0:
                    diff = np.sum(np.abs(b - baseline_i)) / np.sum(np.abs(baseline_i))
                    if diff < threshold:
                        break
                baseline_i = b
            baseline[:, i] = baseline_i
        return spectra - baseline
    
    def dtw_squashing(x, l, k1, k2):
        """动态时间规整(DTW)挤压算法"""
        n_samples, n_features = x.shape
        result = np.zeros_like(x)
        reference = np.mean(x, axis=1)  # 使用平均光谱作为参考
        for i in range(n_features):
            spectrum = x[:, i]
            path, cost = dtw_path(reference, spectrum)
            squashed = np.zeros_like(spectrum)
            for ref_idx, spec_idx in path:
                squashed[ref_idx] += spectrum[spec_idx]
            unique_ref_indices = np.unique([p[0] for p in path])
            for idx in unique_ref_indices:
                count = sum(1 for p in path if p[0] == idx)
                squashed[idx] /= count
            if k1 == "T":
                max_slope = l
                for j in range(1, len(path)):
                    ref_diff = path[j][0] - path[j-1][0]
                    spec_diff = path[j][1] - path[j-1][1]
                    if ref_diff != 0:
                        slope = abs(spec_diff / ref_diff)
                        if slope > max_slope:
                            squashed[path[j][0]] = (squashed[path[j][0]] + squashed[path[j-1][0]]) / 2
            if k2 == "T":
                ref_map_count = {}
                for ref_idx, _ in path:
                    ref_map_count[ref_idx] = ref_map_count.get(ref_idx, 0) + 1
                for ref_idx, count in ref_map_count.items():
                    if count > l:
                        window = min(l, len(spectrum))
                        start = max(0, ref_idx - window//2)
                        end = min(n_samples, ref_idx + window//2 + 1)
                        squashed[ref_idx] = np.mean(spectrum[start:end])
            if l > 1:
                for j in range(n_samples):
                    start = max(0, j - l)
                    end = min(n_samples, j + l + 1)
                    squashed[j] = np.mean(squashed[start:end])
            result[:, i] = squashed
        return result
    
    # 修复：确保所有排列都有first_step_type属性
    def generate_65_permutations(algorithms):
        """
        生成完整的65种算法排列组合，并确保每种排列都有first_step_type属性
        """
        # 为四种算法分配编号1-4
        algorithm_list = [
            (1, "基线校准", algorithms['baseline']),
            (2, "缩放", algorithms['scaling']),
            (3, "滤波", algorithms['filtering']),
            (4, "挤压", algorithms['squashing'])
        ]
        
        all_permutations = []
        
        # 0. 添加"无预处理（原始光谱）"选项（1种）
        all_permutations.append([])  # 空列表表示不使用任何算法
        
        # 1. 生成使用1种算法的排列 (4种)
        for algo in algorithm_list:
            if algo[2] != "无":  # 只包含已选择的算法
                all_permutations.append([algo])
        
        # 2. 生成使用2种算法的排列 (P(4,2)=12种)
        for perm in itertools.permutations(algorithm_list, 2):
            # 确保两种算法都已选择
            if perm[0][2] != "无" and perm[1][2] != "无":
                all_permutations.append(list(perm))
        
        # 3. 生成使用3种算法的排列 (P(4,3)=24种)
        for perm in itertools.permutations(algorithm_list, 3):
            # 确保三种算法都已选择
            if perm[0][2] != "无" and perm[1][2] != "无" and perm[2][2] != "无":
                all_permutations.append(list(perm))
        
        # 4. 生成使用4种算法的排列 (P(4,4)=24种)
        for perm in itertools.permutations(algorithm_list, 4):
            # 确保四种算法都已选择
            if (perm[0][2] != "无" and perm[1][2] != "无" and 
                perm[2][2] != "无" and perm[3][2] != "无"):
                all_permutations.append(list(perm))
        
        # 格式化排列结果，确保每种排列都有first_step_type
        formatted_perms = []
        for i, perm in enumerate(all_permutations):
            # 初始化默认值，确保属性存在
            perm_dict = {
                "name": f"排列方案 {i+1}",
                "order": [],
                "details": perm,
                "count": len(perm),
                "first_step_type": "未知"  # 默认值，确保属性存在
            }
            
            if not perm:  # 无预处理情况
                perm_dict["name"] = f"排列方案 {i+1}: 无预处理（原始光谱）"
                perm_dict["first_step_type"] = "无预处理"
            else:
                # 获取第一步算法的类型名称
                first_step_type = perm[0][1] if perm and len(perm) > 0 else "未知"
                perm_dict["first_step_type"] = first_step_type
                
                # 生成排列名称
                perm_details = []
                for step in perm:
                    perm_details.append(f"{step[0]}.{step[1]}({step[2]})")
                perm_dict["name"] = f"排列方案 {i+1}: " + " → ".join(perm_details)
                perm_dict["order"] = [step[0] for step in perm]
            
            formatted_perms.append(perm_dict)
        
        return formatted_perms
    
    
    # ===== 数据变换函数 =====
    def sigmoid_func(x):
        return 1 / (1 + np.exp(-x))
    
    def squashing_func(x):
        return x / np.sqrt(1 + x**2)
    
    def i_sigmoid_func(x, maxn=10):
        x_norm = x / maxn
        return sigmoid_func(x_norm)
    
    def i_squashing_func(x):
        x_min = np.min(x, axis=0)
        x_max = np.max(x, axis=0)
        x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
        return squashing_func(x_norm)
    
    # ===== LP范数归一化 =====
    def LPnorm(x, p):
        norm = np.linalg.norm(x, ord=p, axis=0)
        norm[norm == 0] = 1  # 避免除零错误
        return x / norm
    
    # ===== 预处理类 =====
    class Preprocessor:
        def __init__(self):
            self.BASELINE_ALGORITHMS = {
                "SD": self._sd_baseline,
                "FD": self._fd_baseline,
                "多项式拟合": polynomial_fit,
                "ModPoly": modpoly,
                "I-ModPoly": imodpoly,
                "PLS": pls,
                "AsLS": asls,
                "airPLS": airpls,
            }
            self.FILTERING_ALGORITHMS = {
                "Savitzky-Golay": self.savitzky_golay,
                "中值滤波(MF)": self.median_filter,
                "移动平均(MAF)": self.moving_average,
                "Lowess": self.lowess_filter,
                "FFT": self.fft_filter,
                "小波变换(DWT)": self.wavelet_filter
            }
            
            self.SCALING_ALGORITHMS = {
                "Peak-Norm": self.peak_norm,
                "SNV": self.snv,
                "MSC": self.msc,
                "M-M-Norm": self.mm_norm,
                "L-范数": self.l_norm
            }
            
            self.SQUASHING_ALGORITHMS = {
                "Sigmoid挤压": sigmoid,
                "改进的Sigmoid挤压": i_sigmoid,
                "逻辑函数": squashing,
                "改进的逻辑函数": i_squashing,
                "DTW挤压": dtw_squashing
            }
    
        def process(self, wavenumbers, data, 
                    baseline_method="无", baseline_params=None,
                    squashing_method="无", squashing_params=None,
                    filtering_method="无", filtering_params=None,
                    scaling_method="无", scaling_params=None,
                    algorithm_order=None):
            """执行预处理流程，支持指定算法顺序，空顺序表示返回原始数据"""
            if baseline_params is None: baseline_params = {}
            if squashing_params is None: squashing_params = {}
            if filtering_params is None: filtering_params = {}
            if scaling_params is None: scaling_params = {}
                
            # 如果算法顺序为空（无预处理），直接返回原始数据
            if algorithm_order is not None and len(algorithm_order) == 0:
                return data.copy(), ["无预处理（原始光谱）"]
                
            y_processed = data.copy()
            method_name = []
            
            # 如果指定了算法顺序，则按顺序执行
            if algorithm_order is not None and len(algorithm_order) > 0:
                # 根据算法编号映射到对应的处理步骤
                step_mapping = {
                    1: ("baseline", baseline_method, baseline_params),
                    2: ("scaling", scaling_method, scaling_params),
                    3: ("filtering", filtering_method, filtering_params),
                    4: ("squashing", squashing_method, squashing_params)
                }
                # 按指定顺序创建步骤列表
                steps = [step_mapping[order] for order in algorithm_order]
            else:
                # 默认顺序：基线 → 挤压 → 滤波 → 缩放（只执行已选择的方法）
                steps = []
                if baseline_method != "无":
                    steps.append(("baseline", baseline_method, baseline_params))
                if squashing_method != "无":
                    steps.append(("squashing", squashing_method, squashing_params))
                if filtering_method != "无":
                    steps.append(("filtering", filtering_method, filtering_params))
                if scaling_method != "无":
                    steps.append(("scaling", scaling_method, scaling_params))
    
            # 按顺序执行预处理步骤
            for step_type, method, params in steps:
                if method == "无":
                    continue
                    
                try:
                    if step_type == "baseline":
                        algorithm_func = self.BASELINE_ALGORITHMS[method]
                        if method in ["多项式拟合", "ModPoly", "I-ModPoly"]:
                            y_processed = algorithm_func(wavenumbers, y_processed, **params)
                        elif method in ["PLS", "AsLS", "airPLS"]:
                            y_processed = algorithm_func(y_processed,** params)
                        else:  # SD、FD 无额外参数
                            y_processed = algorithm_func(y_processed)
                        method_name.append(f"{method}({', '.join([f'{k}={v}' for k, v in params.items()])})")
                            
                    elif step_type == "squashing":
                        algorithm_func = self.SQUASHING_ALGORITHMS[method]
                        if method == "改进的Sigmoid挤压":
                            y_processed = algorithm_func(y_processed, maxn=10)
                            method_name.append(f"{method}(maxn=10)")
                        elif method == "改进的逻辑函数":
                            m = params.get("m", 10)
                            y_processed = algorithm_func(y_processed)
                            method_name.append(f"{method}(m={m})")
                        elif method == "DTW挤压":
                            l = params.get("l", 1)
                            k1 = params.get("k1", "T")
                            k2 = params.get("k2", "T")
                            y_processed = algorithm_func(y_processed, l=l, k1=k1, k2=k2)
                            method_name.append(f"DTW挤压(l={l}, k1={k1}, k2={k2})")
                        else:
                            y_processed = algorithm_func(y_processed)
                            method_name.append(method)
                            
                    elif step_type == "filtering":
                        algorithm_func = self.FILTERING_ALGORITHMS[method]
                        y_processed = algorithm_func(y_processed, **params)
                        params_str = ', '.join([f'{k}={v}' for k, v in params.items()])
                        method_name.append(f"{method}({params_str})")
                        
                    elif step_type == "scaling":
                        algorithm_func = self.SCALING_ALGORITHMS[method]
                        y_processed = algorithm_func(y_processed,** params)
                        params_str = ', '.join([f'{k}={v}' for k, v in params.items()])
                        method_name.append(f"{method}({params_str})")
                        
                except Exception as e:
                    raise ValueError(f"{step_type}处理失败: {str(e)}")
    
            return y_processed, method_name
         
        def _sd_baseline(self, spectra):
            return spectra - np.min(spectra, axis=0)
        
        def _fd_baseline(self, spectra):
            return spectra - np.percentile(spectra, 5, axis=0)
    
        # ===== 滤波算法实现 =====
        def savitzky_golay(self, spectra, k, w):
            return savgol_filter(spectra, w, k, axis=0)
        
        def median_filter(self, spectra, k, w):
            return medfilt(spectra, kernel_size=(w, 1))
        
        def moving_average(self, spectra, k, w):
            kernel = np.ones(w) / w
            return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, spectra)
        
        def lowess_filter(self, spectra, frac):
            result = np.zeros_like(spectra)
            for i in range(spectra.shape[1]):
                smoothed = lowess(spectra[:, i], np.arange(len(spectra)), frac=frac, it=0)
                result[:, i] = smoothed[:, 1]
            return result
        
        def fft_filter(self, spectra, cutoff):
            fft_result = fft(spectra, axis=0)
            frequencies = np.fft.fftfreq(spectra.shape[0])
            filter_mask = np.abs(frequencies) < cutoff
            fft_result[~filter_mask, :] = 0
            return np.real(ifft(fft_result, axis=0))
        
        def wavelet_filter(self, spectra, threshold):
            coeffs = pywt.wavedec(spectra, 'db4', axis=0)
            coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
            return pywt.waverec(coeffs, 'db4', axis=0)
        
        # ===== 缩放算法实现 =====
        def peak_norm(self, spectra):
            return spectra / np.max(spectra, axis=0)
        
        def snv(self, spectra):
            mean = np.mean(spectra, axis=0)
            std = np.std(spectra, axis=0)
            return (spectra - mean) / std
        
        def msc(self, spectra):
            mean_spectrum = np.mean(spectra, axis=1)
            return np.apply_along_axis(lambda x: np.polyval(np.polyfit(mean_spectrum, x, 1), mean_spectrum), 0, spectra)
        
        def mm_norm(self, spectra):
            min_vals = np.min(spectra, axis=0)
            max_vals = np.max(spectra, axis=0)
            return (spectra - min_vals) / (max_vals - min_vals)
        
        def l_norm(self, spectra, p):
            if p == "无穷大":
                return spectra / np.max(np.abs(spectra), axis=0)
            else:
                p_val = float(p)
                norm = np.linalg.norm(spectra, ord=p_val, axis=0)
                norm[norm == 0] = 1  # 避免除零错误
                return spectra / norm
    
    # ===== 文件处理类 =====
    class FileHandler:
        def load_data(self, wavenumber_file, data_file, lines, much):
            wavenumbers = np.loadtxt(wavenumber_file).ravel()
            return wavenumbers, self._getfromone(data_file, lines, much).T 
        
        def _getfromone(self, file, lines, much):
            numb = re.compile(r"-?\d+(?:\.\d+)?")
            ret = np.zeros((lines, much), dtype=float)
            content = file.getvalue().decode("utf-8")
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
            with open(filename, "w") as f:
                for line in data.T:  # 转置回原始格式
                    f.write("\t".join(map(str, line)) + "\n")
    
    # 创建处理器实例
    file_handler = FileHandler()
    preprocessor = Preprocessor()
    
    # 创建三列布局
    col_left, col_mid, col_right = st.columns([1.5, 2.5, 1.2])
    
    # ===== 左侧：数据管理 =====
    with col_left:
        with st.expander("📁 数据管理", expanded=True):
            wavenumber_file = st.file_uploader("上传波数文件", type=['txt'])
            uploaded_file = st.file_uploader("上传光谱数据文件", type=['txt'])
            
            lines = st.number_input("光谱条数", min_value=1, value=1)
            much = st.number_input("每条光谱数据点数", min_value=1, value=2000)

            train_test_ratio = st.slider(
               "训练集测试集划分比例",
               min_value=0.1,
               max_value=0.9,
               value=0.8,
               step=0.1,
               format="%.1f"
            )
            st.session_state.train_test_split_ratio = train_test_ratio
    
            if uploaded_file and wavenumber_file:
                try:
                    st.session_state.raw_data = file_handler.load_data(
                        wavenumber_file, uploaded_file, lines, much
                    )
                    st.success(f"数据加载成功！{lines}条光谱，每条{much}个点")
                except Exception as e:
                    st.error(f"文件加载失败: {str(e)}")
        
        # 系统信息
        if st.session_state.get('raw_data'):
            wavenumbers, y = st.session_state.raw_data
            st.info(f"📊 数据维度: {y.shape[1]}条光谱 × {y.shape[0]}点")
            st.info(f"🔢 训练集比例: {st.session_state.train_test_split_ratio:.1f}，测试集比例: {1 - st.session_state.train_test_split_ratio:.1f}")
            if st.session_state.get('process_method'):
                st.success(f"🛠️ 处理流程: {st.session_state.process_method}")
        
        # 使用说明
        with st.expander("ℹ️ 使用指南", expanded=False):
            st.markdown("""
           **标准操作流程:**
           1. 上传波数文件和光谱数据文件
           2. 设置光谱参数和训练集比例
           3. 在右侧选择预处理方法（可全不选）
           4. 点击"显示排列"按钮，系统会生成65种算法排列组合（含原始光谱）
           5. 在右侧选择一种排列方案并应用
           6. 查看结果并导出
           """)
     
    # ===== 中间：光谱可视化与结果导出 =====
    with col_mid:
        st.subheader("📈 光谱可视化")
        if st.session_state.get('raw_data'):
            wavenumbers, y = st.session_state.raw_data
            # 原始光谱展示
            st.subheader("原始光谱")
            st.caption("(随机显示一条)")
            random_idx = np.random.randint(0, y.shape[1]) if y.shape[1] > 0 else 0
            raw_chart_data = pd.DataFrame({
               "原始光谱": y[:, random_idx]
            }, index=wavenumbers)
            st.line_chart(raw_chart_data)
            
            # 处理结果展示
            if st.session_state.get('selected_arrangement'):
                st.subheader("🔍 处理结果")
                selected_arr = st.session_state.selected_arrangement
                arr_data = st.session_state.arrangement_details[selected_arr]['data']
                arr_method = st.session_state.arrangement_details[selected_arr]['method']
                arr_order = st.session_state.arrangement_details[selected_arr].get('order', [])
                
                st.caption(f"处理方法: {arr_method}")
                st.caption(f"执行顺序: {arr_order if arr_order else '无预处理'}")
                
                # 预处理后的光谱展示
                st.subheader("预处理后的光谱")
                processed_chart_data = pd.DataFrame({
                    "预处理后光谱": arr_data[:, random_idx]
                }, index=wavenumbers)
                st.line_chart(processed_chart_data)
                
                # k值曲线展示（无预处理时不显示）
                if arr_order:  # 只有使用了算法才显示k值曲线
                    st.subheader("k值曲线")
                    k_vals = np.abs(arr_data[:, random_idx] / (y[:, random_idx] + 1e-8))
                    k_chart_data = pd.DataFrame({
                        "k值": k_vals
                    }, index=wavenumbers)
                    st.line_chart(k_chart_data)
                else:
                    st.info("无预处理（原始光谱），不显示k值曲线")
                
                # 原始与处理后对比图
                st.subheader("原始与处理后对比")
                compare_data = pd.DataFrame({
                    "原始光谱": y[:, random_idx],
                    "预处理后光谱": arr_data[:, random_idx]
                }, index=wavenumbers)
                st.line_chart(compare_data)
            elif st.session_state.get('processed_data'):
                # 显示最新处理结果
                _, y_processed = st.session_state.processed_data
                st.subheader("预处理后的光谱")
                processed_chart_data = pd.DataFrame({
                    "预处理后光谱": y_processed[:, random_idx]
                }, index=wavenumbers)
                st.line_chart(processed_chart_data)
            else:
                st.info("请在右侧设置预处理参数并点击'应用处理'或'推荐应用'，或选择排列方案并应用")
            
            # 结果导出
            if st.session_state.arrangement_results or st.session_state.get('processed_data'):
                st.subheader("💾 结果导出")
                export_name = st.text_input("导出文件名", "processed_spectra.txt")
                
                if st.button("导出处理结果", type="secondary"):
                    try:
                        if st.session_state.selected_arrangement:
                            arr_data = st.session_state.arrangement_details[st.session_state.selected_arrangement]['data']
                            file_handler.export_data(export_name, arr_data)
                        else:
                            wavenumbers, y_processed = st.session_state.processed_data
                            file_handler.export_data(export_name, y_processed)
                        st.success(f"结果已导出到 {export_name}")
                    except Exception as e:
                        st.error(f"导出失败: {str(e)}")
        else:
            st.info("请先在左侧上传数据")

    
    # ===== 右侧：预处理设置 + 排列方案选择 =====
    with col_right:
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
                    p = st.selectbox("p(不对称性)", [0.2, 0.1], key="p_asls")
                    lam = st.selectbox("λ(平滑度)", [10**9, 10**6], key="lam_asls")
                    baseline_params["p"] = p
                    baseline_params["lam"] = lam
                elif baseline_method == "airPLS":
                    lam = st.selectbox("λ(平滑度)", [10**7, 10**4, 10**2], key="lam_airpls")
                    baseline_params["lam"] = lam
    
            # 缩放处理
            st.subheader("📏 缩放")
            scaling_method = st.selectbox(
                "缩放方法",
                ["无", "Peak-Norm", "SNV", "MSC", "M-M-Norm", "L-范数"],
                key="scaling_method"
            )
    
            # 缩放参数
            scaling_params = {}
            if scaling_method == "L-范数":
                p = st.selectbox("范数阶数(p)", ["无穷大", "4", "10"], key="p_scaling")
                scaling_params["p"] = p
    
            # 滤波处理
            st.subheader("📶 滤波")
            filtering_method = st.selectbox(
                "滤波方法",
                ["无", "Savitzky-Golay", "中值滤波(MF)", "移动平均(MAF)", "Lowess", "FFT", "小波变换(DWT)"],
                key="filtering_method"
            )
    
            # 滤波参数
            filtering_params = {}
            if filtering_method != "无":
                if filtering_method == "Savitzky-Golay":
                    k = st.selectbox("阶数(k)", [3, 7], key="k_sg")
                    w = st.selectbox("窗口大小(w)", [11, 31, 51], key="w_sg")
                    filtering_params["k"] = k
                    filtering_params["w"] = w
                elif filtering_method in ["中值滤波(MF)", "移动平均(MAF)"]:
                    k = st.selectbox("核大小(k)", [1, 3], key="k_mf")
                    w = st.selectbox("窗口大小(w)", [7, 11], key="w_mf")
                    filtering_params["k"] = k
                    filtering_params["w"] = w
                elif filtering_method == "Lowess":
                    frac = st.selectbox("平滑系数", [0.01, 0.03], key="frac_lowess")
                    filtering_params["frac"] = frac
                elif filtering_method == "FFT":
                    cutoff = st.selectbox("截止频率", [30, 50, 90], key="cutoff_fft")
                    filtering_params["cutoff"] = cutoff
                elif filtering_method == "小波变换(DWT)":
                    threshold = st.selectbox("阈值", [0.1, 0.3, 0.5], key="threshold_dwt")
                    filtering_params["threshold"] = threshold

            # 挤压处理
            st.subheader("🧪 挤压")
            squashing_method = st.selectbox(
                "挤压方法",
                ["无", "Sigmoid挤压", "改进的Sigmoid挤压", "逻辑函数", "改进的逻辑函数", "DTW挤压"],
                key="squashing_method"
            )
    
            # 挤压参数
            squashing_params = {}
            if squashing_method != "无":
                if squashing_method == "改进的逻辑函数":
                    m = st.selectbox("参数m", [10, 20], key="m_improved_squash")
                    squashing_params["m"] = m
                    st.info(f"使用参数: m={m}")
                elif squashing_method == "DTW挤压":
                    l = st.selectbox("参数l", [1, 5], key="l_dtw")
                    k1 = st.selectbox("参数k1", ["T", "F"], key="k1_dtw")
                    k2 = st.selectbox("参数k2", ["T", "F"], key="k2_dtw")
                    squashing_params["l"] = l
                    squashing_params["k1"] = k1
                    squashing_params["k2"] = k2
                    st.info(f"使用参数: l={l}, k1={k1}, k2={k2}")
                elif squashing_method == "改进的Sigmoid挤压":
                    st.info("使用默认参数: maxn=10")
    
            
            # 保存当前选择的算法
            current_algorithms = {
                'baseline_method': baseline_method,
                'baseline_params': baseline_params,
                'scaling_method': scaling_method,
                'scaling_params': scaling_params,
                'filtering_method': filtering_method,
                'filtering_params': filtering_params,
                'squashing_method': squashing_method,
                'squashing_params': squashing_params
            }
            st.session_state.current_algorithms = current_algorithms
            
            # 应用处理与推荐应用按钮
            col_buttons = st.columns(2)
            with col_buttons[0]:
                if st.button("🚀 应用处理", type="primary", use_container_width=True):
                    if st.session_state.raw_data is None:
                        st.warning("请先在左侧上传数据文件")
                    else:
                        try:
                            wavenumbers, y = st.session_state.raw_data
                            processed_data, method_name = preprocessor.process(
                                wavenumbers, y, 
                                baseline_method=baseline_method,
                                baseline_params=baseline_params,
                                squashing_method=squashing_method,
                                squashing_params=squashing_params,
                                filtering_method=filtering_method,
                                filtering_params=filtering_params,
                                scaling_method=scaling_method,
                                scaling_params=scaling_params
                            )
                            
                            arr_name = f"排列_{len(st.session_state.arrangement_results) + 1}"
                            st.session_state.arrangement_results.append(arr_name)
                            st.session_state.arrangement_details[arr_name] = {
                                'data': processed_data,
                                'method': " → ".join(method_name),
                                'params': current_algorithms
                            }
                            st.session_state.selected_arrangement = arr_name
                            st.session_state.processed_data = (wavenumbers, processed_data)
                            st.session_state.process_method = " → ".join(method_name)
                            st.success(f"处理完成: {st.session_state.process_method}")
                        except Exception as e:
                            st.error(f"处理失败: {str(e)}")
        
            with col_buttons[1]:
                if st.button("🌟 推荐应用", type="primary", use_container_width=True):
                    if st.session_state.raw_data is None:
                        st.warning("请先在左侧上传数据文件")
                    else:
                        try:
                            wavenumbers, y = st.session_state.raw_data
                            recommended_params = {
                                'baseline_method': "airPLS",
                                'baseline_params': {'lam': 10**4},
                                'scaling_method': "SNV",
                                'scaling_params': {},
                                'filtering_method': "Savitzky-Golay",
                                'filtering_params': {'k': 3, 'w': 11},
                                'squashing_method': "改进的Sigmoid挤压",
                                'squashing_params': {}
                            }
                            
                            processed_data, method_name = preprocessor.process(
                                wavenumbers, y,** recommended_params
                            )
                            
                            arr_name = f"推荐排列_{len(st.session_state.arrangement_results) + 1}"
                            st.session_state.arrangement_results.append(arr_name)
                            st.session_state.arrangement_details[arr_name] = {
                                'data': processed_data,
                                'method': " → ".join(method_name),
                                'params': recommended_params
                            }
                            st.session_state.selected_arrangement = arr_name
                            st.session_state.processed_data = (wavenumbers, processed_data)
                            st.session_state.process_method = " → ".join(method_name)
                            st.success(f"推荐处理完成: {st.session_state.process_method}")
                        except Exception as e:
                            st.error(f"推荐处理失败: {str(e)}")
        
            # 显示排列按钮
            if st.button("🔍 显示排列", type="secondary", use_container_width=True):
                # 切换显示状态
                st.session_state.show_arrangements = not st.session_state.show_arrangements
                
                # 生成65种排列组合（包含无预处理选项）
                if st.session_state.show_arrangements:
                    # 收集所有算法状态（包括"无"选项）
                    selected_algorithms = {
                        'baseline': baseline_method,
                        'scaling': scaling_method,
                        'filtering': filtering_method,
                        'squashing': squashing_method
                    }
                    
                    # 生成包含原始光谱的65种排列
                    st.session_state.algorithm_permutations = generate_65_permutations(selected_algorithms)
                    # 初始化筛选结果（默认显示全部）
                    st.session_state.filtered_perms = st.session_state.algorithm_permutations
                    st.success(f"已生成{len(st.session_state.algorithm_permutations)}种算法排列组合（含原始光谱）！")
                else:
                    # 隐藏排列时清空筛选结果
                    st.session_state.filtered_perms = []
                
                # 刷新页面以更新布局
                st.experimental_rerun()
            
            # 显示排列方案（仅当show_arrangements为True且有排列数据时）
            if st.session_state.show_arrangements and st.session_state.algorithm_permutations:
                st.subheader("🔄 算法排列方案")
                
                # 修复：安全获取所有第一步算法类型
                try:
                    # 使用集合推导式获取所有第一步类型，并处理可能的缺失值
                    all_first_step_types = list({
                        perm.get("first_step_type", "未知") 
                        for perm in st.session_state.algorithm_permutations
                    })
                    # 排序使显示更一致
                    all_first_step_types.sort()
                except Exception as e:
                    st.error(f"获取排列类型时出错: {str(e)}")
                    all_first_step_types = ["全部", "无预处理", "基线校准", "缩放", "滤波", "挤压"]
                
                selected_first_step = st.selectbox(
                    "选择第一步算法类型",
                    ["全部"] + all_first_step_types,  # 选项：全部 + 所有第一步类型
                    key="first_step_filter"
                )
                
                # 根据选择的第一步算法类型筛选排列
                if selected_first_step == "全部":
                    st.session_state.filtered_perms = st.session_state.algorithm_permutations
                else:
                    # 修复：使用get方法安全访问属性
                    st.session_state.filtered_perms = [
                        p for p in st.session_state.algorithm_permutations 
                        if p.get("first_step_type") == selected_first_step
                    ]
                
                # 排列方案下拉框
                if st.session_state.filtered_perms:
                    st.session_state.selected_perm_idx = st.selectbox(
                        f"选择预处理算法顺序（共{len(st.session_state.filtered_perms)}种）",
                        range(len(st.session_state.filtered_perms)),
                        format_func=lambda x: st.session_state.filtered_perms[x].get("name", f"排列方案 {x+1}"),
                        key="perm_select_box"
                    )
                    
                    # 显示当前选中的排列详情
                    try:
                        selected_perm = st.session_state.filtered_perms[st.session_state.selected_perm_idx]
                        st.caption(f"当前选择: {selected_perm.get('name', '未知排列')}")
                        
                        # 应用选中的排列方案按钮
                        if st.button("✅ 应用此排列方案", type="primary", use_container_width=True):
                            if st.session_state.raw_data is None:
                                st.warning("请先在左侧上传数据文件")
                            else:
                                try:
                                    wavenumbers, y = st.session_state.raw_data
                                    algos = st.session_state.current_algorithms
                                    
                                    # 执行选中的排列方案
                                    processed_data, method_name = preprocessor.process(
                                        wavenumbers, y, 
                                        baseline_method=algos['baseline_method'],
                                        baseline_params=algos['baseline_params'],
                                        squashing_method=algos['squashing_method'],
                                        squashing_params=algos['squashing_params'],
                                        filtering_method=algos['filtering_method'],
                                        filtering_params=algos['filtering_params'],
                                        scaling_method=algos['scaling_method'],
                                        scaling_params=algos['scaling_params'],
                                        algorithm_order=selected_perm.get('order', [])  # 安全获取order属性
                                    )
                                    
                                    # 保存处理结果
                                    arr_name = f"排列_{len(st.session_state.arrangement_results) + 1}"
                                    st.session_state.arrangement_results.append(arr_name)
                                    st.session_state.arrangement_details[arr_name] = {
                                        'data': processed_data,
                                        'method': " → ".join(method_name),
                                        'order': selected_perm.get('order', []),
                                        'params': algos
                                    }
                                    st.session_state.selected_arrangement = arr_name
                                    st.session_state.processed_data = (wavenumbers, processed_data)
                                    st.session_state.process_method = " → ".join(method_name)
                                    st.success(f"排列方案应用完成: {st.session_state.process_method}")
                                except Exception as e:
                                    st.error(f"排列应用失败: {str(e)}")
                    except Exception as e:
                        st.error(f"处理排列方案时出错: {str(e)}")
                else:
                    st.info("暂无符合条件的排列方案（可能未选择该类型的算法）")

if __name__ == "__main__":
    main()
