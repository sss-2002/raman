# arrangement/algorithms/__init__.py
from .baseline import (
    polynomial_fit, modpoly, pls, airpls, baseline_als,
    IModPoly, d2, _sd_baseline, _fd_baseline
)
from .filtering import (
    savitzky_golay, sgolay_filter_custom, median_filter, moving_average,
    MWA, MWM, KalmanF, lowess_filter, fft_filter, Smfft,
    wavelet_filter, waveletlinear
)
from .scaling import (
    peak_norm, snv, MSC, mm_norm, LPnorm, MaMinorm, plotst
)
from .squashing import (
    sigmoid, i_sigmoid, squashing_legacy, squashing,
    i_squashing, dtw_squashing
)

class Preprocessor:
    def __init__(self):
        self.BASELINE_ALGORITHMS = {
            "SD": _sd_baseline,
            "FD": _fd_baseline,
            "多项式拟合": polynomial_fit,
            "ModPoly": modpoly,
            "I-ModPoly": IModPoly,
            "PLS": pls,
            "AsLS": baseline_als,
            "airPLS": airpls,
            "二阶差分(D2)": d2
        }
        self.FILTERING_ALGORITHMS = {
            "Savitzky-Golay": savitzky_golay,
            "sgolayfilt滤波器": sgolay_filter_custom,
            "中值滤波(MF)": median_filter,
            "移动平均(MAF)": moving_average,
            "MWA（移动窗口平均）": MWA,
            "MWM（移动窗口中值）": MWM,
            "卡尔曼滤波": KalmanF,
            "Lowess": lowess_filter,
            "FFT": fft_filter,
            "Smfft傅里叶滤波": Smfft,
            "小波变换(DWT)": wavelet_filter,
            "小波线性阈值去噪": waveletlinear
        }
        self.SCALING_ALGORITHMS = {
            "Peak-Norm": peak_norm,
            "SNV": snv,
            "MSC": MSC,
            "M-M-Norm": mm_norm,
            "L-范数": LPnorm,
            "Ma-Minorm": MaMinorm,
            "标准化(均值0，方差1)": plotst
        }
        self.SQUASHING_ALGORITHMS = {
            "Sigmoid挤压": sigmoid,
            "改进的Sigmoid挤压": i_sigmoid,
            "逻辑函数": squashing_legacy,
            "余弦挤压(squashing)": squashing,
            "改进的逻辑函数": i_squashing,
            "DTW挤压": dtw_squashing
        }

    def process(self, wavenumbers, data,
                baseline_method="无", baseline_params=None,
                squashing_method="无", squashing_params=None,
                filtering_method="无", filtering_params=None,
                scaling_method="无", scaling_params=None,
                algorithm_order=None):
        if baseline_params is None: baseline_params = {}
        if squashing_params is None: squashing_params = {}
        if filtering_params is None: filtering_params = {}
        if scaling_params is None: scaling_params = {}

        if algorithm_order is not None and len(algorithm_order) == 0:
            return data.copy(), ["无预处理（原始光谱）"]

        y_processed = data.copy()
        method_name = []

        if algorithm_order is not None and len(algorithm_order) > 0:
            step_mapping = {
                1: ("baseline", baseline_method, baseline_params),
                2: ("scaling", scaling_method, scaling_params),
                3: ("filtering", filtering_method, filtering_params),
                4: ("squashing", squashing_method, squashing_params)
            }
            steps = [step_mapping[order] for order in algorithm_order]
        else:
            steps = []
            if baseline_method != "无":
                steps.append(("baseline", baseline_method, baseline_params))
            if squashing_method != "无":
                steps.append(("squashing", squashing_method, squashing_params))
            if filtering_method != "无":
                steps.append(("filtering", filtering_method, filtering_params))
            if scaling_method != "无":
                steps.append(("scaling", scaling_method, scaling_params))

        for step_type, method, params in steps:
            if method == "无":
                continue

            try:
                if step_type == "baseline":
                    algorithm_func = self.BASELINE_ALGORITHMS[method]
                    if method in ["多项式拟合", "ModPoly", "I-ModPoly"]:
                        y_processed = algorithm_func(wavenumbers, y_processed,** params)
                    elif method in ["PLS"]:
                        y_processed = algorithm_func(y_processed, **params)
                    elif method == "AsLS":
                        y_processed = algorithm_func(y_processed,** params)
                    elif method == "airPLS":
                        y_processed = algorithm_func(y_processed, **params)
                    elif method == "二阶差分(D2)":
                        y_processed = algorithm_func(y_processed)
                    else:
                        y_processed = algorithm_func(y_processed)
                    method_name.append(f"{method}({', '.join([f'{k}={v}' for k, v in params.items()])})")

                elif step_type == "squashing":
                    algorithm_func = self.SQUASHING_ALGORITHMS[method]
                    if method == "改进的Sigmoid挤压":
                        maxn = params.get("maxn", 10)
                        y_processed = algorithm_func(y_processed, maxn=maxn)
                        method_name.append(f"{method}(maxn={maxn})")
                    elif method == "改进的逻辑函数":
                        y_processed = algorithm_func(y_processed)
                        method_name.append(f"{method}")
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
                    y_processed = algorithm_func(y_processed,** params)
                    params_str = ', '.join([f'{k}={v}' for k, v in params.items()])
                    method_name.append(f"{method}({params_str})")

                elif step_type == "scaling":
                    algorithm_func = self.SCALING_ALGORITHMS[method]
                    y_processed = algorithm_func(y_processed, **params)
                    params_str = ', '.join([f'{k}={v}' for k, v in params.items()])
                    method_name.append(f"{method}({params_str})")

            except Exception as e:
                raise ValueError(f"{step_type}处理失败: {str(e)}")

        return y_processed, method_name
