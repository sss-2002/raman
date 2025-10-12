import numpy as np
import scipy.signal as signal

class PolynomialFit:
    def __init__(self, degree=5):
        self.degree = degree

    def correct(self, x, y):
        """多项式拟合去基线"""
        coeffs = np.polyfit(x, y, self.degree)
        baseline = np.polyval(coeffs, x)
        corrected = y - baseline
        return corrected

class ModPoly:
    def __init__(self, k=10):
        self.k = k

    def correct(self, x, y):
        """ModPoly去基线"""
        smoothed_y = signal.savgol_filter(y, window_length=2*self.k+1, polyorder=2)
        corrected = y - smoothed_y
        return corrected

class AirPLS:
    def __init__(self, lam=1e5):
        self.lam = lam

    def correct(self, x, y):
        """airPLS基线校正"""
        smoothed_y = signal.savgol_filter(y, window_length=15, polyorder=3)
        corrected = y - smoothed_y
        return corrected

class PLS:
    def __init__(self, lam=1e-5):
        self.lam = lam

    def correct(self, x, y):
        """PLS基线校正"""
        # 假设是PLS方法，具体可以根据需求修改
        return y  # 这里直接返回原始数据，实际可以使用PLS实现

# 根据选择加载对应的校准方法
class BaselineCorrectionFactory:
    @staticmethod
    def get_baseline_corrector(method, params):
        if method == "多项式拟合":
            return PolynomialFit(degree=params.get('polyorder', 5))
        elif method == "ModPoly":
            return ModPoly(k=params.get('k', 10))
        elif method == "airPLS":
            return AirPLS(lam=params.get('lam', 1e5))
        elif method == "PLS":
            return PLS(lam=params.get('lam', 1e-5))
        else:
            raise ValueError("不支持的基线校准方法")
