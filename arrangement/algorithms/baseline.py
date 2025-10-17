# arrangement/algorithms/baseline.py
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter

def polynomial_fit(wavenumbers, spectra, polyorder):
    """多项式拟合基线校正"""
    baseline = np.zeros_like(spectra)
    for i in range(spectra.shape[1]):
        coeffs = np.polyfit(wavenumbers, spectra[:, i], deg=polyorder)
        baseline[:, i] = np.polyval(coeffs, wavenumbers)
    return spectra - baseline

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

def pls(spectra, lam):
    """Penalized Least Squares (PLS) 基线校正"""
    n_points = spectra.shape[0]
    baseline = np.zeros_like(spectra)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n_points, n_points - 2))
    D = lam * D.dot(D.transpose())
    for i in range(spectra.shape[1]):
        y = spectra[:, i]
        A = sparse.eye(n_points) + D
        baseline[:, i] = spsolve(A, y)
    return spectra - baseline

def airpls(spectra, lam, max_iter=15, threshold=0.001):
    """Adaptive Iteratively Reweighted Penalized Least Squares (airPLS)"""
    n_points = spectra.shape[0]
    baseline = np.zeros_like(spectra)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(n_points, n_points - 2))
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
                diff = np.sum(np.abs(b - baseline_i)) / np.sum(np.abs(baseline_i)) if np.sum(
                    np.abs(baseline_i)) > 0 else 0
                if diff < threshold:
                    break
            baseline_i = b
        baseline[:, i] = baseline_i
    return spectra - baseline

def baseline_als(y, lam, p, niter=10, tol=1e-6):
    """改进的非对称加权惩罚最小二乘基线校准"""
    y = np.asarray(y, dtype=np.float64)
    L = y.shape[1]
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    result = np.zeros_like(y)
    for j in range(y.shape[0]):
        w = np.ones(L)
        y_curr = y[j].copy()
        for _ in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y_curr)
            if np.max(np.abs(z - y_curr)) < tol:
                break
            w = p * (y[j] > z) + (1 - p) * (y[j] < z)
            y_curr = z
        result[j] = y[j] - z
    return result

def IModPoly(wavenumbers, originalRaman, polyorder, max_iter=100, tolerance=0.005):
    """改进的多项式拟合基线校正"""
    row, col = originalRaman.shape
    corrected = np.zeros((row, col))
    for j in range(row):
        prev_spectrum = originalRaman[j]
        curr_spectrum = prev_spectrum.copy()
        prev_std = 0
        converged = False
        iteration = 1
        while not converged and iteration <= max_iter:
            coeffs = np.polyfit(wavenumbers, curr_spectrum, polyorder)
            fitted = np.polyval(coeffs, wavenumbers)
            residual = curr_spectrum - fitted
            curr_std = np.std(residual)
            if iteration == 1:
                mask = prev_spectrum > (fitted + curr_std)
                curr_spectrum[mask] = fitted[mask] + curr_std
            else:
                mask = prev_spectrum < (fitted + curr_std)
                curr_spectrum = np.where(mask, prev_spectrum, fitted + curr_std)
            relative_change = abs((curr_std - prev_std) / curr_std) if curr_std != 0 else 0
            converged = relative_change < tolerance
            prev_spectrum = curr_spectrum
            prev_std = curr_std
            iteration += 1
        corrected[j] = originalRaman[j] - fitted
    return corrected

def d2(spectra):
    """二阶差分(D2)"""
    row = spectra.shape[0]
    col = spectra.shape[1]
    D2_result = np.zeros((row, col))
    for i in range(row):
        tem = np.diff(spectra[i], 2)
        temp = tem.tolist()
        temp.append(temp[-1])
        temp.append(temp[-1])
        D2_result[i] = temp
    return D2_result

def _sd_baseline(spectra):
    return spectra - np.min(spectra, axis=0)

def _fd_baseline(spectra):
    return spectra - np.percentile(spectra, 5, axis=0)
