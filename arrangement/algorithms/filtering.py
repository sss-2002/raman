# arrangement/algorithms/filtering.py
import numpy as np
import scipy.signal as signal
from scipy.signal import savgol_filter, medfilt
from scipy.fft import fft, ifft
from scipy.fftpack import fft as fftpack_fft, ifft as fftpack_ifft
from statsmodels.nonparametric.smoothers_lowess import lowess
import pywt

def savitzky_golay(spectra, window_length, polyorder):
    return savgol_filter(spectra, window_length, polyorder, axis=0)

def sgolay_filter_custom(spectra, window_length, polyorder):
    if spectra.shape[0] < spectra.shape[1]:
        filtered = savgol_filter(spectra.T, window_length, polyorder, axis=0)
        return filtered.T
    else:
        return savgol_filter(spectra, window_length, polyorder, axis=0)

def median_filter(spectra, k, w):
    return medfilt(spectra, kernel_size=(w, 1))

def moving_average(spectra, k, w):
    kernel = np.ones(w) / w
    return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), 0, spectra)

def MWA(arr, n=6, it=1, mode="full"):
    row = arr.shape[0]
    col = arr.shape[1]
    average = np.zeros((row, col))
    ns = []
    for _ in range(it):
        ns.append(n)
        n -= 2
    for i in range(row):
        average[i] = arr[i].copy()
        nn = ns.copy()
        for _ in range(it):
            n = nn.pop()
            if n > 1:
                tmp = np.convolve(average[i], np.ones((n,)) / n, mode=mode)
                for j in range(1, n):
                    tmp[j - 1] = tmp[j - 1] * n / j
                    tmp[-j] = tmp[-j] * n / j
                j = int(n / 2)
                k = n - j - 1
                average[i] = tmp[j:-k]
    return average

def MWM(arr, n=7, it=1):
    row = arr.shape[0]
    col = arr.shape[1]
    median = np.zeros((row, col))
    ns = []
    for _ in range(it):
        ns.append(n)
        n -= 2
    for i in range(row):
        median[i] = arr[i].copy()
        nn = ns.copy()
        for _ in range(it):
            n = nn.pop()
            if n > 1:
                tmp = signal.medfilt(median[i], n)
                median[i] = tmp
    return median

def Kalman(z, R):
    n_iter = len(z)
    sz = (n_iter,)
    Q = 1e-5
    xhat = np.zeros(sz)
    P = np.zeros(sz)
    xhatminus = np.zeros(sz)
    Pminus = np.zeros(sz)
    K = np.zeros(sz)
    xhat[0] = 0.0
    P[0] = 1.0
    for k in range(1, n_iter):
        xhatminus[k] = xhat[k - 1]
        Pminus[k] = P[k - 1] + Q
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
    return xhat

def KalmanF(xd, R):
    row = xd.shape[0]
    col = xd.shape[1]
    Fxd = np.zeros((row, col))
    for i in range(row):
        Fxd[i] = Kalman(xd[i], R)
    return Fxd

def lowess_filter(spectra, frac):
    result = np.zeros_like(spectra)
    for i in range(spectra.shape[1]):
        smoothed = lowess(spectra[:, i], np.arange(len(spectra)), frac=frac, it=0)
        result[:, i] = smoothed[:, 1]
    return result

def fft_filter(spectra, cutoff):
    fft_result = fft(spectra, axis=0)
    frequencies = np.fft.fftfreq(spectra.shape[0])
    filter_mask = np.abs(frequencies) < cutoff
    fft_result[~filter_mask, :] = 0
    return np.real(ifft(fft_result, axis=0))

def Smfft(arr, row_e=51):
    row = arr.shape[0]
    col = arr.shape[1]
    fftresult = np.zeros((row, col))
    for i in range(row):
        sfft = fftpack_fft(arr[i])
        row_s = len(arr[i])
        sfftn = sfft.copy()
        sfftn[row_e:row_s - row_e] = 0
        result = fftpack_ifft(sfftn)
        real_r = np.real(result)
        fftresult[i] = real_r
    return fftresult

def wavelet_filter(spectra, threshold):
    coeffs = pywt.wavedec(spectra, 'db4', axis=0)
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, 'db4', axis=0)

def waveletlinear(arr, threshold=0.3):
    row = arr.shape[0]
    col = arr.shape[1]
    datarec = np.zeros((row, col))
    w = pywt.Wavelet('db8')
    for i in range(row):
        maxlev = pywt.dwt_max_level(col, w.dec_len)
        coeffs = pywt.wavedec(arr[i], 'db8', level=maxlev)
        for j in range(0, len(coeffs)):
            coeffs[j] = pywt.threshold(coeffs[j], threshold * max(coeffs[j]))
        datarec[i] = pywt.waverec(coeffs, 'db8')
    return datarec
