# arrangement/algorithms/scaling.py
import numpy as np
from sklearn.linear_model import LinearRegression

def peak_norm(spectra):
    return spectra / np.max(spectra, axis=0)

def snv(spectra):
    mean = np.mean(spectra, axis=0)
    std = np.std(spectra, axis=0)
    return (spectra - mean) / std

def MSC(sdata):
    """多元散射校正"""
    n = sdata.shape[0]
    k = np.zeros(sdata.shape[0])
    b = np.zeros(sdata.shape[0])
    M = np.mean(sdata, axis=0)
    for i in range(n):
        y = sdata[i, :].reshape(-1, 1)
        M_reshaped = M.reshape(-1, 1)
        model = LinearRegression()
        model.fit(M_reshaped, y)
        k[i] = model.coef_
        b[i] = model.intercept_
    spec_msc = np.zeros_like(sdata)
    for i in range(n):
        bb = np.repeat(b[i], sdata.shape[1])
        kk = np.repeat(k[i], sdata.shape[1])
        spec_msc[i, :] = (sdata[i, :] - bb) / kk
    return spec_msc

def mm_norm(spectra):
    min_vals = np.min(spectra, axis=0)
    max_vals = np.max(spectra, axis=0)
    return (spectra - min_vals) / (max_vals - min_vals)

def LPnorm(arr, ord):
    """Lp范数归一化"""
    row = arr.shape[0]
    col = arr.shape[1]
    Lpdata = np.zeros((row, col))
    for i in range(row):
        Lp = np.linalg.norm(arr[i, :], ord)
        if Lp != 0:
            Lpdata[i, :] = arr[i, :] / Lp
        else:
            Lpdata[i, :] = arr[i, :]
    return Lpdata

def MaMinorm(Oarr):
    """Ma-Minorm归一化"""
    row = Oarr.shape[0]
    col = Oarr.shape[1]
    MMarr = np.zeros((row, col))
    permax = np.ones((1, col))
    for i in range(row):
        diff = np.max(Oarr[i]) - np.min(Oarr[i])
        if diff != 0:
            MMarr[i] = ((Oarr[i] - permax * np.min(Oarr[i])) / diff) * 10 - 5
        else:
            MMarr[i] = Oarr[i] - permax * np.min(Oarr[i])
    return MMarr

def standardization(Datamat):
    mu = np.average(Datamat)
    sigma = np.std(Datamat)
    if sigma != 0:
        return (Datamat - mu) / sigma
    else:
        return Datamat - mu

def plotst(Data):
    """标准化（均值0，方差1）"""
    row = Data.shape[0]
    col = Data.shape[1]
    st_Data = np.zeros((row, col))
    for i in range(row):
        st_Data[i] = standardization(Data[i])
    return st_Data
