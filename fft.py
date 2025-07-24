import numpy as np
import copy
from scipy.fftpack import fft, ifft


"""__________________________________________
    信号进行傅里叶变换，使高频信号的系数为零，再进行傅里叶逆变换
    转换会时域上的信号便是滤波后的效果。
"""
def Smfft(arr, row_e = 51):
    row = arr.shape[0]
    col = arr.shape[1]
    fftresult = np.zeros((row, col))
    for i in range(row):
        sfft = fft(arr[i])
        row_s = 2000
        #cutfun = np.ones([row_s, 1])
        #cutfun[20:row_s-20] = 0
        sfftn = copy.deepcopy(sfft)
        sfftn[row_e:row_s-row_e] = 0
        result = ifft(sfftn)
        real_r = np.real(result)
        fftresult[i] = real_r
    return fftresult
