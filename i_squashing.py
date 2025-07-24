import numpy as np
import math
import matplotlib.pyplot as plt


# import crossvalidation  as cr
def i_squashing(Data):
    row = Data.shape[0]
    col = Data.shape[1]
    sqData = np.zeros((row, col))
    for i in range(row):
        mi = np.min(Data[i])
        diff = np.max(Data[i]) - mi
        for j in range(col):
            t = (Data[i, j] - mi) / diff
            m = (1 - math.cos(t * math.pi)) / 2
            sqData[i][j] = m * diff + mi
    return sqData
