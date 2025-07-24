

import numpy as np
import math

def squashing(Data):
    row = Data.shape[0]
    col = Data.shape[1]
    sqData = np.zeros((row, col))
    for i in range (row):
        for j in range (col):
            sqData[i][j] =( 1-math.cos(Data[i][j]*math.pi))/2
    return sqData


