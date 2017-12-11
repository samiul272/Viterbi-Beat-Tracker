from numba import jit
import numpy as np

@jit(nopython=True,cache = True)
def matMult(A, B):
    nRow1 = A.shape[0]
    nCol2 = B.shape[1]
    C = np.zeros((nRow1, nCol2))
    for i in range(0, nRow1):
        for j in range(0, nCol2):
            for k in range(0,B.shape[0]):
                C[i, j] += A[i, k] * B[k, j]
    return C

@jit(nopython=True,cache = True)
def conv(A,B):
    lA = len(A)
    lB = len(B)
    C = np.zeros(lA+lB)
    A = np.concatenate((A,np.zeros(lB)))
    B = np.concatenate((B,np.zeros(lA)))
    for i in range(0,lA+lB):
        for j in range(0,lA):
            C[i] += A[j]*B[i-j]
    return C


