import numpy as np


def ftacf(x):
    x = np.array(x)
    M = x.shape[0]
    X = np.fft.fft(x,1024)
    acf = np.fft.irfft(np.absolute(X))
    acf = acf[0:M] / np.transpose(np.arange(M,0,-1))
    return acf

