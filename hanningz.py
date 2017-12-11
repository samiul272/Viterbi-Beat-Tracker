import numpy as np

def hanningz(n):
    w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(0, n - 1).T / n))
    return w
