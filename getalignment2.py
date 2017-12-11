import numpy as np
from numba import jit

# @jit(nopython = True, cache = True)
def getalignment2(dfframe, phwv, period, timesig):
    period = int(np.round(period))-1

    # reverse
    dfframe = dfframe[::-1]
    # output of alignment comb filter
    phcf = np.zeros((period,))
    numelem = int(((len(dfframe)) / period))

    #len of dfframe is slightly different from MATLAB
    # indexing niye confussion ache
    for i in range(0, period):
        for b in range(0,numelem):
            phcf[i] = phcf[i]+dfframe[b * (period) + i] * phwv[i]


    #val = np.max(phcf)
    alignment = np.argmax(phcf)
    #val2 = np.max(phwv)
    #bestguess = np.argmax(phwv)

    if alignment >= 1:
        alignment = alignment - 1
    return alignment
