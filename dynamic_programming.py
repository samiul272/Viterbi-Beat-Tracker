import numpy as np
from scipy.signal import fftconvolve

from ftacf import ftacf
from getalignment2 import getalignment2
from mult import matMult, conv


# from numba import jit,int64
#
#
# @jit(cache = True)
def dynamic_programming(df, p, ppath, mode):
    b = np.zeros(())
    if mode == 1:
        alpha = 0.5
        tightness = 3
    else:
        alpha = 0.90
        tightness = 5

    tempmat = np.matmul(ppath.T, np.ones((1, p.step))).T
    # tempmat = matMult(ppath.T, np.ones((1, p.step))).T
    pd = np.empty(tempmat.size)
    np.round_(tempmat.flatten(), 0, pd)
    mpd = round(np.median(ppath))

    templt = np.exp(-0.5 * np.power((np.arange(-mpd, mpd+0.5) / (mpd / 32.0)), 2))

    localscore = np.convolve(templt, df)
    # localscore = conv(templt, df)
    localscore = localscore[np.round(templt.size / 2) + np.arange(0, df.size)+1]

    backlink = np.zeros((1, localscore.size))
    cumscore = np.zeros((1, localscore.size))

    starting = 1
    for i in range(0, localscore.size):
        prange = np.arange(round(-2 * pd[i]), -round(pd[i] / 2), dtype=int)
        txwt = np.exp(-0.5 * (tightness * np.log(prange / -pd[i])) ** 2)
        timerange = i + prange - 1
        zpad = int(max(0, min(1 - timerange[0], prange.size)))
        # scorecards = np.zeros(txwt.shape)
        # scorecards[:zpad] = 0
        # scorecards[zpad:] = txwt[zpad:] * cumscore[0, timerange[zpad:]]
        scorecards = txwt * np.append(np.zeros((1, zpad)), cumscore[0, (timerange[zpad:])])
        xx = np.argmax(scorecards)
        vv = scorecards[xx]
        cumscore[0, i] = alpha * vv + (1 - alpha) * localscore[i]

        if starting == 1 and localscore[i] < 0.01 * np.max(localscore):
            backlink[0, i] = -1
        else:
            backlink[0, i] = timerange[xx]
            starting = 0

    align = getalignment2(localscore, np.ones(int(1 * pd[-1])), 1 * pd[-1], 0)
    b = np.array([localscore.size - align-1])
    # b = np.hstack((b, float(backlink[0, b])))
    while (backlink[0, int((b[-1])-1)]) > 0:
        b = np.hstack((b, float(backlink[0, int((b[-1])-1)])))
    beats = np.sort(b) * 512.0 / 44100.0

    return beats
