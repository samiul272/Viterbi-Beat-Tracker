import numpy as np
from numba import jit


@jit(nopython=True,cache=True)
def getperiod(acf, wv, timesig, step, pmin, pmax):
    rcf = np.zeros(step)

    if not timesig:  # timesig unknown, must be general state
        numelen = 4

        # indexing niye confusion ache
        for i in range(int(pmin) - 1, int(pmax) - 1):
            for a in range(0, numelen):
                for b in range(1 - a, a - 1):
                    rcf[i] += (acf[a * i + b] * wv[i]) / (2 * a - 1)

    else:
        numelen = timesig

        for i in range(pmin - 1, pmax - 1):
            for a in range(1, numelen):
                for b in range(1 - a, a - 1):
                    rcf[i] = rcf[i] + acf[a * i + b] * wv[i]

    return rcf
