from scipy.stats import norm
from scipy.signal import fftconvolve
import numpy as np
from ftacf import ftacf
from getperiod import getperiod
from sunity import sunity
from viterbi_path import viterbi_path
from adapt_thresh import adapt_thresh


def periodicity_path(df, p, tmat2=None):
    # function to calculate the best periodicity path through time using viterbi decoding
    step = p.step
    winlen = p.winlen

    n = np.arange(1, step + 1)

    # rayleigh weightining curve
    wv = (n / (p.rayparam ** 2)) * np.exp(-n ** 2 / (2 * p.rayparam ** 2))

    # sum to unity
    wv = sunity(wv)

    pin = 0
    pend = len(df) - winlen

    # split df into overlapping frames
    # find the autocorrelation function
    # apply comb filtering and store output in a matrix 'obs'

    ct = 0
    acf = np.zeros((winlen,int(((pend-pin)/step)+1)))  # Declare acf
    obs = np.zeros((128,int(((pend-pin)/step)+1)))

    while pin < pend:
        segment = df[pin:pin+winlen].flatten()
        #----------------------------------------------------
        # acf[:, ct] = fftconvolve(segment,segment[::-1],mode = 'same')
        acf[:,ct] = ftacf(segment)
        rcf = getperiod(acf[:,ct],wv,0,step,p.pmin,p.pmax)
        obs[:,ct] = sunity(adapt_thresh(rcf))
        #----------------------------------------------------
        pin = pin + step
        ct = ct + 1

    '''

    Calculated value of acf in MATLAB and python is not same. Hence the value of obs also differ

    '''

    # make transition matrix
    tmat = np.zeros([step, step])
    # as a diagonal guassian
    for i in range(28, 109):
        tmat[:, i] = norm.pdf(n, i, 8)

    tmat[0:28, :] = 0
    tmat[:, 0:28] = 0
    tmat[108:128, :] = 0
    tmat[:, 108:128] = 0

    if tmat2 != None:
        tmat = tmat2

    # work out best path
    obs = np.array(obs)
    ppath = viterbi_path(wv, tmat, obs +  0.1* np.max(obs) * np.random.random(obs.shape))+1

    # add on a few values at the end, to deal with final overlapping frames
    ppath = np.hstack((ppath, ppath[0][-1]*np.ones((1,4))))
    # abc = np.zeros(np.size(ppath))
    # for xx in range(0,np.size(ppath)):
    #     abc[xx] = ppath[0][xx]

    return ppath
