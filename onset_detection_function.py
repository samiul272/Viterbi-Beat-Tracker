import numpy as np
from scipy.interpolate import interp1d, splev,splrep, interpolate, splprep,pchip
from bt_parms import bt_parms
from hanningz import hanningz
from princarg import princarg
import time
import pyfftw
from scipy.interpolate import UnivariateSpline


def onset_detection_function(x, p):
    p = bt_parms(0.01161)

    o_step = p.winlen * 2
    o_winlen = o_step * 2
    hlfwin = o_winlen / 2

    # formulate hannng window function
    # win = hanningz(o_winlen + 1)
    win = np.hanning(o_winlen)
    # loop parameters
    N = len(x)
    pin = 0
    pend = N - o_winlen

    # vector to store phase and magnitude calculation
    theta1 = np.zeros((int(hlfwin), 1))
    theta2 = np.zeros((int(hlfwin), 1))
    oldmag = np.zeros((int(hlfwin), 1))

    # output onset detection function
    df = []

    # df sample number
    k = -1
    while pin < pend:
        k = k + 1
        # calculate windowed fft frame
        segment = x[pin:(pin + int(o_winlen))]
        X = np.fft.fft(np.fft.fftshift(win * segment))
        # X=np.fft.fft(win * segment)
        # discard first half of the spectrum
        X = X[int((len(X) / 2.0)):len(X)]
        # X = X[1024:0:-1]
        # find the magnitude and phase
        mag = (abs(X)).reshape((int(hlfwin), 1))
        theta = np.angle(X).reshape((int(hlfwin), 1))

        # complex sd part
        dev = princarg(theta - 2 * theta1 + theta2)
        meas = oldmag - (mag * np.exp(1j * dev))
        df.append(np.sum(abs(meas)))

        # update vectors
        theta2 = theta1
        theta1 = theta
        oldmag = mag

        # move to next frame
        pin = pin + o_step

    # now interpolate each detection function by a factor of 2 to get resolution of 11.6ms
    # Result might be different from MATLAB as python used different algorithom
    df = np.array(df)
    l = len(df)
    # tck = splprep(df)
    # df = splev(np.arange(0, l - 0.5, 0.5) * p.timeres / p.fs, tck)
    df = pchip(np.arange(0, l) * p.timeres / p.fs, df)(np.arange(0, l - 0.5, 0.5) * p.timeres / p.fs)
    # dff = np.zeros((len(df)*2-1,))
    # dff[::2]=df
    # dff[1::2]=df[1::]*(1-(df[1::]-df[0:-1:])*df[1::])
    # df = interp1d(np.arange(0, l) * p.timeres / p.fs, df, kind='cubic')(np.arange(0, l - 0.5 , 0.5) * p.timeres / p.fs)
    # df = UnivariateSpline(np.arange(0, l) * p.timeres / p.fs, df)(np.arange(0, l - 0.5, 0.5) * p.timeres / p.fs)
    # df = np.interp(np.arange(0, l - 0.5, 0.5) * p.timeres / p.fs, np.arange(0, l) * p.timeres / p.fs, df)

    return df
