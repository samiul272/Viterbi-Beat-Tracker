import librosa
import numpy as np
from scipy.signal import resample
from bt_parms import bt_parms
from dynamic_programming import dynamic_programming
from onset_detection_function import onset_detection_function
from periodicity_path import periodicity_path
import time


def davies_standard(x, fs):
    # % read wave file
    # %[x fs] = audioread(input);

    # convert to mono
    #x = np.mean(x,axis = 1)

    # if audio is not at 44khz resample
    if fs != 44100:
        x = resample(x, 44100, fs)

    # read beat tracking parameters
    p = bt_parms(0.01161)

    # generate the onset detection function
    # st_time = time.time()
    # df = librosa.onset.onset_strength(y=x,sr=44100)
    df = onset_detection_function(x,p)
    # print 'onset_detection_function runtime'
    # print time.time()-st_time

    # strip any trailing zeros
    while not df[-1]:
        df = df[0:len(df) - 1]

    # get periodicity path
    # st_time = time.time()
    ppath = periodicity_path(df, p)
    # print 'periodicity_path runtime'
    # print time.time()-st_time
    mode = 0  # use this to run normal algorithm
    # find beat locations
    # st_time = time.time()
    beats = dynamic_programming(df, p, ppath, mode)
    # print 'dynamic_programming runtime'
    # print time.time()-st_time
    return beats