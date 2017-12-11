import os
import librosa
import time
from scipy.io import wavfile
from davies_standard import davies_standard
import numpy as np


def veterbiPred(data,fs):
    # if ii<10:
    #     name='C:\Users\\foysal\Desktop\DP_Test\open\open_00'+str(ii)+'.wav'
    # else:
    #     name='C:\Users\\foysal\Desktop\DP_Test\open\open_0'+str(ii)+'.wav'
    #data,fs=librosa.core.load(name, sr=44100)
    ##print data.shape
    sr=fs
    a=data.shape
    data_length= a[0]
    chunk=np.arange(7,29,2)
    ys=data[0:10*fs]
    beats, localscore, cumscore = davies_standard(ys,fs)

    #s,beatss=librosa.beat.beat_track(ys,sr=fs,start_bpm=240,tightness=400)
    #beats=librosa.frames_to_time(beatss, sr=sr)
    ##print len(beats)
    for i in chunk:
        ys=data[0:i*fs]
        b_new, localscore, cumscore = davies_standard(ys,fs)
        #s,b_neww=librosa.beat.beat_track(ys,sr=fs,start_bpm=240,tightness=400)
        #b_new=librosa.frames_to_time(b_neww, sr=sr)
        if (len(b_new)<=1):
            continue

        tempo=sum(b_new[1:]-b_new[0:-1])
        ##print b_new.shape
        tempo=tempo/(len(b_new)-1)
        #print tempo
        while (1):
            new=b_new[-1]+tempo
            b_new=np.append(b_new,new)
            if (new>=i+2):
                break
            if ((new-beats[-1])/((beats[-1]-beats[len(beats)-2])*1.0)>0.5):
                beats=np.append(beats,new)

    return beats
