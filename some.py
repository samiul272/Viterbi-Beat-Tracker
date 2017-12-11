import madmom
import numpy as np
from scipy.io import wavfile
import time
import os

start_time=time.time()

#data, sample_rate = madmom.audio.signal.load_audio_file("open_001.wav", sample_rate=None,
#                                               num_channels=None,
#                                                start=None, stop=None, dtype=None)
fs, data = wavfile.read("open_001.wav", mmap=True)
gap=10
window_s=5
chunk=np.arange(5,29,gap)
data_=data[0:5*fs]
proc=madmom.features.beats.RNNBeatProcessor()
proc2=madmom.features.beats.BeatTrackingProcessor(fps=100)
act=proc(data_)

beats=proc2(act)
print (beats)
for i in chunk:
	ys=data[max(0,(i-window_s))*fs:i*fs]
	act_=proc(ys)
	b_new = proc2(act_)
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
          if (new>=i+gap):
             break
          if ((new-beats[-1])/((beats[-1]-beats[len(beats)-2])*1.0)>0.5):
             beats=np.append(beats,new)
                  ##b_new.append(new)
myfile=open("MADMOM.txt",'w+')
for item in beats:
	myfile.write("%s\n" %item)
myfile.close()


print (beats)
print (np.shape(beats))
print ("-----------Tolat Execution Time %s---------"%(time.time()-start_time))



