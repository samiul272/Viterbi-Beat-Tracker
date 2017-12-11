from scipy.io import wavfile
from davies_standard import davies_standard
import numpy as np
import time
import matlab.engine
from io import StringIO
from viterbi_pred import veterbiPred
import madmom
import os

s = os.curdir
proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
eng = matlab.engine.start_matlab()
t = time.time()
s = 0
open = range(1,26)
closed = [1,2,3,5,6,8,9,10,11,12,13,14,15,16,17,18]
#open/open_
test = open
num = 6
for num in test:
    # fid = './open/open_' + str(num).zfill(3) + '.wav'
    fid = 'C:/Users/Samiul/Desktop/SPC/training_set/open/open_' + str(num).zfill(3) + '.wav'
    ftxt = 'C:/Users/Samiul/Desktop/SPC/training_set/open/open_' + str(num).zfill(3) + '.txt'
    fs, data = wavfile.read(fid, mmap=True)
    data = data / (1.0 * np.max(data))
    # beats, localscore, cumscore = davies_standard(data,fs)
    beats = davies_standard(data, fs)
    truths = np.loadtxt(ftxt)
    # act = madmom.features.beats.RNNBeatProcessor()(fid)
    # beats = proc(act)
    det = matlab.double(list(beats))
    ann = matlab.double(list(truths))
    score = eng.beatEvaluator2(det, ann)
    s += score
    print score
print time.time() - t
print '\n\n\n'
# for beat in beats:
#     print("{0:.4f}".format(round(beat, 4)))


print '\n\n'
# for beat in truths:
#     print("{0:.4f}".format(round(beat, 4)))

print s /float(len(test))
