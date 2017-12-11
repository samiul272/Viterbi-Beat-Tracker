import scipy.io
import scipy.io.wavfile
from davies_standard import davies_standard
fs,data=scipy.io.wavfile.read('C:\\Users\\Irfan\\Documents\\MATLAB\\SP CUP\\training_set\\training_set\\open\\open_003.wav')
beats=davies_standard(data,fs)
