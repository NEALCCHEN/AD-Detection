import librosa
import os
import numpy as np
import matplotlib as plt
# 1. Get the file path to the included audio example
filepath = 'E:\MINGXUAN CHEN\Infinity Lab\Demential Speech Analysis\sample data'
filename =os.path.join(filepath,'0.wav')
# 2. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
y, sr = librosa.load(filename,sr=None)

params = y.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]

# plot the wave
time = np.arange(0,nframes)*(1.0 / framerate)
plt.plot(time,y)
plt.xlabel("Time(s)")
plt.ylabel("Amplitude")
plt.title("Single channel wavedata")
plt.grid('on')#标尺，on：有，off:无。