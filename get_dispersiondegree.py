# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 16:01:59 2018

@author: MINGXUAN CHEN
"""

import librosa
import librosa.display

def Getdispersiondegree(path='E:/MINGXUAN CHEN/Infinity Lab/Demential Speech Analysis/sample data/2.wav', sr=None):
    wave, fs = librosa.load(path, sr=None)
    frame_len = int(20 * fs /1000) # 20ms
    frame_shift = int(10 * fs /1000) # 10ms
# calculate RMS
    rms = librosa.feature.rmse(wave, frame_length=frame_len, hop_length=frame_shift)
    rms = rms[0]
    rms = librosa.util.normalize(rms, axis=0)
    s=sum(rms)
    return(s)

#print(getdispersiondegree())
