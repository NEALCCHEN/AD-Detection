# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:34:31 2018

@author: MINGXUAN CHEN
"""
#import sys

#sys.path.append('E:/MINGXUAN CHEN/Infinity Lab/Demential Speech Analysis/get_dispersiondegree')


import get_dispersiondegree
import os
import matplotlib.pyplot as plt

path='E:/MINGXUAN CHEN/Infinity Lab/wav Dementia Data'
files=os.listdir(path)
s=[]
x1=range(1,len(files)+1)

for file in files:
    s.append(get_dispersiondegree.Getdispersiondegree(os.path.join(path,file)))
    
    
s.sort()


path2='E:/MINGXUAN CHEN/Infinity Lab/wav Control Data'  
files2=os.listdir(path2)
s2=[]
x2=range(1,len(files2)+1)
for file in files2:
    s2.append(get_dispersiondegree.Getdispersiondegree(os.path.join(path2,file)))
    
    
s2.sort()
 
plt.figure(figsize=(5,5))
plt.subplot(1,2,1)
plt.scatter(x1,s,c='r',marker='x',alpha=0.6)
plt.xlabel('RMS energy')
plt.ylabel('number of people')
plt.show()

plt.subplot(1,2,2)
plt.scatter(x2,s2,c='b',marker='o',alpha=0.6)
plt.xlabel('RMS Energy')
plt.ylabel('number of people')
plt.show()

