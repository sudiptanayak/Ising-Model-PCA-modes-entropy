#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 00:12:38 2019

@author: sudipta
"""

import numpy as np
import random as r
import matplotlib.pyplot as plt

size=100
J=1
T=1.9

def initialise_ising(s):
    system=np.random.choice([-1,1],size=s)
    return system

def homoinit(s):
    system=np.ones(s)
    return system

def metropolis_single(system,T):
    a=r.randrange(0,np.size(system))
    left=(a-1)%int(np.size(system))
    right=(a+1)%int(np.size(system))
    dele=2*J*system[a]*(system[left]+system[right])
    p=np.exp(-1*dele/T)
    trial=np.random.uniform()
    if(trial<p):
        system[a]=-1*system[a]

def sentropy(sing):
    N=0
    s=0
    e=0
    for i in range(np.size(sing)):
        N+=1
        s+=sing[i]
    for i in range(np.size(sing)):
        p=sing[i]/s
        e+=-1*p*np.log(p)
    return e/np.log(N)

# =============================================================================
# data=np.zeros((1000,100))
# for i in range(1000):
#     sys=homoinit(100)
#     for j in range(100000):
#         metropolis_single(sys,T)
#     for k in range(100):
#         data[i,k]=sys[k]
#     print(i)
# np.savetxt('/home/sudipta/Documents/RA-Prof Ashwin/code/T'+str(T)+'.txt',data)
#         
# 
# =============================================================================
# =============================================================================
# ar=[0.1]
# t=0.3
# while(t<2):
#     ar.append(t)
#     t=t+0.2
#    
# ar.sort()
# sar=[]
# for i in range(len(ar)):
#     sar.append(str(round(ar[i],1)))
# print(sar)
# PCAentropy=np.zeros((len(sar)))
# for i in range(len(sar)):
#     data=np.loadtxt('/home/sudipta/Documents/RA-Prof Ashwin/code/T'+sar[i]+'.txt')
#     u,s,vh=np.linalg.svd(data)
#     PCAentropy[i]=sentropy(s)
#     print(i)
#     
# np.savetxt('/home/sudipta/Documents/RA-Prof Ashwin/code/PCAentropy1dhomo.txt',PCAentropy)
# 
# plt.figure()
# plt.plot(ar,PCAentropy,label='PCA Entropy 1d')
# plt.xlabel('T')
# plt.ylabel('Entropy')
# plt.legend()
# plt.savefig('/home/sudipta/Documents/RA-Prof Ashwin/code/1disingPCAhomo.png')
#     
# =============================================================================
