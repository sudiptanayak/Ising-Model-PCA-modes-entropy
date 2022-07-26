#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 22:35:21 2019

@author: sudipta
"""

import numpy as np
import random as r
import matplotlib.pyplot as plt

size=100
J=1
T=0.1
def initialise_ising(s):
    system=np.random.choice([-1,1],(s,s))
    return system

def homoinit(s):
    system=np.ones((s,s))
    return system

def oppoinit(s):
    system=np.ones((s,s))
    return -1*system
def metropolis_single(sys,T):
    a=r.randrange(0,np.shape(sys)[0])
    b=r.randrange(0,np.shape(sys)[1])
    top=(a-1)%int(np.shape(sys)[1])
    bottom=(a+1)%int(np.shape(sys)[1])
    left=(b-1)%int(np.shape(sys)[0])
    right=(b+1)%int(np.shape(sys)[0])
    dele=2*J*sys[a,b]*(sys[top,b]+sys[bottom,b]+sys[a,left]+sys[a,right])
    p=np.exp(-1*dele/T)
    trial=np.random.uniform()
    if(trial<p):
        sys[a,b]=-1*sys[a,b]
    

def neighbour(sys,element):
    a=element[0]
    b=element[1]
    top=(a-1)%int(np.shape(sys)[1])
    bottom=(a+1)%int(np.shape(sys)[1])
    left=(b-1)%int(np.shape(sys)[0])
    right=(b+1)%int(np.shape(sys)[0])
    n=[[top,b],[bottom,b],[a,left],[a,right]]
    return n


def wolff(sys,T):
    p=1-np.exp(-2*J/T)
    seedr=r.randrange(0,np.shape(sys)[0])
    seedc=r.randrange(0,np.shape(sys)[1])
    cluster=[[seedr,seedc]]
    oldfrontier=[[seedr,seedc]]
    while(oldfrontier!=[]):
        newfrontier=[]
        for el1 in oldfrontier:
            for el2 in neighbour(sys,el1):
                if el2 not in cluster:
                    if sys[el2[0],el2[1]]==sys[el1[0],el1[1]]:
                        a=r.uniform(0,1)
                        if(a<p):
                            newfrontier.append(el2)
                            cluster.append(el2)
        oldfrontier=newfrontier
    for entry in cluster:
        sys[entry[0],entry[1]]=-1*sys[entry[0],entry[1]]
                    
def magps(sys):
    m=0
    N=0
    for i in range(np.shape(sys)[0]):
        for j in range(np.shape(sys)[1]):
            m+=sys[i][j]
            N+=1
    m=m/N
    return m

def makesys(data,i):
    system=np.zeros((size,size))
    for k in range(size):
        for j in range(size):
            system[k,j]=data[i,k*size+j]
    return system
                
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

def modesentropy(sing,modes):
    N=0
    s=0
    e=0
    for i in range(modes):
        N+=1
        s+=sing[i]
    for i in range(modes):
        p=sing[i]/s
        e+=-1*p*np.log(p)
    return e/np.log(N)

from scipy.integrate import quad

def k(beta):
    result=np.power(np.sinh(2*beta),2)
    result=1/result
    return result
def h(beta,theta):
    result=1+np.power(k(beta),2)-2*k(beta)*np.cos(2*theta)
    result=np.sqrt(result)
    return result

def dk(beta):
    result=-4/(np.power(np.sinh(2*beta),2)*np.tanh(2*beta))
    return result

def L(beta,theta):
    result=dk(beta)*(k(beta)-np.cos(2*theta))/h(beta,theta)
    return result

def integranddg(theta,beta):
    ch=np.cosh(2*beta)
    sh=np.sinh(2*beta)
    n=4*ch*sh+4*sh*ch*h(beta,theta)+np.power(sh,2)*L(beta,theta)
    d=np.power(ch,2)+np.power(sh,2)*h(beta,theta)
    result=n/d
    return result

def dg(beta):
    return quad(integranddg,0,np.pi,args=(beta))[0]

def integrandg(theta,beta):
    ch=np.cosh(2*beta)
    sh=np.sinh(2*beta)
    d=np.power(ch,2)+np.power(sh,2)*h(beta,theta)
    return np.log(d)

def g(beta):
    return quad(integrandg,0,np.pi,args=(beta))[0]

def rentropy(beta):
    result=np.log(2)/2+g(beta)/(2*np.pi)-beta*dg(beta)/(2*np.pi)
    return result
    


# =============================================================================
# system=homoinit(size)
# data=np.zeros((1000,10000))
# for i in range(1000):
#     for j in range(1000000):
#         metropolis_single(system,T)
#     for k in range(size):
#         for l in range(size):
#             data[i,l+size*k]=system[k,l]
#     print(i)
# np.savetxt('/home/sudipta/Documents/RA-Prof Ashwin/code/T'+str(T)+'.txt',data)
# =============================================================================

ar=[0.1,3]
t=0.3
while(t<3):
    ar.append(t)
    t=t+0.2
   
ar.sort()
sar=[]
for i in range(len(ar)):
    sar.append(str(round(ar[i],1)))

num_modes=[5,50,100,500]
snodes=[]
for i in range(len(num_modes)):
    snodes.append(str(num_modes[i]))

# =============================================================================
# entropy_matrix=np.zeros((4,len(sar)))
# for i in range(len(sar)):
#     data=np.loadtxt('/home/sudipta/Documents/RA-Prof Ashwin/code/ising2dmetro/T'+sar[i]+'.txt')
#     u,s,vh=np.linalg.svd(data)
#     for j in range(len(num_modes)):
#         entropy_matrix[j,i]=modesentropy(s,num_modes[j])
#         print(j)
#     
# np.savetxt('/home/sudipta/Documents/RA-Prof Ashwin/code/ising2dmetro/truncatedPCAentropy.txt',entropy_matrix)
# =============================================================================
# =============================================================================
# entropy_matrix=np.loadtxt('/home/sudipta/Documents/RA-Prof Ashwin/code/ising2dmetro/truncatedPCAentropy.txt')
# plt.figure()
# for i in range(len(snodes)):
#     plt.plot(ar,entropy_matrix[i,],label=snodes[i])
# pcafull=np.loadtxt('/home/sudipta/Documents/RA-Prof Ashwin/code/ising2dmetro/PCA Entropy')
# plt.plot(ar,pcafull,label='1000')
#         
# 
# Te=np.linspace(0.1,3,100)
# beta=1/Te
# tentropy=np.zeros(np.size(beta))
# maxe=0
# for i in range(len(tentropy)):
#     tentropy[i]=rentropy(beta[i])
#     maxe=max(maxe,tentropy[i])
# 
# maxe=rentropy(0.0001)
# 
# tentropy=tentropy/maxe
# plt.plot(Te,tentropy,label='Thermo')
# plt.legend()
# plt.xlabel('Temperature')
# plt.ylabel('entropy')
# plt.savefig('/home/sudipta/Documents/RA-Prof Ashwin/code/ising2dmetro/modesentropy.png')
# =============================================================================
