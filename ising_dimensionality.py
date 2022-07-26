# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:13:22 2020

@author: Sudipta Nayak
"""

import numpy as np
import random as r
import matplotlib.pyplot as plt

size=50
J=1
T=3.1

def homoinit(s):
    return np.ones((s,s))

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
        
def sentropy(sing):
    N=0
    s=0
    e=0
    for i in range(np.size(sing)):
        N+=1
        s+=sing[i]
    for i in range(np.size(sing)):
        p=sing[i]/s
        if p!=0:
            
            e+=-1*p*np.log(p)
    return e/np.log(N)

from scipy.integrate import quad

def modesentropy(sing,modes):
    N=0
    s=0
    e=0
    for i in range(modes):
        N+=1
        s+=sing[i]
    for i in range(modes):
        p=sing[i]/s
        if p!=0:
            e+=-1*p*np.log(p)
    return e/np.log(N)

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

ar=[0.1,3.1]
t=0.3
while(t<3):
    ar.append(t)
    t=t+0.2
   
ar.sort()
sar=[]
for i in range(len(ar)):
    sar.append(str(round(ar[i],1)))


for m in range(len(ar))
    system=homoinit(size)
    data_col_size=size*size
    data=np.zeros((1000,data_col_size))
    for i in range(1000):
        for j in range(data_col_size*100):
            metropolis_single(system, ar[m])
        for k in range(size):
            for l in range(size):
                data[i,l+size*k]=system[k,l]
    np.savetxt('C:/Users/Sudipta Nayak/Documents/RA-Prof Ashwin/code/ising_dimensionality_effects/size'+str(size)+'T'+sar[m]+'.txt',data)


# =============================================================================
# num_modes4=[5,10,11,15,16]
# num_modes3=[5,6,7,8,9]
# num_modes50=[5,50,100,500,1000]
# num_modes100=[5,50,100,500]
# entropy_matrix=np.zeros((5,len(sar)))
# for i in range(len(sar)):
#     data=np.loadtxt('C:/Users/Sudipta Nayak/Documents/RA-Prof Ashwin/code/ising_dimensionality_effects/size'+str(size)+'T'+sar[i]+'.txt')
#     u,s,vh=np.linalg.svd(data)
#     for j in range(len(num_modes50)):
#         entropy_matrix[j,i]=modesentropy(s,num_modes50[j])
#         print(j)
#     
# np.savetxt('C:/Users/Sudipta Nayak/Documents/RA-Prof Ashwin/code/ising_dimensionality_effects/modes_entropy_size'+str(size)+'.txt',entropy_matrix)
# =============================================================================

# =============================================================================
# entropy_matrix=np.loadtxt('C:/Users/Sudipta Nayak/Documents/RA-Prof Ashwin/code/ising2dmetro/truncatedPCAentropy.txt')
# plt.figure(figsize=(10,10))
# for i in range(4):
#     plt.plot(ar,entropy_matrix[i,:],label=str(num_modes100[i]))
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
# plt.xlabel('T')
# plt.ylabel('Normalised Entropy')
# plt.title('Size='+str(100))
# plt.axvline(2.269)
# =============================================================================
