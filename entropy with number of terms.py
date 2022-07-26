# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 20:15:39 2020

@author: Sudipta Nayak
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# def eC0(N,alpha):
#     s=0
#     for i in range(1,N+1):
#         s=s+np.exp(-1*alpha*i)
#     return s
# 
# def eC1(N,alpha):
#     s=0
#     for i in range(1,N+1):
#         s=s+np.exp(-1*alpha*i)*i
#     return s
# 
# 
# def Sexp(N,alpha):
#     a=np.log(eC0(N,alpha))/np.log(N)
#     b=eC1(N,alpha)/eC0(N,alpha)
#     c=b*alpha/np.log(N)
#     result=a+c
#     return result
# 
# 
# 
# 
# alpha1=0.5
# alpha2=1
# 
# narray=np.arange(2,10,dtype=int)
# 
# earray1=np.zeros(np.shape(narray))
# earray2=np.zeros(np.shape(narray))
# 
# for i in range(0,8):
#     earray1[i]=Sexp(i+2,alpha1)
#     earray2[i]=Sexp(i+2,alpha2)
#     
# plt.figure(figsize=(10,10))
# plt.plot(narray,earray1,label='alpha=0.01')
# plt.plot(narray,earray2,label='alpha=1')
# plt.legend()
# plt.title('Exponential')
# plt.xlabel('N')
# plt.ylabel('entropy')
# def pC0(N,alpha):
#     s=0
#     for i in range(1,N+1):
#         s=s+1/np.power(i,alpha)
#     return s
# 
# def pC1(N,alpha):
#     s=0
#     for i in range(1,N+1):
#         s=s+(1/np.power(i,1*alpha))*np.log(i)
#     return s
# 
# def Spow(N,alpha):
#     a=np.log(pC0(N,alpha))/np.log(N)
#     b=pC1(N,alpha)/pC0(N,alpha)
#     c=alpha*b/np.log(N)
#     result=a+c
#     return result
# 
# parray1=np.zeros(np.shape(narray))
# parray2=np.zeros(np.shape(narray))
# 
# for i in range(0,8):
#     parray1[i]=Spow(i+2,alpha1)
#     parray2[i]=Spow(i+2,alpha2)
#     
# plt.figure(figsize=(10,10))
# plt.plot(narray,parray1,label='alpha=0.01')
# plt.plot(narray,parray2,label='alpha=0.1')
# plt.legend()
# plt.title('power law')
# plt.xlabel('N')
# plt.ylabel('entropy')
# 
# =============================================================================

data=np.loadtxt('C:/Users/Sudipta Nayak/Documents/RA-Prof Ashwin/code/ising_dimensionality_effects/size'+str(3)+'T'+str(1.7)+'.txt')
u,s,vh=np.linalg.svd(data)

norm=np.sum(s)
s=s/norm
label=['1','2','3','4','5','6','7','8','9']
plt.figure(figsize=(10,10))
plt.bar(label,s)
plt.title('size=3,T=1.7')


def sigma(k,alpha):
    return np.exp(-1*k*alpha)

def Esum(N,alpha):
    s=0
    for i in range(1,N+1):
        s=s+sigma(i,alpha)
    return s

def p(k,N,alpha):
    return sigma(k,alpha)/Esum(N,alpha)

alpha1=0.01
alpha2=0.5

earray1=np.zeros(9)
earray2=np.zeros(9)
for i in range(9):
    earray1[i]=p(i+1,9,alpha1)
    earray2[i]=p(i+1,9,alpha2)
plt.figure(figsize=(10,10))
plt.bar(label,earray1)   
plt.title('exponential,alpha=0.01')
plt.figure(figsize=(10,10))
plt.bar(label,earray2)   
plt.title('exponential,alpha=0.5')

def rho(k,alpha):
    return 1/np.power(k,alpha)

def Psum(N,alpha):
    s=0
    for i in range(1,N+1):
        s=s+rho(i,alpha)
    return s

def p1(k,N,alpha):
    return rho(k,alpha)/Psum(N,alpha)

parray1=np.zeros(9)
parray2=np.zeros(9)
for i in range(9):
    parray1[i]=p1(i+1,9,alpha1)
    parray2[i]=p1(i+1,9,alpha2)
plt.figure(figsize=(10,10))
plt.bar(label,parray1)   
plt.title('power,alpha=0.01')
plt.figure(figsize=(10,10))
plt.bar(label,parray2)   
plt.title('power,alpha=0.5')
