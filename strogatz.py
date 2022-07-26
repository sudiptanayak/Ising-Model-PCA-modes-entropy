#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:13:57 2019

@author: sudipta
"""

import numpy as np
import matplotlib.pyplot as plt

def f(x,y):
    if (x==0) and (y==0):
        return 0
    else:
        return x*x/np.sqrt(x*x+y*y)
    
def g(x,y):
    if (x==0) and (y==0):
        return 0
    else:
        return x*y/np.sqrt(x*x+y*y)
    


mu=3
x=np.linspace(-2,2,100)
y=np.linspace(-2,2,100)
X,Y=np.meshgrid(x,y)
U=X*(1-X*X-Y*Y)+mu*X*X/np.sqrt(X*X+Y*Y)-Y
V=Y*(1-X*X-Y*Y)+mu*X*Y/np.sqrt(X*X+Y*Y)+X
plt.figure(figsize=(6,7))
plt.streamplot(X,Y,U,V,density=[2,2])
#plt.scatter(0,0)


    
    


