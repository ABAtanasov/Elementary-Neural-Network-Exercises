# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 10:11:33 2016

@author: Alex
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

N = 100
Ptot = 20
T = 20
ptot = 10

def sgn(x):
    if x >= 0: return 1.0
    else: return -1.0

def update(v,M):
    l = len(v)
    
    for a in range(l):
        v[a] = sgn(np.sum(M[a,:]*v[:]))
    
def q(v,V,m):
    return np.sum(v[:] * V[m,:])/N

def perturb(v,p):
    v2 = [i for i in v]
    for i in range(len(v)):
        if np.random.random_sample() < p:
            v2[i] = -v[i]
        else: v2[i] = v[i]
    return np.array(v2)

qs = np.ones(T)
M = np.zeros((N,N))
results = np.zeros((Ptot,ptot))

for P in range(1,Ptot):
    for p in range(0,ptot):
        V = np.array([[sgn(np.random.random_sample()-0.5) for i in range(N)] for j in range(P)])
        for i in range(N):
            for j in range(N):
                if i==j: M[i][j] =0
                else:
                    M[i][j] = np.sum(V[:,i]*V[:,j])
                    v = perturb(V[0,:],p/ptot)
        for t in range(T):
            update(v,M)
        print P,p
        results[P,p]=q(v,V,0)


plt.plot(results[4,:])
    