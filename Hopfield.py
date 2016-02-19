import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
N = 100
P = 5
T = 20
p = 0.3

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

V = np.array([[sgn(np.random.random_sample()-0.5) for i in range(N)] for j in range(P)])

M = np.zeros((N,N))

for i in range(N):
    for j in range(N):
        if i==j: M[i][j] =0
        else:
            M[i][j] = np.sum(V[:,i]*V[:,j])


v = perturb(V[0,:],p)

print q(v,V,0)

qs = np.ones(T)

for t in range(T):
    qs[t] = q(v,V,0)
    update(v,M)



plt.plot(qs)
    