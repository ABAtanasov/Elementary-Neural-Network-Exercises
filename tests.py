import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

P = 2
N = 4
def sgn(x):
    if x >= 0: return 1
    else: return -1
    
V = np.array([[sgn(np.random.random_sample()-0.5) for i in range(N)] for j in range(P)])

print V.shape

print V[:,1]
print V[:,2]

print sum(V[:,1]*V[:,2])
print len(V[:,1])