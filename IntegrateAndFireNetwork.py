import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

cutoff = 2
t = np.arange(0,20,.1)
def Input(t):
    return 3 + 1*np.sin(t)

def F(x):
    if x < cutoff:
        return 0
    else:
        return (x-cutoff)
       
def diff(I,t):
    return 0.5 * (- I + Input(t))

vexit = odeint(diff,1,t)

plt.close()
plt.figure(1)
plt.subplot(111)       
plt.plot(t,vexit)
plt.subplot(211)
plt.plot(t,[F(i) for i in vexit])