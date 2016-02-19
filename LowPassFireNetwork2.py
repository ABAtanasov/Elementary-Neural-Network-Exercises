import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

cutoff = 2
t = np.arange(0,20,.1)
def Input(t):
    return 1.93 + .1 * np.sin(t)

def F(x):
    if x < cutoff:
        return 0
    else:
        return (x-cutoff)
       
def diff(v,t):
    return  (- v + F(Input(t)))

vexit = odeint(diff,.15,t)

plt.figure(1)
plt.subplot(111)       
plt.plot(t,vexit)