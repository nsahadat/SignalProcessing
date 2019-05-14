# -*- coding: utf-8 -*-
"""
Inspired by: https://scipy-cookbook.readthedocs.io/items/KalmanFiltering.html

Created on Mon May 13 17:04:29 2019

#simple filtering using kalman filter

@author: MdNazmus
"""
import numpy as np
import matplotlib.pyplot as plt

lenx = 500

#data
x = np.linspace(-10,10,lenx)

#system output: trying different nonlinear and linear system
#y = 5*np.ones(lenx)
y = np.sinc(x)
#y = np.sin(2*3.14*x)
#y = np.cos(2*3.14*x)
#y = np.exp(x)
#y = np.exp(x) + np.exp(-x)
#y = 1/(1 + np.exp(-x))

#system output after adding noise
yn = y + np.random.normal(0,0.4,lenx)

# kalman filter implementation
Q = 1e-5 # process variance

# initialize arrays
xhat=np.zeros(lenx)      # a posteri estimate of x
P=np.zeros(lenx)         # a posteri error estimate
xhatminus=np.zeros(lenx) # a priori estimate of x
Pminus=np.zeros(lenx)    # a priori error estimate
K=np.zeros(lenx)         # gain or blending factor

R = 0.01**2 # estimate of measurement variance, change to see effect

# intial guesses
xhat[0] = 0.0
P[0] = 1.0

for k in range(1,lenx):
    # time update
    xhatminus[k] = xhat[k-1]
    Pminus[k] = P[k-1]+Q

    # measurement update
    K[k] = Pminus[k]/( Pminus[k]+R )
    xhat[k] = xhatminus[k]+K[k]*(yn[k]-xhatminus[k])
    P[k] = (1-K[k])*Pminus[k]




#plotting part
plt.figure()
plt.subplot(2,1,1)
plt.plot(x,yn,color = 'b',linewidth = '1',label="yn")
plt.plot(x,y, color = 'r', linewidth = '1.5', label = "y")
plt.plot(x,xhat, color = 'g', linewidth = '2.5', label = "x^")
plt.title("Noisy, Noiseless system, estimated output")
plt.xlabel("x")
plt.ylabel("y, yn, x^")
plt.legend(loc = "upper left")
plt.grid(True)

# error plotting with iteration
plt.subplot(2,1,2)
plt.plot(range(0,lenx),np.abs(yn-xhat),color = 'b',linewidth = '1',label="Error")
#plt.plot(range(0,lenx),Pminus[range(0,lenx)],color = 'b',linewidth = '1',label="Error")
plt.title("Absolute Error during each Iteration")
plt.xlabel("iteration #")
plt.ylabel("Error")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("outputgraphs.png",dpi=320)
