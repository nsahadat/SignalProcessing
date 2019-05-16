# -*- coding: utf-8 -*-
"""
Created on Wed May 15 21:47:50 2019

@author: mnsah
"""

import numpy as np
x = np.array([[2,2],[3,3],[4,4]])

xnorm = x - np.mean(x.T,axis=1)

print(xnorm)

# find the covariance
Cx = np.dot(xnorm.T,xnorm)/(len(x)-1)
# print the value of covariance matrix
print(Cx)
# calculate eigen value eigen vector for the purpose of diagonalization
Sig, U = np.linalg.eig(Cx)
# print the values of U and Sig
print(U)
print(Sig)

# Now let's use the new basis to find the new transformed values
# old value [1,1]
print(np.dot(U[:,0].T,np.array([[1,1],[5,5],[6,6],[9,9]]).T))

#from 1D to 2D transformation 5*1.41 to [5 5]
print(np.dot(U[:,0],7.0710678))