#! /usr/bin/env python3

import matplotlib.pyplot as plt # plotting functions
import os
import numpy as np
import scipy as sp
import cvxpy as cp
from enum import Enum

#import constants as cst
#import classes as cls
#import functions as fct
import random

aa = np.eye (3)
zz = np.zeros ((3,3))
aa = sp.sparse.coo_matrix (aa)
zz = sp.sparse.coo_matrix (zz)
print (aa)
print (zz)
quit ()

a = np.array ([[1,0], [0,5]])
b = a
a = sp.sparse.coo_matrix (a)
c = a.T @ a + a
print (np.linalg.norm (c.toarray()))
#print (np.linalg.norm (b.T @ a))
#print (np.linalg.norm (a.T @ a))
quit()

a =np.array ([[1,1,2,2,3,3,4,4], [5,5,6,6,7,7,8,8]])
b =a [:,0:2]
cc =np.block ([[np.eye (3), np.eye (3)], [-np.eye (3), np.eye (3)]])
e1 =np.array ([1,2,3,4])
e2 =np.ones ((4))
d =np.block ([e1, e2])
print (b)
quit()

a =np.array ([[1,2], [3,4]])
b =np.array ([[2,4], [6,8]])
c =np.array ([[0,0], [0,0]])
d =np.array ([[0,0], [0,0]])
aa =np.block([[a,b],[c,d]])
print (aa)
quit()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


for _ in range (500):
    print ("experiment:", _)
    pp =np.random.normal (0, 1, (10,50))
    y =np.random.normal (0, 1, (10))
    g = cp.Variable (50)
    prob = cp.Problem (
        cp.Minimize (cp.norm (g, 1)),
        [cp.norm (pp.conj().T @ (pp @ g - y), "inf") <= 0.5])
    prob.solve (solver = cp.CVXOPT)
    g_hat = g.value
    print (np.linalg.norm (g_hat, ord=1))


