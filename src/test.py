#! /usr/bin/env python3

import matplotlib.pyplot as plt # plotting functions
import os
import numpy as np
import cvxpy as cp
from enum import Enum

import constants as cst
import classes as cls
import functions as fct
import random


a=np.zeros ((2,2), dtype='bool')
a[0][0]=1
a[1][1]=1
b =[[3.3,0],[0,4.4]]
print (a@b)

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


