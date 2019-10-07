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



a = cp.Variable (2)
b = cp.Variable (2)
prob = cp.Problem (
    cp.Minimize (cp.norm (2*a+3*b, 2)),
    [cp.norm (a)**3 + cp.norm (b)**3 <= 64, a>=0, b>=0])

prob.solve ()
print (a.value)
print (b.value)

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


