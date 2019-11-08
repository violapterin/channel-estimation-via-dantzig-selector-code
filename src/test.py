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

ver = cls.Version (cls.Size.TEST, cls.Focus.DDSS)

ret =np.zeros ((cst.NN_HH (ver), cst.NN_HH (ver)), dtype=complex)
for _ in range (cst.LL ()):
    alpha = (np.random.normal (0, cst.NN_HH (ver) / cst.LL ())
        + 1J * np.random.normal (0, cst.NN_HH (ver) / cst.LL ()))
    phi = (2 * np.pi * (cst.DIST_ANT () /cst.LAMBDA_ANT ())
        * np.sin (np.random.uniform (0, 2 * np.pi)))
    theta = (2 * np.pi * (cst.DIST_ANT () /cst.LAMBDA_ANT ())
        * np.sin (np.random.uniform (0, 2 * np.pi)))
    ret += alpha * np.outer (fct.arr_resp (phi, ver), fct.arr_resp (theta, ver))
    print (ret)


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


