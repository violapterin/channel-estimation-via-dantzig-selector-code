#! /usr/bin/env python3

import matplotlib.pyplot as plt # plotting functions
import os
import numpy as np
import cvxpy as cp
from enum import Enum

import constants as cst
import classes as cls
import functions as fct

def foo (): baz ()

def baz (): print ("Hey")

foo ()

quit()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

ff_bb = fct.ff_bb()
ff_rr = fct.ff_rr()
ww_bb = fct.ww_bb()
ww_rr = fct.ww_rr()
hh = fct.hh()
zz = fct.zz()
gg = fct.kk().conj().T @ hh @ fct.kk()

pp = np.kron (
    ff_bb.T @ ff_rr.T @ fct.kk().conj(),
    ww_bb @ ww_rr @ fct.kk())
g = fct.vectorize (gg)
z = fct.vectorize (zz)
sigma = 2
y = pp @ g + sigma * z

funs = []
lst_gamma = [1,2,4,8]
for gamma in lst_gamma:
    funs.append (lambda x, y: alg.ddss_complex (x, y * gamma))
for fun in funs:
    est = alg.Estimation (pp, y, hh)
    fun (est, sigma)
    print (est.d)
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


