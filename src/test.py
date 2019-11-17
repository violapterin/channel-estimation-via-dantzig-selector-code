#! /usr/bin/env python3

import matplotlib.pyplot as plt # plotting functions
import os
import numpy as np
import scipy as sp
import cvxpy as cp
import time
from enum import Enum

import constants as cst
import classes as cls
import functions as fct
import random

def foo (a):
    a += 1
    print (a)

def mask_low (arr):
    arr = arr+0.01
    s =arr.shape
    num_supp = int (np.sqrt (s[0] * s[1])) + 1
    arr_abs = abs(arr)
    arr_vec = fct.vectorize (arr)
    arr_abs_vec = fct.vectorize (arr_abs)
    idx_mag = np.argpartition (arr_abs_vec, num_supp)
    idx_low = idx_mag [0:1-num_supp]
    arr_vec [idx_low] = 0
    arr = arr_vec
    arr = fct.inv_vectorize (arr_vec, s [0], s [1])
    return arr


a = 3
foo (a)
foo (a)
print (a)
quit ()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

mm = 20
hold_sum_det =0
for m in range (mm):
    ver = cls.Version (cls.Size.MEDIUM, cls.Focus.ASSORTED)
    ff_bb =fct.pick_mat_bb (ver).T
    ff_rr =fct.pick_mat_rr (ver).T
    ww_bb =fct.pick_mat_bb (ver)
    ww_rr =fct.pick_mat_rr (ver)

    kk = fct.get_kk (ver)
    pp = np.kron (ff_bb.T @ ff_rr.T @ kk.conj(), ww_bb @ ww_rr @ kk)
    d = np.linalg.det (pp.T @ pp)
    hold_sum_det += d
print ("det: ", hold_sum_det / mm)
quit ()


ver = cls.Version (cls.Size.MEDIUM, cls.Focus.ASSORTED)
ww_bb =fct.pick_mat_bb (ver)
ww_rr =fct.pick_mat_rr (ver)
print (np.linalg.det (ww_bb.conj().T @ ww_bb))
print (np.linalg.det (ww_rr.conj().T @ ww_rr))
print (np.linalg.det (ww_bb @ ww_bb.conj().T))
print (np.linalg.det (ww_rr @ ww_rr.conj().T))
quit ()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

aa = np.zeros ((4,3))
aa [2][1] =1
aa [0][1] =3
print (aa)
aa = mask_low (aa)
print (aa)
quit ()


quit ()


mm = 10
t1 = time.time()
for _ in range (mm):
    np.linalg.inv (pp.T @ pp)
    t2 = time.time()
print ("time: ", (t2-t1)/mm)

quit ()
