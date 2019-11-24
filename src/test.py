#! /usr/bin/env python3

import matplotlib.pyplot as plt # plotting functions
import os
import numpy as np
import scipy as sp
import cvxpy as cp
import time
from enum import Enum

#import constants as cst
#import classes as cls
#import functions as fct
#import random

nn_m = 2
nn_p = 5
pp = np.random.normal (size=(nn_m, nn_p))
d, qq = np.linalg.eigh (pp.T @ pp)
idx_mag = np.argsort (abs (d))
bb = np.sort (idx_mag [0 : nn_p - nn_m]).tolist ()
aa = [i for i in range (nn_p) if i not in bb]
d_aa = d [aa]
d_bb = d [bb]
qq_aa = qq [:, aa]
qq_bb = qq [:, bb]
print (qq_aa)
print (qq_bb)

quit ()

nn_m = 3
d = np.array ([4, 3, -100, -6, 7])
k = np.array ([-123, 9, 7, -8, 0])
idx_mag = np.argsort (abs (d))
aa = np.sort (idx_mag [0 : nn_m]).tolist ()
bb = [i for i in range (len (d)) if i not in aa]
#d_aa = np.array ([d[i] for i in aa])
d_aa = d [aa]
d_bb = np.array ([d[i] for i in bb])
k_aa = np.array ([k[i] for i in aa])
k_bb = np.array ([k[i] for i in bb])
cc = np.ones ((5,5))

print (d_aa)
print (cc [:, aa])


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

def mask_low (arr):
    sh =arr.shape
    num_supp = int (np.sqrt (sh [0] * sh [1])) + 1
    arr_abs = abs(arr)
    arr_vec = fct.vectorize (arr)
    arr_abs_vec = fct.vectorize (arr_abs)
    idx_mag = np.argpartition (arr_abs_vec, num_supp)
    idx_low = idx_mag [0:1-num_supp]
    arr_vec [idx_low] = 0
    arr = arr_vec
    arr = fct.inv_vectorize (arr_vec, s [0], s [1])
    return arr


