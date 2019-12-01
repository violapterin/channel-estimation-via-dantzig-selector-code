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

np.random.seed (0)
random.seed (0)

ver = cls.Version (cls.Size.MEDIUM, cls.Focus.DDSS)

ff_bb = fct.pick_mat_bb (ver).T
ff_rr = fct.pick_mat_rr (ver).T
ww_bb = fct.pick_mat_bb (ver)
ww_rr = fct.pick_mat_rr (ver)
hh = fct.pick_hh (ver)
zz = fct.pick_zz (ver)
kk = fct.get_kk (ver)
gg = kk.conj().T @ hh @ kk

pp = np.kron (
    ff_bb.T @ ff_rr.T @ kk.conj(),
    ww_bb @ ww_rr @ kk)
g = fct.vectorize (gg)
z = fct.vectorize (zz)
y = pp @ g + z
pp_rep = fct.find_rep_mat (pp)
y_rep = fct.find_rep_vec (y)
est = cls.Estimation (pp_rep, y_rep, hh, ver)

for h_g in [0.001, 0.1, 0.4, 6.4, 20]:
    fct.ddss_llpp_2 (est, h_g, ver)
    print ("est: ", est.d)
    fct.lasso_qqpp (est, h_g, ver)
    print ("est: ", est.d)
    print ("done")

quit()

for h_g in [0.1,0.3,0.5]:
    fct.oommpp_fixed_times (est, h_g, ver)
    print ("est: ", est.d)
    fct.oommpp_2_norm (est, h_g, ver)
    print ("est: ", est.d)
    fct.oommpp_infty_norm (est, h_g, ver)
    print ("est: ", est.d)

quit()

np.random.seed (0)
pp = np.random.normal (size = (2,5))
print (np.linalg.norm (pp.T @ pp))
g = np.random.normal (size = (5))
z = np.random.normal (size = (2))
y = pp @ g + z

quit()


g_hat = cp.Variable (5)
prob = cp.Problem (
    cp.Minimize (cp.norm (g_hat, 1)),
    [pp.T @ (y - pp @ g_hat) <= 0.422001])
prob.solve ()

print (np.linalg.norm (g_hat.value))

quit ()

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
print (np.linalg.norm (qq @ np.diag (d) @ qq.T - pp.T @ pp))

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


def foo (k, x):
    baz (2*k, x)

def qux (k, x):
    baz (5*k, x)

def baz (k, x):
    print (k*(x**2))

lst_fun = []
for i in range (4):
    lst_fun.append (lambda x, i_0 = i : foo (i_0, x))
for i in range (4):
    lst_fun.append (lambda x, i_0 = i : qux (i_0, x))

for fun in lst_fun:
    fun (3)
