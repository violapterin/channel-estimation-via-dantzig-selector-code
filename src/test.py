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



a = np.random.normal (0, 1) + 1J * np.random.normal (0, 1)
a = a / np.abs (a)
arg = np.angle (a)
scale = 32 / (2 * np.pi)
arg = np.round (scale * arg) / scale
b = np.exp (1J * arg)
print (b - a)

quit ()

nn_iter = 128
s_g = 3
g_g = 1
nn_p = 128
nn_m = 16
e_tot_llss = 0
e_tot_ddss_dir = 0
e_tot_ddss_dia = 0

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

for _ in range (nn_iter):
    g = np.random.normal (0, 1, (nn_p))
    idx_mag = np.argsort (abs (g))
    bb = np.sort (idx_mag [0 : nn_p - nn_m]).tolist ()
    aa = [i for i in range (nn_p) if i not in bb]
    g_aa = np.diag (g [aa])

    z = np.random.normal (0, 1, (nn_m))
    pp = np.random.normal (0, 1, (nn_m, nn_p))
    y = pp @ g + z

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    d_llss = np.linalg.pinv (pp) @ y - g
    #print (d_llss)
    e_tot_llss += np.linalg.norm (d_llss)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    g_hat = cp.Variable (nn_p)
    prob = cp.Problem (
        cp.Minimize (cp.norm (g_hat, 1)),
        [cp.norm (pp.conj().T @ (y - pp @ g_hat), "inf") <= g_g])

    prob.solve (solver = cp.ECOS)
    d_ddss = g - g_hat.value
    #print (d_ddss)
    e_tot_ddss_dir += np.linalg.norm (d_ddss)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    c = np.ones ((nn_p))
    d, qq = np.linalg.eigh (pp.conj().T @ pp)
    k1 = g_g * qq.T @ c + qq.T @ pp.conj().T @ y
    k2 = g_g * qq.T @ c - qq.T @ pp.conj().T @ y

    idx_mag = np.argsort (abs (d))
    bb = np.sort (idx_mag [0 : nn_p - nn_m]).tolist ()
    aa = [i for i in range (nn_p) if i not in bb]
    dd_aa = np.diag (d [aa])
    k1_aa = k1 [aa]
    k1_bb = k1 [bb]
    k2_aa = k2 [aa]
    k2_bb = k2 [bb]
    qq_aa = qq [:, aa]
    qq_bb = qq [:, bb]
    qq_t_c_aa = (qq.T @ c) [aa]

    xp_bb = cp.Variable (nn_p)
    xm_bb = cp.Variable (nn_p)
    yp_aa = cp.Variable (nn_m)
    ym_aa = cp.Variable (nn_m)
    s1_aa = cp.Variable (nn_m)
    s2_aa = cp.Variable (nn_m)

    prob = cp.Problem (
        cp.Minimize (
            qq_t_c_aa @ yp_aa + c @ xp_bb +
            qq_t_c_aa @ ym_aa + c @ xm_bb),
        [dd_aa @ yp_aa - dd_aa @ ym_aa + s1_aa == k1_aa,
            - dd_aa @ yp_aa + dd_aa @ ym_aa + s2_aa == k2_aa,
            - qq_aa @ yp_aa - xp_bb <= 0,
            - qq_aa @ ym_aa - xm_bb <= 0,
            - qq_aa @ s1_aa - qq_bb @ k1_bb <= 0,
            - qq_aa @ s2_aa - qq_bb @ k2_bb <= 0])
    prob.solve (solver = cp.ECOS)

    g_hat = qq_aa @ yp_aa.value + xp_bb.value - qq_aa @ ym_aa.value - xm_bb.value
    d_ddss_dia = g - g_hat
    #print (d_ddss_dia)
    e_tot_ddss_dia += np.linalg.norm (d_ddss_dia)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

e_tot_llss /= nn_iter
e_tot_ddss_dir /= nn_iter
e_tot_ddss_dia /= nn_iter
print ("LS    : ", e_tot_llss)
print ("DS dir: ", e_tot_ddss_dir)
print ("DS dia: ", e_tot_ddss_dia)

quit ()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

kk = np.zeros ((nn_p, nn_p), dtype=complex)
for i in range (nn_p):
    for j in range (nn_p):
        kk [i] [j] = ((1 / np.sqrt (nn_p))
            * np.exp (2 * np.pi * 1J * i * j / nn_p))
aa = np.random.normal ((nn_p, nn_p))
bb = kk.conj().T @ aa @ kk
cc = kk @ aa @ kk.conj().T
m = np.linalg.norm (cc - aa)
print (m)

quit ()

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


