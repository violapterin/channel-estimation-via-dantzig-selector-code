#! /usr/bin/env python3

import numpy as np
import scipy as sp
import cvxpy as cp
import matplotlib.pyplot as plt
from enum import Enum
import os
import time

import constants as cst
import classes as cls
import functions as fct
import random

def test_beta ():
    nn_yy = 6
    nn_hh = 24
    nn_y = nn_yy ** 2
    nn_h = nn_hh ** 2
    ll = 4
    card_ss_est_2 = 2 * nn_yy
    card_ss_est_1 = int ((card_ss_est_2 * nn_h) ** (1/2))
    s_g = 0.2

    err_rel_llss = 0
    err_rel_ddss = 0
    err_rel_ddss_ddss = 0
    err_rel_ddss_ddss_llss = 0
    num_rep = 16

    pp = (np.random.normal (0, 1, (nn_y, nn_h))
            + 1J * np.random.normal (0, 1, (nn_y, nn_h)))
    pp /= np.sqrt (nn_y/np.sqrt(2))

    for _ in range (num_rep):
        print ("experiment", _)

        pp = (np.random.normal (0, 1, (nn_y, nn_h))
                + 1J * np.random.normal (0, 1, (nn_y, nn_h)))
        pp /= np.sqrt (nn_y/np.sqrt(2))

        kk = np.zeros ((nn_hh, nn_hh), dtype = complex)
        for n_1 in range (nn_hh):
            for n_2 in range (nn_hh):
                kk [n_1] [n_2] = ((1 / np.sqrt (nn_hh))
                    * np.exp (2 * np.pi * 1J * n_1 * n_2 / nn_hh))

        hh = np.zeros ((nn_hh, nn_hh), dtype = complex)
        for _ in range (ll):
            alpha = np.random.normal (0, nn_hh / ll) + 1J * np.random.normal (0, nn_hh / ll)
            phi = 2 * np.pi * np.sin (np.random.uniform (0, 2 * np.pi))
            theta = 2 * np.pi * np.sin (np.random.uniform (0, 2 * np.pi))
        hh += (alpha / nn_hh) * np.outer (
                np.array ([np.exp (1J * i * phi) for i in range (nn_hh)]),
                np.array ([np.exp (1J * i * theta) for i in range (nn_hh)]))
        hh = kk @ hh @ kk.conj().T
        h = fct.vectorize (hh)

        z = np.random.normal (0, s_g, (nn_y)) + 1J * np.random.normal (0, s_g, (nn_y))
        y = pp @ h + z
        y_rep = fct.find_rep_vec (y)
        pp_rep = fct.find_rep_mat (pp)
        norm_hh = np.linalg.norm (hh, ord='fro')

        # LS
        h_rep_hat_0th = np.linalg.pinv (pp_rep) @ y_rep
        h_hat_llss = fct.inv_find_rep_vec (h_rep_hat_0th)
        hh_hat_llss = fct.inv_vectorize (h_hat_llss, nn_hh, nn_hh)
        err_rel_llss += np.linalg.norm (hh - hh_hat_llss, ord='fro') / norm_hh

        # DS
        h_rep = cp.Variable (2 * nn_h)
        h_rep_abs = cp.Variable (2 * nn_h)
        k = pp_rep.T @ y_rep
        qq = pp_rep.T @ pp_rep
        c = np.ones ((2 * nn_h))
        g_g = np.sqrt (2 * np.log (2 * nn_h)) * s_g

        prob = cp.Problem (
            cp.Minimize (c.T @ h_rep_abs),
            [h_rep - h_rep_abs <= 0,
                - h_rep - h_rep_abs <= 0,
                qq @ h_rep - g_g * c <= k,
                - qq @ h_rep - g_g * c <= - k])

        prob.solve (solver = cp.ECOS)
        h_rep_hat_1 = h_rep.value
        h_hat_ddss = fct.inv_find_rep_vec (h_rep_hat_1)
        hh_hat_ddss = fct.inv_vectorize (h_hat_ddss, nn_hh, nn_hh)
        err_rel_ddss += np.linalg.norm (hh - hh_hat_ddss, ord='fro') / norm_hh

        # DS, DS
        ss_est_1 = np.sort (np.argsort (np.abs (h_rep_hat_1)) [-card_ss_est_1:])
        pp_rep_1 = pp_rep [:, ss_est_1]

        h_rep = cp.Variable (card_ss_est_1)
        h_rep_abs = cp.Variable (card_ss_est_1)
        k = pp_rep_1.T @ y_rep
        qq = pp_rep_1.T @ pp_rep_1
        c = np.ones ((card_ss_est_1))
        g_g = np.sqrt (2 * np.log (card_ss_est_1)) * s_g

        prob = cp.Problem (
            cp.Minimize (c.T @ h_rep_abs),
            [h_rep - h_rep_abs <= 0,
                - h_rep - h_rep_abs <= 0,
                qq @ h_rep - g_g * c <= k,
                - qq @ h_rep - g_g * c <= - k])

        prob.solve (solver = cp.ECOS)
        h_rep_hat_2 = h_rep.value
        h_rep_hat = np.zeros ((2 * nn_h))
        for i in range (card_ss_est_1):
            h_rep_hat [ss_est_1 [i]] = h_rep_hat_2 [i]
        h_hat_ddss_ddss = fct.inv_find_rep_vec (h_rep_hat)
        hh_hat_ddss_ddss = fct.inv_vectorize (h_hat_ddss_ddss, nn_hh, nn_hh)
        err_rel_ddss_ddss += np.linalg.norm (hh - hh_hat_ddss_ddss, ord='fro') / norm_hh

        # DS, DS, LS
        ss_est_2 = np.sort (np.argsort (np.abs (h_rep_hat_2)) [-card_ss_est_2:])
        pp_rep_2 = pp_rep_1 [:, ss_est_2]

        h_rep_hat_3 = np.linalg.pinv (pp_rep_2) @ y_rep
        h_rep_hat = np.zeros ((2 * nn_h))
        for i in range (card_ss_est_2):
            h_rep_hat [ss_est_1 [ss_est_2 [i]]] = h_rep_hat_3 [i]
        h_hat_ddss_ddss_llss = fct.inv_find_rep_vec (h_rep_hat)
        hh_hat_ddss_ddss_llss = fct.inv_vectorize (h_hat_ddss_ddss_llss, nn_hh, nn_hh)
        err_rel_ddss_ddss_llss += np.linalg.norm (hh - hh_hat_ddss_ddss_llss, ord='fro') / norm_hh


    err_rel_llss /= num_rep
    err_rel_ddss /= num_rep
    err_rel_ddss_ddss /= num_rep
    err_rel_ddss_ddss_llss /= num_rep
    print ("LS:         ", err_rel_llss)
    print ("DS:         ", err_rel_ddss)
    print ("DS, DS:     ", err_rel_ddss_ddss)
    print ("DS, DS, LS: ", err_rel_ddss_ddss_llss)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def test_alpha ():
    nn_yy = 4
    nn_hh = 16
    nn_y = nn_yy ** 2
    nn_h = nn_hh ** 2
    ll = 4
    card_ss_true = 2 * nn_yy # rep.
    card_ss_est_1 = int ((card_ss_true * nn_h) ** (1/2))
    card_ss_est_2 = card_ss_true
    s_g = 0.2
    scale = list (range (nn_h))

    pp = (np.random.normal (0, 1, (nn_y, nn_h))
            + 1J * np.random.normal (0, 1, (nn_y, nn_h)))
    pp /= np.sqrt (nn_y/np.sqrt(2))

    kk = np.zeros ((nn_hh, nn_hh), dtype = complex)
    for n_1 in range (nn_hh):
        for n_2 in range (nn_hh):
            kk [n_1] [n_2] = ((1 / np.sqrt (nn_hh))
                * np.exp (2 * np.pi * 1J * n_1 * n_2 / nn_hh))

    '''
    h_hold = np.random.normal (0, 1, (nn_h)) + 1J * np.random.normal (0, 1, (nn_h))
    ss_true = np.argsort (np.abs (h_hold)) [-card_ss_true:]
    h = np.zeros (nn_h, dtype = complex)
    for i in range (card_ss_true):
        h [ss_true [i]] = h_hold [ss_true[i]]
    '''
    hh = np.zeros ((nn_hh, nn_hh), dtype = complex)
    for _ in range (ll):
        alpha = np.random.normal (0, nn_hh / ll) + 1J * np.random.normal (0, nn_hh / ll)
        phi = 2 * np.pi * np.sin (np.random.uniform (0, 2 * np.pi))
        theta = 2 * np.pi * np.sin (np.random.uniform (0, 2 * np.pi))
    hh += (alpha / nn_hh) * np.outer (
            np.array ([np.exp (1J * i * phi) for i in range (nn_hh)]),
            np.array ([np.exp (1J * i * theta) for i in range (nn_hh)]))
    hh = kk @ hh @ kk.conj().T
    h = fct.vectorize (hh)


    z = np.random.normal (0, s_g, (nn_y)) + 1J * np.random.normal (0, s_g, (nn_y))
    y = pp @ h + z
    y_rep = fct.find_rep_vec (y)
    pp_rep = fct.find_rep_mat (pp)

    # LS
    h_rep_hat_0th = np.linalg.pinv (pp_rep) @ y_rep
    h_hat_llss = fct.inv_find_rep_vec (h_rep_hat_0th)

    # DS
    h_rep = cp.Variable (2 * nn_h)
    h_rep_abs = cp.Variable (2 * nn_h)
    k = pp_rep.T @ y_rep
    qq = pp_rep.T @ pp_rep
    c = np.ones ((2 * nn_h))
    g_g = np.sqrt (2 * np.log (2 * nn_h)) * s_g

    #prob = cp.Problem (cp.Minimize (cp.norm (h_rep, 1)),
    #    [cp.norm (pp_rep.T @ (y_rep - pp_rep @ h_rep), "inf") <= g_g])
    prob = cp.Problem (
        cp.Minimize (c.T @ h_rep_abs),
        [h_rep - h_rep_abs <= 0,
            - h_rep - h_rep_abs <= 0,
            qq @ h_rep - g_g * c <= k,
            - qq @ h_rep - g_g * c <= - k])

    prob.solve (solver = cp.ECOS)
    h_rep_hat_1 = h_rep.value
    h_hat_ddss = fct.inv_find_rep_vec (h_rep_hat_1)

    # DS, DS
    ss_est_1 = np.sort (np.argsort (np.abs (h_rep_hat_1)) [-card_ss_est_1:])
    pp_rep_1 = pp_rep [:, ss_est_1]

    h_rep = cp.Variable (card_ss_est_1)
    h_rep_abs = cp.Variable (card_ss_est_1)
    k = pp_rep_1.T @ y_rep
    qq = pp_rep_1.T @ pp_rep_1
    c = np.ones ((card_ss_est_1))
    g_g = np.sqrt (2 * np.log (card_ss_est_1)) * s_g

    prob = cp.Problem (
        cp.Minimize (c.T @ h_rep_abs),
        [h_rep - h_rep_abs <= 0,
            - h_rep - h_rep_abs <= 0,
            qq @ h_rep - g_g * c <= k,
            - qq @ h_rep - g_g * c <= - k])

    prob.solve (solver = cp.ECOS)
    h_rep_hat_2 = h_rep.value
    h_rep_hat = np.zeros ((2 * nn_h))
    for i in range (card_ss_est_1):
        h_rep_hat [ss_est_1 [i]] = h_rep_hat_2 [i]
    h_hat_ddss_ddss = fct.inv_find_rep_vec (h_rep_hat)

    # DS, DS, LS
    ss_est_2 = np.sort (np.argsort (np.abs (h_rep_hat_2)) [-card_ss_est_2:])
    pp_rep_2 = pp_rep_1 [:, ss_est_2]

    h_rep_hat_3 = np.linalg.pinv (pp_rep_2) @ y_rep
    h_rep_hat = np.zeros ((2 * nn_h))
    for i in range (card_ss_est_2):
        h_rep_hat [ss_est_1 [ss_est_2 [i]]] = h_rep_hat_3 [i]
    h_hat_ddss_ddss_llss = fct.inv_find_rep_vec (h_rep_hat)

    # Plot
    print ("1. LS:         ", np.linalg.norm (h - h_hat_llss) / np.linalg.norm (h))
    print ("2. DS:         ", np.linalg.norm (h - h_hat_ddss) / np.linalg.norm (h))
    print ("3. DS, DS:     ", np.linalg.norm (h - h_hat_ddss_ddss) / np.linalg.norm (h))
    print ("4. DS, DS, LS: ", np.linalg.norm (h - h_hat_ddss_ddss_llss) / np.linalg.norm (h))

    plt.plot (scale, abs (h),
            marker = 'o', # circle
            markersize = 8, linestyle = "None")
    plt.plot (scale, abs (h_hat_ddss),
            marker = 'D', # diamond
            markersize = 4, linestyle = "None")
    plt.plot (scale, abs (h_hat_ddss),
            marker = 'v', # triangle down
            markersize = 5, linestyle = "None")
    plt.plot (scale, abs (h_hat_ddss_ddss),
            marker = '^', # triangle up
            markersize = 6, linestyle = "None")
    plt.plot (scale, abs (h_hat_ddss_ddss_llss),
            marker = 's', # square
            markersize = 7, linestyle = "None")
    plt.show ()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

test_beta ()

quit ()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

pp = np.array ([[5,-6,-7,3,-3,1], [12,-4,-16,-9,9,1]])
g = np.array ([4,2,8,7,-9,0])
idx_sort = np.argsort (np.abs (g))
ss = idx_sort [-2:]
pp_ss = pp [:, ss]
print (pp)
print (g)
print (idx_sort)
print (ss)
print (pp_ss)

quit ()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def mask (arr, threshold):
    s = np.argsort (np.abs (arr))
    num_mask = int (np.round (len (arr) * threshold))
    for i in s [:num_mask]:
        arr [i]=0

a = np.random.randint (-20, 20, size = 10)
print (a)
mask (a, 0.3)
print (a)

quit ()

print (a)
b = np.argsort (np.abs (a))
l = int (np.round (len (a)/2))
for i in b[:l]:
    a [i]=0

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

    g_hat = cp.Variable (nn_p)
    prob = cp.Problem (
        cp.Minimize (cp.norm (g_hat, 1)),
        [cp.norm (pp.conj().T @ (y - pp @ g_hat), "inf") <= g_g])

    prob.solve (solver = cp.ECOS)
    d_ddss = g - g_hat.value
    #print (d_ddss)
    e_tot_ddss_dir += np.linalg.norm (d_ddss)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

