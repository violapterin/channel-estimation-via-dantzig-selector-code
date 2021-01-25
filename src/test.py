#! /usr/bin/env python3

import numpy as np
import scipy as sp
import cvxpy as cp
import random

import constants as cst
import classes as cls
import functions as fct

aa = np.array ([[3,1,5], [2,4,6], [7,7,-7]])
bb = np.array ([[9,7,-8], [-1,0,3]])
x = fct.vectorize (aa)
y = fct.vectorize (bb)
z = np.concatenate ((x,y), axis=0)
cc = np.concatenate ((aa,bb), axis=0)
print (z)
print (cc)

'''
a = np.array ([1,3,4])
b = np.array ([6,-2,-7])
c = np.concatenate ((a,b))
print (a)
print (b)
print (c)
'''


'''
a = np.array ([1,3,4,2,-7]).T
ss = fct.get_largest_index (a, 2)
print ("a = ", a)
print ("ss = ", ss)
#a.T [ss] = np.array ([9,9]).T
b = np.array ([9,8]).T
fct.assign_subvec (a, ss, b)
print ("a = ", a)
quit ()
'''

'''
np.set_printoptions (precision=2)

nn = 5
s = 2
mm = 3
g_ss = np.random.normal (0, 1, (s, 1))
ss = np.random.choice (range (nn), s)
g = np.zeros ((nn, 1))
fct.embed_subvec (g, ss, g_ss)
pp = np.random.normal (0, 1, (mm, nn))
z = np.random.normal (0, 1/4, (mm, 1))
y = pp @ g + z
f_h = np.linalg.pinv (pp) @ y
ss_h = fct.get_supp (f_h, s)
g_h = np.zeros ((nn, 1))
g_h = fct.mask_vec (g_h, f_h)


print (f_h)
print (g_h)
'''

quit ()


def test_beta ():
    nn_yy = 8
    nn_hh = 16
    nn_y = nn_yy ** 2
    nn_h = nn_hh ** 2
    ll = 4
    num_rep = 6
    crd_true = int (np.sqrt (nn_y))
    crd_rep_true = 2 * nn_y # rep.
    crd_rep_0 = 2 * nn_h
    crd_rep_est_2 = crd_rep_true
    c_decay = 2 ^ (-6)
    #crd_est_1 = (crd_est_0 * crd_est_2) ** (1/2)
    crd_rep_est_1 = int ((crd_rep_0 + crd_rep_est_2) / 2)
    s_g = 0.2
    scale = list (range (nn_h))

    os.system ("rm -r ../tmp")
    os.system ("mkdir ../tmp")

    for i_exp in range (num_rep):
        print ("experiment:", i_exp)
        pp = (np.random.normal (0, 1, (nn_y, nn_h))
                + 1J * np.random.normal (0, 1, (nn_y, nn_h)))
        pp /= np.sqrt (nn_y/np.sqrt(2))

        kk = np.zeros ((nn_hh, nn_hh), dtype = complex)
        for n_1 in range (nn_hh):
            for n_2 in range (nn_hh):
                kk [n_1] [n_2] = ((1 / np.sqrt (nn_hh))
                    * np.exp (2 * np.pi * 1J * n_1 * n_2 / nn_hh))

        hh = np.zeros ((nn_hh, nn_hh), dtype = complex)
        for l in range (ll):
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

        prob = cp.Problem (
            cp.Minimize (c.T @ h_rep_abs),
            [h_rep - h_rep_abs <= 0,
                - h_rep - h_rep_abs <= 0,
                qq @ h_rep - g_g * c <= k,
                - qq @ h_rep - g_g * c <= - k])

        prob.solve (solver = cp.ECOS)
        h_rep_hat_1 = h_rep.value
        h_hat_ddss = fct.inv_find_rep_vec (h_rep_hat_1)

        # Plot
        h_abs = np.abs (h)
        h_hat_abs_llss = np.abs (h_hat_llss)
        h_hat_abs_ddss = np.abs (h_hat_ddss)
        h_abs_sort, h_hat_abs_llss_sort, h_hat_abs_ddss_sort = map (
                list, zip (*sorted (zip (h_abs, h_hat_abs_llss, h_hat_abs_ddss),
                reverse = True)))
        sc = np.array (range (len(h_abs_sort)))

        plt.plot(sc, h_abs_sort, 
            marker = 'o', # circle
            markersize = 5, linestyle = "None")
        plt.plot(sc, h_hat_abs_llss_sort,
            marker = 'v', # triangle down
            markersize = 3, linestyle = "None")
        plt.plot(sc, h_hat_abs_ddss_sort,
            marker = 's', # square
            markersize = 4, linestyle = "None")
        nam_fil = "../tmp/" + str(i_exp).zfill(4) + ".png"
        print ("   ", nam_fil)
        plt.savefig (nam_fil)
        plt.close ()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def test_alpha ():
    nn_yy = 4
    nn_hh = 16
    nn_y = nn_yy ** 2
    nn_h = nn_hh ** 2
    ll = 4
    crd_est_2 = 2 * nn_yy
    crd_est_1 = int ((crd_est_2 * 2 * nn_h) ** (1/2))
    s_g = 0.2

    err_rel_llss = 0
    err_rel_ddss = 0
    err_rel_ddss_ddss = 0
    err_rel_ddss_ddss_llss = 0
    num_rep = 8

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
        h_rep_hat_0 = np.linalg.pinv (pp_rep) @ y_rep
        h_hat_llss = fct.inv_find_rep_vec (h_rep_hat_0)
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
        ss_est_1 = np.sort (np.argsort (np.abs (h_rep_hat_1)) [-crd_rep_est_1:])
        pp_rep_1 = pp_rep [:, ss_est_1]

        h_rep = cp.Variable (crd_rep_est_1)
        h_rep_abs = cp.Variable (crd_rep_est_1)
        k = pp_rep_1.T @ y_rep
        qq = pp_rep_1.T @ pp_rep_1
        c = np.ones ((crd_rep_est_1))
        g_g = np.sqrt (2 * np.log (crd_rep_est_1)) * s_g

        prob = cp.Problem (
            cp.Minimize (c.T @ h_rep_abs),
            [h_rep - h_rep_abs <= 0,
                - h_rep - h_rep_abs <= 0,
                qq @ h_rep - g_g * c <= k,
                - qq @ h_rep - g_g * c <= - k])

        prob.solve (solver = cp.ECOS)
        h_rep_hat_ss_2 = h_rep.value
        h_rep_hat_2 = np.zeros ((2 * nn_h))
        for i in range (crd_est_1):
            h_rep_hat_2 [ss_est_1 [i]] = h_rep_hat_ss_2 [i]
        h_hat_ddss_ddss = fct.inv_find_rep_vec (h_rep_hat_2)
        hh_hat_ddss_ddss = fct.inv_vectorize (h_hat_ddss_ddss, nn_hh, nn_hh)
        err_rel_ddss_ddss += np.linalg.norm (hh - hh_hat_ddss_ddss, ord='fro') / norm_hh

        # DS, DS, LS
        ss_est_2 = np.sort (np.argsort (np.abs (h_rep_hat_2)) [-crd_est_2:])
        pp_rep_2 = pp_rep [:, ss_est_2]

        h_rep_hat_ss_3 = np.linalg.pinv (pp_rep_2) @ y_rep
        h_rep_hat_3 = np.zeros ((2 * nn_h))
        for i in range (crd_est_2):
            h_rep_hat_3 [ss_est_2 [i]] = h_rep_hat_ss_3 [i]
        h_hat_ddss_ddss_llss = fct.inv_find_rep_vec (h_rep_hat_3)
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
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

quit ()

