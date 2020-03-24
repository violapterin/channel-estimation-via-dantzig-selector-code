import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import time
import random

import constants as cst
import classes as cls
import cvxpy as cp

def execute (ver):
    arr_s_g = (cst.S_G_INIT (ver)
            * cst.VALUE_SPACING_S_G (ver) ** (np.array (range (cst.NUM_S_G (ver)))))
    lst_h_g = list (cst.VALUE_SPACING_H_G (ver) ** (np.array (range (cst.NUM_H_G (ver)))))
    lst_g_g = list (cst.VALUE_SPACING_G_G (ver) ** (np.array (range (cst.NUM_G_G (ver)))))

    lst_legend = []
    lst_num_rep_meth = []
    lst_meth = [] # arguments: (est, s_g)

    if (ver.focus == cls.Focus.ASSORTED):
        # Least Square
        lst_legend.append ("LS")
        lst_num_rep_meth.append (cst.NUM_REP_LLSS (ver))
        lst_meth.append (lambda x, y: llss (x, ver))
        # Lasso
        lst_legend.append ("Lasso")
        lst_num_rep_meth.append (cst.NUM_REP_LASSO (ver))
        lst_meth.append (lambda x, y: lasso_qqpp (x, cst.G_G_LASSO (ver) * y, ver))

    if (ver.focus == cls.Focus.OOMMPP or ver.focus == cls.Focus.ASSORTED):
        # Orthogonal Matching Pursuit: fixed iteration number
        lst_legend.append ("OMP, $N_H$ times")
        lst_num_rep_meth.append (cst.NUM_REP_OOMMPP (ver))
        lst_meth.append (lambda x, y: oommpp_fixed_times (x, cst.NN_HH (ver), ver))

        # Orthogonal Matching Pursuit: limited l-2 norm
        for h_g in lst_h_g:
            lst_legend.append ("OMP, $l_2$-norm, $\eta$ = " + '%.2f' % h_g + "$\sigma$")
            lst_num_rep_meth.append (cst.NUM_REP_OOMMPP (ver))
            lst_meth.append (
                lambda x, y, h_g = h_g:
                oommpp_2_norm (x, h_g * cst.H_G_OOMMPP_2_NORM (ver) * y, ver))

        # Orthogonal Matching Pursuit: limited l-infinity norm
        for h_g in lst_h_g:
            lst_legend.append ("OMP, $l_\infty$-norm, $\eta$ = " + '%.2f' % h_g + "$\sigma$")
            lst_num_rep_meth.append (cst.NUM_REP_OOMMPP (ver))
            lst_meth.append (
                lambda x, y, h_g = h_g:
                oommpp_infty_norm (x, h_g * cst.H_G_OOMMPP_INFTY_NORM (ver) * y, ver))

    if (ver.focus == cls.Focus.DDSS or ver.focus == cls.Focus.ASSORTED):
        # Dantzig Selector error bound
        lst_legend.append ("DS, theory")
        lst_num_rep_meth.append (cst.NUM_REP_DDSS (ver))
        lst_meth.append (lambda x, y: ddss_theory (x, y, ver))

        # Dantzig Selector: Linear Program
        #for g_g in lst_g_g:
        #    lst_legend.append ("DS, $\gamma$ = " + '%.2f' % g_g + "$\sigma$")
        #    lst_num_rep_meth.append (cst.NUM_REP_DDSS (ver))
        #    lst_meth.append (lambda x, y, g_g = g_g: ddss_llpp (x, g_g * y, ver))
        #lst_num_rep_meth.append (cst.NUM_REP_DDSS (ver))

        # Dantzig Selector: Linear Program, twice
        for g_g in lst_g_g:
            lst_legend.append ("DS twice, $\gamma$ = " + '%.2f' % g_g + "$\sigma$")
            lst_num_rep_meth.append (cst.NUM_REP_DDSS (ver))
            lst_meth.append (lambda x, y, g_g = g_g: ddss_llpp_twice (x, g_g * y, ver))
        lst_num_rep_meth.append (cst.NUM_REP_DDSS (ver))

    assert (len (lst_meth) == len (lst_legend))
    num_meth = len (lst_meth)

    cnt_meth = 0
    cnt_each = 0
    #lst_lst_err_abs = [] # each s_g, each method
    #lst_lst_err_rel = [] # each s_g, each method
    lst_lst_spec_eff = [] # each s_g, each method
    lst_lst_time = [] # each s_g, each method
    time_tot_start = time.time ()

    for i_s_g in range (cst.NUM_S_G (ver)):
        s_g = arr_s_g [i_s_g]
        print ("σ = ", '%.2f' % s_g, " :", sep = '')
        #lst_err_abs = [0] * num_meth
        #lst_err_rel = [0] * num_meth
        lst_spec_eff = [0] * num_meth
        lst_time = [0] * num_meth
        norm_hh = 0
        num_rep_tot = np.array (lst_num_rep_meth).sum () * cst.NUM_REP_HH (ver)

        for i_meth in range (num_meth):
            cnt_meth += 1
            num_rep_meth = lst_num_rep_meth [i_meth] * cst.NUM_REP_HH (ver)

            for _ in range (num_rep_meth):
                cnt_each += 1
                print ("\r", cnt_each, "...", sep = '', end = '', flush = True)

                beam = cls.Beamformer (ver)
                beam.generate ()
                chan = cls.Channel (beam.pp, s_g, ver)
                chan.zero ()
                chan.generate ()
                norm_hh_each = chan.norm_hh
                chan.transmit ()
                est = cls.Estimation (cnt_each, chan.hh, beam.pp, chan.y, s_g, ver)
                est.zero ()

                time_each_start = time.time ()
                lst_meth [i_meth] (est, s_g)
                time_each_stop = time.time ()

                #lst_err_abs [i_meth] += est.d / num_rep_meth
                #lst_err_rel [i_meth] += est.d / (norm_hh_each * num_rep_meth)
                lst_spec_eff [i_meth] += est.rr / num_rep_meth
                lst_time [i_meth] += np.log (time_each_stop - time_each_start) / num_rep_meth

            rate_progress = cnt_each / (cst.NUM_S_G (ver) * num_rep_tot)
            time_hold = time.time ()
            print ('', flush = True)
            print ("    experiment ", cnt_meth, sep = '', flush = True)
            print ("      (", '%.1f' % (100 * rate_progress), "%; ",
                '%.2f' % (
                    (time_hold - time_tot_start) * (1 - rate_progress) / (rate_progress * 60)),
                " min. remaining)",
                sep = '', flush = True)
        #lst_lst_err_abs.append (lst_err_abs)
        #lst_lst_err_rel.append (lst_err_rel)
        lst_lst_spec_eff.append (lst_spec_eff)
        lst_lst_time.append (lst_time)
        print ("                                ", end = '\r') # clear last line
        print ("    done")
    time_tot_stop = time.time ()

    print (cnt_each, "channel instances simulated")
    print (
        "averaged time elapsed for each experiment: ",
        '%.2f' %
            ((time_tot_stop - time_tot_start)
                / (60 * cst.NUM_S_G (ver) * cst.NUM_REP_HH (ver))),
        " (min)", flush = True)
    print (
        "total time elapsed: ",
        '%.2f' % ((time_tot_stop - time_tot_start) / 60),
        " (min)", flush = True)

    arr_x = np.array (np.log (arr_s_g))
    lst_lst_spec_eff = list (np.array (lst_lst_spec_eff).T) # each method, each s_g
    lst_arr_spec_eff = [np.array (lst) for lst in lst_lst_spec_eff]
    label_x = "Std. of noise (log)"
    label_y = "Spectral Efficiency"
    save_table (arr_x, lst_arr_spec_eff,
        label_x, label_y, lst_legend,
        "spectral", ver)
    save_plot (
        arr_x, lst_arr_spec_eff,
        label_x, label_y, lst_legend,
        "spectral", ver)

    '''
    arr_x = np.array (np.log (arr_s_g))
    lst_lst_err_abs = list (np.array (lst_lst_err_abs).T) # each method, each s_g
    lst_arr_err_abs = [np.array (lst) for lst in lst_lst_err_abs]
    label_x = "Std. of noise (log)"
    label_y = "Absolute Frobenius norm error"
    save_table (arr_x, lst_arr_err_abs,
        label_x, label_y, lst_legend,
        "absolute", ver)
    save_plot (
        arr_x, lst_arr_err_abs,
        label_x, label_y, lst_legend,
        "absolute", ver)

    arr_x = np.array (np.log (arr_s_g))
    lst_lst_err_rel = list (np.array (lst_lst_err_rel).T) # each method, each s_g
    lst_arr_err_rel = [np.array (lst) for lst in lst_lst_err_rel]
    label_x = "Std. of noise (log)"
    label_y = "Relative Frobenius norm error"
    save_table (
        arr_x, lst_arr_err_rel,
        label_x, label_y, lst_legend,
        "relative", ver)
    save_plot (
        arr_x, lst_arr_err_rel,
        label_x, label_y, lst_legend,
        "relative", ver)
    '''

    arr_x = np.array (np.log (arr_s_g))
    lst_lst_time = list (np.array (lst_lst_time).T) # each meth, each s_g
    # Don't plot the DS theory's time usage (this must be plotted last).
    if "DS, theory" in lst_legend:
        hold_idx = lst_legend.index("DS, theory")
        del lst_legend [hold_idx]
        del lst_lst_time [hold_idx]
    lst_arr_time = [np.array (lst) for lst in lst_lst_time]
    label_x = "Std. of noise (log)"
    label_y = "Time in seconds (log)"
    save_table (
        arr_x, lst_arr_time,
        label_x, label_y, lst_legend,
        "time", ver)
    save_plot (
        arr_x, lst_arr_time,
        label_x, label_y, lst_legend,
        "time", ver)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def llss (est, ver):
    pp_rep = find_rep_mat (est.pp)
    y_rep = find_rep_vec (est.y)
    try:
        pp_rep_inv = np.linalg.pinv (pp_rep)
        est.g_rep_hat = pp_rep_inv @ y_rep
    except np.linalg.LinAlgError as err:
        print ("Least Square fails "
                "because Moore-Penrose inverse does not exist!", flush = True)
        print (err)
        est.zero ()

    est.convert ()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def lasso_qqpp (est, g_g, ver):
    pp_rep = find_rep_mat (est.pp)
    y_rep = find_rep_vec (est.y)
    nn = 2 * cst.NN_H (ver)
    c = np.ones ((nn))
    k = - 2 * pp_rep.T @ y_rep
    qq = pp_rep.T @ pp_rep
    g_rep = cp.Variable ((nn))
    g_rep_abs = cp.Variable ((nn))

    prob = cp.Problem (
        cp.Minimize ((1/nn) * (cp.quad_form (g_rep, qq) + k.T @ g_rep)),
        [g_rep - g_rep_abs <= 0,
            - g_rep - g_rep_abs <= 0,
            c.T @ g_rep_abs <= g_g])

    try:
        prob.solve (solver = cp.ECOS,
            max_iters = cst.CVX_ITER_MAX (ver),
            abstol = cst.CVX_TOL_ABS (ver),
            reltol = cst.CVX_TOL_REL (ver),
            feastol = cst.CVX_TOL_FEAS (ver))
        est.g_rep_hat = g_rep.value
    except (cp.error.SolverError, cp.error.DCPError) as err:
        print ("Lasso fails "
                "when solving the convex program!", flush = True)
        print (err)
        est.g_rep_hat = np.linalg.pinv (pp_rep) @ y_rep

    est.convert ()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def ddss_theory (est, s_g, ver):
    #est.d = s_g * (cst.LL (ver) ** (1/2)) * (
    #        3.29 * np.log (cst.NN_HH (ver))
    #        + 4.56 * (np.log (cst.NN_HH (ver)) ** (3/2)))
    #est.d = s_g * (cst.LL (self.ver) ** (1/2)) * (np.log (cst.NN_HH (self.ver)) ** (3/2))
    est.find_ddss_theory (s_g)

def ddss_llpp (est, g_g, ver):
    pp_rep = find_rep_mat (est.pp)
    y_rep = find_rep_vec (est.y)

    nn = 2 * cst.NN_H (ver)

    g_rep = cp.Variable (nn)
    g_rep_abs = cp.Variable (nn)
    k = pp_rep.T @ y_rep
    qq = pp_rep.T @ pp_rep
    c = np.ones ((nn))

    prob = cp.Problem (cp.Minimize (c.T @ g_rep_abs),
        [g_rep - g_rep_abs <= 0,
            - g_rep - g_rep_abs <= 0,
            qq @ g_rep - g_g * c <= k,
            - qq @ g_rep - g_g * c <= - k])

    try:
        #prob.solve (solver = cp.ECOS)
        prob.solve (solver = cp.ECOS,
            max_iters = cst.CVX_ITER_MAX (ver),
            abstol = cst.CVX_TOL_ABS (ver),
            reltol = cst.CVX_TOL_REL (ver),
            feastol = cst.CVX_TOL_FEAS (ver))
        est.g_rep_hat = g_rep.value
    except (cp.error.SolverError, cp.error.DCPError) as err:
        print ("Dantzig Selector fails "
                "when solving the convex program!", flush = True)
        print (err)
        est.g_rep_hat = np.linalg.pinv (pp_rep) @ y_rep

    est.convert ()

def ddss_llpp_twice (est, g_g, ver):
    pp_rep = find_rep_mat (est.pp)
    y_rep = find_rep_vec (est.y)

    nn_0 = 2 * cst.NN_H (ver)
    nn_2 = int (2 * np.sqrt (np.log (cst.NN_HH (ver))))
    #nn_2 = 2 * cst.NN_HH (ver)
    nn_1 = int ((nn_2 * nn_0) ** (1/2))
    #nn_1 = int ((nn_2 + nn_0) / 2)

    g_rep = cp.Variable (nn_0)
    g_rep_abs = cp.Variable (nn_0)
    k = pp_rep.T @ y_rep
    qq = pp_rep.T @ pp_rep
    c = np.ones ((nn_0))

    prob = cp.Problem (cp.Minimize (c.T @ g_rep_abs),
        [g_rep - g_rep_abs <= 0,
            - g_rep - g_rep_abs <= 0,
            qq @ g_rep - g_g * c <= k,
            - qq @ g_rep - g_g * c <= - k])

    try:
        #prob.solve (solver = cp.ECOS)
        prob.solve (solver = cp.ECOS,
            max_iters = cst.CVX_ITER_MAX (ver),
            abstol = cst.CVX_TOL_ABS (ver),
            reltol = cst.CVX_TOL_REL (ver),
            feastol = cst.CVX_TOL_FEAS (ver))
        g_rep_hat_0 = g_rep.value
    except (cp.error.SolverError, cp.error.DCPError) as err:
        print ("Dantzig Selector fails "
                "when solving the convex program!", flush = True)
        print (err)
        g_rep_hat_0 = np.linalg.pinv (pp_rep) @ y_rep

    # DS, DS
    ss_est_1 = np.sort (np.argsort (np.abs (g_rep_hat_0)) [-nn_1:])
    pp_rep_1 = pp_rep [:, ss_est_1]

    g_rep = cp.Variable (nn_1)
    g_rep_abs = cp.Variable (nn_1)
    k = pp_rep_1.T @ y_rep
    qq = pp_rep_1.T @ pp_rep_1
    c = np.ones ((nn_1))

    prob = cp.Problem (
        cp.Minimize (c.T @ g_rep_abs),
        [g_rep - g_rep_abs <= 0,
            - g_rep - g_rep_abs <= 0,
            qq @ g_rep - g_g * c <= k,
            - qq @ g_rep - g_g * c <= - k])

    try:
        #prob.solve (solver = cp.ECOS)
        prob.solve (solver = cp.ECOS,
            max_iters = cst.CVX_ITER_MAX (ver),
            abstol = cst.CVX_TOL_ABS (ver),
            reltol = cst.CVX_TOL_REL (ver))
        g_rep_hat_ss_1 = g_rep.value
    except (cp.error.SolverError, cp.error.DCPError) as err:
        print ("Dantzig Selector fails "
                "when solving the convex program!", flush = True)
        print (err)
        est.g_rep_hat = np.linalg.pinv (pp_rep) @ y_rep

    g_rep_hat_1 = np.zeros ((nn_0))
    for i in range (nn_1):
        g_rep_hat_1 [ss_est_1 [i]] = g_rep_hat_ss_1 [i]

    # DS, DS, LS
    ss_est_2 = np.sort (np.argsort (np.abs (g_rep_hat_1)) [-nn_2:])
    pp_rep_2 = pp_rep [:, ss_est_2]

    g_rep_hat_ss_2 = np.linalg.pinv (pp_rep_2) @ y_rep
    g_rep_hat_2 = np.zeros ((nn_0))
    for i in range (nn_2):
        g_rep_hat_2 [ss_est_2 [i]] = g_rep_hat_ss_2 [i]

    est.g_rep_hat = g_rep_hat_2
    est.convert ()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def oommpp_fixed_times (est, times, ver):
    pp_rep = find_rep_mat (est.pp)
    y_rep = find_rep_vec (est.y)
    est.zero ()
    r = y_rep # remainder
    tt = range (cst.NN_H (ver)) # list of column indices
    ss = [] # extracted column indices
    count_iter = 0
    pp_rep_ss_inv = np.zeros ((cst.NN_H (ver), cst.NN_Y (ver)))
    while True:
        count_iter += 1
        lst_match = [abs (pp_rep [:, i].T @ r) for i in tt]
        s = np.argmax (lst_match)
        ss.append (s)
        ss = list (sorted (set (ss)))
        pp_rep_ss = pp_rep [:, ss]

        try:
            pp_rep_ss_inv = np.linalg.pinv (pp_rep_ss)
        except np.linalg.LinAlgError as err:
            print ("Orthogonal mathcing pursuit fails when estimating support "
                    "because Moore-Penrose inverse does not exist!", flush = True)
            print (err)
            est.g_rep_hat = np.linalg.pinv (pp_rep) @ y_rep
            est.convert ()
            return

        r = y_rep - pp_rep_ss @ pp_rep_ss_inv @ y_rep
        if (count_iter >= times):
            break
    g_rep_hat_ss = pp_rep_ss_inv @ y_rep
    for i in range (len(ss)):
        est.g_rep_hat [ss [i]] = g_rep_hat_ss [i] 

    est.convert ()

def oommpp_2_norm (est, h_g, ver):
    pp_rep = find_rep_mat (est.pp)
    y_rep = find_rep_vec (est.y)
    est.zero ()
    r = y_rep # remained vector
    tt = range (cst.NN_H (ver)) # list of column indices
    ss = [] # extracted column indices
    count_iter = 0
    pp_rep_ss_inv = np.zeros ((cst.NN_H (ver), cst.NN_Y (ver)))
    while True:
        count_iter += 1
        lst_match = [abs (pp_rep [:, i].T @ r) for i in tt]
        s = np.argmax (lst_match)
        ss.append (s)
        ss = list (sorted (set (ss)))
        pp_rep_ss = pp_rep [:, ss]

        try:
            pp_rep_ss_inv = np.linalg.pinv (pp_rep_ss)
        except np.linalg.LinAlgError as err:
            print ("Orthogonal mathcing pursuit fails when estimating support "
                    "because Moore-Penrose inverse does not exist!", flush = True)
            print (err)
            est.g_rep_hat = np.linalg.pinv (pp_rep) @ y_rep
            est.convert ()
            return

        r = y_rep - pp_rep_ss @ pp_rep_ss_inv @ y_rep
        if (np.linalg.norm (r, ord = 2) <= h_g
            or (count_iter >= cst.ITER_MAX_OOMMPP (ver))):
            break
    g_rep_hat_ss = pp_rep_ss_inv @ y_rep
    for i in range (len(ss)):
        est.g_rep_hat [ss [i]] = g_rep_hat_ss [i]

    est.convert ()

def oommpp_infty_norm (est, h_g, ver):
    pp_rep = find_rep_mat (est.pp)
    y_rep = find_rep_vec (est.y)
    est.zero ()
    r = y_rep # remainder
    tt = range (cst.NN_H (ver)) # list of column indices
    ss = [] # extracted column indices
    count_iter = 0
    pp_rep_ss_inv = np.zeros ((cst.NN_H (ver), cst.NN_Y (ver)))
    while True:
        count_iter += 1
        lst_match = [abs (pp_rep [:, i].T @ r) for i in tt]
        s = np.argmax (lst_match)
        ss.append (s)
        ss = list (sorted (set (ss)))
        pp_rep_ss = pp_rep [:, ss]

        try:
            pp_rep_ss_inv = np.linalg.pinv (pp_rep_ss)
        except np.linalg.LinAlgError as err:
            print ("Orthogonal mathcing pursuit fails when estimating support "
                    "because Moore-Penrose inverse does not exist!", flush = True)
            print (err)
            est.g_rep_hat = np.linalg.pinv (pp_rep) @ y_rep
            est.convert ()
            return

        r = y_rep - pp_rep_ss @ pp_rep_ss_inv @ y_rep
        if (np.linalg.norm (pp_rep.T @ r, ord = np.inf) <= h_g
            or count_iter >= cst.ITER_MAX_OOMMPP (ver)):
            break
    g_rep_hat_ss = pp_rep_ss_inv @ y_rep
    for i in range (len (ss)):
        est.g_rep_hat [ss [i]] = g_rep_hat_ss [i]

    est.convert ()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def mat_complex_normal (nn_1, nn_2):
    return ((np.random.normal (0, 1, (nn_1, nn_2))
        +1J * np.random.normal (0, 1, (nn_1, nn_2))))

def pick_zz (ver):
    return mat_complex_normal (cst.NN_YY (ver), cst.NN_YY (ver))

def pick_mat_bb (ver):
    return (mat_complex_normal (cst.NN_YY (ver), cst.NN_RR (ver)) / np.sqrt (cst.NN_YY (ver)))

def pick_mat_rr (ver):
    kk = np.sqrt (cst.NN_HH (ver)) * get_kk (ver)
    kk_ss = kk [random.sample (list (range (cst.NN_HH (ver))), cst.NN_RR (ver)), :]
    return (np.sqrt (cst.NN_HH (ver) / cst.NN_RR (ver))) * kk_ss

def pick_hh (ver):
    ret = np.zeros ((cst.NN_HH (ver), cst.NN_HH (ver)), dtype = complex)
    for _ in range (cst.LL (ver)):
        alpha = (np.random.normal (0, cst.NN_HH (ver) / cst.LL (ver))
            + 1J * np.random.normal (0, cst.NN_HH (ver) / cst.LL (ver)))
        phi = (2 * np.pi * (cst.DIST_ANT (ver) /cst.LAMBDA_ANT (ver))
            * np.sin (np.random.uniform (0, 2 * np.pi)))
        theta = (2 * np.pi * (cst.DIST_ANT (ver) /cst.LAMBDA_ANT (ver))
            * np.sin (np.random.uniform (0, 2 * np.pi)))
        ret += alpha * np.outer (arr_resp (phi, ver), arr_resp (theta, ver))
    return ret

def get_kk (ver):
    ret = np.zeros ((cst.NN_HH (ver), cst.NN_HH (ver)), dtype = complex)
    for n_1 in range (cst.NN_HH (ver)):
        for n_2 in range (cst.NN_HH (ver)):
            ret [n_1] [n_2] = ((1 / np.sqrt (cst.NN_HH (ver)))
                * np.exp (2 * np.pi * 1J * n_1 * n_2 / cst.NN_HH (ver)))
    return ret

def arr_resp (t, ver):
    return ((1 / np.sqrt (cst.NN_HH (ver)))
        * np.array ([np.exp (1J * i * t) for i in range (cst.NN_HH (ver))]))

def find_rep_vec (v):
    ret =np.zeros ((2 * len (v)))
    for i in range (len (v)):
        ret [2 * i] = np.real (v [i])
        ret [2 * i + 1] = np.imag (v [i])
    return ret

def inv_find_rep_vec (v):
    assert (len (v) % 2 == 0)
    len_v =int (len (v) / 2)
    v_re =np.array ([v [2 * i] for i in range (len_v)])
    v_im =np.array ([v [2 * i + 1] for i in range (len_v)])
    return v_re +1J *v_im

def find_rep_mat (aa):
    ret =np.zeros ((2 * (aa.shape[0]), 2 * (aa.shape[1])))
    for i in range (aa.shape[0]):
        for j in range (aa.shape[1]):
            ret [2 * i] [2 * j] = np.real (aa [i, j])
            ret [2 * i + 1] [2 * j] = np.imag (aa [i, j])
            ret [2 * i] [2 * j + 1] = -np.imag (aa [i, j])
            ret [2 * i + 1] [2 * j + 1] = np.real (aa [i, j])
    return ret

def indic_vec (nn, i):
    ret = np.zeros ((nn), dtype = 'bool')
    ret [i] = 1
    return ret

def indic_rep_mat (nn, i):
    ret = np.zeros ((2 * nn, 2 * nn), dtype = 'bool')
    ret [2 * i] [2 * i] = 1
    ret [2 * i + 1] [2 * i + 1] = 1
    return ret

def vectorize (v):
    return np.reshape (v, (1, -1)) [0]

def inv_vectorize (v, nn_1, nn_2):
    assert (len (v) == nn_1 * nn_2)
    return np.reshape (v, (nn_1, -1))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def save_plot (arr_x, lst_arr_y, label_x, label_y, lst_legend, title, ver):
    full_title = title + ", "
    switcher_focus = {
        cls.Focus.OOMMPP: "OMP",
        cls.Focus.DDSS: "DS",
        cls.Focus.ASSORTED: "assorted"}
    full_title = full_title + switcher_focus [ver.focus] + ", "
    switcher_size = {
        cls.Size.TEST: "test",
        cls.Size.SMALL: "small",
        cls.Size.MEDIUM: "medium",
        cls.Size.BIG: "big"}
    full_title = full_title + switcher_size [ver.size]

    plt.close ("all")
    plt.title (full_title, fontsize = 15)
    plt.xlabel (label_x, fontsize = 12)
    plt.ylabel (label_y, fontsize = 12)

    num_style = 3
    lst_style = ['-', '--', ':']
        # '-', '--', '-.', ':':
        # solid, long dotted, long-short dotted, short dotted
    num_color = 4
    lst_color = ['r', 'g', 'b', 'k']
        # 'r', 'g', 'c', 'b', 'k':
        # red, green, cyan, blue, black
    num_marker = 5
    lst_marker = ['v', '^', 'o', 's', 'D']
        # 'v', '^', 'o', 's', '*', 'D':
        # triangle down, triangle up, circle, square, star, diamond
    size_marker = 6
    width_line = 2

    assert (len (lst_arr_y) == len (lst_legend))
    for i_met in range (len (lst_arr_y)):
        arr_y = lst_arr_y [i_met]
        assert (len (arr_x) == len (arr_y))
        plt.plot (
            arr_x,
            arr_y,
            markersize = size_marker,
            linewidth = width_line,
            linestyle = lst_style [int (i_met % num_style)],
            color = lst_color [int (i_met % num_color)],
            marker = lst_marker [int (i_met % num_marker)],
            label = lst_legend [i_met])
    plt.legend (
        bbox_to_anchor = (1.05, 1),
        loc = 'upper left',
        borderaxespad = 0.)

    full_title = title
    switcher_size = {
        cls.Size.TEST: "test",
        cls.Size.SMALL: "small",
        cls.Size.MEDIUM: "medium",
        cls.Size.BIG: "big"}
    full_title = switcher_size [ver.size] + "-" + full_title
    switcher_focus = {
        cls.Focus.OOMMPP: "oommpp",
        cls.Focus.DDSS: "ddss",
        cls.Focus.ASSORTED: "assorted"}
    full_title = switcher_focus [ver.focus] + "-" + full_title + ".png"

    os.system ("mkdir -p ../plt") # To create new directory only if nonexistent
    path_plot_out = (
        os.path.abspath (os.path.join (os.getcwd (), os.path.pardir))
        + "/plt/" + full_title)
    if os.path.isfile (path_plot_out):
        os.system ("rm -f " + path_plot_out)
    plt.savefig (path_plot_out, bbox_inches = "tight")
    plt.close ()

def save_table (arr_x, lst_arr_y, label_x, label_y, lst_legend, title, ver):
    full_title = title + "-"
    switcher_focus = {
        cls.Focus.OOMMPP: "OMP",
        cls.Focus.DDSS: "DS",
        cls.Focus.ASSORTED: "assorted"}
    full_title = full_title + switcher_focus [ver.focus] + "-"
    switcher_size = {
        cls.Size.TEST: "test",
        cls.Size.SMALL: "small",
        cls.Size.MEDIUM: "medium",
        cls.Size.BIG: "big"}
    full_title = full_title + switcher_size [ver.size] + ".txt"
    full_title = (full_title.replace (" ", "-"))

    os.system ("mkdir -p ../dat") # To create new directory only if nonexistent
    path_table_out = (
        os.path.abspath (os.path.join (os.getcwd (), os.path.pardir))
        + "/dat/" + full_title)
    if os.path.isfile (path_table_out):
        os.system ("rm -f " + path_table_out)

    with open (path_table_out, 'w') as the_file:
        the_file.write (label_x + '\t')
        the_file.write ('\t'.join (map (str, arr_x)) + '\n')
        assert (len (lst_arr_y) == len (lst_legend))
        for i in range (len (lst_arr_y)):
            the_file.write (
                lst_legend [i] + '\t'
                    + '\t'.join (map (str, lst_arr_y [i])) + '\n')


