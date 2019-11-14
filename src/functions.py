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
    arr_sigma = multiple_values (cst.NUM_SIGMA ())
    lst_legend = []
    lst_method = [] # arguments: (estimation, sigma)

    if (ver.focus == cls.Focus.ASSORTED):
        # Least Square
        lst_legend.append ("LS")
        lst_method.append (lambda x, y: llss (x, ver))
        # Lasso
        lst_legend.append ("Lasso")
        lst_method.append (lambda x, y: lasso_ssooccpp (x, cst.GAMMA_LASSO (ver) * y, ver))

    if (ver.focus == cls.Focus.OOMMPP or ver.focus == cls.Focus.ASSORTED):
        # Orthogonal Matching Pursuit: fixed iteration number
        lst_legend.append ("OMP, $L$ times")
        lst_method.append (lambda x, y:
                oommpp_fixed_times (x, cst.LL (), ver))
        # Orthogonal Matching Pursuit: limited l-2 norm
        for c_0 in multiple_values (cst.NUM_ETA (ver)):
            lst_legend.append ("OMP, $l_2$-norm, $\eta$ = " + '%.2f' % c_0 + "$\sigma$")

            lst_method.append (
                lambda x, y, c = c_0:
                oommpp_2_norm (x, c * cst.ETA_OOMMPP_2_NORM (ver) * y, ver))
        # Orthogonal Matching Pursuit: limited l-infinity norm
        for c_0 in multiple_values (cst.NUM_ETA (ver)):
            lst_legend.append ("OMP, $l_\infty$-norm, $\eta$ = " + '%.2f' % c_0 + "$\sigma$")
            lst_method.append (
                lambda x, y, c = c_0:
                oommpp_infty_norm (x, c * cst.ETA_OOMMPP_INFTY_NORM (ver) * y, ver))

    # Dantzig Selector error bound
    if (ver.focus == cls.Focus.DDSS or ver.focus == cls.Focus.ASSORTED):
        lst_legend.append ("DS, theory")
        lst_method.append (lambda x, y: ddss_theory (x, y, ver))
        for c_0 in multiple_values (cst.NUM_GAMMA_DS (ver)):
            lst_legend.append ("DS, $\gamma$ = " + '%.2f' % c_0 + "$\sigma$")
            lst_method.append (lambda x, y, c = c_0: ddss_llpp (x, c * y, ver))

    assert (len (lst_method) == len (lst_legend))
    num_method = len (lst_method)

    count_prog = 0
    lst_lst_err_abs = [] # each sigma, each method
    lst_lst_err_rel = [] # each sigma, each method
    lst_lst_time = [] # each sigma, each method

    time_tot_start = time.time()
    for i_sigma in range (cst.NUM_SIGMA ()):
        sigma = arr_sigma [i_sigma]
        print ("σ = ", '%.2f' % sigma, " :", sep = '')
        lst_err_abs = [0] * num_method
        lst_time = [0] * num_method
        norm_hh = 0
        for _ in range (cst.NUM_REPEAT (ver)):
            count_prog += 1
            ff_bb = pick_ff_bb (ver)
            ff_rr = pick_ff_rr (ver)
            ww_bb = pick_ww_bb (ver)
            ww_rr = pick_ww_rr (ver)
            hh = pick_hh (ver)
            zz = pick_zz (ver)
            kk = get_kk (ver)
            gg = kk.conj().T @ hh @ kk
            norm_hh += np.log (np.linalg.norm (hh, ord=2))

            pp = np.kron (
                ff_bb.T @ ff_rr.T @ kk.conj(),
                ww_bb @ ww_rr @ kk)
            g = vectorize (gg)
            z = vectorize (zz)
            y = pp @ g + (sigma / np.sqrt(2)) * z
            pp_rep = find_rep_mat (pp)
            y_rep = find_rep_vec (y)
            est = cls.Estimation (pp_rep, y_rep, hh, ver)

            for i in range (num_method):
                time_each_start = time.time ()
                lst_method [i] (est, sigma)
                time_each_stop = time.time ()
                lst_time [i] += np.log ((time_each_stop - time_each_start) / 60)
                lst_err_abs [i] += np.log (est.d)

            rate_progress = count_prog / (cst.NUM_SIGMA () * cst.NUM_REPEAT (ver))
            time_hold = time.time ()
            # Use `end = '\r\r'` to force carriage to return to line start
            print (
                "    experiment ", count_prog,
                sep = '', flush = True)
            print (
                "      (", '%.1f' % (100 * rate_progress), "%; ",
                '%.2f' % (
                    (time_hold - time_tot_start) * (1 - rate_progress) / (rate_progress * 60)),
                " min. remaining)",
                sep = '', flush = True)
        lst_err_abs = list (np.array (lst_err_abs) / cst.NUM_REPEAT (ver))
        norm_hh /= cst.NUM_REPEAT (ver)
        lst_err_rel = list (np.array (lst_err_abs) / norm_hh)
        lst_time = list (np.array (lst_time) / cst.NUM_REPEAT (ver))
        lst_lst_err_abs.append (lst_err_abs)
        lst_lst_err_rel.append (lst_err_rel)
        lst_lst_time.append (lst_time)
        print ("                                ", end = '\r') # clear last line
        print ("    done")
    time_tot_stop = time.time ()

    print (
        "averaged time elapsed for each experiment: ",
        '%.2f' %
            ((time_tot_stop - time_tot_start) / (60 * cst.NUM_SIGMA () * cst.NUM_REPEAT (ver))),
        " (min)", flush = True)
    print (
        "total time elapsed: ",
        '%.2f' % ((time_tot_stop - time_tot_start) / 60),
        " (min)", flush = True)

    arr_x = np.array (np.log (arr_sigma))
    lst_lst_err_abs = list (np.array (lst_lst_err_abs).T) # each method, each sigma
    lst_arr_err_abs = [np.array (lst) for lst in lst_lst_err_abs]
    label_x = "Std. of Noise (Log)"
    label_y = "Absolute 2-Norm error (Log)"
    save_table (arr_x, lst_arr_err_abs,
        label_x, label_y, lst_legend,
        "Absolute", ver)
    save_plot (
        arr_x, lst_arr_err_abs,
        label_x, label_y, lst_legend,
        "Absolute", ver)

    lst_lst_err_rel = list (np.array (lst_lst_err_rel).T) # each method, each sigma
    lst_arr_err_rel = [np.array (lst) for lst in lst_lst_err_rel]
    label_x = "Std. of Noise (Log)"
    label_y = "Relative 2-Norm error"
    save_table (
        arr_sigma, lst_arr_err_rel,
        label_x, label_y, lst_legend,
        "Relative", ver)
    save_plot (
        arr_sigma, lst_arr_err_rel,
        label_x, label_y, lst_legend,
        "Relative", ver)

    lst_lst_time = list (np.array (lst_lst_time).T) # each method, each sigma
    # Don't plot the DS theory's time usage (this must be plotted last).
    if "DS, theory" in lst_legend:
        hold_idx = lst_legend.index("DS, theory")
        del lst_legend [hold_idx]
        del lst_lst_time [hold_idx]
    lst_arr_time = [np.array (lst) for lst in lst_lst_time]
    label_x = "Std. of Noise (Log)"
    label_y = "Time in minute (Log)"
    save_table (
        arr_sigma, lst_arr_time,
        label_x, label_y, lst_legend,
        "Time", ver)
    save_plot (
        arr_sigma, lst_arr_time,
        label_x, label_y, lst_legend,
        "Time", ver)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def llss (est, ver):
    est.refresh ()
    try:
        pp_rep_inv = np.linalg.pinv (est.pp_rep)
        est.g_rep_hat = pp_rep_inv @ est.y_rep
    except np.linalg.LinAlgError as e:
        print ("Least square fails due to singularity!", flush = True)
        print (e)
        est.refresh ()
        return
    est.convert ()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def lasso_direct (est, gamma, ver):
    est.refresh ()
    g = cp.Variable (2 * cst.NN_H (ver))

    prob = cp.Problem (
        cp.Minimize (cp.norm (est.pp_rep @ g - est.y_rep, 2)),
        [cp.norm (g, 1) <= gamma])

    try:
        prob.solve (solver = cp.ECOS)
        est.g_rep_hat = g.value
    except cp.error.SolverError:
        print ("Lasso fails to solve the program!", flush = True)
        est.g_rep_hat = np.linalg.pinv (est.pp_rep) @ est.y_rep
    est.convert ()

def lasso_ssooccpp (est, gamma, ver):
    est.refresh ()
    nn = 2 * cst.NN_H (ver)
    zz = np.zeros ((nn, nn))
    aa = np.block ([est.pp_rep, -est.pp_rep, np.zeros ((2 * cst.NN_Y (ver), 1))])
    b = np.block ([np.zeros ((2 * nn)), 1])
    c = np.block ([np.ones ((2 * nn)), 0])
    x = cp.Variable (2 * nn + 1)

    prob = cp.Problem (
        cp.Minimize (b.T @ x),
        [x >= 0,
            c @ x <= gamma,
            cp.norm (aa @ x -est.y_rep, 2) <= b @ x])

    try:
        prob.solve (solver = cp.ECOS)
        est.g_rep_hat = (x.value) [0 : nn] - (x.value) [nn : 2*nn]
    except cp.error.SolverError:
        print ("Lasso fails to solve the program!", flush = True)
        est.g_rep_hat = np.linalg.pinv (est.pp_rep) @ est.y_rep
    est.convert ()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def ddss_theory (est, sigma, ver):
    est.refresh ()
    est.d = sigma * (cst.LL () ** (1/2)) * (
            3.29 * np.log (cst.NN_HH (ver))
            + 4.56 * (np.log (cst.NN_HH (ver)) ** (3/2)))

def ddss_direct (est, gamma, ver):
    est.refresh ()
    g = cp.Variable (2 * cst.NN_H (ver))

    prob = cp.Problem (
        cp.Minimize (cp.norm (g, 1)),
        [cp.norm (est.pp_rep.T @ (est.y_rep - est.pp_rep @ g), "inf")
            <= gamma])

    try:
        prob.solve (solver = cp.ECOS)
        est.g_rep_hat = g.value
    except cp.error.SolverError:
        print ("Dantzig Selector fails to solve the program!", flush = True)
        est.g_rep_hat = np.linalg.pinv (est.pp_rep) @ est.y_rep
    est.convert ()

def ddss_llpp (est, gamma, ver):
    est.refresh ()
    nn = 2 * cst.NN_H (ver)
    zz = np.zeros ((nn, nn))
    i = np.block ([np.zeros ((nn)), np.ones ((nn))])
    k = np.block (
        [est.pp_rep.T @ est.y_rep - gamma * np.ones ((nn)),
            -est.pp_rep.T @ est.y_rep - gamma * np.ones ((nn))])
    aa = np.block ([[np.eye (nn), np.eye (nn)], [-np.eye (nn), np.eye (nn)], [zz, np.eye (nn)]])
    cc = np.block ([[est.pp_rep.T @ est.pp_rep, zz], [-est.pp_rep.T @ est.pp_rep, zz]])
    aa = sp.sparse.coo_matrix (aa)
    cc = sp.sparse.coo_matrix (cc)
    x = cp.Variable (2 * nn)

    prob = cp.Problem (
        cp.Minimize (i.T @ x),
        [aa @ x >= 0,
            cc @ x >= k])

    try:
        prob.solve (solver = cp.CVXOPT)
        est.g_rep_hat = (x.value) [0 : nn]
    except cp.error.SolverError:
        print ("Dantzig Selector fails to solve the program!", flush = True)
        est.g_rep_hat = np.linalg.pinv (est.pp_rep) @ est.y_rep
    est.convert ()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def oommpp_fixed_times (est, times, ver):
    est.refresh ()
    r = est.y_rep # remainder
    tt = range (cst.NN_H (ver)) # list of column indices
    ss = [] # extracted column indices
    count_iter = 0
    pp_rep_ss_inv = np.zeros ((cst.NN_H (ver), cst.NN_Y (ver)))
    while True:
        count_iter += 1
        lst_match = [abs (est.pp_rep [:, i].T @ r) for i in tt]
        s = np.argmax (lst_match)
        ss.append (s)
        ss = list (sorted (set (ss)))
        pp_rep_ss = est.pp_rep [:, ss]
        try:
            pp_rep_ss_inv = np.linalg.pinv (pp_rep_ss)
        except np.linalg.LinAlgError as e:
            print ("Orthogonal mathcing pursuit fails due to singularity!", flush = True)
            print (e)
            est.g_rep_hat = np.linalg.pinv (est.pp_rep) @ est.y_rep
            break
        r = est.y_rep - pp_rep_ss @ pp_rep_ss_inv @ est.y_rep
        if (count_iter >= times):
            break
    g_hat_ss = pp_rep_ss_inv @ est.y_rep
    for i in range (len(ss)):
        est.g_hat [ss[i]] = g_hat_ss[i] 
    est.convert ()

def oommpp_2_norm (est, eta, ver):
    est.refresh ()
    r = est.y_rep # remained vector
    tt = range (cst.NN_H (ver)) # list of column indices
    ss = [] # extracted column indices
    count_iter = 0
    pp_rep_ss_inv = np.zeros ((cst.NN_H (ver), cst.NN_Y (ver)))
    while True:
        count_iter += 1
        lst_match = [abs (est.pp_rep [:, i].T @ r) for i in tt]
        s = np.argmax (lst_match)
        ss.append (s)
        ss = list (sorted (set (ss)))
        pp_rep_ss = est.pp_rep [:, ss]
        try:
            pp_rep_ss_inv = np.linalg.pinv (pp_rep_ss)
        except np.linalg.LinAlgError as e:
            print ("Orthogonal mathcing pursuit fails due to singularity!", flush = True)
            print (e)
            est.g_rep_hat = np.linalg.pinv (est.pp_rep) @ est.y_rep
            break
        r = est.y_rep - pp_rep_ss @ pp_rep_ss_inv @ est.y_rep
        if (np.linalg.norm (r, ord = 2) <= eta
            or (count_iter >= cst.ITER_MAX_OOMMPP (ver))):
            break
    g_hat_ss = pp_rep_ss_inv @ est.y_rep
    for i in range (len(ss)):
        est.g_hat [ss[i]] = g_hat_ss[i] 
    est.convert ()

def oommpp_infty_norm (est, eta, ver):
    est.refresh ()
    r = est.y_rep # remained vector
    tt = range (cst.NN_H (ver)) # list of column indices
    ss = [] # extracted column indices
    count_iter = 0
    pp_rep_ss_inv = np.zeros ((cst.NN_H (ver), cst.NN_Y (ver)))
    while True:
        count_iter += 1
        lst_match = [abs (est.pp_rep [:, i].T @ r) for i in tt]
        s = np.argmax (lst_match)
        ss.append (s)
        ss = list (sorted (set (ss)))
        pp_rep_ss = est.pp_rep [:, ss]
        try:
            pp_rep_ss_inv = np.linalg.pinv (pp_rep_ss)
        except np.linalg.LinAlgError as e:
            print ("Orthogonal mathcing pursuit fails due to singularity!", flush = True)
            print (e)
            est.g_rep_hat = np.linalg.pinv (est.pp_rep_rep) @ est.y_rep
            break
        r = est.y_rep - pp_rep_ss @ pp_rep_ss_inv @ est.y_rep
        if (np.linalg.norm (pp_rep_ss.T @ r, ord = np.inf) <= eta
            or count_iter >= cst.ITER_MAX_OOMMPP):
            break
    g_hat_ss = pp_rep_ss_inv @ est.y_rep
    for i in range (len(ss)):
        est.g_hat [ss[i]] = g_hat_ss[i] 
    est.convert ()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def multiple_values (n):
    return list (cst.VALUE_SPACING ()
        ** (np.array (range (n)) - (n - 1) / 2))

def mat_complex_normal (nn_1, nn_2):
    return ((np.random.normal (0, 1, (nn_1, nn_2))
        +1J * np.random.normal (0, 1, (nn_1, nn_2))))

def pick_zz (ver):
    return mat_complex_normal (cst.NN_YY (ver), cst.NN_YY (ver))

def pick_ff_bb (ver):
    return (mat_complex_normal (cst.NN_RR (ver), cst.NN_YY (ver))
    / np.sqrt (cst.NN_YY (ver)))

def pick_ww_bb (ver):
    return (mat_complex_normal (cst.NN_YY (ver), cst.NN_RR (ver))
    / np.sqrt (cst.NN_YY (ver)))

def pick_ff_rr (ver):
    kk = get_kk (ver)
    kk_ss = kk [:, random.sample (list (range (cst.NN_HH(ver))), cst.NN_RR(ver))]
    return kk_ss / np.sqrt (cst.NN_RR (ver))

def pick_ww_rr (ver):
    kk = get_kk (ver)
    kk_ss = kk [random.sample (list (range (cst.NN_HH(ver))), cst.NN_RR(ver)), :]
    return kk_ss / np.sqrt (cst.NN_RR (ver))

def pick_hh (ver):
    ret =np.zeros ((cst.NN_HH (ver), cst.NN_HH (ver)), dtype=complex)
    for _ in range (cst.LL ()):
        alpha = (np.random.normal (0, cst.NN_HH (ver) / cst.LL ())
            + 1J * np.random.normal (0, cst.NN_HH (ver) / cst.LL ()))
        phi = (2 * np.pi * (cst.DIST_ANT () /cst.LAMBDA_ANT ())
            * np.sin (np.random.uniform (0, 2 * np.pi)))
        theta = (2 * np.pi * (cst.DIST_ANT () /cst.LAMBDA_ANT ())
            * np.sin (np.random.uniform (0, 2 * np.pi)))
        ret += alpha * np.outer (arr_resp (phi, ver), arr_resp (theta, ver))
    return ret

def get_kk (ver): # DFT matrix
    ret = np.zeros ((cst.NN_HH (ver), cst.NN_HH (ver)), dtype=complex)
    for i in range (cst.NN_HH (ver)):
        for j in range (cst.NN_HH (ver)):
            ret [i] [j] = ((1 / np.sqrt (cst.NN_HH (ver)))
                * np.exp (2 * np.pi * 1J * i * j / cst.NN_HH (ver)))
    return ret

def arr_resp (t, ver):
    return ((1 / np.sqrt (cst.NN_HH (ver)))
        * np.array ([np.exp (1J * i * t) for i in range (cst.NN_HH (ver))]))

def find_rep_vec (v):
    ret =np.zeros ((2*len (v)))
    for i in range (len (v)):
        ret [2*i] =np.real (v [i])
        ret [2*i+1] =np.imag (v [i])
    return ret

def inv_find_rep_vec (v):
    assert (len (v)%2 == 0)
    len_v =int (len (v)/2)
    v_re =np.array ([v [2*i] for i in range (len_v)])
    v_im =np.array ([v [2*i+1] for i in range (len_v)])
    return v_re +1J *v_im

def find_rep_mat (aa):
    ret =np.zeros ((2 *(aa.shape[0]), 2 *(aa.shape[1])))
    for i in range (aa.shape[0]):
        for j in range (aa.shape[1]):
            ret [2*i] [2*j] =np.real (aa [i,j])
            ret [2*i+1] [2*j] =np.imag (aa [i,j])
            ret [2*i] [2*j+1] =-np.imag (aa [i,j])
            ret [2*i+1] [2*j+1] =np.real (aa [i,j])
    return ret

def indic_vec (nn, i):
    ret =np.zeros ((nn), dtype = 'bool')
    ret [i] =1
    return ret

def indic_rep_mat (nn, i):
    ret =np.zeros ((2*nn, 2*nn), dtype = 'bool')
    ret [2*i] [2*i] =1
    ret [2*i+1] [2*i+1] =1
    return ret

def vectorize (v):
    return np.reshape (v, (1,-1)) [0]

def inv_vectorize (v, nn_1, nn_2):
    assert (len (v) == nn_1 * nn_2)
    return np.reshape (v, (nn_1, -1))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def save_plot (arr_x, lst_arr_y, label_x, label_y, lst_legend, title, ver):
    full_title = title + ", "
    switcher = {
        cls.Focus.OOMMPP: "OMP",
        cls.Focus.DDSS: "DS",
        cls.Focus.ASSORTED: "assorted"}
    full_title += switcher [ver.focus] + ", "
    switcher = {
        cls.Size.TEST: "test",
        cls.Size.SMALL: "small",
        cls.Size.MEDIUM: "medium",
        cls.Size.BIG: "big"}
    full_title += switcher [ver.size]

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
    for i_method in range (len (lst_arr_y)):
        arr_y = lst_arr_y [i_method]
        assert (len (arr_x) == len (arr_y))
        plt.plot (
            arr_x,
            arr_y,
            markersize = size_marker,
            linewidth = width_line,
            linestyle = lst_style [int (i_method % num_style)],
            color = lst_color [int (i_method % num_color)],
            marker = lst_marker [int (i_method % num_marker)],
            label = lst_legend [i_method])
    plt.legend (
        bbox_to_anchor = (1.05, 1),
        loc = 'upper left',
        borderaxespad = 0.)

    full_title = (full_title.replace (" ", "_"))
    os.system ("mkdir -p ../plt") # To create new directory only if nonexistent
    path_plot_out = (
        os.path.abspath (os.path.join (os.getcwd (), os.path.pardir))
        + "/plt/" + full_title + ".png")
    if os.path.isfile (path_plot_out):
        os.system ("rm -f " + path_plot_out)
    plt.savefig (path_plot_out, bbox_inches = "tight")
    plt.close ("all")

def save_table (arr_x, lst_arr_y, label_x, label_y, lst_legend, title, ver):
    full_title = title + ", "
    switcher = {
        cls.Focus.OOMMPP: "OMP",
        cls.Focus.DDSS: "DS",
        cls.Focus.ASSORTED: "assorted"}
    full_title += switcher [ver.focus] + ", "
    switcher = {
        cls.Size.TEST: "test",
        cls.Size.SMALL: "small",
        cls.Size.MEDIUM: "medium",
        cls.Size.BIG: "big"}
    full_title += switcher [ver.size] + ".txt"

    full_title = (title.replace (" ", "_") + ",_")
    switcher = {
        cls.Size.TEST: "test",
        cls.Size.SMALL: "small",
        cls.Size.MEDIUM: "medium",
        cls.Size.BIG: "big"}
    full_title = full_title + switcher [ver.size] + ",_"
    switcher = {
        cls.Focus.OOMMPP: "OMP",
        cls.Focus.DDSS: "DS",
        cls.Focus.ASSORTED: "assorted"}
    full_title = full_title + switcher [ver.focus] + ".txt"

    full_title = (full_title.replace (" ", "_"))
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


