import numpy as np
import matplotlib.pyplot as plt
import os
import time

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
        lst_method.append (lambda x, y: lasso (x, GAMMA_LASSO (ver) * y, ver))

    if (ver.focus == cls.Focus.OOMMPP or ver.focus == cls.Focus.ASSORTED):
        # Orthogonal Matching Pursuit: fixed iteration number
        lst_legend.append ("OMP, $L$ times")
        lst_method.append (lambda x, y:
                oommpp_fixed_times (x, cst.LL (ver), ver))
        # Orthogonal Matching Pursuit: limited l-2 norm
        for c_0 in multiple_values (cst.NUM_ETA (ver)):
            lst_legend.append ("OMP, $l_2$-norm")
            lst_method.append (
                lambda x, y, c = c_0:
                oommpp_2_norm (x, c * cst.ETA_OOMMPP_2_NORM (ver) * y, ver))
        # Orthogonal Matching Pursuit: limited l-infinity norm
        for c_0 in multiple_values (cst.NUM_ETA (ver)):
            lst_legend.append ("OMP, $l_\infty$-norm")
            lst_method.append (
                lambda x, y, c = c_0:
                oommpp_infty_norm (x, c * ETA_OOMMPP_INFTY_NORM (ver) * y, ver))

    # Dantzig Selector error bound
    if (ver.focus == cls.Focus.DDSS or ver.focus == cls.Focus.ASSORTED):
        lst_legend.append ("DS, theory")
        lst_method.append (lambda x, y: ddss_theory (x, y, ver))
        for c_0 in multiple_values (cst.NUM_GAMMA_DS (ver)):
            lst_legend.append ("DS, $\gamma$ = " + '%.2f' % c_0 + "$\sigma$")
            lst_method.append (lambda x, y, c = c_0: ddss (x, c * y, ver))

    assert (len (lst_method) == len (lst_legend))
    num_method = len (lst_method)

    count_prog = 0
    time_start = time.time()
    lst_lst_err_abs = [] # each sigma, each method
    lst_lst_err_rel = [] # each sigma, each method
    for i_sigma in range (cst.NUM_SIGMA ()):
        sigma = arr_sigma [i_sigma]
        print ("σ = ", '%.2f' % sigma, " :", sep = '')
        lst_err_abs = [0] * num_method
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
            y = pp @ g + sigma * z
            est = cls.Estimation (pp, y, hh, ver)

            for i in range (num_method):
                lst_method [i] (est, sigma)
                lst_err_abs [i] += np.log (est.d)

            percent_progress = 100 * count_prog / (cst.NUM_SIGMA () * cst.NUM_REPEAT (ver))
            print (
                "    experiment ", count_prog, " (",
                '%.1f' % percent_progress, "%)",
                sep = '', end = '\r')
        lst_err_abs = list ((np.array (lst_err_abs) / cst.NUM_REPEAT (ver)))
        norm_hh /= cst.NUM_REPEAT (ver)
        lst_err_rel = list ((np.array (lst_err_abs) / norm_hh))
        lst_lst_err_abs.append (lst_err_abs)
        lst_lst_err_rel.append (lst_err_rel)
        print ("                                ", end = '\r') # clear last line
        print ("    done")
    time_stop = time.time ()

    print (
        "averaged time elapsed for each experiment: ",
        '%.2f' %
            ((time_stop - time_start) / (60 * cst.NUM_SIGMA () * cst.NUM_REPEAT (ver))),
        " (min)")
    print (
        "total time elapsed: ",
        '%.2f' % ((time_stop - time_start) / 60),
        " (min)")

    arr_x = np.array (np.log (arr_sigma))
    lst_lst_y_abs = list (np.array (lst_lst_err_abs).T) # each method, each sigma
    lst_arr_y_abs = [np.array (lst) for lst in lst_lst_y_abs]
    lst_lst_y_rel = list (np.array (lst_lst_err_rel).T) # each method, each sigma
    lst_arr_y_rel = [np.array (lst) for lst in lst_lst_y_rel]

    label_x = "Std. of Noise (Log)"
    label_y = "Absolute 2-Norm error (Log)"
    title = "Channel Estimation Performance, Absolute"
    save_table (arr_x, lst_arr_y_abs, label_x, label_y, lst_legend, title)
    save_plot (arr_x, lst_arr_y_abs, label_x, label_y, lst_legend, title)
    label_x = "Std. of Noise (Log)"
    label_y = "Relative 2-Norm error"
    title = "Channel Estimation Performance, Relative"

    save_table (arr_sigma, lst_arr_y_rel, label_x, label_y, lst_legend, title)
    save_plot (arr_sigma, lst_arr_y_rel, label_x, label_y, lst_legend, title)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def llss (est, ver):
    est.refresh ()
    try:
        pp_inv = np.linalg.pinv (est.pp)
    except np.linalg.LinAlgError:
        print ("Least square fails due to singularity!", flush = True)
        return
    est.g_hat = pp_inv @ est.y
    est.convert ()

def lasso (est, gamma, ver):
    est.refresh ()
    g = cp.Variable (cst.NN_H (ver), complex = True)
    prob = cp.Problem (
        cp.Minimize (cp.norm (est.pp @ g - est.y, 2)),
        [cp.norm (g, 1) <= gamma])
    try:
        prob.solve ()
    except cp.error.SolverError:
        print ("Lasso fails to solve the program!", flush = True)
        est.g_hat = np.linalg.pinv (est.pp) @ est.y
        return
    est.g_hat = g.value
    est.convert ()

def oommpp_fixed_times (est, times, ver):
    est.refresh ()
    r = est.y # remainder
    tt = range (cst.NN_H (ver)) # list of column indices
    ss = [] # extracted column indices
    count_iter = 0
    pp_ss_inv = np.zeros ((cst.NN_H (ver), cst.NN_Y (ver)))
    while True:
        count_iter += 1
        lst_match = [abs (est.pp [:, i].conj().T @ r) for i in tt]
        s = np.argmax (lst_match)
        ss.append (s)
        ss = list (sorted (set (ss)))
        pp_ss = est.pp [:, ss]
        try:
            pp_ss_inv = np.linalg.pinv (pp_ss)
        except np.linalg.LinAlgError:
            print ("Orthogonal mathcing pursuit fails due to singularity!", flush = True)
            return
        r = est.y - pp_ss @ pp_ss_inv @ est.y
        if (count_iter >= times):
            break
    g_hat_ss = pp_ss_inv @ est.y
    for i in range (len(ss)):
        est.g_hat [ss[i]] = g_hat_ss[i] 
    est.convert ()

def oommpp_2_norm (est, eta, ver):
    est.refresh ()
    r = est.y # remained vector
    tt = range (cst.NN_H (ver)) # list of column indices
    ss = [] # extracted column indices
    count_iter = 0
    pp_ss_inv = np.zeros ((cst.NN_H (ver), cst.NN_Y (ver)))
    while True:
        count_iter += 1
        lst_match = [abs (est.pp [:, i].conj().T @ r) for i in tt]
        s = np.argmax (lst_match)
        ss.append (s)
        ss = list (sorted (set (ss)))
        pp_ss = est.pp [:, ss]
        try:
            pp_ss_inv = np.linalg.pinv (pp_ss)
        except np.linalg.LinAlgError:
            print ("Orthogonal mathcing pursuit fails due to singularity!", flush = True)
            return
        r = est.y - pp_ss @ pp_ss_inv @ est.y
        if (np.linalg.norm (r, 2) <= eta):
            break
        if (count_iter >= cst.MAX_ITER_OOMMPP (ver)):
            break
    g_hat_ss = pp_ss_inv @ est.y
    for i in range (len(ss)):
        est.g_hat [ss[i]] = g_hat_ss[i] 
    est.convert ()

def oommpp_infty_norm (est, eta, ver):
    est.refresh ()
    r = est.y # remained vector
    tt = range (cst.NN_H (ver)) # list of column indices
    ss = [] # extracted column indices
    count_iter = 0
    pp_ss_inv = np.zeros ((cst.NN_H (ver), cst.NN_Y (ver)))
    while True:
        count_iter += 1
        lst_match = [abs (est.pp [:, i].conj().T @ r) for i in tt]
        s = np.argmax (lst_match)
        ss.append (s)
        ss = list (sorted (set (ss)))
        pp_ss = est.pp [:, ss]
        try:
            pp_ss_inv = np.linalg.pinv (pp_ss)
        except np.linalg.LinAlgError:
            print ("Orthogonal mathcing pursuit fails due to singularity!", flush = True)
            return
        r = est.y - pp_ss @ pp_ss_inv @ est.y
        if (np.linalg.norm (pp_ss.conj().T @ r, 2) <= eta
            or count_iter >= cst.MAX_ITER_OOMMPP):
            break
    g_hat_ss = pp_ss_inv @ est.y
    for i in range (len(ss)):
        est.g_hat [ss[i]] = g_hat_ss[i] 
    est.convert ()

def ddss_theory (est, sigma, ver):
    est.refresh ()
    est.d = 8 * np.sqrt (cst.LL (ver) * np.log (cst.NN_HH (ver))) * sigma

def ddss (est, gamma, ver):
    est.refresh ()
    g = cp.Variable (cst.NN_H (ver), complex = True)
    prob = cp.Problem (
        cp.Minimize (cp.norm (g, 1)),
        [cp.norm (est.pp.conj().T @ (est.pp @ g - est.y), "inf")
            <= gamma])
    try:
        prob.solve ()
    except cp.error.SolverError:
        print ("Complex DS fails to solve the program!", flush = True)
        est.g_hat = (np.linalg.pinv (est.pp) @ est.y)
        return
    est.g_hat = g.value
    est.convert ()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def multiple_values (n):
    return list (cst.VALUE_SPACING ()
        ** (np.array (range (n)) - (n - 1) / 2))

def mat_complex_normal (nn_1, nn_2):
    return ((np.random.normal (0, 1, (nn_1, nn_2))
        +1J * np.random.normal (0, 1, (nn_1, nn_2))))

def mat_uniform_phase (nn_1, nn_2):
    return ( np.exp (
        2 * np.pi * 1J
            * np.random.randint ( cst.NUM_GRID_PHASE (), size = (nn_1, nn_2))
            / cst.NUM_GRID_PHASE ()))

def pick_zz (ver):
    return mat_complex_normal (cst.NN_YY (ver), cst.NN_YY (ver))

def pick_ff_bb (ver):
    return (mat_complex_normal (cst.NN_RR (ver), cst.NN_YY (ver))
    / np.sqrt (cst.NN_YY (ver)))

def pick_ww_bb (ver):
    return (mat_complex_normal (cst.NN_YY (ver), cst.NN_RR (ver))
    / np.sqrt (cst.NN_YY (ver)))

def pick_ff_rr (ver):
    return (mat_uniform_phase (cst.NN_HH (ver), cst.NN_RR (ver))
    / np.sqrt (cst.NN_RR (ver)))

def pick_ww_rr (ver):
    return (mat_uniform_phase (cst.NN_RR (ver), cst.NN_HH (ver))
    / np.sqrt (cst.NN_RR (ver)))

def pick_hh (ver):
    ret =np.zeros ((cst.NN_HH (ver), cst.NN_HH (ver)), dtype=complex)
    for _ in range (cst.LL (ver)):
        alpha = (np.random.normal (0, cst.NN_HH (ver) / cst.LL (ver))
            + 1J * np.random.normal (0, cst.NN_HH (ver) / cst.LL (ver)))
        phi = (2 * np.pi * (cst.DIST_ANT () /cst.LAMBDA_ANT ())
            * np.sin (np.random.uniform (0, 2 * np.pi)))
        theta = (2 * np.pi * (cst.DIST_ANT () /cst.LAMBDA_ANT ())
            * np.sin (np.random.uniform (0, 2 * np.pi)))
        ret += alpha * arr_resp (phi, ver) @ arr_resp (theta, ver).conj().T
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

def vectorize (v):
    return np.reshape (v, (1,-1)) [0]

def inv_vectorize (v, nn_1, nn_2):
    assert (len (v) == nn_1 * nn_2)
    return np.reshape (v, (nn_1, -1))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def save_plot (arr_x, lst_arr_y, label_x, label_y, lst_legend, title):
    plt.close ("all")
    plt.title (title, fontsize = 15)
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

    path_plot_out =(
        os.path.abspath (os.path.join (os.getcwd (), os.path.pardir))
            + "/plt/" + title.replace (" ", "_") + ".png")
    if os.path.isfile(path_plot_out):
        os.system ("rm -f " + path_plot_out)
    plt.savefig (path_plot_out, bbox_inches = "tight")

def save_table (arr_x, lst_arr_y, label_x, label_y, lst_legend, title):
    path_table_out =(
        os.path.abspath (os.path.join (os.getcwd (), os.path.pardir))
        + "/dat/" + title.replace (" ", "_") + ".txt")

    with open (path_table_out, 'w') as the_file:
        the_file.write (label_x + '\t')
        the_file.write ('\t'.join (map (str, arr_x)) + '\n')
        assert (len (lst_arr_y) == len (lst_legend))
        for i in range (len (lst_arr_y)):
            the_file.write (
                lst_legend [i] + '\t'
                    + '\t'.join (map (str, lst_arr_y [i])) + '\n')


