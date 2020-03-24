import numpy as np
import classes as cls

def NN_HH (ver):
    switcher = {
        cls.Size.TEST : 8,
        cls.Size.SMALL : 16,
        cls.Size.MEDIUM : 20,
        cls.Size.BIG : 24}
    return switcher [ver.size]

def NN_YY (ver):
    return int (NN_HH (ver) / 3)

def NN_RR (ver):
    return int (np.sqrt (NN_HH (ver) * NN_YY (ver)))

def NN_Y (ver):
    return NN_YY (ver) ** 2

def NN_H (ver):
    return NN_HH (ver) ** 2

def SS_supp_est (ver):
    return NN_YY (ver)

def D_MAX (ver):
    return 4

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def NUM_GRID_PHASE (ver):
    return 16

def LAMBDA_ANT (ver):
    return 1

def DIST_ANT (ver):
    return 3

def LL (ver):
    return 3

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def NUM_REP_LLSS (ver):
    return 48

def NUM_REP_LASSO (ver):
    return 1

def NUM_REP_DDSS (ver):
    return 1

def NUM_REP_OOMMPP (ver):
    return 16

def NUM_REP_HH (ver):
    return 12

def S_G_INIT (ver):
    return 2 ** (-5)

def NUM_S_G (ver):
    return 7

def VALUE_SPACING_S_G (ver):
    return np.sqrt (3)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def VALUE_SPACING_G_G (ver):
    return 3

def NUM_G_G (ver): # DS
    if (ver.focus == cls.Focus.DDSS):
        return 3
    elif (ver.focus == cls.Focus.ASSORTED):
        return 1
    else: # cls.Focus.OOMMPP
        return 0

def G_G_DDSS (ver):
    return 2 * np.sqrt (np.log (NN_HH (ver)))

def G_G_LASSO (ver): # same as `G_G_DDSS`
    return 2 * np.sqrt (np.log (NN_HH (ver)))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def VALUE_SPACING_H_G (ver):
    return 3

def NUM_H_G (ver): # OMP
    if (ver.focus == cls.Focus.OOMMPP):
        return 3
    elif (ver.focus == cls.Focus.ASSORTED):
        return 1
    else: # cls.Focus.DDSS
        return 0

def H_G_OOMMPP_2_NORM (ver):
    return np.sqrt (3 * NN_YY (ver))

def H_G_OOMMPP_INFTY_NORM (ver):
    return 2 * np.sqrt (np.log (NN_HH (ver)))

def ITER_MAX_OOMMPP (ver):
    return 4 * NN_HH (ver)

def D_G_PRECISION (ver):
    return 1e-8

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def CVX_ITER_MAX (ver):
    return 48
    # default: 100

def CVX_TOL_ABS (ver):
    return 1e-7
    # default: 1e-7

def CVX_TOL_REL (ver):
    return 1e-6
    # default: 1e-6

def CVX_TOL_FEAS (ver):
    return 1e-7
    # default: 1e-7

