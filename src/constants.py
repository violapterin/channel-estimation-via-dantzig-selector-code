import numpy as np
import classes as cls

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def NUM_GRID_PHASE (ver):
    return 16

def LAMBDA_ANT (ver):
    return 1

def DIST_ANT (ver):
    return 3

def LL (ver):
    return 4

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def NUM_REP_LLSS (ver):
    return 24

def NUM_REP_OOMMPP (ver):
    return 12

def NUM_REP_LASSO (ver):
    return 1

def NUM_REP_DDSS (ver):
    return 1

def NUM_REP_HH (ver):
    return 4

def VALUE_SPACING_S_G (ver):
    return 2

def VALUE_SPACING_H_G (ver):
    return 4

def VALUE_SPACING_G_G (ver):
    return 4

def S_G_INIT (ver):
    return 2 ** (-6)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def NN_HH (ver):
    switcher = {
        cls.Size.TEST : 8,
        cls.Size.SMALL : 16,
        cls.Size.MEDIUM : 24,
        cls.Size.BIG : 32}
    return switcher [ver.size]

def NN_RR (ver):
    return int (NN_HH (ver) / np.sqrt (2))

def NN_YY (ver):
    return int (NN_HH (ver) / 2)

def NN_Y (ver):
    return NN_YY (ver) ** 2

def NN_H (ver):
    return NN_HH (ver) ** 2

def SS_supp_est (ver):
    return NN_YY (ver)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def NUM_S_G (ver):
    return 7

def NUM_H_G (ver): # OMP
    if (ver.focus == cls.Focus.OOMMPP):
        return 3
    elif (ver.focus == cls.Focus.ASSORTED):
        return 1
    else: # cls.Focus.DDSS
        return 0

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

def ITER_MAX_OOMMPP (ver):
    return 4 * NN_HH (ver)

def H_G_OOMMPP_2_NORM (ver):
    return np.sqrt (3) * NN_YY (ver)

def H_G_OOMMPP_INFTY_NORM (ver):
    return 2 * np.sqrt (np.log (NN_HH (ver)))

