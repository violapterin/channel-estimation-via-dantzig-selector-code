import numpy as np
import classes as cls

def NUM_GRID_PHASE ():
    return 16

def LAMBDA_ANT ():
    return 1

def DIST_ANT ():
    return 3

def LL ():
    return 4

def VALUE_SPACING ():
    return 2

def NUM_SIGMA ():
    return 7

def ITER_MAX_CVX ():
    return 32

def TOLERANCE_ABS_CVX ():
    return 1e-4

def TOLERANCE_REL_CVX ():
    return 1e-3

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def NN_HH (ver):
    switcher = {
        cls.Size.TEST : 8,
        cls.Size.SMALL : 16,
        cls.Size.MEDIUM : 32,
        cls.Size.BIG : 64}
    return switcher [ver.size]

def NN_RR (ver):
    return int (np.round (np.log (NN_HH (ver)) ** 2))

def NN_YY (ver):
    return int (np.round (np.log (NN_HH (ver))))

def NN_Y (ver):
    return NN_YY (ver) ** 2

def NN_H (ver):
    return NN_HH (ver) ** 2

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def NUM_REPEAT (ver):
    switcher = {
        cls.Size.TEST : 16,
        cls.Size.SMALL : 64,
        cls.Size.MEDIUM : 64,
        cls.Size.BIG : 64}
    return switcher [ver.size]

def NUM_ETA (ver): # OMP
    if (ver.focus == cls.Focus.OOMMPP):
        return 3
    elif (ver.focus == cls.Focus.ASSORTED):
        return 1
    else: # cls.Focus.DDSS
        return 0

def NUM_GAMMA_DS (ver): # DS
    if (ver.focus == cls.Focus.DDSS):
        return 3
    elif (ver.focus == cls.Focus.ASSORTED):
        return 1
    else: # cls.Focus.OOMMPP
        return 0

def ITER_MAX_OOMMPP (ver):
    return 4 * NN_HH (ver)

# To avoid division by zero.
def EPSILON ():
    return 10 ** (-12)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def GAMMA_DDSS (ver):
    return 2 * np.sqrt (np.log (NN_HH (ver)))

def GAMMA_LASSO (ver): # just copying `GAMMA_DDSS`
    return 2 * np.sqrt (np.log (NN_HH (ver)))

def ETA_OOMMPP_2_NORM (ver):
    return np.sqrt (3) * NN_Y (ver)

def ETA_OOMMPP_INFTY_NORM (ver):
    return 2 * np.sqrt (np.log (NN_HH (ver)))

