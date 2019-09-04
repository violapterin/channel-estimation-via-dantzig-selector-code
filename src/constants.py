import numpy as np
import classes as cls

def NUM_GRID_PHASE ():
    return 16

def LAMBDA_ANT ():
    return 3

def DIST_ANT ():
    return 2

def VALUE_SPACING ():
    return 2

def NUM_SIGMA ():
    return 7

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def NN_YY (ver):
    switcher = {
        cls.Size.TEST : 3,
        cls.Size.SMALL : 5,
        cls.Size.MEDIUM : 7,
        cls.Size.BIG : 9}
    return switcher [ver.size]

def NN_RR (ver):
    switcher = {
        cls.Size.TEST : 6,
        cls.Size.SMALL : 10,
        cls.Size.MEDIUM : 14,
        cls.Size.BIG : 18}
    return switcher [ver.size]

def NN_HH (ver):
    switcher = {
        cls.Size.TEST : 12,
        cls.Size.SMALL : 20,
        cls.Size.MEDIUM : 28,
        cls.Size.BIG : 36}
    return switcher [ver.size]

def LL (ver):
    switcher = {
        cls.Size.TEST : 1,
        cls.Size.SMALL : 2,
        cls.Size.MEDIUM : 3,
        cls.Size.BIG : 4}
    return switcher [ver.size]

def NN_Y (ver):
    return NN_YY (ver) ** 2

def NN_H (ver):
    return NN_HH (ver) ** 2

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def NUM_REPEAT (ver):
    switcher = {
        cls.Size.TEST : 8,
        cls.Size.SMALL : 32,
        cls.Size.MEDIUM : 32,
        cls.Size.BIG : 32}
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
        return 5
    elif (ver.focus == cls.Focus.ASSORTED):
        return 1
    else: # cls.Focus.OOMMPP
        return 0

def MAX_ITER_OOMMPP (ver):
    return 4 * NN_HH (ver)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def GAMMA_DDSS (ver):
    return 2 * np.sqrt (np.log (cst.NN_HH (ver)))

def GAMMA_LASSO (ver): # just copying `GAMMA_DDSS`
    return 2 * np.sqrt (np.log (cst.NN_HH (ver)))

def ETA_OOMMPP_2_NORM (ver):
    return np.sqrt (3) * cst.NN_Y (ver)

def ETA_OOMMPP_INFTY_NORM (ver):
    return 2 * np.sqrt (np.log (cst.NN_HH (ver)))
