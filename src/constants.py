import numpy as np
import classes as cls

def NN_HH (ver):
   switcher = {
      cls.Size.SMALL : 12,
      cls.Size.MEDIUM : 18,
      cls.Size.BIG : 24,
      }
   return switcher [ver.size]

def NN_H (ver):
   return NN_HH (ver) ** 2

def NN_YY_t (ver):
   switcher = {
      cls.Ratio.TALL : int (NN_HH (ver) /2),
      cls.Ratio.WIDE : int (NN_HH (ver) /3),
      cls.Ratio.SQUARE : int (NN_HH (ver) /3),
      }
   return switcher [ver.ratio]

def NN_YY_r (ver):
   switcher = {
      cls.Ratio.TALL : int (NN_HH (ver) /3),
      cls.Ratio.WIDE : int (NN_HH (ver) /2),
      cls.Ratio.SQUARE : int (NN_HH (ver) /3),
      }
   return switcher [ver.ratio]

def NN_Y_t (ver):
   return NN_Y_t (ver) ** 2

def NN_Y_r (ver):
   return NN_Y_r (ver) ** 2

def SS_SUPP_H (ver):
   return int (L * np.log (NN_HH (ver)))

def DIFF_SP (ver):
   switcher = {
      cls.Stage.TWO: 3,
      cls.Stage.THREE: 2,
      cls.Stage.SIX: 1}
   return switcher [ver.stage] * int (NN_H (ver) /9)

def RELAX_THRESHOLD (ver):
   return 3

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

def NUM_MET ():
    return 5

def NUM_STAGE (ver):
   switcher = {
      cls.Stage.TWO: 2,
      cls.Stage.THREE: 3,
      cls.Stage.SIX: 6}
   return switcher [ver.stage]

def NUM_CHAN_BASIC ():
   return 16

def NUM_CHAN_MET (met):
   switcher = {
      cls.Method.LLSS : 12,
      cls.Method.OOMMPP_TWO : 6,
      cls.Method.OOMMPP_INFTY : 6,
      cls.Method.LASSO : 2,
      cls.Method.DDSS : 1,
      }
   return switcher [met] * NUM_CHAN_BASIC ()

def S_G_INIT ():
   return 2 ** (-5)

def NUM_S_G ():
   return 7

def SCALE_S_G ():
   return np.sqrt (2)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def G_G_DDSS (ver):
   return 2 * np.sqrt (np.log (NN_HH (ver)))

def G_G_LASSO (ver):
   return NN_YY_t (ver) * NN_YY_r (ver) / 8

def H_G_OOMMPP_TWO (ver):
   return np.sqrt (3 * NN_YY_t (ver) * NN_YY_r (ver))

def H_G_OOMMPP_INFTY (ver):
   return 2 * np.sqrt (np.log (NN_HH (ver)))

def ITER_MAX_OOMMPP (ver):
   return 4 * NN_HH (ver)

def MAX_NORM (ver):
   return 8 * NN_HH (ver)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def CVX_ITER_MAX (ver):
   return 32 # default: 100

def CVX_TOL_ABS (ver):
   return 5e-7 # default: 1e-7

def CVX_TOL_REL (ver):
   return 5e-6 # default: 1e-6

def CVX_TOL_FEAS (ver):
   return 5e-7 # default: 1e-7

