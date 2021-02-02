import numpy as np
import classes as cls

def NN_HH (ver):
   switcher = {
      cls.Size.SMALL : 6,
      cls.Size.MEDIUM : 12,
      cls.Size.BIG : 18,
      }
   return switcher [ver.size]

def NN_H (ver):
   return NN_HH (ver) ** 2

def NN_YY_t (ver):
   switcher = {
      cls.Ratio.TALL : int (NN_HH (ver) /3),
      cls.Ratio.WIDE : int (NN_HH (ver) /2),
      cls.Ratio.SQUARE : int (NN_HH (ver) /2),
      }
   return switcher [ver.ratio]

def NN_YY_r (ver):
   switcher = {
      cls.Ratio.TALL : int (NN_HH (ver) /2),
      cls.Ratio.WIDE : int (NN_HH (ver) /3),
      cls.Ratio.SQUARE : int (NN_HH (ver) /2),
      }
   return switcher [ver.ratio]

def NN_Y_t (ver):
   return NN_Y_t (ver) ** 2

def NN_Y_r (ver):
   return NN_Y_r (ver) ** 2

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def LAMBDA_ANT (ver):
   return 1

def DIST_ANT (ver):
   return 3

def LL (ver):
   return NN_HH (ver)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def NUM_MET ():
    return 5

def NUM_STAGE (ver):
   switcher = {
      cls.Stage.ONE: 1,
      cls.Stage.TWO: 2,
      cls.Stage.FOUR: 4}
   return switcher [ver.stage]

def NUM_CHAN_BASIC ():
   return 128

def NUM_CHAN_MET (met):
   switcher = {
      cls.Method.LLSS : 8,
      cls.Method.OOMMPP_TWO : 4,
      cls.Method.OOMMPP_INFTY : 4,
      cls.Method.LASSO : 2,
      cls.Method.DDSS : 1,
      }
   return switcher [met] * NUM_CHAN_BASIC ()

def S_G_INIT ():
   return 2 ** (-3)

def NUM_S_G ():
   return 6

def SCALE_S_G ():
   return 2

def LST_MET ():
   return [cls.Method.DDSS,
         cls.Method.LASSO,
         cls.Method.OOMMPP_TWO,
         cls.Method.OOMMPP_INFTY,
         cls.Method.LLSS]

def LEGEND ():
   return ["Dantzig Selector",
         "Lasso",
         "OMP, two norm",
         "OMP, infinity norm",
         "least square"]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def G_G_DDSS (ver):
   return 2 * np.sqrt (np.log (NN_HH (ver)))

def G_G_LASSO (ver):
   return 2 * np.sqrt (np.log (NN_HH (ver)))

def H_G_OOMMPP_TWO (ver):
   return np.sqrt (3) * (NN_YY_t (ver) * NN_YY_r (ver)) ** (1/4)

def H_G_OOMMPP_INFTY (ver):
   return 2 * np.sqrt (np.log (NN_HH (ver)))

def ITER_MAX_OOMMPP (ver):
   return 2 * NN_H (ver)

def MAX_NORM (ver):
   return 4 * NN_H (ver)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def CVX_ITER_MAX (ver):
   return 32 # default: 100

def CVX_TOL_ABS (ver):
   return 5e-7 # default: 1e-7

def CVX_TOL_REL (ver):
   return 5e-6 # default: 1e-6

def CVX_TOL_FEAS (ver):
   return 5e-7 # default: 1e-7

