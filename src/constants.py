import numpy as np
import classes as cls

def NN_YY (ver):
   switcher = {
      cls.Data.SMALL : 2,
      cls.Data.MEDIUM : 4,
      cls.Data.BIG : 6,
   }
   return switcher [ver.data]

def NN_RR (ver):
   return int ((3/2) * NN_YY (ver))

def NN_HH_t (ver):
   switcher = {
      cls.Channel.SQUARE : 3 * NN_YY (ver),
      cls.Channel.TALL : 3 * NN_YY (ver),
      cls.Channel.WIDE : 4 * NN_YY (ver),
   }
   return switcher [ver.channel]

def NN_HH_r (ver):
   switcher = {
      cls.Channel.SQUARE : 3 * NN_YY (ver),
      cls.Channel.TALL : 4 * NN_YY (ver),
      cls.Channel.WIDE : 3 * NN_YY (ver),
   }
   return switcher [ver.channel]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def LAMBDA_ANT (ver):
   return 1

def DIST_ANT (ver):
   return 3

def LL (ver):
   return int (np.sqrt (NN_HH_t (ver) * NN_HH_r (ver)) / 2)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def NUM_MET (ver):
   switcher = {
      cls.Threshold.USUAL: 5,
      cls.Threshold.OOMMPP : 8,
      cls.Threshold.LASSO : 6,
      cls.Threshold.DDSS : 6,
   }
   return switcher [ver.threshold]

def NUM_STAGE (ver):
   switcher = {
      cls.Stage.THREE : 3,
      cls.Stage.SIX : 6,
      cls.Stage.NINE : 9
   }
   return switcher [ver.stage]

def NUM_CHAN_BASIC ():
   return 96

def NUM_CHAN_MET (met):
   switcher = {
      cls.Method.LLSS : 8,
      cls.Method.OOMMPP_TWO_LAX : 4,
      cls.Method.OOMMPP_TWO : 4,
      cls.Method.OOMMPP_TWO_TENSE : 4,
      cls.Method.OOMMPP_INFTY_LAX : 4,
      cls.Method.OOMMPP_INFTY : 4,
      cls.Method.OOMMPP_INFTY_TENSE : 4,
      cls.Method.LASSO_LAX : 2,
      cls.Method.LASSO : 2,
      cls.Method.LASSO_TENSE : 2,
      cls.Method.DDSS_LAX : 1,
      cls.Method.DDSS : 1,
      cls.Method.DDSS_TENSE : 1,
   }
   return switcher [met] * NUM_CHAN_BASIC ()

def S_G_INIT ():
   return 2 ** (-2)

def NUM_S_G ():
   return 6

def SCALE_S_G ():
   return np.sqrt (2)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def G_G_DDSS (ver):
   return 2 * np.sqrt (np.log (np.sqrt (NN_HH_t (ver) * NN_HH_r (ver))))

def G_G_LASSO (ver):
   return 2 * np.sqrt (np.log (np.sqrt (NN_HH_t (ver) * NN_HH_r (ver))))

def H_G_OOMMPP_TWO (ver):
   return np.sqrt (3) * np.sqrt (NN_YY (ver))

def H_G_OOMMPP_INFTY (ver):
   return 2 * np.sqrt (np.log (np.sqrt (NN_HH_t (ver) * NN_HH_r (ver))))

def ITER_MAX_OOMMPP (ver):
   return 2 * NN_HH_t (ver) * NN_HH_r (ver)

def MAX_NORM (ver):
   return 4 * NN_HH_t (ver) * NN_HH_r (ver)

def CVX_ITER_MAX (ver):
   return 32 # default : 100

def CVX_TOL_ABS (ver):
   return 5e-7 # default : 1e-7

def CVX_TOL_REL (ver):
   return 5e-6 # default : 1e-6

def CVX_TOL_FEAS (ver):
   return 5e-7 # default : 1e-7

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def LST_MET (ver):
   if (ver.threshold == cls.Threshold.USUAL):
      result = [
         cls.Method.DDSS,
         cls.Method.LASSO,
         cls.Method.OOMMPP_TWO,
         cls.Method.OOMMPP_INFTY,
         cls.Method.LLSS,
      ]
   if (ver.threshold == cls.Threshold.OOMMPP):
      result = [
         cls.Method.DDSS,
         cls.Method.LASSO,
         cls.Method.OOMMPP_TWO,
         cls.Method.OOMMPP_TWO_LAX,
         cls.Method.OOMMPP_TWO_TENSE,
         cls.Method.OOMMPP_INFTY,
         cls.Method.OOMMPP_INFTY_LAX,
         cls.Method.OOMMPP_INFTY_TENSE,
      ]
   if (ver.threshold == cls.Threshold.LASSO):
      result = [
         cls.Method.DDSS,
         cls.Method.LASSO,
         cls.Method.LASSO_LAX,
         cls.Method.LASSO_TENSE,
         cls.Method.OOMMPP_TWO,
         cls.Method.OOMMPP_INFTY,
      ]
   if (ver.threshold == cls.Threshold.DDSS):
      result = [
         cls.Method.DDSS,
         cls.Method.DDSS_LAX,
         cls.Method.DDSS_TENSE,
         cls.Method.LASSO,
         cls.Method.OOMMPP_TWO,
         cls.Method.OOMMPP_INFTY,
      ]
   return result

def LEGEND (ver):
   if (ver.threshold == cls.Threshold.USUAL):
      result = [
         "DS",
         "Lasso",
         "OMP, 2 norm",
         "OMP, $\infty$ norm",
         "LS",
      ]
   if (ver.threshold == cls.Threshold.OOMMPP):
      result = [
         "DS",
         "Lasso",
         "OMP, 2 norm",
         "OMP, 2 norm, twice $\gamma$",
         "OMP, 2 norm, half $\gamma$",
         "OMP, $\infty$ norm",
         "OMP, $\infty$ norm, twice $\gamma$",
         "OMP, $\infty$ norm, half $\gamma$",
      ]
   if (ver.threshold == cls.Threshold.LASSO):
      result = [
         "DS",
         "Lasso",
         "Lasso, twice $\gamma$",
         "Lasso, half $\gamma$",
         "OMP, 2 norm",
         "OMP, $\infty$ norm",
      ]
   if (ver.threshold == cls.Threshold.DDSS):
      result = [
         "DS",
         "DS, twice $\gamma$",
         "DS, half $\gamma$",
         "Lasso",
         "OMP, 2 norm",
         "OMP, $\infty$ norm",
      ]
   return result

