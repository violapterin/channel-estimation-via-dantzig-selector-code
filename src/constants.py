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
   switcher = {
      cls.Radio.EQUAL : NN_YY (ver),
      cls.Radio.TWICE : 2 * NN_YY (ver),
      }
   return switcher [ver.radio]

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
   return int (np.sqrt (NN_HH_t (ver) * NN_HH_r (ver)))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def NUM_MET (ver):
   switcher = {
      cls.Threshold.USUAL: 5,
      cls.Threshold.OOMMPP : 9,
      cls.Threshold.LASSO : 7,
      cls.Threshold.DDSS : 7,
      }
   return switcher [ver.threshold]

def NUM_STAGE (ver):
   switcher = {
      cls.Stage.TWO : 2,
      cls.Stage.FOUR : 4,
      cls.Stage.SIX : 6
      }
   return switcher [ver.stage]

def NUM_CHAN_BASIC ():
   return 128

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
   return 5

def SCALE_S_G ():
   return 2

def LST_MET (ver):
   result = [cls.Method.DDSS,
         cls.Method.LASSO,
         cls.Method.OOMMPP_TWO,
         cls.Method.OOMMPP_INFTY,
         cls.Method.LLSS]
   if (ver.threshold == cls.Threshold.OOMMPP):
      result.append (cls.Method.OOMMPP_TWO_LAX)
      result.append (cls.Method.OOMMPP_TWO_TENSE)
      result.append (cls.Method.OOMMPP_INFTY_LAX)
      result.append (cls.Method.OOMMPP_INFTY_TENSE)
   if (ver.threshold == cls.Threshold.LASSO):
      result.append (cls.Method.LASSO_LAX)
      result.append (cls.Method.LASSO_TENSE)
   if (ver.threshold == cls.Threshold.DDSS):
      result.append (cls.Method.DDSS_LAX)
      result.append (cls.Method.DDSS_TENSE)
   return result

def LEGEND (ver):
   result = ["DS",
         "Lasso",
         "OMP, 2 norm",
         "OMP, $\infty$ norm",
         "LS"]
   if (ver.threshold == cls.Threshold.OOMMPP):
      result.append ("OMP, 2 norm, twice $\gamma$")
      result.append ("OMP, 2 norm, half $\gamma$")
      result.append ("OMP, $\infty$ norm, twice $\gamma$")
      result.append ("OMP, $\infty$ norm, half $\gamma$")
   if (ver.threshold == cls.Threshold.LASSO):
      result.append ("Lasso, twice $\gamma$")
      result.append ("Lasso, half $\gamma$")
   if (ver.threshold == cls.Threshold.DDSS):
      result.append ("DS, twice $\gamma$")
      result.append ("DS, half $\gamma$")
   return result



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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def CVX_ITER_MAX (ver):
   return 32 # default : 100

def CVX_TOL_ABS (ver):
   return 5e-7 # default : 1e-7

def CVX_TOL_REL (ver):
   return 5e-6 # default : 1e-6

def CVX_TOL_FEAS (ver):
   return 5e-7 # default : 1e-7

