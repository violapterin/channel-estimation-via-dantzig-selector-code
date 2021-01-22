import numpy as np
from enum import Enum
import sys

import constants as cst
import functions as fct

'''
class Beamformer:
   def __init__ (s, ver):
      s.ver = ver
      s.pp = np.zeros ((cst.NN_Y (s.ver), cst.NN_H (s.ver)))

   def generate (s):
      kk = fct.get_kk (s.ver)
      ff_bb = fct.pick_mat_bb (s.ver).T
      ff_rr = fct.pick_mat_rr (s.ver).T
      ww_bb = fct.pick_mat_bb (s.ver)
      ww_rr = fct.pick_mat_rr (s.ver)
      pp = np.kron (ff_bb.T @ ff_rr.T @ kk.conj (), ww_bb @ ww_rr @ kk)
      s.pp = pp
'''

'''
class Channel:
   def __init__ (s, pp, s_g, ver):
      s.ver = ver
      s.pp = pp
      s.s_g = s_g
      s.hh = np.zeros ((cst.NN_HH (s.ver), cst.NN_HH (s.ver)), dtype = complex)
      s.y = np.zeros ((cst.NN_Y (s.ver)))

   def generate (s):
      s.hh = fct.pick_hh (s.ver)
      s.norm_hh = np.linalg.norm (s.hh, ord = 'fro')

   def zero (s):
      s.hh = np.zeros ((cst.NN_HH (s.ver), cst.NN_HH (s.ver)), dtype = complex)

   def transmit (s):
      kk = fct.get_kk (s.ver)
      zz = fct.pick_zz (s.ver)
      gg = kk.conj ().T @ s.hh @ kk
      g = fct.vectorize (gg)
      z = fct.vectorize (zz)
      s.y = s.pp @ g + (s.s_g / np.sqrt(2)) * z
'''

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

'''
class Estimation:
   def __init__ (s, hh, s_g, ver):
      s.ver = ver
      s.hh = hh
      s.supp = range (2 * cst.NN_H (ver))
      s.s_g = s_g
      s.g_r_h = np.zeros (2 * (cst.NN_H (s.ver)))
      s.hh_h = np.zeros ((cst.NN_HH (s.ver), cst.NN_HH (s.ver)), dtype = complex)
      s.rr = 0

   def convert (s):
      g_h = fct.inv_find_rep_vec (s.g_r_h)
      gg_h = fct.inv_vectorize (g_h, cst.NN_HH (s.ver), cst.NN_HH (s.ver))
      s.hh_h = (fct.get_kk (s.ver) @ gg_h @ fct.get_kk (s.ver).conj().T)
      nor_hh = np.linalg.norm (s.hh, ord = 'fro')
      nor_ee = np.linalg.norm (s.hh - s.hh_h, ord = 'fro')
      s.rr = nor_hh / (nor_ee + 2 * cst.NN_H (ver) * s.s_g)

   def find_ddss_theory (s, s_g):
      nor_hh = np.linalg.norm (s.hh, ord = 'fro')
      nor_ee = 8 * s_g * (cst.LL (s.ver) ** (1/2)) * (np.log (cst.NN_HH (s.ver)) ** (3/2))
      s.rr = nor_hh / (nor_ee + s.hh * s.s_g)
'''

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class Version:
   def __init__ (s, size, ratio, stage):
      s.size = size
      s.ratio = ratio
      s.round = stage

class Method (Enum):
   LLSS = 1
   LASSO = 2
   OOMMPP_TWO = 3
   OOMMPP_INFTY = 4
   DDSS = 5

class Size (Enum):
   SMALL = 1
   MEDIUM = 2
   BIG = 3

class Ratio (Enum):
   TALL = 1
   WIDE = 2
   SQUARE = 3

class Stage (Enum):
   ONE = 1
   TWO = 2
   THREE = 3

