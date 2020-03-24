import numpy as np
from enum import Enum
import sys

import constants as cst
import functions as fct


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
        pp /= np.linalg.norm (pp, ord = 'fro')
        s.pp = pp

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


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class Estimation:
    def __init__ (s, cnt_each, hh, pp, y, s_g, ver):
        s.ver = ver
        s.cnt_each = cnt_each
        s.hh = hh
        s.pp = pp
        s.y = y
        s.s_g = s_g
        s.g_rep_hat = np.zeros (2 * (cst.NN_H (s.ver)))
        s.hh_hat = np.zeros ((cst.NN_HH (s.ver), cst.NN_HH (s.ver)), dtype = complex)
        #s.d = 0
        s.rr = 0

    def zero (s):
        s.g_rep_hat = np.zeros (2 * (cst.NN_H (s.ver)))
        s.hh_hat = np.zeros ((cst.NN_HH (s.ver), cst.NN_HH (s.ver)), dtype = complex)
        #s.d = 0
        s.rr = 0

    def set_g_rep_hat (s, g_rep_hat):
        s.g_rep_hat = g_rep_hat

    def convert (s):
        nor = np.linalg.norm (s.g_rep_hat, ord = 1)
        if (nor > cst.D_MAX (s.ver) * cst.NN_H (s.ver)):
            s.zero ()
        g_hat = fct.inv_find_rep_vec (s.g_rep_hat)
        gg_hat = fct.inv_vectorize (g_hat, cst.NN_HH (s.ver), cst.NN_HH (s.ver))
        s.hh_hat = (fct.get_kk (s.ver) @ gg_hat @ fct.get_kk (s.ver).conj().T)
        #s.d = np.linalg.norm (s.hh_hat - s.hh, ord = 'fro')

        nor_hh = np.linalg.norm (s.hh, ord = 2)
        nor_ee  = np.linalg.norm (s.hh - s.hh_hat, ord = 2)
        hh_tot = (np.sqrt (cst.NN_HH (s.ver))
            * (s.s_g + 2 * nor_ee / np.sqrt (cst.NN_HH (s.ver))) ** (-1)
            * s.hh @ s.hh.conj().T)
        s.rr = np.log2 (np.abs (np.linalg.det (np.eye (cst.NN_HH (s.ver)) + hh_tot)))

    def find_ddss_theory (s, s_g):
        nor_hh = np.linalg.norm (s.hh, ord = 2)
        nor_ee = 8 * s_g * (cst.LL (s.ver) ** (1/2)) * (np.log (cst.NN_HH (s.ver)) ** (3/2))
        hh_tot = (np.sqrt (cst.NN_HH (s.ver))
            * (s.s_g + 2 * nor_ee / np.sqrt (cst.NN_HH (s.ver))) ** (-1)
            * s.hh @ s.hh.conj().T)
        s.rr = np.log2 (np.abs (np.linalg.det (np.eye (cst.NN_HH (s.ver)) + hh_tot)))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class Version:
    def __init__ (s, size, focus):
        s.size = size
        s.focus = focus

class Size (Enum):
    TEST = 1
    SMALL = 2
    MEDIUM = 3
    BIG = 4

class Focus (Enum):
    OOMMPP = 1
    DDSS = 2
    ASSORTED = 3
