import numpy as np
from enum import Enum
import sys

import constants as cst
import functions as fct

class Estimation:
    def __init__ (self, pp_rep, y_rep, hh, ver):
        self.ver = ver
        self.hh = hh
        self.y_rep = y_rep
        self.pp_rep = pp_rep
        self.g_rep_hat = np.zeros (2 * (cst.NN_H (self.ver)))
        self.g_hat = np.zeros ((cst.NN_H (self.ver)), dtype = complex)
        self.gg_hat = np.zeros ((cst.NN_HH (self.ver), cst.NN_HH (self.ver)), dtype = complex)
        self.hh_hat = np.zeros ((cst.NN_HH (self.ver), cst.NN_HH (self.ver)), dtype = complex)
        self.d = 0

    def zero (self):
        self.g_rep_hat = np.zeros (2 * (cst.NN_H (self.ver)))
        self.g_hat = np.zeros ((cst.NN_H (self.ver)), dtype = complex)
        self.gg_hat = np.zeros ((cst.NN_HH (self.ver), cst.NN_HH (self.ver)), dtype = complex)
        self.hh_hat = np.zeros ((cst.NN_HH (self.ver), cst.NN_HH (self.ver)), dtype = complex)
        self.d = 0

    def convert (self):
        self.g_hat = fct.inv_find_rep_vec (self.g_rep_hat)
        self.gg_hat = fct.inv_vectorize (self.g_hat, cst.NN_HH (self.ver), cst.NN_HH (self.ver))
        self.hh_hat = (fct.get_kk (self.ver) @ self.gg_hat @ fct.get_kk (self.ver).conj().T)
        self.d = np.linalg.norm (self.hh_hat - self.hh, ord = 'fro')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class Version:
    def __init__ (self, size, focus):
        self.size = size
        self.focus = focus

class Size (Enum):
    TEST = 1
    SMALL = 2
    MEDIUM = 3
    BIG = 4

class Focus (Enum):
    OOMMPP = 1
    DDSS = 2
    ASSORTED = 3
