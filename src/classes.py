import numpy as np
import sys
from enum import Enum

import constants as cst
import functions as fct

class Estimation:
    def __init__ (self, pp, y, hh, ver):
        self.pp = pp
        self.y = y
        self.hh = hh
        self.ver = ver
        self.g_hat = np.zeros ((cst.NN_H (ver)), dtype = complex)
        self.hh_hat = np.zeros ((cst.NN_HH (ver), cst.NN_HH (ver)), dtype = complex)
        self.d = 0

    def refresh (self):
        self.g_hat = np.zeros ((cst.NN_H (self.ver)), dtype = complex)
        self.hh_hat = np.zeros ((cst.NN_HH (self.ver), cst.NN_HH (self.ver)), dtype = complex)
        self.d = 0

    def convert (self):
        self.hh_hat = (fct.get_kk (self.ver)
            @ fct.inv_vectorize (self.g_hat, cst.NN_HH (self.ver), cst.NN_HH (self.ver))
            @ fct.get_kk (self.ver).conj().T)
        self.d = np.linalg.norm (self.hh_hat - self.hh, ord=2)

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
