import numpy as np
from enum import Enum
import sys

import constants as cst
import functions as fct

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class Version:
   def __init__ (s, size, ratio, stage):
      s.size = size
      s.ratio = ratio
      s.stage= stage

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
   FOUR = 3

