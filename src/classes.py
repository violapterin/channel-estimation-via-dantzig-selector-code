import numpy as np

import constants as cst
import functions as fct

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class Version (object):
   def __init__ (s, data, channel, stage, threshold):
      s.data = data
      s.channel = channel
      s.threshold = threshold
      s.stage= stage

class Method (object):
   LLSS = 1
   OOMMPP_TWO_LAX = 2
   OOMMPP_TWO = 3
   OOMMPP_TWO_TENSE = 4
   OOMMPP_INFTY_LAX = 5
   OOMMPP_INFTY = 6
   OOMMPP_INFTY_TENSE = 7
   LASSO_LAX = 8
   LASSO = 9
   LASSO_TENSE = 10
   DDSS_LAX = 11
   DDSS = 12
   DDSS_TENSE = 13

class Data (object):
   SMALL = 1
   MEDIUM = 2
   BIG = 3

class Channel (object):
   SQUARE = 1
   TALL = 2
   WIDE = 3

class Stage (object):
   THREE = 1
   SIX = 2
   NINE = 3

class Threshold (object):
   USUAL = 1
   OOMMPP = 2
   LASSO = 3
   DDSS = 4

