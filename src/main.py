#! /usr/bin/env python3

import os

import functions as fct
import classes as cls

NUM_REP_FIG = 12
for i in range (NUM_REP_FIG):
    idx_hold = i + 1
    ver = cls.Version (
        cls.Size.MEDIUM,
        cls.Focus.ASSORTED,
        str (idx_hold))
    fct.execute (ver)


