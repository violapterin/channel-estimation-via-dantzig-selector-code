#! /usr/bin/env python3

import os

import functions as fct
import classes as cls

ver = cls.Version (
     cls.Size.SMALL,
     cls.Ratio.SQUARE,
     cls.Stage.TWO)

fct.execute (ver)

ver = cls.Version (
     cls.Size.SMALL,
     cls.Ratio.TALL,
     cls.Stage.TWO)

fct.execute (ver)

ver = cls.Version (
     cls.Size.SMALL,
     cls.Ratio.WIDE,
     cls.Stage.TWO)

fct.execute (ver)
