#! /usr/bin/env python3

import os

import functions as fct
import classes as cls

ver = cls.Version (cls.Size.SMALL, cls.Ratio.TALL, cls.Stage.ONE)
fct.execute (ver)
ver = cls.Version (cls.Size.SMALL, cls.Ratio.WIDE, cls.Stage.ONE)
fct.execute (ver)
ver = cls.Version (cls.Size.SMALL, cls.Ratio.SQUARE, cls.Stage.ONE)
fct.execute (ver)
