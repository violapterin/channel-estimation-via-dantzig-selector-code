#! /usr/bin/env python3

import os

import functions as fct
import classes as cls

ver = cls.Version (cls.Data.SMALL, cls.Channel.TALL, cls.Stage.TWO, cls.Threshold.USUAL)
fct.execute (ver)

ver = cls.Version (cls.Data.SMALL, cls.Channel.TALL, cls.Stage.FOUR, cls.Threshold.USUAL)
fct.execute (ver)

ver = cls.Version (cls.Data.SMALL, cls.Channel.TALL, cls.Stage.SIX, cls.Threshold.USUAL)
fct.execute (ver)
