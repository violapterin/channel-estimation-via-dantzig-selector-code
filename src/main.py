#! /usr/bin/env python3

import functions as fct
import classes as cls

ver = cls.Version (cls.Size.TEST, cls.Focus.DDSS)
#ver = cls.Version (cls.Size.SMALL, cls.Focus.DDSS)
fct.execute (ver)

#ver = cls.Version (cls.Size.TEST, cls.Focus.ASSORTED)
#fct.execute (ver)


