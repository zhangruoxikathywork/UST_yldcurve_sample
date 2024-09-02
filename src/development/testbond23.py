#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 13:18:22 2024

@author: tcoleman
"""

# Testing bond 23 from December 1932. After running "readCRSP_Optimize_TSC.py"
# This file should be under "tests" but too complicated to get the directories forimports and reading data wording


import pvfn as pv
from readCRSP_Optimize_TSC import parms, quotedate

#%%

xPycsab_001 = pv.bondPriceFromYield_callable(xyield=.001,  settledate=quotedate,parms=parms[23:24], 
                               freq=2, parmflag=True,callflag=True,vol=0.2)
xPycsab_05 = pv.bondPriceFromYield_callable(xyield=.05,  settledate=quotedate,parms=parms[23:24], 
                               freq=2, parmflag=True,callflag=True,vol=0.2)
xPycsab_m01 = pv.bondPriceFromYield_callable(xyield=-0.01,  settledate=quotedate,parms=parms[23:24], 
                               freq=2, parmflag=True,callflag=True,vol=0.2)

print(xPycsab_001,xPycsab_05,xPycsab_m01)

