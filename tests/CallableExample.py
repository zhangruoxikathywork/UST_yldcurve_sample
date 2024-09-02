# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: tcoleman """

# magic %reset resets by erasing variables, etc. 

import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.optimize as so
import importlib as imp
import cProfile
import time as time
import matplotlib.pyplot as plt

sys.path.append('../src/package')

#%%

import discfact as df
import pvfn as pv
import DateFunctions_1 as dates
import pvcover as pvc


imp.reload(dates)
imp.reload(pv)
#imp.reload(df)
imp.reload(pvc)


#%% Create parameter vectors sab

# Semi bond, 4.7% coupon, maturity 1-sep-2045, first call 1-sep-2018
parms = [[4.7,dates.YMDtoJulian([2045,9,1])[0],2,100.,"A/A","eomyes",dates.YMDtoJulian([2018,9,1])[0],False],
         [4.7,dates.YMDtoJulian([2045,9,1])[0],2,100.,"A/A","eomyes",dates.YMDtoJulian([2018,9,1])[0],True]]

parms = pd.DataFrame(parms)
ai = 2.20797

# settle 19-feb-2016
quotedate = dates.YMDtoJulian([2016,2,19])
settledate = quotedate
# First call date 1-sep-2018
calldate = dates.YMDtoJulian([2018,9,1])

# Set the prices - clean P then dirty P
prices_call = np.array([[95.80,98.00797],
 [95.80,98.00797]])

sparsenc = pv.bondParmsToMatrix(quotedate,parms)
xcfncsab = sparsenc[0].toarray()
xdtncsab = dates.JuliantoYMD(sparsenc[2])


#%%


xYncsab = pv.bondYieldFromPrice_parms(dirtyprice=prices_call[0][1],  settledate=quotedate,parms=parms[1:2], freq=2, parmflag=True)
xPncsab = pv.bondPriceFromYield_parms(xYncsab,  settledate=quotedate,parms=parms[0:1], freq=2, parmflag=True)
xPncsab = xPncsab - ai

xFPncsab = pv.bondPriceFromYield_parms(xYncsab,  settledate=calldate,parms=parms[0:1], freq=2, parmflag=True)

xPycsab = pv.bondPriceFromYield_callable(xyield=xYncsab,  settledate=quotedate,parms=parms[1:2], 
                               freq=2, parmflag=True,callflag=True,vol=0.2)
xPycsab = xPycsab - ai
xPycsab4 = pv.bondPriceFromYield_callable(xyield=.04,  settledate=quotedate,parms=parms[1:2], 
                               freq=2, parmflag=True,callflag=True,vol=0.2)
xPycsab4 = xPycsab4 - ai
xYycsab = pv.bondYieldFromPrice_callable(dirtyprice=prices_call[1][1],settledate=quotedate,parms=parms[1:2], 
                               freq=2, parmflag=True,callflag=True,vol=0.2)

xPncOAYsab = pv.bondPriceFromYield_parms(xYycsab,  settledate=quotedate,parms=parms[0:1], freq=2, parmflag=True)
xPncOAYsab = xPncOAYsab - ai

xFPncOAYsab = pv.bondPriceFromYield_parms(xYycsab,  settledate=calldate,parms=parms[0:1], freq=2, parmflag=True)


print([xPncsab,xYncsab,xPycsab,xPycsab4,xYycsab,xFPncsab,xPncOAYsab,xFPncOAYsab])

# From by-hand calculations with 4.972%sab, CP 95.80, AI 2.20797
#   HP calculator: 4.972%sab -> CP = 95.80, DP = 98.0080, CP 95.80 -> 4.9727%sab
#   Excel calculations (sheet "testyldcalc.xls", 365.25 days/yr) PV = 98.01224
#   Python matches PV exactly, python DP=98.007967 -> yld = 4.97229%sab, DP=98.01224 -> yld = 4.972000%


#%% Create parameter vectors ab

# Annual bond, 4.7% coupon, maturity 1-sep-2045, first call 1-sep-2018
parms = [[4.7,53205.,1,100.,"A/A","eomyes",43343,False],
         [4.7,53205.,1,100.,"A/A","eomyes",43343,True]]

parms = pd.DataFrame(parms)


# settle 19-feb-2016
quotedate = 42418
settledate = quotedate

# Set the prices - clean P then dirty P
prices_call = np.array([[95.80,98.00117],
 [95.80,98.00117]])

sparsenc = pv.bondParmsToMatrix(quotedate,parms)
xcfncab = sparsenc[0].toarray()
xdtncab = dates.JuliantoYMD(sparsenc[2])


#%%

yieldab = .04972

xPncab = pv.bondPriceFromYield_parms(yieldab,  settledate=quotedate,parms=parms[0:1], freq=1, parmflag=True)
xYncab = pv.bondYieldFromPrice_parms(dirtyprice=prices_call[0][1],  settledate=quotedate,parms=parms[1:2], freq=1, parmflag=True)

xPycab = pv.bondPriceFromYield_callable(xyield=yieldab,  settledate=quotedate,parms=parms[1:2], 
                               freq=1, parmflag=True,callflag=True,vol=0.2)
xYycab = pv.bondYieldFromPrice_callable(dirtyprice=prices_call[1][1],settledate=quotedate,parms=parms[1:2], 
                               freq=1, parmflag=True,callflag=True,vol=0.2)


print([xPncab,xYncab,xPycab,xYycab])

# From by-hand calculations with 4.972%ab, CP 95.80, AI 2.1959
#   HP calculator: 4.972%sab -> CP = 95.80, DP = 97.9959, CP 95.80 -> 4.9724%ab
#   Excel calculations (sheet "testyldcalc.xls", 365.25 days/yr) PV = 98.001166
#   Python matches PV exactly, python DP=97.9959 -> yld = 4.97235%sab, DP=98.00117 -> yld = 4.972000

