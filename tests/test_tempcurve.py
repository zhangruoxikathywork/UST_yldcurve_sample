# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 21:04:28 2024

@author: tcoleman & joyma"""

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
import priceyieldfn1 as pyld


imp.reload(dates)
imp.reload(pv)
#imp.reload(df)
imp.reload(pvc)


#%% Create parameter vectors

#    three par bonds: 2yr 1.162%, 5yr 1.721%, 10yr 2.183%  
# 0      1.162% of 15-feb-2018 non-call       4 half-yrs + 1 maturity = 5          
# 1      1.721.5% of 15-feb-2021 non-call        10 half-yrs + 1 maturity = 11
# 2      2.183% of 15-feb-2026 non-call       20 half-year + 1 maturity = 21

# First version of parms does not have tax status. 
parms = [[1.162,43145,2,100.,"A/A","eomyes",0,False],
         [1.721,44241,2,100.,"A/A","eomyes",0,False],
         [2.183,46067,2,100.,"A/A","eomyes",0,False]]

parms = pd.DataFrame(parms)


# settle 15-feb-2016
quotedate = 42414

# Set the prices - par bonds so clean & dirty for all 100. 
#Flat fwds 1.162%, 2.111%, 2.703% (sab); 1.1586% 2.0999%, 2.6849% (cc)
prices_call = np.array([[100.0,100.0],
 [100.0,100.0],
 [100.0,100.0]])





#%% Make curves 


ratessab = np.array([.01162,.02111,.02703])
ratescc = 2.*np.log(1+ratessab/2.)
ratescc6 = np.copy(ratescc)
ratescc6 = ratescc + 0.01         # Set them away from the true values, so that the optimization will have something to do


breaks = quotedate + np.array([2.,5.,11.])*365.25
curvepwcf = ['pwcf',0,breaks,ratescc]
curvepwlz = ['pwlz',0,breaks,ratescc]
curvepwtf = ['pwtf',0,breaks,ratescc]

yvols = np.array([0,0,0.10,0.433,0,0.10,0,0,0])


#%%
xmat = np.array([2])
xpmt = np.array([1.262])
xfv = np.array([100.])
xfreq = np.array([2])

[pyld.PVBondStubFromCurve_vec(xmat,curvepwcf,xpmt,xfv,xfreq), pyld.PVBondStubFromCurve_vec(xmat,curvepwlz,xpmt,xfv,xfreq), 
 pyld.PVBondStubFromCurve_vec(xmat,curvepwtf,xpmt,xfv,xfreq)]


