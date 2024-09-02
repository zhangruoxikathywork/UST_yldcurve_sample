#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 15:04:34 2024

@author: tcoleman
"""
#%%
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


#%%

sys.path.append('../../src/package')
sys.path.append('../../../BondsTable')
sys.path.append('../../tests')
sys.path.append('../../data')

import discfact as df
import pvfn as pv
import DateFunctions_1 as dates
import Curve_Plotting as cp

#%%  build pwcf curve

quotedate=dates.YMDtoJulian(19321231)
breaks = np.array([0.0833, 0.5, 1.,2.,5.,10.,20.,30.])
breakdates = dates.CalAdd(quotedate,nyear=breaks)

plot_points_yr = np.arange(.01,30,.01)
#plot_points_yr = np.arange(.1,30,.1)


rates = np.array([0.02,.025,.03,.035,.04,.045,.05,.06] )
ratescc = 2.*np.log(1+rates/2.)

curvepwcf = ['pwcf',quotedate,breakdates,ratescc]
curvepwlz = ['pwlz',quotedate,breakdates,ratescc]
curvepwtf = ['pwtf',quotedate,breakdates,ratescc+0.005]

#curvepwcf = ['pwcf',quotedate,breaks,ratescc]
#curvepwlz = ['pwlz',quotedate,breaks,ratescc]
#curvepwtf = ['pwtf',quotedate,breaks,ratescc]

fwdcurve_list = [curvepwcf,curvepwlz,curvepwtf]

#%% Testing plotting


cp.zero_curve_plot(fwdcurve_list,plot_points_yr)

ft0 = time.perf_counter()
cp.forward_curve_plot(fwdcurve_list,plot_points_yr)
ft1 = time.perf_counter()

pt0 = time.perf_counter()
cp.parbond_curve_plot(fwdcurve_list,plot_points_yr,twostep=True)
pt1 = time.perf_counter()
cp.parbond_curve_plot(fwdcurve_list,plot_points_yr)
pt2 = time.perf_counter()

print('Forwards: ', ft1-ft0)
print('two-step time: ', pt1-pt0, 'one-step time: ',pt2-pt1)


#cp.parbond_curve_plot_old(breakdates, rates, plot_points_yr, quotedate)


#%%

#xx1 = coupon_rates*pv_annuity + df_maturity
#xx2 = xx1 - coupon_rates * accrued_interest

#x1 = 100*(coupon_rates - 100 * pbrates)
#x1ssq = np.sqrt(sum(x1*x1) / len(x1))

#coupon_rates = 100 * pbrates
#parms_pb[0] = coupon_rates
#dirtyprice = coupon_rates * pv_annuity + df_maturity
#for j in range(len(pbrates)):
#    pbrates[j] = pv.bondYieldFromPrice_parms(dirtyprice[j], 
#                    parms=parms_pb.loc[j:j], 
#                    settledate=xquotedate, parmflag=True, padj=False)

#x2 = 100*(coupon_rates - 100 * pbrates)
#x2ssq = np.sqrt(sum(x2*x2) / len(x2))


