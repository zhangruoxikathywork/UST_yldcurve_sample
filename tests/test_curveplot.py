#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 17:18:40 2017

@author: joyma
"""
##this is a file to plot zero_curves for 3 types of curve inputs
##taxibility is the next problem to be solved
##joy's first 100 lines of code!

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append('../src/package')

#%%

import pvfn as pv
import DateFunctions_1 as dates
import discfact as df
import Curve_Plotting as plot

#%%
breaks = np.array([0.5,1,2,5,10,])
rates = np.array([0.01,0.02,0.025,0.037,0.042])
curve_points = np.arange(.25,10,0.25)

#%%
#pwcf_rates three curves

pwcf_curve = ['pwcf',0.0, breaks,rates]
pwcf_discount_curve = df.discFact (curve_points, pwcf_curve)

#%%
print (pwcf_discount_curve)
plt.plot(curve_points, pwcf_discount_curve, color='red')
plt.show()

#%%
pwcf_zero_curve = np.log(pwcf_discount_curve)/(-curve_points)
print (pwcf_zero_curve)
plt.plot(curve_points, pwcf_zero_curve, color='red')
plt.show()

#%%
#pwlz_rates three curves
pwlz_curve = ['pwlz',0.0, breaks, rates]
pwlz_discount_curve = df.discFact (curve_points, pwlz_curve)


#%%
print (pwlz_discount_curve)
plt.plot(curve_points, pwlz_discount_curve, color='green')
plt.show()

#%%
pwlz_zero_curve = np.log(pwlz_discount_curve)/(-curve_points)
print (pwlz_zero_curve)
plt.plot(curve_points, pwlz_zero_curve, color='green')


#%%
#pwtf_rates three curves
pwtf_curve = ['pwtf',0.0, breaks,rates]
pwtf_discount_curve = df.discFact (curve_points, pwtf_curve)


#%%
print (pwtf_discount_curve)
plt.plot(curve_points, pwtf_discount_curve, color='blue')
plt.show()

#%%
pwtf_zero_curve = np.log(pwtf_discount_curve)/(-curve_points)
print (pwtf_zero_curve)
plt.plot(curve_points, pwtf_zero_curve, color='blue')
plt.show()

#%%

print ('rates', rates)

print ('pwcf_discount_curve', pwcf_discount_curve)
print ('pwlz_discount_curve', pwlz_discount_curve)
print ('pwtf_discount_curve', pwtf_discount_curve)


print ('pwcf_zero_curve', pwcf_zero_curve)
print ('pwlz_zero_curve', pwlz_zero_curve)
print ('pwtf_zero_curve', pwtf_zero_curve)


plt.plot(curve_points,pwcf_discount_curve, color='red', label='pwcf')
plt.plot(curve_points,pwlz_discount_curve, color='green', label='pwlz')
plt.plot(curve_points,pwtf_discount_curve, color='blue', label='pwtf')
plt.legend()
plt.show()

plt.plot(curve_points, pwcf_zero_curve, color='red', label='pwcf')
plt.plot(curve_points, pwlz_zero_curve, color='green', label='pwlz')
plt.plot(curve_points, pwtf_zero_curve, color='blue', label='pwtf')
plt.legend()
plt.show()

#%% Plot 3 zero curves
breaks_test = np.array([0.5,1,2,5,10,])
rates_test = np.array([0.01,0.02,0.025,0.037,0.042])
curve_points_test = np.arange(.25,10,0.25)


plot.zero_curve_plot_old (breaks_test,rates_test,curve_points_test)


#%% Plot 3 forward curves
breaks_test = np.array([0.5,1,2,5,10,])
rates_test = np.array([0.01,0.02,0.025,0.037,0.042])
curve_points_test = np.arange(.25,10,0.25)

plot.forward_curve_plot_old(breaks_test,rates_test,curve_points_test)


#%% Plot 1 par bond curve pwcf
breaks_test = np.array([0.5,1,2,5,10,])
rates_test = np.array([0.01,0.02,0.025,0.037,0.042])
#curve_points_test = np.arange(.25,10,0.25)
#curve_points_test = np.array([1.,2.])
nmonth = np.arange(3,120,3)
quote_date = dates.YMDtoJulian(20171012)
curve_points_test = dates.CalAdd(quote_date,"add",nmonth=nmonth)
breaks_test = breaks_test*365.25 + quote_date

stored_curve = ['pwcf',quote_date,breaks_test,rates_test]

parms_maturity= []
for i in curve_points_test:
    parms_maturity_single = [0,i,2,100.,"A/A","eomyes",0, False]
    parms_maturity.append(parms_maturity_single)
parms_maturity = pd.DataFrame(parms_maturity)

df_maturity = pv.pvBondFromCurve(stored_curve, settledate = quote_date, parms = parms_maturity)[:,1]

parms_annuity = []
for i in curve_points_test:
    parms_annuity_single = [1.,i,2,0.,"A/A","eomyes",0, False]
    parms_annuity.append(parms_annuity_single)
parms_annuity = pd.DataFrame(parms_annuity)
    
xx = pv.pvBondFromCurve(stored_curve, settledate = quote_date, parms = parms_annuity)
pv_annuity = xx[:,1]

accrued_interest = xx[:,1]-xx[:,0]
c = (100-df_maturity)/(pv_annuity - accrued_interest)


#%% Plot 3 par bond curves

breaks_test = np.array([0.5,1,2,5,10,])
rates_test = np.array([0.01,0.02,0.025,0.037,0.042])
#curve_points_test = np.arange(.25,10,0.25)
#curve_points_test = np.array([1.,2.])
nmonth = np.arange(3,120,3)
quote_date = dates.YMDtoJulian(20171012)
curve_points_test = dates.CalAdd(quote_date,"add",nmonth=nmonth)
breaks_test = breaks_test*365.25 + quote_date


plot.par_bond_curve_plot (breaks=breaks_test, rates=rates_test, curve_points=curve_points_test, quotedate=quote_date)


#%% Plot forward curves with 3 taxability types 

curve_points_yr = np.arange(.25,40,.1)
pwcf_fwd_curve_tax = [['pwcf',
  np.array([12053.]),
  np.array([12418., 12783., 13879., 15705., 23010., 85103.]),
  np.array([-0.0009846 ,  0.03461944,  0.02852191,  0.03915178,  0.02835638,
          0.01335326])],
 ['pwcf',
  np.array([12053.]),
  np.array([12418., 12783., 13879., 15705., 23010., 85103.]),
  np.array([0.00280699, 0.03841103, 0.0323135 , 0.04294337, 0.03214797,
         0.01714485])],
 ['pwcf',
  np.array([12053.]),
  np.array([12418., 12783., 13879., 15705., 23010., 85103.]),
  np.array([-0.00123616,  0.03436787,  0.02827034,  0.03890021,  0.02810481,
          0.01310169])]]


plot.forward_curve_plot_w_taxability(pwcf_fwd_curve_tax, curve_points_test)