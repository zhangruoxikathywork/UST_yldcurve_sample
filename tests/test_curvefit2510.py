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



#%% Build matrixes from parameters for all bonds and callable bonds
sparsenc = pv.bondParmsToMatrix(quotedate,parms)


cfsparse = sparsenc[0]          # This should already be written as a sparse array
cfdata = sparsenc[1]
datedata = sparsenc[2]       # This is vector of data (to feed into discFact())
aivector = sparsenc[3]      
cfindex = sparsenc[4]
cfptr = sparsenc[5]
bondcount = sparsenc[6]
maxcol = sparsenc[7]
maturity = sparsenc[8]    
finalpmt = sparsenc[9]
coupon = sparsenc[10]
frequency = sparsenc[11]
calldate = sparsenc[12]
callflag = sparsenc[13]


xcfnc = cfsparse.toarray()


if (callflag.any()) :                                # Process callables if any
    xcalldate = calldate.copy()
    xcalldate[~callflag] = dates.YMDtoJulian(20991231)   # For non-callable bonds set call date beyond any maturity
    sparseyc = pv.bondParmsToMatrix(xcalldate,parms)

    cfsparse = sparseyc[0]          # This should already be written as a sparse array
    cfdata = sparseyc[1]
    datedata = sparseyc[2]       # This is vector of data (to feed into discFact())
    aivector = sparseyc[3]      
    cfindex = sparseyc[4]
    cfptr = sparseyc[5]
    bondcount = sparseyc[6]
    maxcol = sparseyc[7]
    maturity = sparseyc[8]    
    finalpmt = sparseyc[9]
    coupon = sparseyc[10]
    frequency = sparseyc[11]
    calldate = sparseyc[12]
    callflag = sparseyc[13]

    xcfyc = cfsparse.toarray()
else:
    sparseyc = 0


#%% Make curves 


ratessab = np.array([.01162,.02111,.02703])
ratescc = 2.*np.log(1+ratessab/2.)
ratescc6 = np.copy(ratescc)
ratescc6 = ratescc + 0.01         # Set them away from the true values, so that the optimization will have something to do


breaks = quotedate + np.array([2.,5.,9.])*365.25        # debug with short final break
curvepwcf = ['pwcf',quotedate,breaks,ratescc]
curvepwlz = ['pwlz',quotedate,breaks,ratescc]
curvepwtf = ['pwtf',quotedate,breaks,ratescc]

yvols = np.array([0,0,0.10,0.433,0,0.10,0,0,0])


#%% New PV functions
xxPVBondCurvepwcf_new = pv.pvBondFromCurve(curvepwcf,sparsenc)
xxPVBondCurvepwlz_new = pv.pvBondFromCurve(curvepwlz,sparsenc)
xxPVBondCurvepwtf_new = pv.pvBondFromCurve(curvepwtf,sparsenc)


#cProfile.run('pv.pvBondFromCurve(curvepwcf,xarray)',sort='tottime')


#%% Test ssq:
    
curve_est_pwcf = list(curvepwcf)     # I think this just makes a copy of curve - this is not to convert type
curve5 = list(curvepwcf)
curve_est_pwcf[3] = ratescc + 0.01   # This bumps up the curve so that we have something to optimize


# New


#New with curve and prices the same (ssq should be zero)
ssq5 = pvc.pvCallable_lnYCover(ratescc, prices_call, curve5, yvols, sparsenc, sparseyc)


#%% Try to optimize no taxability
curve_est_pwcf = list(curvepwcf)     # I think this just makes a copy of curve - this is not to convert type
curve5 = list(curvepwcf)
curve_est_pwcf[3] = ratescc + 0.01   # This bumps up the curve so that we have something to optimize
curve_est_pwlz = list(curvepwlz)     # I think this just makes a copy of curve - this is not to convert type
curve_est_pwtf = list(curvepwtf)     # I think this just makes a copy of curve - this is not to convert type


xxfitted_pwcf = so.minimize(pvc.pvCallable_lnYCover, ratescc6, args=(prices_call, curve_est_pwcf, 
                                yvols, sparsenc, sparseyc), method="Nelder-Mead", jac=False)
xxfitted_pwlz = so.minimize(pvc.pvCallable_lnYCover, ratescc6, args=(prices_call, curve_est_pwlz, 
                                yvols, sparsenc, sparseyc), method="Nelder-Mead", jac=False)
xxfitted_pwtf = so.minimize(pvc.pvCallable_lnYCover, ratescc6, args=(prices_call, curve_est_pwtf, 
                                yvols, sparsenc, sparseyc), method="Nelder-Mead", jac=False)

xxPVBondCurvepwtf_new = pv.pvBondFromCurve(curve_est_pwtf,sparsenc)



#%% What follows is curve plotting originally written by Joy Dantong Ma
breaks = quotedate + np.array([2.,5.,10.])*365.25
# Curves defined here and duplicated below (to match original curveplot code)
pwcf_curve = curve_est_pwcf
pwlz_curve = curve_est_pwlz
pwtf_curve = curve_est_pwtf

rates = np.array([0.01,0.02,0.025,0.037,0.042])

curve_points_yr = np.arange(.1,10,0.1)
curve_points = quotedate + curve_points_yr*365.25

#%%
#pwcf_rates three curves

pwcf_curve = curve_est_pwcf
pwcf_discount_curve = df.discFact (curve_points, pwcf_curve)


print (pwcf_discount_curve)
plt.plot(curve_points_yr, pwcf_discount_curve, color='red')
plt.title('PWCF discount curve')
plt.show()



pwcf_zero_curve = np.log(pwcf_discount_curve)/(-curve_points_yr)
print (pwcf_zero_curve)
plt.plot(curve_points_yr, pwcf_zero_curve, color='red')
plt.title('PWCF Zero Curve')
plt.show()

pwcf_fwd_curve = -365*np.log((df.discFact((curve_points+1), pwcf_curve))/(df.discFact(curve_points, pwcf_curve)))
print (pwcf_fwd_curve)
plt.plot(curve_points_yr, pwcf_fwd_curve, color='red')
plt.title('PWCF Forward Rates')
plt.show()


#%%
#pwlz_rates three curves
pwlz_curve = curve_est_pwlz
pwlz_discount_curve = df.discFact (curve_points, pwlz_curve)



print (pwlz_discount_curve)
plt.plot(curve_points_yr, pwlz_discount_curve, color='green')
plt.title('pwlz discount factor')
plt.show()


pwlz_zero_curve = np.log(pwlz_discount_curve)/(-curve_points_yr)
print (pwlz_zero_curve)
plt.title('pwlz zero curve')
plt.plot(curve_points_yr, pwlz_zero_curve, color='green')
plt.show()

pwlz_fwd_curve = -365*np.log((df.discFact((curve_points+1), pwlz_curve))/(df.discFact(curve_points, pwlz_curve)))
print (pwcf_fwd_curve)
plt.plot(curve_points_yr, pwlz_fwd_curve, color='green')
plt.title('PWLZ Forward Rates')
plt.show()


#%%
#pwtf_rates three curves
pwtf_curve = curve_est_pwtf
pwtf_discount_curve = df.discFact (curve_points, pwtf_curve)



print (pwtf_discount_curve)
plt.plot(curve_points_yr, pwtf_discount_curve, color='blue')
plt.title('PWTF Discount factors')
plt.show()


pwtf_zero_curve = np.log(pwtf_discount_curve)/(-curve_points_yr)
print (pwtf_zero_curve)
plt.plot(curve_points_yr, pwtf_zero_curve, color='blue')
plt.title('PWTF Zero Rates')
plt.show()

pwtf_fwd_curve = -365*np.log((df.discFact((curve_points+1), pwtf_curve))/(df.discFact(curve_points, pwtf_curve)))
print (pwtf_fwd_curve)
plt.plot(curve_points_yr, pwtf_fwd_curve, color='blue')
plt.title('PWTF Forward Rates')
plt.show()



#%%

print ('rates', rates)

print ('pwcf_discount_curve', pwcf_discount_curve)
print ('pwlz_discount_curve', pwlz_discount_curve)
print ('pwtf_discount_curve', pwtf_discount_curve)


print ('pwcf_zero_curve', pwcf_zero_curve)
print ('pwlz_zero_curve', pwlz_zero_curve)
print ('pwtf_zero_curve', pwtf_zero_curve)


plt.plot(curve_points_yr,pwcf_discount_curve, color='red', label='pwcf')
plt.plot(curve_points_yr,pwlz_discount_curve, color='green', label='pwlz')
plt.plot(curve_points_yr,pwtf_discount_curve, color='blue', label='pwtf')
plt.title('Discount Factors')
plt.legend()
plt.show()

plt.plot(curve_points_yr, pwcf_zero_curve, color='red', label='pwcf')
plt.plot(curve_points_yr, pwlz_zero_curve, color='green', label='pwlz')
plt.plot(curve_points_yr, pwtf_zero_curve, color='blue', label='pwtf')
plt.title('Zero Rates')
plt.legend()
plt.show()

#%% creating a function to produce all three curves at once

def zero_curve_plot (curves,curve_points_yr):
## this function takes in input curves (breaks, rates), and plot zero curves at a quarterly 
## frequency (curve_points' step is 0.25). it plots 3 types of curve all at once

    colors = ['red', 'green', 'blue']
    ## pwcf = red, pwlz = green, pwtf = blue. this will be consistent for all curve plotting
    xquotedate = curves[0][1]
    curve_points = xquotedate + curve_points_yr*365.25
    for i in range(len(curves)):
#        stored_curve = [ctype,0.0,breaks,rates]
        xcurve = curves[i]
        discount_curve = df.discFact(curve_points,xcurve)
        zero_curve = np.log(discount_curve)/(-curve_points_yr)
        plt.plot(curve_points_yr, zero_curve, color=colors[i], label=xcurve[0])
    plt.legend()
    plt.title('Zero Curves')
    plt.show()

curve_list = [curve_est_pwcf,curve_est_pwlz,curve_est_pwtf]

zero_curve_plot (curve_list,curve_points_yr)



#%% Plot 3 forward curves

breaks_test = np.array([0.5,1,2,5,10,])
rates_test = np.array([0.01,0.02,0.025,0.037,0.042])
curve_points_test = np.arange(.1,10,0.1)


def forward_curve_plot (curves,curve_points_yr):
## this function takes in input curves (breaks, rates), and plot forward curves at a quarterly 
## frequency (curve_points' step is 0.25). it plots 3 types of curve all at once

    colors = ['red','green','blue']
    ## pwcf = red, pwlz = green, pwtf = blue. this will be consistent for all curve plotting
    xquotedate = curves[0][1]
    curve_points = xquotedate + curve_points_yr*365.25

    for i in range(len(curves)):
        xcurve = curves[i]
        forward_curve = -365*np.log((df.discFact((curve_points+1), xcurve))/(df.discFact(curve_points, xcurve)))
        plt.plot(curve_points_yr, forward_curve, color = colors[i],label=xcurve[0])
    plt.legend()
    plt.title("Forward Curves")
    plt.show()

forward_curve_plot (curve_list,curve_points_yr)



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


def par_bond_curve_plot (breaks, rates, curve_points, settledate):
    
    curve_types = ['pwcf','pwlz','pwtf']
    colors = ['red','green','blue']
    ## pwcf = red, pwlz = green, pwtf = blue. this will be consistent for all curve plotting

    for ctype,color in zip(curve_types, colors):
        stored_curve = [ctype,settledate,breaks,rates]
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
        coupon_rates = (100-df_maturity)/(pv_annuity - accrued_interest)
        plt.plot(curve_points, coupon_rates, color = color,label=ctype)
    plt.legend()
    plt.title('Par Bond Curves')
    plt.show()

par_bond_curve_plot (breaks = breaks_test, rates = rates_test, curve_points = curve_points_test,settledate = quote_date)
