# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 21:04:28 2016

@author: tcoleman"""

# magic %reset resets by erasing variables, etc. 

import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.optimize as so
import importlib as imp
import cProfile
import time as time
import timeit as timeit

sys.path.append('../src/package')

#%%

import discfact as df
import pvfn as pv
import DateFunctions_1 as dates
import pvcover as pvc


imp.reload(dates)
imp.reload(pv)
#reload(df)
imp.reload(pvc)


#%% Create parameter vectors

#    eight bonds, 
# 0      2.25% of 15-nov-2025 non-call       20 half-yrs + 1 maturity = 21          
# 1      2.5% of 15-feb-2026 non-call        20 half-yrs + 1 maturity = 21
# 2      2.25% of 15-nov-2025 callable 15-nov-2020  20 half-year + 1 maturity = 21
# 3      2.25% of 15-nov-2025 callable 15-nov-2020  20 half-year + 1 maturity = 21 (duplicate)
# 4      5% of 15-nov-2025 non-call          20 half-yrs + 1 maturity = 21
# 5      5% of 15-nov-2025 callable 15-nov-2020  20 half-yrs + 1 maturity = 21
# 6      4% of 15-nov-2022                   14 half-yrs + 1 maturity = 15
# 7      3% of 15-feb-2020                   8 half-yrs + 1 maturity = 9
# 8      0% of 15-may-2016 (bill)            no coupons + 1 maturity = 1

# First version of parms does not have tax status. 2nd version has tax status. Make 4, 5, 6 partially tax-exempt, 7 fully exempt
parms = [[2.25,45975,2,100.,"A/A","eomyes",0,False],[2.5,46067,2,100.,"A/A","eomyes",0,False],
         [2.25,45975,2,100.,"A/A","eomyes",44149,True],[2.25,45975,2,100.,"A/A","eomyes",44149,True],
	[5.,45975,2,100.,"A/A","eomyes",0,False],[5.,45975,2,100.,"A/A","eomyes",44149,True],
    [5.,44879,2,100.,"A/A","eomyes",0,False],[5.,43875,2,100.,"A/A","eomyes",0,False],
    [0.,42504,0,100.,"A/A","eomyes",0,False]]
tparms = [[2.25,45975,2,100.,"A/A","eomyes",0,False,1],[2.5,46067,2,100.,"A/A","eomyes",0,False,1],
         [2.25,45975,2,100.,"A/A","eomyes",44149,True,1],[2.25,45975,2,100.,"A/A","eomyes",44149,True,1],
	[5.,45975,2,100.,"A/A","eomyes",0,False,2],[5.,45975,2,100.,"A/A","eomyes",44149,True,2],
    [5.,44879,2,100.,"A/A","eomyes",0,False,2],[5.,43875,2,100.,"A/A","eomyes",0,False,3],
    [0.,42504,0,100.,"A/A","eomyes",0,False,1]]

parms = pd.DataFrame(parms)
tparms = pd.DataFrame(tparms)


# settle 19-feb-2016
quotedate = 42418

# Set the prices - this is for flat curve of 5%sab (4.9385%cc)
prices_call = np.array([[78.99438458023005, 79.58779117363665],
 [80.52620711688378, 80.55367964435631],
 [78.99431003817156, 79.58771663157816],
 [77.86150987717079, 78.45491647057739],
 [99.98503514032244, 101.30371645900375],
 [98.48395850630824, 99.80263982498955],
 [99.98640702787986, 101.30508834656118],
 [99.99989442134074, 100.05483947628579],
 [98.84393375126179, 98.84393375126179]])



#%% Build matrixes from parameters for all bonds and callable bonds
sparsenc = pv.bondParmsToMatrix(quotedate,parms)

#timeit.timeit('10*5',number=1000)
#timeit.timeit('pv.bondParmsToMatrix(quotedate,parms)',number=10)

#%timeit pv.bondParmsToMatrix(quotedate,parms)

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



#%% Make curves 


ratessab = np.array([0.05,0.05,0.05,0.05])
ratescc = 2.*np.log(1+ratessab/2.)
ratescc6 = np.copy(ratescc)
ratescc6 = ratescc + 0.01         # Set them away from the 5% flat, so that the optimization will have something to do


breaks = quotedate + np.array([2.,5.,10.,20.])*365.
curvepwcf = ['pwcf',quotedate,breaks,ratescc]
tcurvepwcf = ['pwcf',quotedate,breaks,ratescc,0.001,0.002]   # Version with spreads for taxability (type 2 and type 3)
curvepwlz = ['pwlz',quotedate,breaks,ratescc]
curvepwtf = ['pwtf',quotedate,breaks,ratescc]

yvols = np.array([0,0,0.10,0.433,0,0.10,0,0,0])


#%% New PV functions
xxPVBondCurvepwcf_new = pv.pvBondFromCurve(curvepwcf,sparsenc)

print("PV straight bonds, sparse matrices vs parms")
%timeit pv.pvBondFromCurve(curvepwcf,sparsenc)
%timeit pv.pvBondFromCurve(curvepwcf,settledate = quotedate, parms=parms)

#cProfile.run('pv.pvBondFromCurve(curvepwcf,xarray)',sort='tottime')

# Test values (for flat 5% curve):
testnoncall = np.array([[  78.99438458,   79.58779117],
       [  80.52620712,   80.55367964],
       [  78.99438458,   79.58779117],
       [  78.99438458,   79.58779117],
       [  99.98503514,  101.30371646],
       [  99.98503514,  101.30371646],
       [  99.98640703,  101.30508835],
       [  99.99989442,  100.05483948],
       [  98.84393375,   98.84393375]])

xtest1 = np.abs(xxPVBondCurvepwcf_new - testnoncall) > .0001 
xtest1.any()                               # This should be false

#%% Test callable - new

xxPVBondCurvepwcfCall_new = pv.pvCallable_lnY(curvepwcf, yvols, sparsenc = sparsenc, sparseyc = sparseyc)

print("PV straight bond, sparse matrices, some callable sparse, some callable parms")
%timeit pv.pvBondFromCurve(curvepwcf,sparsenc)
%timeit pv.pvCallable_lnY(curvepwcf, yvols, sparsenc = sparsenc, sparseyc = sparseyc)
%timeit pv.pvCallable_lnY(curvepwcf,yvols,settledate = quotedate, parms=parms)



# For pwcf 5% sab with 365.25-day year (rates quoted at cc, 365.25-day year)
# (PV from spread-sheet, AI from HP17B, price as difference)
#            2.25% of 15-nov-25   2.5% 15-feb-26    2.25% 15-nov-25   2.25% 15-nov-25  5% 15-nov-25  5% 15-nov-25
#                                                  call 15-nov-20    call 15-nov-20                 call 15-nov-20
#  YES call       0                    1                 2                3                  4           5            6    7
#  yvol                                               10%                43.3%                           10%
# Price      78.994385             80.526207       78.994310          77.8617          99.9850       98.484061
# PV         79.587791             80.55368        79.587717          78.4510         101.303716     99.80274 
# AI         0.593407              0.027473 Call   0.593407           0.593407            1.318681    1.318681 
#    call value from BLACKSWPN.xls                 0.00007            1.1327                          1.50113
#    call value from python                        0.00007            1.1327                          1.50097
# Converting yvol to pvol for bond 3. DV01=4.065, y=5%, fp=87.97, Pvol=DV01*Yvol*Y/P ~ 10%
#    

# 18-jul-17, need to check pwcf callable price with spread. With 10bp cc spread, 
#                5% 15-nov-25
#               call 15-nov-20
# Price           98.870027
# PV              99.18895 (almost matches above, opt value = 1.501)
#  This seems to pass the preliminary check of just adding spread of 10bp to the whole curve

xtest2 = np.abs(prices_call - xxPVBondCurvepwcfCall_new) > 0.0001
xtest2.any()              # Should be False

#%% Test ssq:
    
curve_est = list(curvepwcf)     # I think this just makes a copy of curve - this is not to convert type
curve5 = list(curvepwcf)
curve_est[3] = ratescc + 0.01   # This bumps up the curve so that we have something to optimize


# New

print("ssq (pvCallable_lnYCover) sparse matrices")
%timeit pvc.pvCallable_lnYCover(ratescc6, prices_call, curve_est, yvols, sparsenc, sparseyc)

#New with curve and prices the same (ssq should be zero)
ssq5 = pvc.pvCallable_lnYCover(ratescc, prices_call, curve5, yvols, sparsenc, sparseyc)

# Timings: 
#  pvCallabel_lnYCover_old    0.126 sec
#  pvCallable_lnYCover        0.0044 sec
# speed-up about 29x

#%% Try to optimize no taxability




print("optimize, straight bonds")
%timeit so.minimize(pvc.pvCallable_lnYCover, ratescc6, args=(prices_call, curve_est, yvols, sparsenc, sparseyc), method="Nelder-Mead", jac=False)

# Result array should be 0.49385 or close (5%sab)
#   Function result (ssq) 2.37e-06
# Old version 21.0 sec, another time 13.6sec
# New (matrix) version 0.487 sec, another time 0.52sec
# Speed-up of about 43x (26x)



#%% Test taxability. First, create CF matrixes for type 1, type 2, type 3


calldate = np.array(tparms.iloc[:,6])                 # Assume that dates are already Julian
callflag = np.array(tparms.iloc[:,7])
if (callflag.any()) :                                # Process callables if any
    xcalldate = calldate.copy()
    xcalldate[~callflag] = dates.YMDtoJulian(20991231)   # For non-callable bonds set call date beyond any maturity

taxtype = tparms.iloc[:,8]
x1 = np.array(taxtype == 1)
sparsenc1 = pv.bondParmsToMatrix(quotedate,tparms[x1])
sparseyc1 = pv.bondParmsToMatrix(xcalldate[x1],tparms[x1])
x2 = np.array(taxtype == 2)
sparsenc2 = pv.bondParmsToMatrix(quotedate,tparms[x2])
sparseyc2 = pv.bondParmsToMatrix(xcalldate[x2],tparms[x2])
x3 = np.array(taxtype == 3)
sparsenc3 = pv.bondParmsToMatrix(quotedate,tparms[x3])
sparseyc3 = pv.bondParmsToMatrix(xcalldate[x3],tparms[x3])

#%% Test pvCallable_lnYCover_tax

tcurve = curvepwcf.copy()
ratestax = tcurve[3].tolist()
ratestax.extend([.0,.0])
ratestax = np.array(ratestax)

# New non-tax

print("ssq (pvCallable_lnYCover) with no tax and taxability")
%timeit pvc.pvCallable_lnYCover(ratescc, prices_call, curve_est, yvols, sparsenc, sparseyc)
%timeit pvc.pvCallable_lnYCover_tax(ratestax, tcurve, prices1=prices_call[x1],vols1=yvols[x1], sparsenc1 = sparsenc1, sparseyc1 = sparseyc1,prices2=prices_call[x2],vols2=yvols[x2],sparsenc2 = sparsenc2,sparseyc2 = sparseyc2,prices3=prices_call[x3],vols3=yvols[x3],sparsenc3 = sparsenc3,sparseyc3 = sparseyc3) 

# 0.00548 seconds (lnYCover non-tax)
# 0.00999 seconds (ssq tax)

#%% Test optimization with and without taxability


print("optimizie no tax and taxability")
%timeit so.minimize(pvc.pvCallable_lnYCover, ratescc, args=(prices_call, curve_est, yvols, sparsenc, sparseyc), method="Nelder-Mead", jac=False)
%timeit so.minimize(pvc.pvCallable_lnYCover_tax, ratestax, args=(tcurve, prices_call[x1],yvols[x1], sparsenc1, sparseyc1,prices_call[x2],yvols[x2],sparsenc2,sparseyc2,prices_call[x3],yvols[x3],sparsenc3,sparseyc3), method="Nelder-Mead", jac=False)
#    1.10836 seconds (optimize new)
#    2.03824 seconds (optimize new, tax)

#%% Garbage code - development. Delete





