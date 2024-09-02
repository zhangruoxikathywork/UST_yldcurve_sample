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

sys.path.append('../src/package')

#%%

import discfact as df
import pvfn as pv
import DateFunctions_1 as dates

imp.reload(dates)
imp.reload(pv)
#reload(df)

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

# First version of parms does not have tax status. 2nd version has tax status. Make 4th & 5th bond partially tax-exempt
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
#tparms = [[2.25,45975,2,100.,"A/A","eomyes",0,False,1],[2.5,46067,2,100.,"A/A","eomyes",0,False,1],
#          [2.25,45975,2,100.,"A/A","eomyes",44149,1],[2.25,45975,2,100.,"A/A","eomyes",44149,1],
#	[5.,45975,2,100.,"A/A","eomyes",0,False,2],[5.,45975,2,100.,"A/A","eomyes",44149,2],
#    [0.,42504,0,100.,"A/A","eomyes",0,False,1]]
parms = pd.DataFrame(parms)
tparms = pd.DataFrame(tparms)
# settle 19-feb-2016
quotedate = 42418

#%% Test building matrixes from parameters for just standard bonds

t0 = time.perf_counter()
xarray = pv.bondParmsToMatrix(quotedate,parms)
t1 = time.perf_counter()
print("%.5f seconds (parms)" % (t1-t0))


cfsparse = xarray[0]
cfdata = xarray[1]
datedata = xarray[2]
aivector = xarray[3]
cfindex = xarray[4]
cfptr = xarray[5]
bondcount = xarray[6]
maxcol = xarray[7]
maturity = xarray[8]
finalpmt = xarray[9]
coupon = xarray[10]
frequency = xarray[11]
calldate = xarray[12]
callflag = xarray[13]


xcf = sp.csr_matrix((cfdata, cfindex, cfptr), shape=(bondcount, maxcol)).toarray()

#%% Make curves 


ratessab = np.array([0.05,0.05,0.05,0.05])
ratescc = 2.*np.log(1+ratessab/2.)
#ratescc = ratescc + 0.001

breaks = quotedate + np.array([2.,5.,10.,20.])*365.
curvepwcf = ['pwcf',quotedate,breaks,ratescc]
tcurvepwcf = ['pwcf',quotedate,breaks,ratescc,0.001,0.002]   # Version with spreads for taxability (type 2 and type 3)
curvepwlz = ['pwlz',quotedate,breaks,ratescc]
curvepwtf = ['pwtf',quotedate,breaks,ratescc]

#%% New PV functions
t0 = time.perf_counter()
xxPVBondCurvepwcf_new = pv.pvBondFromCurve(curvepwcf,xarray)
t1 = time.perf_counter()
print("%.5f seconds (cf)" % (t1-t0))

t0 = time.perf_counter()
xxPVBondCurvepwcf_parms = pv.pvBondFromCurve(curvepwcf,settledate = quotedate, parms=parms)
t1 = time.perf_counter()
print("%.5f seconds (new parms)" % (t1-t0))

#cProfile.run('pv.pvBondFromCurve(curvepwcf,xarray)',sort='tottime')

#%% Test price-from-yield and yield-from-price

datesparse = sp.csr_matrix((datedata, cfindex, cfptr), shape=(bondcount, maxcol)) 
 
#cfvec = sp.find(cfsparse[0,:])[2]  # CF vector as numpy array
#datevec = sp.find(datesparse[0,:])[2]  # Date vector
#xpricei = pv.bondPriceFromYield_matrix(0.05, cfvec, datevec, quotedate,0.0)  # Default dirtyprice=0 used
#dirtyprice = xpricei
#xyieldi = pv.bondYieldFromPrice_matrix(dirtyprice, cfvec, datevec, quotedate)
#xyieldi = so.brentq(pv.bondPriceFromYield_matrix, -0.03, 0.5, args=(cfvec, datevec, quotedate, dirtyprice), xtol=1e-12, rtol=9.0e-16, maxiter=100, full_output=True, disp=True)
 
#xpricei = pv.BondPriceFromYield(0.05,0.0, quotedate, parms.iloc[0,0], parms.iloc[0,1], parms.iloc[0,2], parms.iloc[0,3], parms.iloc[0,4], parms.iloc[0,5])  # Default dirtyprice=0 used
#price = xpricei
#xyieldi = pv.BondYieldFromPrice(price, quotedate, parms.iloc[0,0], parms.iloc[0,1], parms.iloc[0,2], parms.iloc[0,3], parms.iloc[0,4], parms.iloc[0,5])
 
#    xprice = []
#    xyield = []
#    t0 = time.perf_counter()
#    for i in range(0,bondcount) : 
#        cfvec = sp.find(cfsparse[i,:])[2]  # CF vector as numpy array
#    datevec = sp.find(datesparse[i,:])[2]  # Date vector
#    xpricei = pv.bondPriceFromYield_matrix(0.05, cfvec, datevec, quotedate)  # Default dirtyprice=0 used
#    xprice.append(xpricei)
#    xyieldi = pv.bondYieldFromPrice_matrix(xpricei, cfvec, datevec, quotedate)
#    xyield.append(xyieldi)
#t1 = time.perf_counter()
#print("%.5f seconds (matrix)" % (t1-t0))

# Test "parms" version - those take in parms and then convert to cf vector 
# First test passing through the cf vector
xpriceparms1 = []
xyieldparms1 = []
t0 = time.perf_counter()
for i in range(0,bondcount) : 
    cfvec = sp.find(cfsparse[i,:])[2]  # CF vector as numpy array
    datevec = sp.find(datesparse[i,:])[2]  # Date vector
    xpricei = pv.bondPriceFromYield_parms(0.05, cfvec=cfvec, datevec=datevec, settledate=quotedate)  # Default dirtyprice=0 used
    xpriceparms1.append(xpricei)
    xyieldi = pv.bondYieldFromPrice_parms(xpricei, cfvec=cfvec, datevec=datevec, settledate=quotedate)
    xyieldparms1.append(xyieldi)
t1 = time.perf_counter()
print("%.5f seconds (parms, pass cfvec)" % (t1-t0))
 
# Second test passing the parms vector
xpriceparms2 = []
xyieldparms2 = []
t0 = time.perf_counter()
for i in range(0,bondcount) : 
    cfvec = sp.find(cfsparse[i,:])[2]  # CF vector as numpy array
    datevec = sp.find(datesparse[i,:])[2]  # Date vector
    xpricei = pv.bondPriceFromYield_parms(0.05, settledate=quotedate, parms=parms.iloc[[i]])  # Default dirtyprice=0 used
    xpriceparms2.append(xpricei)
    xyieldi = pv.bondYieldFromPrice_parms(xpricei, settledate=quotedate, parms=parms.iloc[[i]])
    xyieldparms2.append(xyieldi)
t1 = time.perf_counter()
print("%.5f seconds (parms, pass parms)" % (t1-t0))
 
 
# Test different root-finding methods
# The first (using so.brentq) is used in pv.bondYieldFromPrice_matrix. The so.newton is alternative

#xyieldBrent = []
#t0 = time.perf_counter()
#for i in range(0,bondcount) : 
#    cfvec = sp.find(cfsparse[i,:])[2]  # CF vector as numpy array
#    datevec = sp.find(datesparse[i,:])[2]  # Date vector
#    xpricei = pv.bondPriceFromYield_parms(0.05, cfvec=cfvec,datevec=datevec,settledate=quotedate)  # Default dirtyprice=0 used
#    xyieldi = so.brentq(pv.bondPriceFromYield_parms, -0.03, 0.5, args=(cfvec, datevec, quotedate , xpricei ),full_output=True, disp=True)
#    xyieldBrent.append(xyieldi)
#t1 = time.perf_counter()
#print("%.5f seconds (Brentq)" % (t1-t0))

#xyieldNewton = []
#xyield0 = np.maximum(coupon,1.) / 100.
#t0 = time.perf_counter()
#for i in range(0,bondcount) : 
#    cfvec = sp.find(cfsparse[i,:])[2]  # CF vector as numpy array
#    datevec = sp.find(datesparse[i,:])[2]  # Date vector
#    xpricei = pv.bondPriceFromYield_matrix(0.05, cfvec, datevec, quotedate)  # Default dirtyprice=0 used
#    xyieldi = so.newton(pv.bondPriceFromYield_matrix, xyield0[i], args=(cfvec, datevec, quotedate , xpricei ))
#    xyieldNewton.append(xyieldi)
#t1 = time.perf_counter()
#print("%.5f seconds (Newton)" % (t1-t0))

# There appears to be no appreciable speed advantage of Newton over Brentq. The yield
# root-finding is well-behaved so Newton should work reliably but it seems to be only
# about 20% faster - probably not enough to move from the more reliable BrentQ. 

# The version taking in parms runs about 6x slower:
#  take in parms about 6x slower
#  take in cf no substantial penalty
# Thus can get rid of _matrix versions and use _parms but feed in CF vector


#%% Test option-adjusted YTM

vols = np.array([0,0,0.10,0.433,0,0.10,0,0,0])
coupon = np.array(parms.iloc[:,0])                   # vector of coupons
if (callflag.any()) :                                # Process callables if any
    xcalldate = calldate.copy()
    xcalldate[~callflag] = dates.YMDtoJulian(20991231)   # For non-callable bonds set call date beyond any maturity
    sparseyc = pv.bondParmsToMatrix(xcalldate,parms)

cfsparseyc = sparseyc[0]          # This should already be written as a sparse array
cfdatayc = sparseyc[1]
datedatayc = sparseyc[2]       # This is vector of data (to feed into discFact())
aivectoryc = sparseyc[3]
cfindexyc = sparseyc[4]
cfptryc = sparseyc[5]
bondcountyc = sparseyc[6]
maxcolyc = sparseyc[7]
maturityyc = sparseyc[8]              # If True we are processing for partially or wholly tax exempt (spreads)
finalpmtyc = sparseyc[9]
couponyc = sparseyc[10]
frequencyyc = sparseyc[11]
calldateyc = sparseyc[12]
callflagyc = sparseyc[13]

datesparseyc = sp.csr_matrix((datedatayc, cfindexyc, cfptryc), shape=(bondcount, maxcolyc)) 

xcfsparse = cfsparse.toarray()


xpricecall1 = []
xyieldcall1 = []
t0 = time.perf_counter()
for i in range(0,bondcount) : 
    cfvecnc = sp.find(cfsparse[i,:])[2]  # CF vector as numpy array
    datevecnc = sp.find(datesparse[i,:])[2]  # Date vector
    cfvecyc = sp.find(cfsparseyc[i,:])[2]  # CF vector as numpy array
    datevecyc = sp.find(datesparseyc[i,:])[2]  # Date vector
    xpricei = pv.bondPriceFromYield_callable(0.05, cfvecnc=cfvecnc, datevecnc=datevecnc, settledate = quotedate, cfvecyc=cfvecyc, datevecyc = datevecyc, calldate= calldate[i],  freq=frequencyyc[i], vol = vols[i], callflag = callflagyc[i], parms=0, parmflag=False)  # Default dirtyprice=0 used
    xpricecall1.append(xpricei)
    xyieldi = pv.bondYieldFromPrice_callable(xpricei, cfvecnc=cfvecnc, datevecnc=datevecnc, settledate = quotedate, cfvecyc=cfvecyc, datevecyc = datevecyc, calldate= calldate[i],  freq=frequencyyc[i], vol = vols[i], callflag = callflagyc[i], parms=0, parmflag=False)  # Default dirtyprice=0 used
#    xyieldi = pv.bondYieldFromPrice_callable(xpricei, settledate=quotedate, vol = vols[i], parms=parms.iloc[[i]])
    xyieldcall1.append(xyieldi)
t1 = time.perf_counter()
print("%.5f seconds (callable yield, pass cfvecs)" % (t1-t0))

# Now calculate price from yield with parms
xpricecall2 = []
xyieldcall2 = []
t0 = time.perf_counter()
for i in range(0,bondcount) : 
    xpricei = pv.bondPriceFromYield_callable(0.05, settledate=quotedate, vol = vols[i], parms=parms.iloc[[i]])  # Default dirtyprice=0 used
    xpricecall2.append(xpricei)
    xyieldi = pv.bondYieldFromPrice_callable(xpricei, settledate=quotedate, vol = vols[i], parms=parms.iloc[[i]])
    xyieldcall2.append(xyieldi)
t1 = time.perf_counter()
print("%.5f seconds (callable yield, pass parms)" % (t1-t0))

# Now calculate price from yield with parms, but price is calculated from yield-to-maturity (not using callable version)
xpricecall3 = []
xyieldcall3 = []
t0 = time.perf_counter()
for i in range(0,bondcount) : 
    xpricei = pv.bondPriceFromYield_parms(0.05, settledate=quotedate, parms=parms.iloc[[i]])  # Default dirtyprice=0 used
    xpricecall3.append(xpricei)
    xyieldi = pv.bondYieldFromPrice_callable(xpricei, settledate=quotedate, vol = vols[i], parms=parms.iloc[[i]])
    xyieldcall3.append(xyieldi)
t1 = time.perf_counter()
print("%.5f seconds (callable yield from non-call, pass parms)" % (t1-t0))



#%% Test Option-adjusted-YTM versus yield-to-worst

# Yield-to-worst:
# - Maturity date for P<100 
# - Call date for P>100 

# Above, with input ytm=5%, all bonds were below par. Output xpricecall2

xdprice5 = np.array(xpricecall2)
xprice5 = np.copy(xdprice5)
xprice5 = xprice5 - aivector


# Now calculate option-adjusted yield, yield to mat, yield to call for 5%
xpricecall5 = []
xyieldcall5 = []
xyieldcall5m = []
xyieldcall5c = []
#t0 = time.perf_counter()
for i in range(0,bondcount) : 
    xparms = parms.iloc[[i]].copy()
    xpricei = pv.bondPriceFromYield_callable(0.05, settledate=quotedate, vol = vols[i], parms=xparms)  # Default dirtyprice=0 used
    xpricecall5.append(xpricei)
    xyieldi = pv.bondYieldFromPrice_callable(xpricei, settledate=quotedate, vol = vols[i], parms=xparms)
    xyieldcall5.append(xyieldi)
    xyieldi = pv.bondYieldFromPrice_parms(xpricei, settledate=quotedate,  parms=xparms)
    xyieldcall5m.append(xyieldi)
    if (bool(np.array(xparms[7]))):
        xparms[1] = xparms[6]
    xyieldi = pv.bondYieldFromPrice_parms(xpricei, settledate=quotedate,  parms=xparms)
    xyieldcall5c.append(xyieldi)
t1 = time.perf_counter()
#print("%.5f seconds (callable yield, pass parms)" % (t1-t0))

xpricecall5 = np.array(xpricecall5)
xyieldcall5m = np.array(xyieldcall5m)
xyieldcall5c = np.array(xyieldcall5c)
xpricecall5 = xpricecall5 - aivector
x1 = np.logical_and(callflag,(xpricecall5 > 100))
xyieldcall5m[x1] = xyieldcall5c[x1]
xyieldcall5 = np.array([xyieldcall5,xyieldcall5m,xpricecall5,callflag]).T

# Now calculate option-adjusted yield, yield to mat, yield to call for 5%
xpricecall2 = []
xyieldcall2 = []
xyieldcall2m = []
xyieldcall2c = []
#t0 = time.perf_counter()
for i in range(0,bondcount) : 
    xparms = parms.iloc[[i]].copy()
    xpricei = pv.bondPriceFromYield_callable(0.02, settledate=quotedate, vol = vols[i], parms=xparms)  # Default dirtyprice=0 used
    xpricecall2.append(xpricei)
    xyieldi = pv.bondYieldFromPrice_callable(xpricei, settledate=quotedate, vol = vols[i], parms=xparms)
    xyieldcall2.append(xyieldi)
    xyieldi = pv.bondYieldFromPrice_parms(xpricei, settledate=quotedate,  parms=xparms)
    xyieldcall2m.append(xyieldi)
    if (bool(np.array(xparms[7]))):
        xparms[1] = xparms[6]
    xyieldi = pv.bondYieldFromPrice_parms(xpricei, settledate=quotedate,  parms=xparms)
    xyieldcall2c.append(xyieldi)
t1 = time.perf_counter()
#print("%.2f seconds (callable yield, pass parms)" % (t1-t0))

xpricecall2 = np.array(xpricecall2)
xyieldcall2m = np.array(xyieldcall2m)
xyieldcall2c = np.array(xyieldcall2c)
xpricecall2 = xpricecall2 - aivector
x1 = np.logical_and(callflag,(xpricecall2 > 100))
xyieldcall2m[x1] = xyieldcall2c[x1]
xyieldcall2 = np.array([xyieldcall2,xyieldcall2m,xpricecall2,callflag]).T


#%% Test callable - new

# The new 



t0 = time.perf_counter()
xxPVBondCurvepwcfCall_new = pv.pvCallable_lnY(curvepwcf, vols, sparsenc = xarray, sparseyc = sparseyc)
t1 = time.perf_counter()
print("%.5f seconds (new callable)" % (t1-t0))

# Timing (7-oct-17) - alternative calls for calculating yield within bondYieldFromPrice:
#  - About .00484 sec When use bondPriceFromYield_matrix 
#  - About .00787 sec when use bondPriceFromYield_parms

# For pwcf 5% sab with 365.25-day year (rates quoted at cc, 365.25-day year)
# Quotedate (settledate) 19-feb-2016
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



#%% Profile

#cProfile.run('pv.pvCallable_lnY(settledate,parms,curvepwcf,vols)',sort='tottime',file='xx.out')
#cProfile.run('pv.pvCallable_lnY(settledate,parms,curvepwcf,vols)','profile_pvCallable.out')
#
#import pstats
#p = pstats.Stats('profile_pvCallable.out')
#p.strip_dirs()
#p.sort_stats('cumulative').print_stats(30)
#p.sort_stats('time').print_stats(30)

