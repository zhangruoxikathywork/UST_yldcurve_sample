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

#%% ------ Sample data for curves ------
# These are various test maturity dates
mat = np.array([0.25,0.5,0.75,0.9,0.99,1.0,1.2])
# These are the sab forward rates, to be used for 0.5, 1, 2, 5 yrs (see below)
ratessab = np.array([0.01,0.02005,0.02512,0.03724])
ratescc = 2.*np.log(1+ratessab/2.)
# Define curve types 
#   pwcf - piece-wise constant forward (forward rates constant)
#   pwlz - piece-wise linear zero (zero retes linearly interpolated)
#   pwtf - piece-wise twisted forward (forward rate slope depends on level on either side)
curvepwcf = ['pwcf',0.0,np.array([0.5,1.,2.,5.]),ratescc]
curvepwlz = ['pwlz',0.0,np.array([0.5,1.,2.,5.]),ratescc]
curvepwtf = ['pwtf',0.0,np.array([0.5,1.,2.,5.]),ratescc]

xxpwcf = df.dfpwcf(mat,0.,curvepwcf[2],curvepwcf[3])
xxpwlz = df.dfpwlz(mat,0.,curvepwlz[2],curvepwlz[3])     # From some simple hand checks, seems to work
xxpwtf = df.dfpwtf(mat,0.,curvepwtf[2],curvepwtf[3])
xx2pwcf = df.discFact(mat,curvepwcf)
xx2pwlz = df.discFact(mat,curvepwlz)
xx2pwtf = df.discFact(mat,curvepwtf)
#
#


#%% Test building matrixes from parameters

#    seven bonds, 
#       2.25% of 15-nov-2025 non-call       20 half-yrs + 1 maturity = 21          
#       2.5% of 15-feb-2026 non-call        20 half-yrs + 1 maturity = 21
#       2.25% of 15-nov-2025 callable 15-nov-2020  20 half-year + 1 maturity = 21
#       5% of 15-nov-2025 non-call          20 half-yrs + 1 maturity = 21
#       5% of 15-nov-2025 callable 15-nov-2020  20 half-yrs + 1 maturity = 21
#       4% of 15-nov-2022                   14 half-yrs + 1 maturity = 15
#       3% of 15-feb-2020                   8 half-yrs + 1 maturity = 9


# First version of parms does not have tax status. 2nd version has tax status. Make 4th & 5th bond partially tax-exempt
parms = [[2.25,45975,2,100.,"A/A","eomyes",0],[2.5,46067,2,100.,"A/A","eomyes",0,False],[2.25,45975,2,100.,"A/A","eomyes",44149],
	[5.,45975,2,100.,"A/A","eomyes",0],[5.,45975,2,100.,"A/A","eomyes",44149],
    [5.,44879,2,100.,"A/A","eomyes",0],[5.,43875,2,100.,"A/A","eomyes",0]]
tparms = [[2.25,45975,2,100.,"A/A","eomyes",0,1],[2.5,46067,2,100.,"A/A","eomyes",0,1],[2.25,45975,2,100.,"A/A","eomyes",44149,1],
	[5.,45975,2,100.,"A/A","eomyes",0,2],[5.,45975,2,100.,"A/A","eomyes",44149,2]]
parms = pd.DataFrame(parms)
tparms = pd.DataFrame(tparms)
# settle 19-feb-2016
settledate = 42418

#%% Test with forward settle date

# settle 19-feb-2020 - this shold force the last 2 bonds to have no entries
xsettledate = dates.YMDtoJulian(20230219)

xarray = pv.bondParmsToMatrix(xsettledate,parms)


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

# Make dense (filled out) arrays from sparse

xcf = sp.csr_matrix((cfdata, cfindex, cfptr), shape=(bondcount, maxcol)).toarray()


#%% Test with taxability

xarray = pv.bondParmsToMatrix(settledate,tparms)

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

# Make dense (filled out) arrays from sparse

xcf = sp.csr_matrix((cfdata, cfindex, cfptr), shape=(bondcount, maxcol)).toarray()

#%% Test converting bond parameters to (sparse) arrays

t0 = time.perf_counter()
xarray = pv.bondParmsToMatrix(settledate,parms)
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

xcf = sp.csr_matrix((cfdata, cfindex, cfptr), shape=(bondcount, maxcol)).toarray()

#%% Test converting for callable - with forward date

# Set the calldate for non-calls beyond the maturity date (20991231 works even for consol)
# and the function bondParmsToMatrix will skip that bond - setting the whole row to zero in sparse matrix

calldate = np.array(parms.iloc[:,6])
callflag = calldate > 0
xcalldate = np.copy(calldate)
xcalldate[~callflag] = dates.YMDtoJulian(20991231)

xarray_call = pv.bondParmsToMatrix(xcalldate,parms)

#%% Make curves and test PV functionality

# For pwcf 5% sab with 365.25-day year (rates quoted at cc, 365.25-day year)
# (PV from spread-sheet, AI from HP17B, price as difference)
#          2.25% of 15-nov-25   2.5% 15-feb-26    2.25% 15-nov-25  5% 15-nov-25  5% 15-nov-25
#                                                 call 15-nov-20                 call 15-nov-20
# Price     78.994385             80.526207       76.235              99.9850
# PV        79.587791             80.55368        76.829             101.303716   99.7997 (2.188% Pvol, 1.504)
# AI         0.593407              0.027473 Call   2.759 (10% Pvol)    1.318681   99.8067 (10% Yvol, 1.497)

# TSC 18-jul-17, testing spread curves (with pwcf only, spread 10bp cc)
#                                                                  5% 15-nov-25  5% 15-nov-25
#                                                 call 15-nov-20                 call 15-nov-20
# Price                                                               99.205755  99.205755
# PV                                                                 100.524436 100.524436 
# AI                                                                   1.318681 1.318681
#   This seems to be correct. The non-spread version gives 5.000952%sab, 4.939451%cc, 
#   the spread version 5.103497%sab, 5.039470%cc - i.e. 10bp cc 


ratessab = np.array([0.05,0.05,0.05,0.05])
ratescc = 2.*np.log(1+ratessab/2.)
#ratescc = ratescc + 0.001

breaks = settledate + np.array([2.,5.,10.,20.])*365.
curvepwcf = ['pwcf',settledate,breaks,ratescc]
tcurvepwcf = ['pwcf',settledate,breaks,ratescc,0.001,0.002]   # Version with spreads for taxability (type 2 and type 3)
curvepwlz = ['pwlz',settledate,breaks,ratescc]
curvepwtf = ['pwtf',settledate,breaks,ratescc]

#%% New PV functions
t0 = time.perf_counter()
xxPVBondCurvepwcf_new = pv.pvBondFromCurve(curvepwcf,xarray)
t1 = time.perf_counter()
print("%.5f seconds (cf)" % (t1-t0))

t0 = time.perf_counter()
xxPVBondCurvepwcf_parms = pv.pvBondFromCurve(curvepwcf,settledate = settledate, parms=parms)
t1 = time.perf_counter()
print("%.5f seconds (new parms)" % (t1-t0))

#cProfile.run('pv.pvBondFromCurve(curvepwcf,xarray)',sort='tottime')

#%% Old PV functions
t0 = time.perf_counter()
xxPVBondCurvepwcf = pv.PVBondFromCurve_vec(settledate,parms,curvepwcf)
t1 = time.perf_counter()
print("%.5f seconds (old)" % (t1-t0))
xxPVBondCurvepwlz = pv.PVBondFromCurve_vec(settledate,parms,curvepwlz)
xxPVBondCurvepwtf = pv.PVBondFromCurve_vec(settledate,parms,curvepwtf)
xxPVBondCurvepwcftax = pv.PVBondFromCurve_vec(settledate,tparms,tcurvepwcf)

#%% Test price-from-yield and yield-from-price

datesparse = sp.csr_matrix((datedata, cfindex, cfptr), shape=(bondcount, maxcol)) 
 
cfvec = sp.find(cfsparse[0,:])[2]  # CF vector as numpy array
datevec = sp.find(datesparse[0,:])[2]  # Date vector
xpricei = pv.bondPriceFromYield_matrix(0.05, cfvec, datevec, settledate,0.0)  # Default dirtyprice=0 used
dirtyprice = xpricei
xyieldi = pv.bondYieldFromPrice_matrix(dirtyprice, cfvec, datevec, settledate)
xyieldi = so.brentq(pv.bondPriceFromYield_matrix, -0.03, 0.5, args=(cfvec, datevec, settledate, dirtyprice), xtol=1e-12, rtol=9.0e-16, maxiter=100, full_output=True, disp=True)
 
xpricei = pv.BondPriceFromYield(0.05,0.0, settledate, parms.iloc[0,0], parms.iloc[0,1], parms.iloc[0,2], parms.iloc[0,3], parms.iloc[0,4], parms.iloc[0,5])  # Default dirtyprice=0 used
price = xpricei
xyieldi = pv.BondYieldFromPrice(price, settledate, parms.iloc[0,0], parms.iloc[0,1], parms.iloc[0,2], parms.iloc[0,3], parms.iloc[0,4], parms.iloc[0,5])
 
xprice = []
xyield = []
t0 = time.perf_counter()
for i in range(0,bondcount) :
    cfvec = sp.find(cfsparse[i,:])[2]  # CF vector as numpy array
    datevec = sp.find(datesparse[i,:])[2]  # Date vector
    xpricei = pv.bondPriceFromYield_matrix(0.05, cfvec, datevec, settledate)  # Default dirtyprice=0 used
    xprice.append(xpricei)
    xyieldi = pv.bondYieldFromPrice_matrix(xpricei, cfvec, datevec, settledate)
    xyield.append(xyieldi)
t1 = time.perf_counter()
print("%.5f seconds (new)" % (t1-t0))
 
xprice_old = []
xyield_old = []
t0 = time.perf_counter()
for i in range(0,bondcount) :
    xpricei = pv.BondPriceFromYield(0.05,0.0, settledate, parms.iloc[i,0], parms.iloc[i,1], parms.iloc[i,2], parms.iloc[i,3], parms.iloc[i,4], parms.iloc[i,5])  # Default dirtyprice=0 used
    xprice_old.append(xpricei)
    xyieldi = pv.BondYieldFromPrice(price, settledate, parms.iloc[i,0], parms.iloc[i,1], parms.iloc[i,2], parms.iloc[i,3], parms.iloc[i,4], parms.iloc[i,5])
    xyield_old.append(xyieldi)
t1 = time.perf_counter()
print("%.5f seconds (old)" % (t1-t0))
 


#%% Test callable - old

#vols = np.array([0,0,0.10,0,0.02188])

#xxPVBondCurvepwcfCall = pv.pvCallable_lnP(settledate,parms,curvepwcf,vols)
#xxPVBondCurvepwlzCall = pv.pvCallable_lnP(settledate,parms,curvepwlz,vols)
#xxPVBondCurvepwtfCall = pv.pvCallable_lnP(settledate,parms,curvepwtf,vols)

#vols = np.array([0,0,0.10,0,0.10,0,0])
#
#xxPVBondCurvepwcfCall_lnY = pv.pvCallable_lnY_old(settledate,parms,curvepwcf,vols)
#xxPVBondCurvepwlzCall_lnY = pv.pvCallable_lnY_old(settledate,parms,curvepwlz,vols)
#xxPVBondCurvepwtfCall_lnY = pv.pvCallable_lnY_old(settledate,parms,curvepwtf,vols)
#xxPVBondCurvepwcfCall_lnYtax = pv.pvCallable_lnY_old(settledate,tparms,tcurvepwcf,vols)

# 29-jun-17, need to check the pwcf callable prices. The non-callable all match the above,
# but the callable (for 10% yvol) give 
#               2.25% 15-nov-25    5% 15-nov-25
#               call 15-nov-20     call 15-nov-20
# Price           78.9943           98.4841
# PV              79.5877           99.8027 (almost matches above, opt value = 1.501)

# 18-jul-17, need to check pwcf callable price with spread. With 10bp cc spread, 
#                5% 15-nov-25
#               call 15-nov-20
# Price           98.870027
# PV              99.18895 (almost matches above, opt value = 1.501)
#  This seems to pass the preliminary check of just adding spread of 10bp to the whole curve

#%% Test callable - new

#vols = np.array([0,0,0.10,0,0.02188])

#xxPVBondCurvepwcfCall = pv.pvCallable_lnP(settledate,parms,curvepwcf,vols)
#xxPVBondCurvepwlzCall = pv.pvCallable_lnP(settledate,parms,curvepwlz,vols)
#xxPVBondCurvepwtfCall = pv.pvCallable_lnP(settledate,parms,curvepwtf,vols)

vols = np.array([0,0,0.10,0,0.10,0,0])
coupon = np.array(parms.iloc[:,0])

xxPVBondCurvepwcfCall_lnY = pv.pvCallable_lnY(curvepwcf, vols, coupon, calldate, 
                                              callflag, sparsenc = xarray, sparseyc = xarray_call)
#xxPVBondCurvepwlzCall_lnY = pv.pvCallable_lnY(settledate,parms,curvepwlz,vols)
#xxPVBondCurvepwtfCall_lnY = pv.pvCallable_lnY(settledate,parms,curvepwtf,vols)
#xxPVBondCurvepwcfCall_lnYtax = pv.pvCallable_lnY(settledate,tparms,tcurvepwcf,vols)

# 29-jun-17, need to check the pwcf callable prices. The non-callable all match the above,
# but the callable (for 10% yvol) give 
#               2.25% 15-nov-25    5% 15-nov-25
#               call 15-nov-20     call 15-nov-20
# Price           78.9943           98.4841
# PV              79.5877           99.8027 (almost matches above, opt value = 1.501)

# 18-jul-17, need to check pwcf callable price with spread. With 10bp cc spread, 
#                5% 15-nov-25
#               call 15-nov-20
# Price           98.870027
# PV              99.18895 (almost matches above, opt value = 1.501)
#  This seems to pass the preliminary check of just adding spread of 10bp to the whole curve

#%% Profile

cProfile.run('pv.pvCallable_lnY(settledate,parms,curvepwcf,vols)',sort='tottime',file='xx.out')
cProfile.run('pv.pvCallable_lnY(settledate,parms,curvepwcf,vols)','profile_pvCallable.out')

import pstats
p = pstats.Stats('profile_pvCallable.out')
p.strip_dirs()
p.sort_stats('cumulative').print_stats(30)
p.sort_stats('time').print_stats(30)

