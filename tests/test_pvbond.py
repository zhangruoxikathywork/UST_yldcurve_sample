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
settledate = 42418

#%% Test building matrixes from parameters with forward settle date

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
coupon = xarray[10]
frequency = xarray[11]
calldate = xarray[12]
callflag = xarray[13]

# Make dense (filled out) arrays from sparse

xcf = sp.csr_matrix((cfdata, cfindex, cfptr), shape=(bondcount, maxcol)).toarray()


#%% Test converting for callable - with forward date

# Set the calldate for non-calls beyond the maturity date (20991231 works even for consol)
# and the function bondParmsToMatrix will skip that bond - setting the whole row to zero in sparse matrix

calldate = np.array(parms.iloc[:,6])
callflag = calldate > 0
xcalldate = np.copy(calldate)
xcalldate[~callflag] = dates.YMDtoJulian(20991231)

xarray = pv.bondParmsToMatrix(xcalldate,parms)

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

#%% Test building matrixes from parameters for just standard bonds

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
coupon = xarray[10]
frequency = xarray[11]
calldate = xarray[12]
callflag = xarray[13]

xcf = sp.csr_matrix((cfdata, cfindex, cfptr), shape=(bondcount, maxcol)).toarray()

#%% Make curves 


ratessab = np.array([0.05,0.05,0.05,0.05])
ratescc = 2.*np.log(1+ratessab/2.)
#ratescc = ratescc + 0.001

breaks = settledate + np.array([2.,5.,10.,20.])*365.
curvepwcf = ['pwcf',settledate,breaks,ratescc]
tcurvepwcf = ['pwcf',settledate,breaks,ratescc,0.001,0.002]   # Version with spreads for taxability (type 2 and type 3)
curvepwlz = ['pwlz',settledate,breaks,ratescc]
curvepwtf = ['pwtf',settledate,breaks,ratescc]

#%% and test PV functionality for New PV Bond functions
t0 = time.perf_counter()
xxPVBondCurvepwcf_new = pv.pvBondFromCurve(curvepwcf,xarray)
t1 = time.perf_counter()
print("%.5f seconds (cf)" % (t1-t0))

t0 = time.perf_counter()
xxPVBondCurvepwcf_parms = pv.pvBondFromCurve(curvepwcf,settledate = settledate, parms=parms)
t1 = time.perf_counter()
print("%.5f seconds (new parms)" % (t1-t0))

# For pwcf 5% sab with 365.25-day year (rates quoted at cc, 365.25-day year)
# (PV from spread-sheet, AI from HP17B, price as difference)
#          2.25% of 15-nov-25   2.5% 15-feb-26    2.25% 15-nov-25   2.25% 15-nov-25  5% 15-nov-25  5% 15-nov-25             Bill
#                                                 call 15-nov-20    call 15-nov-20                 call 15-nov-20           15-may-2016
#  NO call      0                    1                 2                3                  4           5            6    7     8
# Price     78.994385             80.526207       78.994385         78.994385           99.9850     99.9850                  98.844
# PV        79.587791             80.55368        79.587791         79.587791         101.303716   101.303716
# AI         0.593407              0.027473       0.593407          0.593407           1.318681      1.318681

# TSC 18-jul-17, testing spread curves (with pwcf only, spread 10bp cc)
#                                                                  5% 15-nov-25  5% 15-nov-25
#                                                 call 15-nov-20                 call 15-nov-20
# Price                                                               99.205755  99.205755
# PV                                                                 100.524436 100.524436 
# AI                                                                   1.318681 1.318681
#   This seems to be correct. The non-spread version gives 5.000952%sab, 4.939451%cc, 
#   the spread version 5.103497%sab, 5.039470%cc - i.e. 10bp cc 


#cProfile.run('pv.pvBondFromCurve(curvepwcf,xarray)',sort='tottime')

#%% Old PV functions

# Will not work with Bills, so exclude the bill from testing
xparms = parms[0:8]

t0 = time.perf_counter()
xxPVBondCurvepwcf = pv.PVBondFromCurve_vec(settledate,xparms,curvepwcf)
t1 = time.perf_counter()
print("%.5f seconds (old)" % (t1-t0))
xxPVBondCurvepwlz = pv.PVBondFromCurve_vec(settledate,xparms,curvepwlz)
xxPVBondCurvepwtf = pv.PVBondFromCurve_vec(settledate,xparms,curvepwtf)
#xxPVBondCurvepwcftax = pv.PVBondFromCurve_vec(settledate,tparms,tcurvepwcf)

