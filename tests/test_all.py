# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 21:04:28 2016

@author: tcoleman"""

import sys
import numpy as np
import pandas as pd
import importlib as imp

sys.path.append('../src/package')

#%%

import discfact as df
import pvfn as pv
import DateFunctions_1 as dates

imp.reload(dates)
imp.reload(pv)
imp.reload(df)

#%% ------ Sample data for curves ------
# Sample maturity dates for testing dates and bonds
mat = np.array([0.25,0.5,0.75,0.9,0.99,1.0,1.2])
# Rates
ratessab = np.array([0.01,0.02005,0.02512,0.03724])
ratescc = 2.*np.log(1+ratessab/2.)
# Create curve lists - type, anchor point (quote date), break points, rates
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

#%% ------ Sample data for calculating bond PV ------

i = np.array([5,5,5,5,5,5,5])
pmt = np.array([6,6,6,6,6,6,6])
fv = np.array([100,100,100,100,100,100,100])
freq = np.array([2,2,2,2,2,2,2])
xx1PVBondYld = pv.PVBondStubFromYld_vec(mat,i,pmt,fv,freq) 
xx2PVBondYld = pv.PVBondStubFromCurve_vec(mat,i,pmt,fv,freq) 
# The results for this should be: (mat, i, pmt, fv, freq, -> PV)
#  0.25, 5%, 6%, 100, 2 -> 100.25455
#  0.5, 5%, 6%, 100, 2 -> 100.487805
#  0.75, 5%, 6%, 100, 2 -> 100.736373
#  0.9, 5%, 6%, 100, 2, -> 0.4yr stub, 1 half-yr period, 100.875293
#  0.99, 5%, 6%, 100, 2, -> 2 per exact, 100.963712
#  1.01, 5%, 6%, 100, 2, -> 2 per exact, 100.963712
#  1.2, 5%, 6%, 100, 2, -> .4 per stub, 2 per exact, 101.159603
xxPVBondCurvepwcf = pv.PVBondStubFromCurve_vec(mat,curvepwcf,pmt,fv,freq) 
xxPVBondCurvepwlz = pv.PVBondStubFromCurve_vec(mat,curvepwlz,pmt,fv,freq) 
xxPVBondCurvepwtf = pv.PVBondStubFromCurve_vec(mat,curvepwtf,pmt,fv,freq) 
# For pwcf curve I think the answers should be:
# 0.25, 101.24347
# 0.5, 102.48756
# 0.75, 103.47021
# 0.9, 104.04677
# 0.99 (1), 104.4554
# 1.01 (1), 104.4554
# 1.2, 105.1334

xxPVBondCurvepwcf2 = pv.PVBondApproxFromCurve_vec(mat,curvepwcf,pmt,fv,freq) 


#%% -------------- Sample date for testing date function ----------------
# Julian dates are days from 31-dec-1899 
#  15-feb-1996  35,109 0
#  15-feb-1996  35,109 1
#  15-feb-1996  35,109 2
#  15-feb-1996  35,109 3
#  15-feb-1996  35,109 4
#  15-feb-1996  35,109 5
#  29-feb-2000  36,584 6
#  28-feb-1999  36,218 7
#  30-apr-1999  36,279 8
#  30-apr-1999  36,279 9
#  30-apr-1999  36,279 10
#  31-dec-1932  12,053 11
#  15-nov-1996  35,383 12
jdate1 = np.array((35109,35109,35109,35109,35109,35109,36584,36218,36279,36279,36279,12053,35383))  #
#                         Diff    eom=y eom=n
#                          y m d   d     d
#  15-feb-1996  35,109 0   0 0 0
#  15-mar-1996  35,138 1   0 1 0 
#  16-feb-1997  35,476 2   1 0 1
#  16-mar-1997  35,504 3   1 1 1
#  17-feb-2000  36,572 4   4 0 2
#  17-mar-2000  36,601 5   4 1 2
#  31-aug-2000  36,768 6   0 6     0     2
#  31-aug-1999  36,402 7   0 6     0     3
#  31-oct-1999  36,463 8   0 6     0     1
#  30-oct-1999  36,462 9   0 6     0     0
#  29-oct-1999  36,461 10  0 6    -1    -1
#  1-jan-1946   16,802 11  14 -11 -30
#  5-jan-2000   36,529 12  4 -10 -10
jdate2 = np.array((35109,35138,35476,35504,36572,36601,36768,36402,36463,36462,36461,16802,36529))  #

ymd1 = dates.JuliantoYMD(jdate1)
ymd2 = dates.JuliantoYMD(jdate2)

# This calculates date difference (y m d) WITHOUT eom convention. It should match the "eom=n" above
diff = ymd2 - ymd1
# This calculates date differene WITH eom convention. It should match the "eom=y" above
eomdiff = dates.DateDiff(jdate1,jdate2)

mthdiff = 12.*eomdiff[0] + eomdiff[1] + eomdiff[2]/30.

np.remainder(mthdiff,6)

#%% Testing stuff for bond calculations

## 2.25% of 15-nov-2025 and 2.5% of 15-feb-2026
#parms = [[2.25,45975,2,100.,"A/A","eomyes"],[2.5,46067,2,100.,"A/A","eomyes"]]
## settle 19-feb-2016
#settledate = 42418
#
#
#xlen = len(parms)
#
#coupon = np.array([parms[i][0] for i in range(0,xlen)])  # This seems the way to 'slice' along the 2nd axis
#maturity = np.array([parms[i][1] for i in range(0,xlen)])  # With numpy array this would be x[,0] but doesn't work
#frequency = np.array([parms[i][2] for i in range(0,xlen)]) # with lists. And these cannot be numpy arrays because
#finalpmt = np.array([parms[i][3] for i in range(0,xlen)])  # some of the entries (daycount, eom) are strings
#daycount = [parms[i][4] for i in range(0,xlen)]
#eom = [parms[i][5] for i in range(0,xlen)]
#
#xfrequency = frequency.astype(float)
#xcoupon = coupon.astype(float) / xfrequency
#xsettle = np.tile(settledate,xlen)
#eomdiff = dates.DateDiff(xsettle,maturity)
#
#
#xmthdiff = 12.*eomdiff[0] + eomdiff[1] + eomdiff[2]/30.    # Settle to maturity in "months"
#xmthper = 12. / xfrequency          # The number of months per coupon period
#x1 = xmthdiff / xmthper   # Fractional number of "periods" to settle date
#xperprior = np.ceil(x1)  # Number of "periods" to prior coupon paymen (payment before settle)
#xpernext = np.floor(x1)   # Number of "period" to next payment date

#%% Test PVBondFromCurve_vec

#    Five bonds, 
#       2.25% of 15-nov-2025 non-call
#       2.5% of 15-feb-2026 non-call
#       2.25% of 15-nov-2025 callable 15-nov-2020
#       5% of 15-nov-2025 non-call
#       5% of 15-nov-2025 callable 15-nov-2020

# For pwcf 5% sab with 365.25-day year (rates quoted at cc, 365.25-day year)
# (PV from spread-sheet, AI from HP17B, price as difference)
#          2.25% of 15-nov-25   2.5% 15-feb-26    2.25% 15-nov-25  5% 15-nov-25  5% 15-nov-25
#                                                 call 15-nov-20                 call 15-nov-20
# Price     78.994385             80.526207       76.235              99.9850
# PV        79.587791             80.55368        76.829             101.303716   99.7997 (2.188% Pvol, 1.504)
# AI         0.593407              0.027473 Call   2.759 (10% Pvol)    1.318681   99.8067 (10% Yvol, 1.497)

parms = [[2.25,45975,2,100.,"A/A","eomyes",0],[2.5,46067,2,100.,"A/A","eomyes",0],[2.25,45975,2,100.,"A/A","eomyes",44149],
	[5.,45975,2,100.,"A/A","eomyes",0],[5.,45975,2,100.,"A/A","eomyes",44149]]
parms = pd.DataFrame(parms)
# settle 19-feb-2016
settledate = 42418

ratessab = np.array([0.05,0.05,0.05,0.05])
ratescc = 2.*np.log(1+ratessab/2.)

breaks = settledate + np.array([2.,5.,10.,20.])*365.
curvepwcf = ['pwcf',settledate,breaks,ratescc]
curvepwlz = ['pwlz',settledate,breaks,ratescc]
curvepwtf = ['pwtf',settledate,breaks,ratescc]


xxPVBondCurvepwcf = pv.PVBondFromCurve_vec(settledate,parms,curvepwcf)
xxPVBondCurvepwlz = pv.PVBondFromCurve_vec(settledate,parms,curvepwlz)
xxPVBondCurvepwtf = pv.PVBondFromCurve_vec(settledate,parms,curvepwtf)

# For pwcf 5% sab with 365.25-day year (rates quoted at cc, 365.25-day year)
# (PV from spread-sheet, AI from HP17B, price as difference)
#          2.25% of 15-nov-25   2.5% 15-feb-26
# Price     78.994385             80.526207
# PV        79.587791             80.55368
# AI         0.593407              0.027473


#%% Test callable

#vols = np.array([0,0,0.10,0,0.02188])

#xxPVBondCurvepwcfCall = pv.pvCallable_lnP(settledate,parms,curvepwcf,vols)
#xxPVBondCurvepwlzCall = pv.pvCallable_lnP(settledate,parms,curvepwlz,vols)
#xxPVBondCurvepwtfCall = pv.pvCallable_lnP(settledate,parms,curvepwtf,vols)

vols = np.array([0,0,0.10,0,0.10])

xxPVBondCurvepwcfCall_lnY = pv.pvCallable_lnY(settledate,parms,curvepwcf,vols)
xxPVBondCurvepwlzCall_lnY = pv.pvCallable_lnY(settledate,parms,curvepwlz,vols)
xxPVBondCurvepwtfCall_lnY = pv.pvCallable_lnY(settledate,parms,curvepwtf,vols)

# 29-jun-17, need to check the pwcf callable prices. The non-callable all match the above,
# but the callable (for 10% yvol) give 
#               2.25% 15-nov-25    5% 15-nov-25
#               call 15-nov-20     call 15-nov-20
# Price           78.9943           98.4841
# PV              79.5877           99.8027 (almost matches above, opt value = 1.501)


#%% Test PVBondFromCurve_vec

ratessab = np.array([0.03,0.035,0.045,0.05])
ratescc = 2.*np.log(1+ratessab/2.)

breaks = settledate + np.array([2.,5.,10.,20.])*365.
curvepwcf = ['pwcf',settledate,breaks,ratescc]
curvepwlz = ['pwlz',settledate,breaks,ratescc]
curvepwtf = ['pwtf',settledate,breaks,ratescc]


xxPVBondCurvepwcf = pv.PVBondFromCurve_vec(settledate,parms,curvepwcf)
xxPVBondCurvepwlz = pv.PVBondFromCurve_vec(settledate,parms,curvepwlz)
xxPVBondCurvepwtf = pv.PVBondFromCurve_vec(settledate,parms,curvepwtf)




#%% Testing CalAdd

JulianDate = np.array((40250,40250,40250,40250,40250,40250,40250,40250))
xdates = np.array((40000,40100,40000,40100,40200,40389,39871,39505))
xymd1 = dates.JuliantoYMD(xdates)
nyear = np.array((0,0,0,1,2,1,1,0))
nmonth = np.array((0,0,0,6,6,6,6,6))
nday = np.array((0,5,100,0,0,0,0,0))
x1 = np.size(nyear)
x2 = np.size(nmonth)
x3 = np.size(nday)
y1 = max([x1,x2,x3])

xymds = dates.JuliantoYMD(JulianDate)
xleap = dates.IsALeapYear(xymds[0])
dd = xymds[2]
mm = xymds[1]
xleap = dates.IsALeapYear(xymd1[0])
dd = xymd1[2]
mm = xymd1[1]

xdd = dd - nday

#%% Sample date for testing calladd
# Julian dates are days from 31-dec-1899 
#  15-feb-1996  35,109 0
#  15-feb-1996  35,109 1
#  29-feb-2000  36,584 2
#  29-feb-2000  36,584 3
#  27-feb-1999  36,217 4
#  28-feb-1999  36,218 5
#  28-feb-1999  36,218 6
#  28-feb-1999  36,218 7
#  30-apr-1999  36,279 8
#  30-apr-1999  36,279 9
#  30-apr-1999  36,279 10
#  31-oct-1999  36,463 11
#  30-oct-1999  36,462 12
jdate1 = np.array((35109,
                   35109,
                   36584,
                   36584,
                   36217,
                   36218,
                   36218,
                   36218,
                   36279,
                   36279,
                   36279,
                   36463,
                   36462))  #

nday = np.array((23,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0))
nmonth = np.array((0,
                   6,
                   6,
                   6,
                   6,
                   6,
                   12,
                   0,
                   6,
                   12,
                   18,
                   6,
                   6))

nyear = np.array((0,
                  1,
                  0,
                  1,
                  0,
                  0,
                  0,
                  1,
                  0,
                  0,
                  0,
                  0,
                  0))

xx1 = dates.CalAdd(jdate1,"add",nyear=nyear,nmonth=nmonth,nday=nday)

# Results should be  
#  9-mar-1996   35,132 0
#  15-aug-1997  35,656 1
#  31-aug-2000  36,768 2
#  31-aug-2001  37,133 3
#  27-aug-1999  36,398 4
#  31-aug-1999  36,402 5
#  29-feb-2000  36,584 6
#  29-feb-2000  36,584 7
#  31-oct-1999  36,463 8
#  30-apr-2000  36,645 9
#  31-oct-2000  36,829 10
#  30-apr-2000  36,645 11
#  30-apr-2000  36,645 12

xx2 = dates.JuliantoYMD(xx1)
#%% Now test CalAdd and should fail
nday = np.array((23,
                   23,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0))

dates.CalAdd(jdate1,"add",nyear=nyear,nmonth=nmonth,nday=nday)
# Now should fail because adding days & months
