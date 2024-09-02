# -*- coding: utf-8 -*-
"""


@author: tcoleman
"""

import numpy as np
import pandas as pd
import importlib as imp
import scipy.optimize as so

#%%
import discfact as df
import pvfn as pv

imp.reload(df)
imp.reload(pv)



#%% ------ PV (stub coupon) as Function of Yield that works for odd dates - NO LONGER USED ------
# -----------------
# Behaves like the HP17b function TVM by returning PV given Yield, except 
# 1) takes in maturity and then figures out the number of periods
# 2) all arguments are vectors (must be same length)
# 3) pmt is annual payment (converted to periodic payment with freq)
# 4) takes in yld and returns pv (doesn't take in pv and return yld)
# For non-exact periods, puts a stub at beginning
#    For example, for maturity 1.25 years:
#             c/4        c/2       c/2 + 100
#       |-----|----------|----------|
#       0    0.25      0.75        1.25
#    In other words when abs(freq*mat - round(freq*mat) > 0.05)
#    Then put fraction of coupon: (freq*mat - int(freq*mat))/freq
# When maturity is on a coupon interval, then do usual:
#                 c/2        c/2      c/2 + 100
#       |----------|----------|---...---|
#       0         0.5        1.0       2.5
# otherwise behaves the HP17B function - to calculate DV01
# Takes in payment per year and then divides by frequency (not as in HP17B)

def PVBondStubFromYld_vec(mat,i,pmt,fv,freq):
    "PV function - behaves like HP17B TVM function but: 1) mat in yrs; 2) args vectors; 3) pmt ann 4) only produces PV given i, ..."
    # mat, i, mpt, fv, freq must all be vectors of same length
    # mat will be maturity in years
    # i is interest rate (as in 5.0 for 5%)
    # pmt is periodic payment (as in 3.0 for 6% sab bond)
    # freq may be different for different bonds
    xpmt = pmt.astype(float) / freq
    noperiodsdbl = freq * mat.astype(float)         # This will be the no. of coupon periods. 
                                   # This may not always be an integer - short bonds
                                   # will be zeros, longer may have short stub (put it at beginning)
    x1 = np.absolute(noperiodsdbl - np.round(noperiodsdbl))    # This will be a small positive when very near to integer
    x2 = x1 > 0.05            # Checks when not integer periods - true for integer (or very close)
    x2 = x2.astype(int)       # Converts boolean to 0, 1. x2 of 1 means yes stub
    stubper = noperiodsdbl - noperiodsdbl.astype(int)    # For mat not near integer, length of short stub
    noperiodsint = (1-x2)*np.round(noperiodsdbl) + x2*np.round(noperiodsdbl - stubper)  # The first thought would
                              # be to simply convert the original noperiodsdbl to integer. But if a maturity
                              # is 1.98yrs we want to round that to 2. So, for those where noperiodsdbl is 
                              # close to integer (this will be x2 TRUE) we round, while for others (say mat=1.25
                              # and noperiodsdbl = 2.5) we subtract the non-integer part (and round for safety)
    noperiodsint = noperiodsint.astype(int)
    stubper = x2*stubper            # Zero out all elements that are exact periods (non-stub)
    stubpmt = xpmt * stubper       # Define stub payment. For non-stub will be zero
    beta = 1/(1+i.astype(float)/(100.0*freq))      # discount factor - assume i is in percent (2.50 for 2.5%)
    xx = np.power(beta,noperiodsint)
    pvreg = beta*xpmt*(1-xx)/(1-beta) + fv*xx
    stubdisc = np.power(beta,stubper)
    pv = stubdisc*(pvreg + stubpmt)
    return(-pv)
#PV_vec = np.vectorize(PV) # NB - apparently np.vectorize is for convenience rather than performance
                           #      it essentially implements a "for" loop
# -----------------
#%% ------ PV (stub coupon) as Function of Yield or Curve that works for odd dates ------
# -----------------
# Behaves like the HP17b TVM function by returning PV given Yield, except 
# 1) takes in maturity and then figures out the number of periods
# 2) all arguments are vectors (must be same length)
# 3) pmt is annual payment (converted to periodic payment with freq)
# 4) takes in yld and returns pv (doesn't take in pv and return yld)
# For non-exact periods, puts a stub at beginning. This means price
# will (approximately) be the clean price
#    For example, for maturity 1.25 years:
#             c/4        c/2       c/2 + 100
#       |-----|----------|----------|
#       0    0.25      0.75        1.25
#    In other words when abs(freq*mat - round(freq*mat) > 0.05)
#    Then put fraction of coupon: (freq*mat - int(freq*mat))/freq
# When maturity is on a coupon interval, then do usual:
#                 c/2        c/2      c/2 + 100
#       |----------|----------|---...---|
#       0         0.5        1.0       2.5
# otherwise behaves the HP17B function - to calculate DV01
# Takes in payment per year and then divides by frequency (not as in HP17B)

def PVBondStubFromCurve_vec(mat,curve,pmt,fv,freq):
    "PV function - behaves somewhat like HP17B TVM function but: 1) mat in yrs; 2) args vectors; 3) pmt ann 4) curve may be number or curve; 5) only produces PV given i, ..."
    # if type(curve) = tuple then curve is actually a vector of ytms
    # if type(curve) = list then curve is list:
    #  1. crvtype - (string) name to identify the type of curve
    #       pwcf - piece-wise constant forward (flat forwards, continusuously-compounded)
    #       pwlz - piece-wise linear zero (cc)
    #       pwtf - piece-wise twisted forward (linear between breaks but not connected at breaks)
    #  2. rates (numeric vector) - numbers like 0.025 for 2.5% - assumed cc at 365.25 annual rate
    #  3. the breaks (vector, same length as rates, of number of years to curve break)
    #
    # mat, curve(if vector), pmt, fv, freq must all be vectors of same length
    # mat will be maturity in years
    # i is interest rate (as in 5.0 for 5%)
    # pmt is periodic payment (as in 6.0 for 6% sab bond)
    # freq may be different for different bonds
    xfreq = freq.astype(float)
    xpmt = pmt.astype(float) / xfreq
    noperiodsdbl = freq * mat.astype(float)         # This will be the no. of coupon periods. 
                                   # This may not always be an integer - short bonds
                                   # will be zeros, longer may have short stub (put it at beginning)
    x1 = np.absolute(noperiodsdbl - np.round(noperiodsdbl))    # This will be a small positive when very near to integer
    stubflag = x1 > 0.05            # Checks when not integer periods - true for integer (or very close)
    stubflag = stubflag.astype(int)       # Converts boolean to 0, 1. stubflag of 1 means yes stub
    stubper = noperiodsdbl - noperiodsdbl.astype(int)    # For mat not near integer, length of short stub
    noperiodsint = (1-stubflag)*np.round(noperiodsdbl) + stubflag*np.round(noperiodsdbl - stubper)  # The first thought would
                              # be to simply convert the original noperiodsdbl to integer. But if a maturity
                              # is 1.98yrs we want to round that to 2. So, for those where noperiodsdbl is 
                              # close to integer (this will be stubflag TRUE) we round, while for others (say mat=1.25
                              # and noperiodsdbl = 2.5) we subtract the non-integer part (and round for safety)
    stubper = stubflag*stubper            # Zero out all elements that are exact periods (non-stub)
    xmat = (stubper + noperiodsint) / xfreq   # the maturity adjusted to exact payment intervals
    noperiodsint = noperiodsint.astype(int)
    stubpmt = xpmt * stubper       # Define stub payment. For non-stub will be zero
    if (type(curve) == np.ndarray) :       # section for ytm
        beta = 1/(1+curve.astype(float)/(100.0*xfreq))      # discount factor - assume i is in percent (2.50 for 2.5%)
        dfmat = np.power(beta,noperiodsint)
        stubdisc = np.power(beta,stubper)  
        xpvann = beta*(1-dfmat)/(1-beta)   # PV of annuity from ytm
        xpvann = stubdisc * xpvann 
        dfmat = dfmat * stubdisc
    else:
        dfmat = df.discFact(xmat,curve)      # dfs for maturity dates
        stubdisc = df.discFact(stubper,curve)  # dfs for stub periods (may be 1 because may be no stub)
        xpvann = []                       # Create empty list
        for j in range(np.size(noperiodsint)) :
            if (noperiodsint[j] == 0):    # no coupon periods, so just one (final) payment
                x2 = 0.0                  # pv of coupons is 0.0
            else :           # if stub payment up-front, then do range from 0 to 1+noperiods
                             # If no stub, range from 1 to 1+noperiods
                             # e.g. is there is 1 integral payment and yes stub, then need 2 payments (stub + final)
                             # if no stub, just 1 payment (final)
                x1 = (stubper[j] + np.array(range(1,(1 + noperiodsint[j])))) / xfreq[j]
                x2 = sum(df.discFact(x1,curve))       # cumulates up the coupon payments
            xpvann.append(x2)   # 
        xpvann = np.array(xpvann)
    pv = stubdisc*stubpmt + xpvann*xpmt + fv*dfmat
    return(-pv)
    
    
#%% ------ PV (approximate) as Function of Yield or Curve that works for odd dates ------
# -----------------
# Behaves like the HP17b TVM function by returning PV given Yield, except 
# 1) takes in maturity and then figures out the number of periods
# 2) all arguments are vectors (must be same length)
# 3) pmt is annual payment (converted to periodic payment with freq)
# 4) takes in yld and returns pv (doesn't take in pv and return yld)
# For non-exact periods, puts full first coupon, so PV is dirty price
#    For example, for maturity 1.25 years:
#             c/2        c/2       c/2 + 100
#       |-----|----------|----------|
#       0    0.25      0.75        1.25
# otherwise behaves the HP17B function - to calculate DV01
# Takes in payment per year and then divides by frequency (not as in HP17B)

def PVBondApproxFromCurve_vec(mat,curve,pmt,fv,freq):
    "PV function - behaves somewhat like HP17B TVM function but: 1) mat in yrs; 2) args vectors; 3) pmt ann 4) curve may be number or curve; 5) only produces PV given i, ..."
    # if type(curve) = tuple then curve is actually a vector of ytms
    # if type(curve) = list then curve is list:
    #  1. crvtype - (string) name to identify the type of curve
    #       pwcf - piece-wise constant forward (flat forwards, continusuously-compounded)
    #       pwlz - piece-wise linear zero (cc)
    #       pwtf - piece-wise twisted forward (linear between breaks but not connected at breaks)
    #  2. rates (numeric vector) - numbers like 0.025 for 2.5% - assumed cc at 365.25 annual rate
    #  3. the breaks (vector, same length as rates, of number of years to curve break)
    #
    # mat, curve(if vector), pmt, fv, freq must all be vectors of same length
    # mat will be maturity in years
    # i is interest rate (as in 5.0 for 5%)
    # pmt is periodic payment (as in 6.0 for 6% sab bond)
    # freq may be different for different bonds
    xcurve = curve
    if (type(curve) == np.ndarray) :       # section for ytm, create curve
        xcurve = ['pwcf',0.0,np.array([100]),np.array(2.*np.log(1+curve[0]/2.))]
    
    xfreq = freq.astype(float)
    xpmt = pmt.astype(float) / xfreq
    xmat = mat.astype(float)
    noperiodsdbl = freq * xmat         # This will be the no. of coupon periods.             # For exact dates, get noperiodsdbl using DateDiff & 12*yrs + mths + days/30 (mat has to be date)
                                   # This may not always be an integer - short bonds
                                   # will be zeros, longer will have full first coupon
    noperiodsint = np.ceil(noperiodsdbl)      # The number of coupon periods
    noperiodsint = noperiodsint.astype(int)
    stubper = noperiodsdbl - noperiodsdbl.astype(int)    # Length of short stub (may be zero)
    x1 = noperiodsdbl == np.floor(noperiodsdbl)     # checks if we are on a coupon date (exact yr)
    stubper = stubper + x1.astype(int)       # if so, then add one coupon period 
    dfmat = df.discFact(xmat,curve)      # dfs for maturity dates
    xpvann = []                       # Create empty list
    for j in range(np.size(noperiodsint)) :    # loop over bonds
            x1 = (stubper[j] + np.array(range(0,noperiodsint[j]))) / xfreq[j]   # Generates vector of periods
                                         # (yrs) for coupon dates.
                                         # To do this for exact dates, we could generate similar vector and use 
                                         # CalAdd to subtract vector of 6mths from maturity date. The main subtelty
                                         # is month-end issues - need to calculate the no. of years from mth-end
                                         # to mth-end, not day-of-month to day-of-month. But the new function
                                         # "DateDiff" calculates difference in yrs, mths, days, with days adjusted
                                         # to mth end when flag is set on. If we use that to calculat no. of mths
                                         # then we can generate vector of correct length (being integer mths for end-to-end, 
                                         # non-integer when not end-to-end)
            x2 = sum(df.discFact(x1,xcurve))       # cumulates up the coupon payments
            xpvann.append(x2)   # 
    xpvann = np.array(xpvann)
    pv = xpvann*xpmt + fv*dfmat
    return(-pv)
    

#%% ------ Price of bond from yield, and yield from price ------

def BondPriceFromYield(xyield, price, settledate, coupon, maturity, frequency, redemption, daycount, eomyes):
    "Price function for single (non-callable) bond - calls PVBondFromCurve_vec"

# 23-jul-17 TSC modified to use new version of pvBondFromCurve (but called with parms)

# Takes settle, parms & yield for a single bond. Creates a curve (flat forwards) and feeds to PVBondFromCurve_vec"
# Parms can be an array of bonds. Parms will be pandas later
# For now, here is the definition of "parms"
#   - list of lists or nested lists - one list for each bond, with each bond being:
#     - coupon (numeric)
#     - Maturity date (Julian, such as 45975 for 15-nov-2025)
#     - Frequency: 2 = semi, 4 = quarterly
#     - Redemption amount - generally 100., but allows for 0. and then it's an annuity
#     - daycount: "A/A" for "Actual/Actual" - at the moment this is assumed implicitely
#     - "eomyes" for end-of-month convention of UST: going from 28-feb-1999 to 31-aug-1999 to 29-feb-2000 to 31-aug-2000
#     - if callable, first call date. If not callable, can be not there or zero. If not there, will be set to zero. 

    parms = pd.DataFrame([[coupon, maturity, frequency, redemption, daycount , eomyes, 0, 1]])
    if (frequency == 0) :
        xfrequency = 1
    else :
        xfrequency = 1.*frequency
    curve = ['pwcf',settledate,np.array([settledate + 100*365]),np.array([xfrequency*np.log(1. + xyield/xfrequency)])]
    xpv = pv.pvBondFromCurve(curve,settledate = settledate, parms=parms)
    diff = xpv[0,0] - price
    return(diff)

def BondYieldFromPrice(price, settledate, coupon, maturity, frequency, redemption, daycount, eomyes):
    "Yield from price for single (non-callable) bond - uses scipy.optimize.brentq and PVBondFromCurve_vec"

    xyield = so.brentq(BondPriceFromYield, -0.03, 0.5, args=(price, settledate, coupon, maturity, frequency, redemption, daycount, eomyes), xtol=1e-12, rtol=9.0e-16, maxiter=100, full_output=False, disp=True)

    return(xyield)



