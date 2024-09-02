
import numpy as np
import pandas as pd
import scipy.stats as ss
import scipy.sparse as sp
import scipy.optimize as so
from scipy.optimize import fsolve
import discfact as df
import DateFunctions_1 as dt
import pvfn as pv


#%% Normal Yield

def pvCallable_nY(tcurve, vols, sparsenc=0, sparseyc=0., settledate=0,
                   parms=0):
    """
    Return PV & Price function for bonds, callable & non-call (using Normal Yield
    model).

    Args
    ----
    tcurve : list
        The list contains:
            0. crvtype - (string) name to identify the type of curve
                pwcf - piece-wise constant forward (flat forwards,
                       continusuously-compounded)
                pwlz - piece-wise linear zero (cc)
                pwtf - piece-wise twisted forward (linear between breaks but
                       not connected at breaks)
            1. valuedate - curve value date. Will be zero (in which case use
                           offsets) or date (as days or date)
            2. For pwcf, pwlz, pwtf the breaks (vector, same length as rates,
               of either dates or years from today, based on whether
               valuedate is yrs (zero) or date)
            3. For pwcf, pwlz, pwtf the rates (numeric vector) - numbers like
               0.025 for 2.5% - assumed cc at 365.25 annual rate
              For dailydf a zoo object of dfs (dates, offsets, dfs)
            4. s2 spread for tax=2
            5. s3 spread for tax=3
            For 4-5, this will be made into three curves to apply to the
            three tax types:
              1 - fully taxable - breaks & rates
              2 - partially taxable - breaks & rates + s2
              - If no spreads entered (len(tcurve)=4) then only fully taxable
                and no extra processing for taxability
              - If only one spread then make s2 & s3 the same. (This will
                accomodate case where there is only partial or wholly
                tax exempt. Which it is will have to be handled outside
                of optimization.)
    vols : array
        Volatilities corresponding to each bond. vols must be same length as
        parms (a vol for every bond) - not used for non-callable bonds
    sparsenc : array, optional
        Sparse array for all bonds (to final maturity date). Default is 0.
    sparseyc : scipy.sparse matrix, optional
        Sparse array for callable bonds. Default is 0.
    settledate : numeric or date, optional
        The settlement date for the bond transactions. Default is 0.
    parms : dataframe
        For each bond, its parameters contain:
            0 coupon (numeric)
            1 Maturity date (Julian, such as 45975 for 15-nov-2025)
            2 Frequency: 2 = semi, 4 = quarterly
            3 Redemption amount - generally 100., but allows for 0. and then
              it's an annuity
            4 daycount: "A/A" for "Actual/Actual" - at the moment this is
              assumed implicitely
            5 "eomyes" for end-of-month convention of UST: going from
              28-feb-1999 to 31-aug-1999 to 29-feb-2000 to 31-aug-2000
            6 if callable, first call date. If not callable, can be not there
              or zero. If not there, will be set to zero
            7 Call flag - True if calable
            8 tax status
              - 1 = fully taxable
              - 2 = partially exempt
              - 3 = fully tax exempt
        Default is 0. Used if sparse matrices are not provided.

    Returns
    -------
    array
        2-column array: 0th col price, 1st col pv.

    Notes
    -----
        23-jul-17 TSC. Re-write to work from sparse arrays of CF & Dates,
                       and using new versions of pricefromyield
        18-jul-17 TSC use new version of PVBondFromCurve_vec which has
                      argument "asofquotedate" to force the calcuation of
                      the forward bond as of the forward date.
    """
    # First check if sparsenc (CF & date arrays) have been handed in. If not, then
    # get all the other needed data (such as coupon, calldate, callflag) from parms
    if ((type(parms) == pd.DataFrame) and (type(sparsenc) != list) ) :  # When parms are passed in and sparesearray not, then process parms
        sparsenc = pv.bondParmsToMatrix(settledate,parms)
        coupon = np.array(parms.iloc[:,0])                   # vector of coupons
        calldate = np.array(parms.iloc[:,6])                 # Assume that dates are already Julian
        callflag = np.array(parms.iloc[:,7])
        if (callflag.any()) :                                # Process callables if any
            xcalldate = calldate.copy()
            xcalldate[~callflag] = dt.YMDtoJulian(20991231)   # For non-callable bonds set call date beyond any maturity #are you changing everything to callable? do all of them have to go through the same yield functions?
            sparseyc = pv.bondParmsToMatrix(xcalldate,parms)

    bondpv = pv.pvBondFromCurve(tcurve, sparsearray = sparsenc)     # calculate the PV of all bonds to the final maturity date
                                # TSC 13-jan-24 Do NOT put in settledate - not needed after creating sparse array
    callflag = sparsenc[13]              # Need callflag before processing any callables
    aivector = sparsenc[3]               # Need aivector for original, nor forward, date
    if (callflag.any()) :                 # There are some callable bonds

        cfsparse = sparseyc[0]          # This should already be written as a sparse array
        cfdata = sparseyc[1]
        datedata = sparseyc[2]       # This is vector of data (to feed into discFact())
#        aivector = sparseyc[3]      # Need to use aivector for non-callable
        cfindex = sparseyc[4]
        cfptr = sparseyc[5]
        bondcount = sparseyc[6]
        maxcol = sparseyc[7]
        maturity = sparseyc[8]    
        finalpmt = sparseyc[9]
        coupon = sparseyc[10]
        freq = sparseyc[11]
        calldate = sparseyc[12]
#        callflag = sparseyc[13]

        icall = np.array(list(range(bondcount)))[callflag]     # NB - callflag must be np.array
        bondpvfwd = pv.pvBondFromCurve(tcurve,sparsearray = sparseyc, settledate = calldate, asofsettle = True)
        pvmaturity = df.discFact(maturity[icall],tcurve) * finalpmt[icall]
        dfcall = df.discFact(calldate[icall],tcurve)
        pvmaturity = pvmaturity / dfcall    # fwd value of final payment
        pvfwdann = (bondpvfwd[icall,1] - pvmaturity) / coupon[icall]        # Get pvfwdann from bond less final. What if coupon = 0. Need to handle that externally
        datesparse = sp.csr_matrix((datedata, cfindex, cfptr), shape=(bondcount, maxcol)) 
        fwdyld = []
        for i in icall :
            cfvec = sp.find(cfsparse[i,:])[2]  # CF vector as numpy array
            datevec = sp.find(datesparse[i,:])[2]  # Date vector
            xyieldi = pv.bondYieldFromPrice_parms(bondpvfwd[i,1], cfvec=cfvec, datevec=datevec, settledate=calldate[i])            
            fwdyld.append(xyieldi)
        fwdyld = np.array(fwdyld)
        fwdyld = freq[icall]*(np.exp(fwdyld/freq[icall]) - 1)      # Convert to appropriate bond basis
       
        # fwdyld -- did it take logarithms?
        vol = vols[icall]
        expiry = (calldate[icall] - tcurve[1])/365.2  # Expiry goes from quote date of the curve
        strike = coupon[icall] / 100.                     # exercise yield (the coupon)
        ### whether to perform approx conversion vol = vol * (fwdyld + strike)/2
    
        d1 = (fwdyld - strike) / (vol * np.sqrt(expiry))
        d2 = d1 - vol * np.sqrt(expiry)
        call = dfcall * pvfwdann * (ss.norm.cdf(-d2) * strike - ss.norm.cdf(-d1) * fwdyld) * 100.
        pvcall = bondpv[icall, 1] - call
        bondpv[icall, 1] = pvcall
        bondpv[icall, 0] = pvcall - aivector[icall]

    return (bondpv)


#%%
def pvCallable_lnP(tcurve, vols, sparsenc=0, sparseyc=0., settledate=0,
                   parms=0):
    """
    Return PV & Price function for bonds, callable & non-call (using Log Normal Price
    model).

    Args
    ----
    tcurve : list
        The list contains:
            0. crvtype - (string) name to identify the type of curve
                pwcf - piece-wise constant forward (flat forwards,
                       continusuously-compounded)
                pwlz - piece-wise linear zero (cc)
                pwtf - piece-wise twisted forward (linear between breaks but
                       not connected at breaks)
            1. valuedate - curve value date. Will be zero (in which case use
                           offsets) or date (as days or date)
            2. For pwcf, pwlz, pwtf the breaks (vector, same length as rates,
               of either dates or years from today, based on whether
               valuedate is yrs (zero) or date)
            3. For pwcf, pwlz, pwtf the rates (numeric vector) - numbers like
               0.025 for 2.5% - assumed cc at 365.25 annual rate
              For dailydf a zoo object of dfs (dates, offsets, dfs)
            4. s2 spread for tax=2
            5. s3 spread for tax=3
            For 4-5, this will be made into three curves to apply to the
            three tax types:
              1 - fully taxable - breaks & rates
              2 - partially taxable - breaks & rates + s2
              - If no spreads entered (len(tcurve)=4) then only fully taxable
                and no extra processing for taxability
              - If only one spread then make s2 & s3 the same. (This will
                accomodate case where there is only partial or wholly
                tax exempt. Which it is will have to be handled outside
                of optimization.)
    vols : array
        Volatilities corresponding to each bond. vols must be same length as
        parms (a vol for every bond) - not used for non-callable bonds
    sparsenc : array, optional
        Sparse array for all bonds (to final maturity date). Default is 0.
    sparseyc : scipy.sparse matrix, optional
        Sparse array for callable bonds. Default is 0.
    settledate : numeric or date, optional
        The settlement date for the bond transactions. Default is 0.
    parms : dataframe
        For each bond, its parameters contain:
            0 coupon (numeric)
            1 Maturity date (Julian, such as 45975 for 15-nov-2025)
            2 Frequency: 2 = semi, 4 = quarterly
            3 Redemption amount - generally 100., but allows for 0. and then
              it's an annuity
            4 daycount: "A/A" for "Actual/Actual" - at the moment this is
              assumed implicitely
            5 "eomyes" for end-of-month convention of UST: going from
              28-feb-1999 to 31-aug-1999 to 29-feb-2000 to 31-aug-2000
            6 if callable, first call date. If not callable, can be not there
              or zero. If not there, will be set to zero
            7 Call flag - True if calable
            8 tax status
              - 1 = fully taxable
              - 2 = partially exempt
              - 3 = fully tax exempt
        Default is 0. Used if sparse matrices are not provided.

    Returns
    -------
    array
        2-column array: 0th col price, 1st col pv.

    Notes
    -----
        23-jul-17 TSC. Re-write to work from sparse arrays of CF & Dates,
                       and using new versions of pricefromyield
        18-jul-17 TSC use new version of PVBondFromCurve_vec which has
                      argument "asofquotedate" to force the calcuation of
                      the forward bond as of the forward date.
    """
    # First check if sparsenc (CF & date arrays) have been handed in. If not, then
    # get all the other needed data (such as coupon, calldate, callflag) from parms
    if ((type(parms) == pd.DataFrame) and (type(sparsenc) != list) ) :  # When parms are passed in and sparesearray not, then process parms
        sparsenc = pv.bondParmsToMatrix(settledate,parms)
        coupon = np.array(parms.iloc[:,0])                   # vector of coupons
        calldate = np.array(parms.iloc[:,6])                 # Assume that dates are already Julian
        callflag = np.array(parms.iloc[:,7])
        if (callflag.any()) :                                # Process callables if any
            xcalldate = calldate.copy()
            xcalldate[~callflag] = dt.YMDtoJulian(20991231)   # For non-callable bonds set call date beyond any maturity #are you changing everything to callable? do all of them have to go through the same yield functions?
            sparseyc = pv.bondParmsToMatrix(xcalldate,parms)

    bondpv = pv.pvBondFromCurve(tcurve, sparsearray = sparsenc)     # calculate the PV of all bonds to the final maturity date
                                # TSC 13-jan-24 Do NOT put in settledate - not needed after creating sparse array
    callflag = sparsenc[13]              # Need callflag before processing any callables
    aivector = sparsenc[3]               # Need aivector for original, nor forward, date
    if (callflag.any()) :                 # There are some callable bonds

        cfsparse = sparseyc[0]          # This should already be written as a sparse array
        cfdata = sparseyc[1]
        datedata = sparseyc[2]       # This is vector of data (to feed into discFact())
#        aivector = sparseyc[3]      # Need to use aivector for non-callable
        cfindex = sparseyc[4]
        cfptr = sparseyc[5]
        bondcount = sparseyc[6]
        maxcol = sparseyc[7]
        maturity = sparseyc[8]    
        finalpmt = sparseyc[9]
        coupon = sparseyc[10]
        freq = sparseyc[11]
        calldate = sparseyc[12]
#        callflag = sparseyc[13]

        icall = np.array(list(range(bondcount)))[callflag]     # NB - callflag must be np.array
        bondpvfwd = pv.pvBondFromCurve(tcurve,sparsearray = sparseyc, settledate = calldate, asofsettle = True)
        pvmaturity = df.discFact(maturity[icall],tcurve) * finalpmt[icall]
        dfcall = df.discFact(calldate[icall],tcurve)
        pvmaturity = pvmaturity / dfcall    # fwd value of final payment
        pvfwdann = (bondpvfwd[icall,1] - pvmaturity) / coupon[icall]        # Get pvfwdann from bond less final. What if coupon = 0. Need to handle that externally
        datesparse = sp.csr_matrix((datedata, cfindex, cfptr), shape=(bondcount, maxcol)) 
        fwdyld = []
        for i in icall :
            cfvec = sp.find(cfsparse[i,:])[2]  # CF vector as numpy array
            datevec = sp.find(datesparse[i,:])[2]  # Date vector
            xyieldi = pv.bondYieldFromPrice_parms(bondpvfwd[i,1], cfvec=cfvec, datevec=datevec, settledate=calldate[i])            
            fwdyld.append(xyieldi)
        fwdyld = np.array(fwdyld)
        fwdyld = freq[icall]*(np.exp(fwdyld/freq[icall]) - 1)      # Convert to appropriate bond basis
       
        # fwdyld -- did it take logarithms?
        vol = vols[icall]
        expiry = (calldate[icall] - tcurve[1])/365.2  # Expiry goes from quote date of the curve
        strike = coupon[icall] / 100.                     # exercise yield (the coupon)
        ### whether to perform approx conversion vol = vol * (fwdyld + strike)/2
    
        d1 = (fwdyld - strike) / (vol * np.sqrt(expiry))
        d2 = d1 - vol * np.sqrt(expiry)
        call = dfcall * pvfwdann * (ss.norm.cdf(-d2) * strike - ss.norm.cdf(-d1) * fwdyld) * 100.
        pvcall = bondpv[icall, 1] - call
        bondpv[icall, 1] = pvcall
        bondpv[icall, 0] = pvcall - aivector[icall]

    return (bondpv)
