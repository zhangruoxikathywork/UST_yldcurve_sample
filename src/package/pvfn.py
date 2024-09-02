##########################################
# This contains functions that calculate #
# PV for bonds. Also translates from     #
# parms to sparse CF matrix.             #
# Also has option code                   #
##########################################


# -*- coding: utf-8 -*-
"""
Created on Mon October 12 10:22:56 2015
Modified October 10 2015

@author: tcoleman
"""
import sys
import numpy as np
import pandas as pd
import scipy.stats as ss
import scipy.sparse as sp
import scipy.optimize as so
from scipy.optimize import fsolve
import discfact as df
import DateFunctions_1 as dt

sys.path.append('../../src/package')
sys.path.append('../../data')
sys.path.append('../development')

import crsp_data_processing as data_processing
import produce_inputs as inputs

# # Debugger setup
# yield_to_worst = True
# settledate = 19460330
# #settledate = dt.YMDtoJulian(settledate)[0]
# filepath = '../../data/USTMonthly.csv'  # '../../data/1916to2024_YYYYMMDD.csv'
# bonddata = data_processing.clean_crsp(filepath)
# # bonddata = data_processing.create_weight(bonddata, bill_weight=1)

# parms = inputs.read_and_process_csvdata(bonddata, settledate, 0)
# parms = inputs.filter_yield_to_worst_parms(19460330, parms, yield_to_worst=False)
# parms = inputs.create_weight(parms, wgttype=1, lam1=1, lam2=2)

#%%

def bondParmsToMatrix(settledate, parms, padj=False):
    """
    Convert bond parms into cf & date CSR (sparse) matrixes.

    Args
    ----
    settledate : int
        The settlement date for the bond transactions, be Julian date.
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
        These last 3 (callability and tax status) are not actually used in
        this function. The callable data are used in # "pvCallable_lnY" and
        the taxability is used in the cover function for optimization.
            9 bond average bid and ask price

        Example:
           eight bonds,
        0      2.25% of 15-nov-2025 non-call       20 half-yrs + 1 maturity = 21
        1      2.5% of 15-feb-2026 non-call        20 half-yrs + 1 maturity = 21
        2      2.25% of 15-nov-2025 callable 15-nov-2020  20 half-year + 1 maturity = 21
        3      2.25% of 15-nov-2025 callable 15-nov-2020  20 half-year + 1 maturity = 21 (duplicate)
        4      5% of 15-nov-2025 non-call          20 half-yrs + 1 maturity = 21
        5      5% of 15-nov-2025 callable 15-nov-2020  20 half-yrs + 1 maturity = 21
        6      4% of 15-nov-2022                   14 half-yrs + 1 maturity = 15
        7      3% of 15-feb-2020                   8 half-yrs + 1 maturity = 9
        (e.g., bond 0: pd.DataFrame([[2.25,45975,2,100.,"A/A","eomyes",0,1]]))

    Returns
    -------
    list
        A list containing the following elements in order, also the data for
        the sparse matrixes:
        0) cfsparse: cash flows (coupon + principal) made into a sparse matrix.
           This will be used later for multiplying discount factors. Dates are
           not made into matrix because dates we want as a long vector to feed
           into discFact
        1) cfdata: cash flow data as one long vector
        2) datedata: dates as one vector
        3) aivector: accrued interest vector (one entry for each bond)
        4) cfindex: column indexes for translating cfdata & datedata into
           sparse matrixes
        5) cfptr: pointer for rows (see below)
        6) bondcount: number of bonds (rows)
        7) maxcol: maximum number of columns

        And then there is a bunch of junk we need to carry around because
        we need it to value callable bonds:
        8) maturity: vector of maturity dates (one entry for each bond) we need
           to get the PVAnnuity for each callable bond, and the easiest way
           seems to be to calculate the PV of the principal and thend
           subtract from total PV
        9) finalmpt: vector of principal amounts (to get PV of principal)
        10) coupon: to get PV of annuity
        11) frequency: to convert yield from cc to bond basis
        12) calldate:
        13) callflag: True if bond is callable

        Example:
        >>> import numpy as np
        >>> from scipy.sparse import csr_matrix
        >>> row = np.array([0, 0, 1, 2, 2, 2])
        >>> col = np.array([0, 2, 2, 0, 1, 2])
        >>> data = np.array([1, 2, 3, 4, 5, 6])
        >>> csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
        array([[1, 0, 2],
            [0, 0, 3],
            [4, 5, 6]])

    Methodology
    -----------
    This function translates the parms into scipy sparse matrixes
    ("Compressed Sparse Row" csr_matrix)
    The idea is for a matrix with one row for each bond (n bonds => n rows) and
    columns equal to the max number of cf dates (max for longest bond)
    Let's say there are three bonds. The first and second have some dates that
    overlap but bond2 is longer. Bond3 has some different dates that are in
    between dates for first two.
    Let's say we are sitting on 19-feb-2016 (Julian = 42418)
    bond1      3% of 15-may-2017 non-call       3 half-yrs
    bond2      5% of 15-nov-2018 non-cal        6 half-years
    bond3      7% of 15-dec-2016 non-call        2 half-yrs
    In this case the cash flow matrix (measured in years) would look like:
      Dates    d1   d2   d3   d4   d5   d6   d7   d8
      bond1    cf   0    cf   0    cf   0    0    0
      bond2    cf   0    cf   0    cf   cf  cf    cf
      bond3     0   cf   0    cf   0    0    0    0

      Date (yrs) .24  .32  .74   .82   1.24   1.74  2.24  2.74
      Date:      5/16 6/16 11/16 12/16 5/17  11/17  5/18  11/18
      bond1 cf:  1.5   0   1,5    0    101,5   0     0     0
      bond2 cf:  2.5   0   2.5    0     2.5   2.5   2.5  102.5
      bond3 cf:   0   3.5   0   103.5    0     0     0     0

    There are zeros in this cf matrix because some CF dates do not overlap
    (bonds 1&2 vs bond3) and because one bond is longer than the others (bond3)

    It is convenient to store this as a sparse matrix (csr_matrix) because
    then the date data can be stored as one long vector which can then be
    processed (in one go) by the vectorized discFact function, then expanded
    into a sparse matrix.
    In this case the date vector would be
      Date (yrs) .24  .32  .74   .82   1.24   1.74  2.24  2.74
      Date:      5/16 6/16 11/16 12/16 5/17  11/17  5/18  11/18

    The data for the sparse matrixes are:
      1) cfdata      data for (sparse) array of cash flows: one long vector
                     which unravelled (or exapanded) into a sparse matrix
                     using the indexes below
      2) datedata    data for (sparse) array of dates for the cfs
                     - again long vector
      3) aivector    vector of accrued interest (length n = number of bonds)
      4) col_ind     column indexes (will be same as cfindex)
      5) row_ind     row indexes
      here datematrix[row_ind[k],col_ind[k]] = datedata[k]
      6) taxvector   vector of tax status (length n = number of bonds) since
                     need to process bonds with different curves according
                     to tax status
    or
      4) cfindex     col indexes of bond cfs & dates (same as col_index above)
      5) cfindptr    pointer to locations
      here column indexes for row i are stored in
      cfindex[cfindptr[i]:cfindptr[i+1]]
      and data are in cfdata[indptr[i]:indptr[i+1]]
      (this is apparently the standard CSR representation)
    see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html

    CSR sparse matrices can be created using either row / column indeces,
    or index pointer / indices.
    Using the sparse matrixes is important - for 31-dec-1932 the sparse
    data are 1427 elements, while the dense matrix is 33x667 or 22,011
    elements.Reduction by factor of 15. (Because of the consol - assumed
    to mature 20990401 - 667 cols)

    EXAMPLE
    >>> row = np.array([0, 0, 1, 2, 2, 2])
    >>> col = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
    array([[1, 0, 2],
          [0, 0, 3],
          [4, 5, 6]])

    >>> indptr = np.array([0, 2, 3, 6])
    >>> indices = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()
    array([[1, 0, 2],
          [0, 0, 3],
          [4, 5, 6]])

    Notes
    -----
        The last 3 (callability and tax status) of parms are not used in the
        "pvBondFromCurve" function. The callable data are used in
        "pvCallable_lnY" and the taxability is used in the cover
        function for optimization.

        ??? Include argumet "taxcurve" so that we can skip tax processing if
        there are no partially or fully tax-exempt bonds (after abut the 1940s)
    """
    

    
    # iin contains the number of days in the months */
    iin = np.array( ((31,28,31,30,31,30,31,31,30,31,30,31),
            (31,29,31,30,31,30,31,31,30,31,30,31)) )

    # if yield_to_worst == True:
    #     # If current price < 100, leave maturity date as maturity date
    #     # If current price > 100, replace maturity date with first call data
    #     # If days to first call date < 365 then discard

    #     # Check conditions based on the current price and days to first call
    #     parms['days_to_first_call'] = parms['First Call Date at Time of Issue'] - settledate
    #     parms = parms[(parms['CallFlag'] == False) | ((parms['CallFlag'] == True) & (parms['days_to_first_call'] >= 365))]
    #     parms.loc[(parms['Price'] >= 100) & (parms['CallFlag'] == True), 'Maturity Date at time of Issue'] = parms['First Call Date at Time of Issue']
    #     parms.drop('days_to_first_call', axis=1, inplace=True)
    #     parms = parms.reset_index(drop=True)
    #     parms['CallFlag'] = False


    # Create empty lists (which will later be converted to np.array) to hold the output data
    cfdata = []
    datedata = []
    aivector = []
    cfindex = []
    cfptr = [0]
    princfdata = []
    prinindex = []
    prinptr = [0]
    prindate = []
    #    taxvector = []
    #    maturity = False
    maxcol = 0             # will hold the maxium number of cf & date entries (max no. of cols)
    bondcount = len(parms)        # counter for number of bonds

    frequency = np.array(parms.iloc[:,2])
    coupon = np.array(parms.iloc[:,0])
    xcoupon = coupon / np.maximum(frequency,1)
    maturity = np.array(parms.iloc[:,1])
    lmaturity = maturity.tolist()
    finalpmt = np.array(parms.iloc[:,3])
    eom = np.array(parms.iloc[:,5])
    calldate = np.array(parms.iloc[:,6])
    callflag = np.array(parms.iloc[:,7])
    # Check if prices column exists. If not, set prices to 100
    if 'Prices' in parms.columns:
        prices = np.array(parms['Price'])
    else:
        prices = coupon.copy()* 0. + 100.   # This set to 100, same length as all others
        


#    if (parms.shape[1] == 8) :                     # Test for taxability (if any fully or partially tax exempt) 
#        x1 = pd.Series.tolist(parms.iloc[:,7])
#        if ((2 in x1) or (3 in x1)) :              # Yes there are some fully or partially tax exempt
#            taxvector = parms.iloc[:,7]            # Put tax status into taxvector
#            taxflag = True

    if hasattr(settledate,"__len__")  :            # if settledate is not a list then make it an array
        if (len(settledate) == 1):                 # If it is a singleton then make it a vector
            xsettle = np.tile(settledate,bondcount)
        else :                                     # Otherwise check if it's the right length
            if (len(settledate) != bondcount) :
                raise ValueError("error in bondParmsToMatrix: len of settledate not same as number of bonds")
            else :                                 # If it is len > 1 and right length, then make it array
                xsettle = np.array(settledate)
    else:
        xsettle = np.tile(settledate,bondcount)

    eomdiff = dt.DateDiff(xsettle,maturity)
    xsettleymd = dt.JuliantoYMD(xsettle)
    leap = dt.IsALeapYear(xsettleymd[0])
    leap = leap.astype(int)
    daysinmth = iin[leap,(xsettleymd[1].astype(int)-1)]
    xmthdiff = 12.*eomdiff[0] + eomdiff[1] + eomdiff[2]/daysinmth    # Settle to maturity in "months"
    xmthper = 12. / np.maximum(frequency,1)          # The number of months per coupon period. Use max to handle bills
    x1 = xmthdiff / xmthper   # Fractional number of "periods" to settle date
    xperprior = np.ceil(x1)  # Number of "periods" to prior coupon paymen (payment before settle)
    xpernext = np.floor(x1)   # Number of "period" to next payment date
    xperprior = xperprior.astype(int)
    xpernext = xpernext.astype(int)

    for j in range(len(xpernext)) :   # This loops over bonds, for coupons only (maturities done in vector above)
                                      # Is it worth the complication of doing bonds in parallel?
        if (maturity[j] >= xsettle[j]) :

            # If the bond maturity is at or beyond settle date, do this
            if ( (xcoupon[j] > 0) and (frequency[j] > 0) ) :   # This is a coupon bond - not a bill
                x1 = np.array(range(0,(xperprior[j]+1))) / frequency[j]     # vector of years back from maturity date for coupons
                x2 = np.modf(x1)                                         # 0th array are fractions, 1st is yrs
                x1 = x2[0]*12                                            # no. months
                x2 = x2[1]                                               # no. yrs
                xcpdates = dt.CalAdd(maturity[j],add="sub",nmonth=x1,nyear=x2,eom=eom[j])  # creates vector of coupon dates
                                           # Create vector of yrs & vector of mnths because
                            # CalAdd is more efficient if don't have to go back
                            # over lots of months. If current mth - nmonth < 0 (say going back 6
                            # mths from March) then CalAdd has to subtract a year and add 12 months.
                            # (If go back 36 mths from March have to step back year-by-year until
                            # mths becomes positive). This way minimizes the looping required
                x1 = len(xcpdates)
                x2 = xcoupon[j]*(xsettle[j] - xcpdates[(x1-1)])/(xcpdates[(x1-2)]-xcpdates[(x1-1)])   # AI factor, discounted back to today
                aivector.append(x2)
                datedata.extend(np.ndarray.tolist(xcpdates[:(x1-1)]))          # The cf dates for this bond coupons
                cfdata.extend([xcoupon[j]]*(x1-1))             # This makes a list of xcoupon repeated (x1-1) times
            else:
                aivector.append(0.)
                x1 = 1           # This just sets the counter correctly for no coupos
# Put in maturity & final payment for all bonds
            datedata.append(lmaturity[j])                # cf date for maturity
            cfdata.extend([finalpmt[j]])                     # Append final payment
            cfindex.extend(list(range(x1)))    # The indexes of the coupon CFs + maturity
            cfptr.append(x1+cfptr[-1])
            maxcol = max(maxcol,x1)                   # This is 1+#coup because there will be the maturity to do on
            # Now do everything duplicate for maturity dates
            prindate.append(lmaturity[j])                # cf date for maturity
            princfdata.extend([finalpmt[j]])                     # Append final payment
            prinindex.extend(list(range(1)))    # The indexes of the coupon CFs + maturity
            prinptr.append(1+prinptr[-1])
#            maxcol = max(maxcol,x1)                   # This is 1+#coup because there will be the maturity to do on

        else :   # If the bond maturity is less than (earlier than) settle date, then there will be no entries for this row
            cfptr.append(cfptr[-1])              # This should make the pointer do nothing for this bond (this row of the CF matrix)
            aivector.append(0.)
# Now convert everything to numpy arrays
    cfdata = np.array(cfdata)
    datedata = np.array(datedata)
    aivector = np.array(aivector)
    cfindex = np.array(cfindex)
    cfptr = np.array(cfptr)
#    taxvector = np.array(taxvector)
    maturity = np.array(maturity)
    finalpmt = np.array(finalpmt)
    cfsparse = sp.csr_matrix((cfdata, cfindex, cfptr), shape=(bondcount, maxcol))   # Create the sparse matrix for cf here - only need to do this once

    return_list = [cfsparse, cfdata, datedata, aivector, cfindex, cfptr, bondcount,
                   maxcol, maturity, finalpmt, coupon, frequency, calldate,
                   callflag]
    
    if padj == True:
        # Principal cf sparse array - Only for bonds with Price < 100, Principal Amount * (Principal - Price)/100
        pricecheck = prices < 100
        cfp = np.zeros_like(prices)
        cfp[pricecheck] = finalpmt[pricecheck] * (finalpmt[pricecheck] - prices[pricecheck]) / 100
        cfpadj = sp.csr_matrix((cfp, prinindex, prinptr), shape=(sum(pricecheck), 1))   # Subject to change
        return_list.append(cfpadj)
    
    return (return_list)


#%% ------ New PV Function of Yield or Curve - works from parms or sparse matrixes ------

def pvBondFromCurve(curve, sparsearray=0, settledate=0, parms=0,
                    asofsettle=False, padj=False, padjparm=0):  # also need padj value
    """
    Calculate PV & Price function for vector of (non-callable) bonds.

    Args
    ----
    curve : list
        The yield curve data, which can be a curve (breaks, rates, etc)
        or a single number (YTM).
        ??? Include argument "taxcurve" so that we can
        skip tax processing if there are no partially or fully tax-exempt
        bonds (after abut the 1940s)
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

            Also take in "tcurve" to handle multiple tax status.
            tcurve has structure (0-3 same as curve, 4-5 added)
            type         SD          breaks  rates    s2                     s3
            "pwcf" etc   0 or date   vector  vector   spread for tax=2  spread for tax=3
            This will be made into three curves to apply to the three tax
            types:
              1 - fully taxable - breaks & rates
              2 - partially taxable - breaks & rates + s2
              - If no spreads entered (len(tcurve)=4) then only fully taxable
                and no extra processing for taxability
              - If only one spread then make s2 & s3 the same. (This will
                accomodate case where there is only partial or wholly
                tax exempt. Which it is will have to be handled outside
                of optimization.)
        Example:
        - curvepwlz = ['pwlz',settledate,breaks,ratescc]
        - tcurvepwcf = ['pwcf',settledate,breaks,ratescc,0.001,0.002]
          # Version with spreads for taxability (type 2 and type 3)
    sparsearray : list, optional if parms is passed
        Sparce matrix, calculated from bondParmsToMatrix function.
        Default is 0.
        The list contains:
        0) cfsparse: cash flows (coupon + principal) made into a sparse matrix.
           This will be used later for multiplying discount factors. Dates are
           not made into matrix because dates we want as a long vector to feed
           into discFact
        1) cfdata: cash flow data as one long vector
        2) datedata: dates as one vector
        3) aivector: accrued interest vector (one entry for each bond)
        4) cfindex: column indexes for translating cfdata & datedata into
           sparse matrixes
        5) cfptr: pointer for rows (see below)
        6) bondcount: number of bonds (rows)
        7) maxcol: maximum number of columns
    settledate : int
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
    asofsettle : bool, optional
        If False, calculates PV as of the quote date of the curve; if True,
        calculates PV as of the settledate. Default is False.
        The reason for this is so that PVCallable does not have to worry about
        the tax status of individual bonds (does not have to worry about the
        curve or call discFact)

    Returns
    -------
    array
        2-element arrays: 0th bond price, 1st bond pv

    Methodology
    -----------
    This function returns PV using sparse matrixes, will also work with parms
    if passed in parms.

    EXAMPLE
    >>> Five bonds,
          2.25% of 15-nov-2025 non-call
          2.5% of 15-feb-2026 non-call
          2.25% of 15-nov-2025 callable 15-nov-2020
          5% of 15-nov-2025 non-call
          5% of 15-nov-2025 callable 15-nov-2020
    For pwcf 5% sab with 365.25-day year (rates quoted at cc, 365.25-day year)
    (PV from spread-sheet, AI from HP17B, price as difference)
              2.25% of 15-nov-25   2.5% 15-feb-26    2.25% 15-nov-25  5% 15-nov-25  5% 15-nov-25
                                                    call 15-nov-20                 call 15-nov-20
    Price     78.994385             80.526207       76.235              99.9850
    PV        79.587791             80.55368        76.829             101.303716   99.7997 (2.188% Pvol, 1.504)
    AI         0.593407              0.027473 Call   2.759 (10% Pvol)    1.318681   99.8067 (10% Yvol, 1.497)
    """
# This is round-about but want to check if a settledate has been passed in
    settflag = False
    if (hasattr(settledate,"__len__")) :
        settflag = True
    elif (settledate > 0) :
        settflag = True

# Check if sparsearray (CF & date arrays) have been handed in
    if ((type(parms) == pd.DataFrame) and (type(sparsearray) != list) ) :  # When parms are passed in and sparesearray not, then process parms
        sparsearray = bondParmsToMatrix(settledate, parms, padj)
 # We should not feed in settledate with CF matrix UNLESS we need to FV as of settledate (and set flag asofsettle)
    elif ((type(sparsearray == list)) and (settflag and not(asofsettle)) ) : 
        raise ValueError("error in pvBondFromCurve: you are putting in a settledate with CF matrix - dangerous")

    cfsparse = sparsearray[0]          # This should already be written as a sparse array
    cfdata = sparsearray[1]
    datedata = sparsearray[2]       # This is vector of data (to feed into discFact())
    aivector = sparsearray[3]
    cfindex = sparsearray[4]
    cfptr = sparsearray[5]
    bondcount = sparsearray[6]
    maxcol = sparsearray[7]
    
    if padj == True:
        cfpadj = sparsearray[14]


#    taxcurve = False

    if (asofsettle) :                  # We need settle date because we are going to forward value
        if hasattr(settledate,"__len__")  :            # if settledate is not a list then make it an array
            if (len(settledate) == 1):                 # If it is a singleton then make it a vector
                xsettle = np.tile(settledate,bondcount)
            else :                                     # Otherwise check if it's the right length
                if (len(settledate) != bondcount) :
                    raise ValueError("error in PVBondFromCurve_vec: len of settledate not same as number of bonds")
                else :                                 # If it is len > 1 and right length, then make it array
                    xsettle = np.array(settledate)
        else:
            xsettle = np.tile(settledate,bondcount)

    dfdata = df.discFact(datedata,curve)      # dfs for maturity dates
    dfsparse = sp.csr_matrix((dfdata, cfindex, cfptr), shape=(bondcount, maxcol))      # dfs for settle dates
    pvsparse = cfsparse.multiply(dfsparse)
    
    ## Principal adjustment
    if padj == True:
        pvsparse = pvsparse + padjparm * (cfpadj.multiply(dfsparse))  
    
    pv = pvsparse.sum(1)
    pv = np.ravel(pv)
    price = pv - aivector
    if (asofsettle) :
        dfsettle = df.discFact(xsettle, curve)
        pv = pv / dfsettle
        price = price / dfsettle

    return (np.array([price,pv]).T)



# #%% ------ Price of bond from yield, and yield from price to work from CF matrix ------

# def bondPriceFromYield_matrix(xyield, cfvec, datevec, settledate = 0., dirtyprice = 0.):
#     "Price function for single bond using (non-sparse) cf & date vectors"

# # 23-jul-17 TSC new versions to work from (non-sparse) cf & date vectors

#     pv = np.dot(np.exp(-xyield*(datevec-settledate)/365.25),cfvec) # why is it continuously compounding here? is it because the input of so.brentq has to be continuous?
#     diff = pv - dirtyprice
#     return(diff)

# def bondYieldFromPrice_matrix(dirtyprice, cfvec, datevec, settledate = 0.):
#     "Yield from price for single bond from cf & date vectors - uses scipy.optimize.brentq"

#     xyield = so.brentq(bondPriceFromYield_matrix, -0.03, 0.5, args=(cfvec, datevec, settledate , dirtyprice ), xtol=1e-12, rtol=9.0e-16, maxiter=100, full_output=False, disp=True)

#     return(xyield)


#%% ------ Price of bond from yield, and yield from price to work from cfvec or parms, then use functions above ------

def bondPriceFromYield_parms(xyield, cfvec=0, datevec=0, settledate=0,
                             dirtyprice=0, parms=0, freq=2, parmflag=False, padj=False):
    """
    Calculate price for single bond using (non-sparse) cf & date vectors or
    parms - yield bond basis for parms.

    Args
    ----
    xyield : number
        Bond yield
    cfvec : vector
        Cash flow. Default is 0.
    datevec : vector
        Dates. Default is 0.
    settledate : int
        The settlement date for the bond transactions. Default is 0.
    dirtyprice : number
        Bond dirty price. Default is 0.
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
        Default is 0.
    freq : number
        Payment frequency. Default is 2 (semi-annual).
    parmflag : boolean
        Check if parms are passed in. Default is False, will automaticall turn
        to True if parms are successfully passed in.

    Returns
    -------
    number
        Difference between bond PV and dirty price

    NOTES
        7-oct-17 TSC There seems to be penalty (6x on a test) for taking
        in parameters, but not penalty when pass in CF vectors
    """
# Check if cfvec has been handed in
    if (parmflag or ((type(parms) == pd.DataFrame) and (type(cfvec) != np.ndarray)) ) :  # When parms are passed in and cfvec not, then process parms
        parmflag = True
        sparsearray = bondParmsToMatrix(settledate, parms, padj)  
        cfvec = sparsearray[1]       # pick out the only two elements needed
        datevec = sparsearray[2]
        freq = int(parms.iloc[0, 2])        # must be converted to int from pandas DataFrame
    if (freq == 0):
        freq = 2                 # Bills have freq = 0 - make them sab
    xyield = freq*np.log(1+xyield/freq)     # This converts from bond basis (eq sab) to cc
    pv = np.dot(np.exp(-xyield*(datevec-settledate)/365.25),cfvec) # why is it continuously 
                                            #compounding here? is it because the input of 
                                            #so.brentq has to be continuous?
    diff = pv - dirtyprice
    return (diff)


def bondYieldFromPrice_parms(dirtyprice, cfvec=0, datevec=0, settledate=0,
                             parms=0, freq=2, parmflag=False, padj=False):
    """
    Calculate yield from price for single bond from cf & date or parms
    - yield bond basis for parms - uses scipy.optimize.brentq.

    Args
    ----
    dirtyprice : number
        Bond dirty price.
    cfvec : vector
        Cash flow. Default is 0.
    datevec : vector
        Dates. Default is 0.
    settledate : int
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
        Default is 0.
    freq : number
        Payment frequency. Default is 2 (semi-annual).
    parmflag : boolean
        Check if parms are passed in. Default is False, will automaticall turn
        to True if parms are successfully passed in.

    Returns
    -------
    number
        Bond yield
    """
# Check if cfvec has been handed in
    if (parmflag or ((type(parms) == pd.DataFrame) and (type(cfvec) != np.ndarray)) ) :  # When parms are passed in and cfvec not, then process parms
        parmflag = True
        sparsearray = bondParmsToMatrix(settledate, parms, padj)  
        cfvec = sparsearray[1]       # pick out the only two elements needed
        datevec = sparsearray[2]
        freq = int(parms.iloc[0, 2])  # int(parms[2]), must be converted to int from pandas DataFrame
    if (freq == 0):
        freq = 2                 # Bills have freq = 0 - make them sab
    # Need to set parms=0 and parmflag=False in the call to bondPriceFromYield_parms to force no parms
    # In this case, no penalty relative to calling the _matrix version

    # Debug: Print the values at the endpoints
    # f_a = bondPriceFromYield_parms(-0.1, cfvec, datevec, settledate, dirtyprice, 0., freq, False, padj)
    # f_b = bondPriceFromYield_parms(0.5, cfvec, datevec, settledate, dirtyprice, 0., freq, False, padj)
    # print(f"Function value at -0.1: {f_a}")
    # print(f"Function value at 0.5: {f_b}")

    try:
        xyield = so.brentq(bondPriceFromYield_parms, -0.03, 0.5, args=(cfvec, datevec, 
                        settledate , dirtyprice, 0., freq, False, padj), xtol=1e-12, rtol=9.0e-16, 
                        maxiter=100, full_output=False, disp=True)  # This one uses _parms
    except ValueError as e:
        print(f"Error: {e}")
        return None

#    if (parmflag):
#        xyield = freq*(np.exp(xyield/freq) - 1.)     # This converts from cc to bond basis (eq sab)
#    xyield = freq*(np.exp(xyield/freq) - 1.)     # This converts from cc to bond basis (eq sab)
    return (xyield)

# ------ Bond from yield and yield from bond for callable bonds ------
# 7-oct-17 TSC
#  - Works with either CF vectors or parms
#  - If parms then create CF vectors and set callflag as appropriate
#  - If read in CF vectors then check if callflag 
#    - if False then call bondPriceFromYield
#    - if True then do call option calculation


def bondPriceFromYield_callable(xyield, cfvecnc=0, datevecnc=0, settledate=0,
                                cfvecyc=0, datevecyc=0, calldate=0,
                                dirtyprice=0, freq=2, vol=0,
                                callflag=False, parms=0, parmflag=False, padj=False):
    """
    Calculate price for single callable bond using (non-sparse) cf & date
    vectors or parms - yield bond basis for parms.

    Args
    ----
    xyield : number
        Bond yield
    cfvecnc : vector
        Cash flow for non-callable bond. Default is 0.
    datevecnc : vector
        Dates for non-callable bond. Default is 0.
    settledate : int
        The settlement date for the bond transactions. Default is 0.
    cfvecyc : vector
        Cash flow for callable bond. Default is 0.
    datevecyc : vector
        Dates for callable bond. Default is 0.
    dirtyprice : number
        Bond dirty price. Default is 0.
    freq : number
        Payment frequency. Default is 2 (semi-annual).
    vol : number
        Volatilities corresponding to a bond. Default is 0.
    callflag : boolean
        Whether the bond is callable. Default is False, automatically become
        True if parms[7] is True.
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
        Default is 0.
    parmflag: boolean
        Check if parms are passed in. Default is False, will automaticall turn
        to True if parms are successfully passed in.

    Returns
    -------
    number
        Difference between callable bond PV and dirty price
    """
# Check if cfvec has been handed in
    if (parmflag or ((type(parms) == pd.DataFrame) and (type(cfvecnc) != np.ndarray)) ) :  # When parms are passed in and cfvec not, then process parms
        parmflag = True
        sparsearray = bondParmsToMatrix(settledate, parms, padj)  
        cfvecnc = sparsearray[1]       # pick out the only two elements needed
        datevecnc = sparsearray[2]
        freq = int(parms.iloc[0, 2])        # must be converted to int from pandas DataFrame
        callflag = bool(np.array(parms.iloc[0, 7]))      # I don't understand - I want to coerce to boolean but first have to convert
                                                # from some kind of Pandas thing to np.array (single element) then boolean
        if (callflag):
            calldate = int(parms.iloc[0, 6])                 # Assume that dates are already Julian
            sparsearray = bondParmsToMatrix(calldate, parms, padj)  
            cfvecyc = sparsearray[1]       # pick out the only two elements needed
            datevecyc = sparsearray[2]

    if (freq == 0):
        freq = 2    
    xyield = freq*np.log(1+xyield/freq)     # This converts from bond basis (eq sab) to cc
    pvnc = np.dot(np.exp(-xyield*(datevecnc-settledate)/365.25),cfvecnc) # why is it continuously compounding here? is it because the input of so.brentq has to be continuous?
    if (callflag or (type(cfvecyc) == np.ndarray)):    # process for callable
        if (np.size(cfvecyc) > 0):             # Do this double-check because may pass in np arrays of length zero for non-call bonds
    # Need many elements for valuing call option by LNY model:
    # - pv forward annuity
    # - df call date
    # - ytm for forward bond. But since this is flat ytm, fwd ytm = ytm (which is input)
            pvyc = np.dot(np.exp(-xyield*(datevecyc-calldate)/365.25),cfvecyc)  # PV of fwd bond
            pvmaturity = np.exp(-xyield*(datevecyc[-1]-calldate)/365.25)*cfvecyc[-1]  # This should be PV of principal
            pvfwdann = (pvyc - pvmaturity) / (freq * cfvecyc[0])      # This ASSUMES the the first element of the CF vec is the periodic (not annual) coupon
            dfcall = np.exp(-xyield*(calldate-settledate)/365.25)
            fwdyld = float(xyield)
            fwdyld = freq*(np.exp(fwdyld/freq) - 1)      # Convert to appropriate bond basis
            expiry=(calldate - settledate )/365.2        # Expiry goes from quote date of the curve  QUOTE DATE
            strike=cfvecnc[1]*freq / 100.                     # exercise yield (the coupon) - ASSUMES first element
            d1=(np.log(fwdyld/strike) + 0.5*(vol**2)*expiry)/(vol*np.sqrt(expiry))
            d2=d1-vol*np.sqrt(expiry)
            call = dfcall * pvfwdann * (ss.norm.cdf(-d2)*strike - ss.norm.cdf(-d1)*fwdyld) * 100.
    #        aicallable = bondpv[x6,1] - bondpv[x6,0]
            pvnc = pvnc - call
    diff = pvnc - dirtyprice
    return (diff)


def bondYieldFromPrice_callable(dirtyprice, cfvecnc=0, datevecnc=0,
                                settledate=0, cfvecyc=0, datevecyc=0,
                                calldate=0, freq=2, vol=0., callflag=False,
                                parms=0, parmflag=False, padj=False):
    """
    Calculate yield from price for single callable bond from cf & date or parms
    - yield bond basis for parms - uses scipy.optimize.brentq.

    Args
    ----
    dirtyprice : number
        Callable bond dirty price
    cfvecnc : vector
        Cash flow for non-callable bond. Default is 0.
    datevecnc : vector
        Dates for non-callable bond. Default is 0.
    settledate : int
        The settlement date for the bond transactions. Default is 0.
    cfvecyc : vector
        Cash flow for callable bond. Default is 0.
    datevecyc : vector
        Dates for callable bond. Default is 0.
    freq : number
        Payment frequency. Default is 2 (semi-annual).
    vol : number
        Volatilities corresponding to a bond. Default is 0.
    callflag : boolean
        Whether the bond is callable. Default is False, automatically become
        True if parms[7] is True.
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
            7 Call flag - True if callable
            8 tax status
              - 1 = fully taxable
              - 2 = partially exempt
              - 3 = fully tax exempt
        Default is 0.
    parmflag : boolean
        Check if parms are passed in. Default is False, will automaticall turn
        to True if parms are successfully passed in.

    Returns
    -------
    number
        Callable bond yield
    """
# Check if cfvec has been handed in
    if (parmflag or ((type(parms) == pd.DataFrame) and (type(cfvecnc) != np.ndarray)) ) :  # When parms are passed in and cfvec not, then process parms
        parmflag = True
        sparsearray = bondParmsToMatrix(settledate, parms, padj)  
        cfvecnc = sparsearray[1]       # pick out the only two elements needed
        datevecnc = sparsearray[2]
        freq = int(parms.iloc[0, 2])  # int(parms[2]), must be converted to int from pandas DataFrame
        callflag = bool(np.array(parms.iloc[0, 7])) # bool(np.array(parms[7]))      # I don't understand - I want to coerce to boolean but first have to convert
                                                # from some kind of Pandas thing to np.array (single element) then boolean
        if (callflag):
            calldate = int(parms.iloc[0, 6])                 # Assume that dates are already Julian
            sparsearray = bondParmsToMatrix(calldate, parms, padj) 
            cfvecyc = sparsearray[1]       # pick out the only two elements needed
            datevecyc = sparsearray[2]

    if (freq == 0):
        freq = 2                 # Bills have freq = 0 - make them sab
    # Need to set parms=0 and parmflag=False in the call to bondPriceFromYield_parms to force no parms
    # In this case, no penalty relative to calling the _matrix version
    # xyield = so.brentq(bondPriceFromYield_callable, 0.00001, 0.5, args=(cfvecnc, datevecnc,  # error: ValueError: f(a) and f(b) must have different signs
    #                 settledate, cfvecyc, datevecyc, calldate, dirtyprice, 
    #                 freq, vol, callflag, 0., False ), 
    #                 xtol=1e-12, rtol=9.0e-16, maxiter=100, full_output=False, disp=True)  
                                # This one uses _parms ???

    # Initial guess for the yield
    initial_guess = 0.01

    # Solve for the yield using fsolve
    xyield = fsolve(bondPriceFromYield_callable, initial_guess, args=(cfvecnc, datevecnc, settledate, cfvecyc,
                                                                       datevecyc, calldate, dirtyprice, freq,
                                                                       vol, callflag, 0., False, padj))[0]

#    xyield = freq*(np.exp(xyield/freq) - 1.)     # This converts from cc to bond basis (eq sab)

    return (xyield)



#%%


def pvCallable(tcurve, vols, opt_type="LnY", sparsenc=0, sparseyc=0., settledate=0,
                   parms=0, padj=False, padjparm=0):
    """
    Return PV & Price function for bonds, callable & non-call (using ln Yield
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
        sparsenc = bondParmsToMatrix(settledate,parms,padj)
        coupon = np.array(parms.iloc[:,0])                   # vector of coupons
        calldate = np.array(parms.iloc[:,6])                 # Assume that dates are already Julian
        callflag = np.array(parms.iloc[:,7])
        if (callflag.any()) :                                # Process callables if any
            xcalldate = calldate.copy()
            xcalldate[~callflag] = dt.YMDtoJulian(20991231)   # For non-callable bonds set call date beyond any maturity #are you changing everything to callable? do all of them have to go through the same yield functions?
            sparseyc = bondParmsToMatrix(xcalldate,parms,padj)

    bondpv = pvBondFromCurve(tcurve, sparsearray=sparsenc, padj=padj, padjparm=padjparm)     # calculate the PV of all bonds to the final maturity date
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
        icall = np.array(list(range(bondcount)))[callflag == 1]  # NB - callflag must be np.array. 
            # Including the column names of all the callable bonds. All np.arrays should be in the same shape of [icall]

        if opt_type == "LnY" or opt_type == "NormY" or opt_type == "SqrtY":
            bondpvfwd = pvBondFromCurve(tcurve, sparsearray=sparseyc, settledate=calldate, asofsettle=True, padj=padj, padjparm=padjparm) # from T1 to Tm
            pvmaturity = df.discFact(maturity[icall],tcurve) * finalpmt[icall]
            dfcall = df.discFact(calldate[icall],tcurve)  # From t to T1
            pvmaturity = pvmaturity / dfcall    # fwd value of final payment. From T1 to Tm
            pvfwdann = (bondpvfwd[icall,1] - pvmaturity) / coupon[icall]  # Get pvfwdann from bond less final. What if coupon = 0. Need to handle that externally
            datesparse = sp.csr_matrix((datedata, cfindex, cfptr), shape=(bondcount, maxcol)) 
            vol = np.full(bondcount, vols)[icall]
          
            fwdyld = []  # nu at T1 of yield
            for i in icall:
                cfvec = sp.find(cfsparse[i,:])[2]  # CF vector as numpy array
                datevec = sp.find(datesparse[i,:])[2]  # Date vector
                xyieldi = bondYieldFromPrice_parms(bondpvfwd[i,1], cfvec=cfvec, datevec=datevec, settledate=calldate[i], padj=padj)            
                fwdyld.append(xyieldi)
            fwdyld = np.array(fwdyld)
            fwdyld = freq[icall]*(np.exp(fwdyld/freq[icall]) - 1)      # Convert to appropriate bond basis, strike is also annualized
            
            expiry=(calldate[icall] - tcurve[1])/365.2        # Expiry goes from quote date of the curve
            strike=coupon[icall] / 100.                     # exercise yield (the coupon)
            
            if opt_type == "LnY": 
                d1 = (np.log(fwdyld/strike) + 0.5*(vol**2)*expiry) / (vol*np.sqrt(expiry))
                d2 = d1-vol*np.sqrt(expiry)
                call = dfcall * pvfwdann * (ss.norm.cdf(-d2)*strike - ss.norm.cdf(-d1)*fwdyld) * 100.  # Put on yield
                
            elif opt_type == "NormY": 
                sigmaT = vol*np.sqrt(expiry)  # vol column here should start at a different number
                d = (fwdyld-strike) / sigmaT
                call = dfcall * pvfwdann * sigmaT * (-d*ss.norm.cdf(-d) + ss.norm.pdf(d)) * 100.  # Put on yield
                
            elif opt_type == "SqrtY":
                
                
                
                d2=d1-vol*np.sqrt(expiry)
                call = dfcall * pvfwdann * (ss.norm.cdf(-d2)*strike - ss.norm.cdf(-d1)*fwdyld) * 100.
                pass
            
        elif opt_type == "LnP": 
            bondpvfwd = pvBondFromCurve(tcurve, sparsearray=sparseyc, settledate=calldate, asofsettle=True, padj=padj, padjparm=padjparm)
            vol = np.full(bondcount, vols)[icall]
            dfcall = df.discFact(calldate[icall],tcurve)
            strike = finalpmt[icall]
            expiry=(calldate[icall] - tcurve[1])/365.2
            
            d1 = (np.log(bondpvfwd[icall,1]/strike) + 0.5*(vol**2)*expiry) / (vol*np.sqrt(expiry))
            d2 = d1 - vol*np.sqrt(expiry)
            call = dfcall * (ss.norm.cdf(d1)*bondpvfwd[icall,1] - ss.norm.cdf(d2)*strike)  # Call on bond
            
        else: 
            print(opt_type)
            raise ValueError("The value of opt_type must be LnY, NormY, SqrtY or LnP")
            
#        aicallable = bondpv[x6,1] - bondpv[x6,0]
        pvcall = bondpv[icall,1] - call
        bondpv[icall,1] = pvcall
        bondpv[icall,0] = pvcall - aivector[icall]
        
    return (bondpv)


def pvCallable_lnY(tcurve, vols, sparsenc=0, sparseyc=0., settledate=0,
                   parms=0, padj=False, padjparm=0):
    """
    Return PV & Price function for bonds, callable & non-call (using ln Yield
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
        sparsenc = bondParmsToMatrix(settledate,parms,padj)
        coupon = np.array(parms.iloc[:,0])                   # vector of coupons
        calldate = np.array(parms.iloc[:,6])                 # Assume that dates are already Julian
        callflag = np.array(parms.iloc[:,7])
        if (callflag.any()) :                                # Process callables if any
            xcalldate = calldate.copy()
            xcalldate[~callflag] = dt.YMDtoJulian(20991231)   # For non-callable bonds set call date beyond any maturity #are you changing everything to callable? do all of them have to go through the same yield functions?
            sparseyc = bondParmsToMatrix(xcalldate,parms,padj)

    bondpv = pvBondFromCurve(tcurve, sparsearray=sparsenc, padj=padj, padjparm=padjparm)     # calculate the PV of all bonds to the final maturity date
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

        icall = np.array(list(range(bondcount)))[callflag == 1]     # NB - callflag must be np.array
        bondpvfwd = pvBondFromCurve(tcurve, sparsearray=sparseyc, settledate=calldate, asofsettle=True, padj=padj, padjparm=padjparm)
        pvmaturity = df.discFact(maturity[icall],tcurve) * finalpmt[icall]
        dfcall = df.discFact(calldate[icall],tcurve)
        pvmaturity = pvmaturity / dfcall    # fwd value of final payment
        pvfwdann = (bondpvfwd[icall,1] - pvmaturity) / coupon[icall]        # Get pvfwdann from bond less final. What if coupon = 0. Need to handle that externally
        datesparse = sp.csr_matrix((datedata, cfindex, cfptr), shape=(bondcount, maxcol)) 
        fwdyld = []
        for i in icall :
            cfvec = sp.find(cfsparse[i,:])[2]  # CF vector as numpy array
            datevec = sp.find(datesparse[i,:])[2]  # Date vector
            xyieldi = bondYieldFromPrice_parms(bondpvfwd[i,1], cfvec=cfvec, datevec=datevec, settledate=calldate[i], padj=padj)
            if xyieldi is None:
                raise ValueError("fwdyld is None, cannot perform division")
            fwdyld.append(xyieldi)
            
        fwdyld = np.array(fwdyld)
        fwdyld = freq[icall]*(np.exp(fwdyld/freq[icall]) - 1)      # Convert to appropriate bond basis
        
        

        vol = vols[icall]
        expiry=(calldate[icall] - tcurve[1])/365.2        # Expiry goes from quote date of the curve
        strike=coupon[icall] / 100.                     # exercise yield (the coupon)
        d1=(np.log(fwdyld/strike) + 0.5*(vol**2)*expiry)/(vol*np.sqrt(expiry))
        d2=d1-vol*np.sqrt(expiry)
        call = dfcall * pvfwdann * (ss.norm.cdf(-d2)*strike - ss.norm.cdf(-d1)*fwdyld) * 100.
#        aicallable = bondpv[x6,1] - bondpv[x6,0]
        pvcall = bondpv[icall,1] - call
        bondpv[icall,1] = pvcall
        bondpv[icall,0] = pvcall - aivector[icall]

    return (bondpv)


#%% 
def pvCallable_lnYCover_tax_old(rates, prices, settledate, tparms, tcurve, yvols) :
    """
    Cover function for callable bond function, that inserts curve parameters.
    """
# This version assumes spreads for type-2 and type-3 tax status

    tcurve[4:6] = rates[-2:]       # Inserts last 2 elements of "rates" (the spreads) into 
                                  # the appropriate spots in "curve"
    tcurve[3] = rates[:-2]         # Inserts the other elements into "rates"
    bondpv = pvCallable_lnY(settledate,tparms,tcurve,yvols)
    diffs = bondpv[:,0] - prices[:,0]
    diffs = sum(diffs ** 2)

    return (diffs)

#%%


#x1 = bondParmsToMatrix(settledate, parms, padj=False)