##########################################
# This is to produce inputs--parms,
# weights, and prices for rates
# calculations
##########################################

'''
Overview
-------------
1. Create inputs--parms, weights, and prices for rates calculations from the cleaned bond dataset

Requirements
-------------
../../data/WRDS CRSP UST dataset in csv format
crsp_data_processing.py

'''

import sys
import os
import numpy as np
import pandas as pd
import datetime
import importlib as imp
import scipy.optimize as so
import scipy.sparse as sp
import time as time
import cProfile

sys.path.append('../../src/package')
sys.path.append('../../../BondsTable')
sys.path.append('../../tests')
sys.path.append('../../data')

import DateFunctions_1 as dates


# %% Get bond parameters for a quote date/month

def read_and_process_csvdata(bonddata, quotedate, calltype=0):
    """Get bond parameter dataframe for a quote date using cleaned CRSP bond data."""

    bonddata = bonddata[bonddata['MCALDT'] == quotedate]
    bonddata['TMATDT_julian'] = bonddata['TMATDT'].apply(dates.YMDtoJulian)
    bonddata['TMATDT_julian'] = bonddata['TMATDT_julian'].apply(lambda x: x[0] if len(x) > 0 else None)
    bonddata['Redemption Amount'] = 100
    bonddata['Daycount'] = "A/A"
    bonddata['eom'] = "eomyes"
    bonddata['TFCALDT'] = bonddata['TFCALDT'].fillna(0)
    bonddata['First Call Date at Time of Issue'] = bonddata['TFCALDT'].apply(lambda x: dates.YMDtoJulian([x])
                                                                             if x != 0 else 0)
    bonddata['First Call Date at Time of Issue'] = bonddata['First Call Date at Time of Issue'].apply(lambda x: x[0] if
                                                                                                      x != 0 else 0)
    parms = bonddata[['TCOUPRT', 'TMATDT_julian', 'TNIPPY', 'Redemption Amount',
                      'Daycount', 'eom', 'First Call Date at Time of Issue', 'callflag', 'ITAX', 'IBILL', 'TMNOMPRC', 'TMDURATN', 'Spread']]
    # Rename the columns
    column_mapping = {
    'TCOUPRT': 'Coupon Rate (percent per annum)',
    'TMATDT_julian': 'Maturity Date at time of Issue',
    'TNIPPY': 'Number of Interest Payments per Year',
    'Redemption Amount': 'Redemption Amount',
    'Daycount': 'Daycount',
    'eom': 'eom',
    'First Call Date at Time of Issue': 'First Call Date at Time of Issue',
    'callflag': 'CallFlag',
    'ITAX': 'Taxability of Interest',
    'IBILL': 'IBILL',
    'TMNOMPRC': 'Price',
    'TMDURATN': 'DUR',
    'Spread': 'Spread'
    }
    parms = parms.rename(columns=column_mapping)
    
    if calltype == 1:
        parms = parms[parms['CallFlag'] == 1]
        bonddata = bonddata[bonddata['First Call Date at Time of Issue'] != 0]
    if calltype == 2:
        parms = parms[parms['CallFlag'] == 0]
        bonddata = bonddata[bonddata['First Call Date at Time of Issue'] == 0]
    
    parms = parms.reset_index(drop=True)
    
    # prices = ( np.array(bonddata[["TMBID"]]) + np.array(bonddata[["TMASK"]]) ) / 2.
    # 'PRICE' is very similar to 'TMNOMPRC' in the CRSP dataset, so we will now use 'TMNOMPRC' inst
    # VALUE USED IN CRSP CALCULATIONS, MOST OFTEN THE BID AND ASK AVERAGE. PRIOR TO 1960, BIDS AND SALES WERE USED.
    # prices =  np.array(bonddata[["TMNOMPRC"]])
    # parms['Price'] = prices
    # weights = bonddata['weight1']
    
    return parms


def read_and_process_fortrandata(filepath, quotedate, calltype=0):
    xxtemp = pd.read_csv(filepath, sep='\t', index_col=['Quote Date', 'CRSP Issue Identification Number',
                                                    'Coupon Rate (percent per annum)', 'Uniqueness Number'])
    xxtemp['Face Value Outstanding'] = xxtemp['Face Value Outstanding'].str.replace(r'[^-+\d.]', '', regex=True).astype(float)
    xxtemp['Publicly Held Face Value Outstanding'] = xxtemp['Publicly Held Face Value Outstanding'].str.replace(r'[^-+\d.]', '', regex=True).astype(float)
    #To "index" into different parts of datafrme we can use indexes. For levels other than 0 have to specify level
    # These will be "series"
    xx1 = xxtemp.xs(quotedate)['Number of Interest Payments per Year']

    xx2 = xxtemp.xs(19330131)[['Number of Interest Payments per Year','Maturity Date at time of Issue']]

    xx3 = xxtemp.xs(quotedate)['First Price, usually Bid']

    xx4 = xxtemp.xs(19340502.2,level=1)['First Price, usually Bid']

    # To get all data for a specific quote date
    # This will remain "DataFrame"
    xx3212 = xxtemp.xs(19321231)

    # To index multiple columns
    xx6 = xx3212[['First Price, usually Bid','Face Value Outstanding']]

    xx7 = xxtemp.xs(19340502.2,level=1)[['First Price, usually Bid','Face Value Outstanding']]

    # removes the indexes - applied to xxtemp this is the same as not reading in with indexes
    # But here it's applied to one quote date
    yy3212 = xx3212.reset_index()

    #zz1 = pd.concat([xx3212,xx3301],axis=1)

    # Want to do a copy so that it does not refer back to the original data frame
    zz1 = yy3212[['Coupon Rate (percent per annum)','Maturity Date at time of Issue',
        'Number of Interest Payments per Year','First Call Date at Time of Issue',
        "First Price, usually Bid","Second Price, usually Ask","Taxability of Interest"]].copy()
        
    x1 = pd.isnull(zz1['First Call Date at Time of Issue'])     # TRUE when call is nan

    zz1.loc[x1,'First Call Date at Time of Issue'] = 0.0      # Replace nan s with 0 s

    x1 = zz1['Number of Interest Payments per Year'] == 0       # Check for no interest payments (Bills)

    x2 = zz1["Second Price, usually Ask"] < 0.
    zz1.loc[x2,"Second Price, usually Ask"] = zz1.loc[x2,"First Price, usually Bid"]

    zz2 = zz1.loc[np.logical_not(x1)].copy()                           # Get rid of Bills
    zz2 = zz1.copy()                                                   # Keep Bills

    zz2['Redemption Amount'] = pd.Series(100.0, index=zz2.index)  # Append a new column (using original index)

    zz3 = np.array(zz2)                                         # Convert from Pandas DataFrame to NumPy Array
                                # But cannot convert to NumPy Array once we put in string variables. 

    zz2['Daycount'] = pd.Series("A/A", index=zz2.index)
    zz2['eom'] = pd.Series("eomyes", index=zz2.index)
    zz20 = zz2.reset_index()
    #zz21 = zz20[['Coupon Rate (percent per annum)','Number of Interest Payments per Year', 
    #    'Redemption Amount','Daycount','eom']]

    d1 = zz20['Maturity Date at time of Issue']
    d1 = d1.tolist()                                              # Convert Maturity Date to Julian
    D1 = dates.YMDtoJulian(d1)

    d2 = zz20['First Call Date at Time of Issue']   
    d2 = d2.tolist()
    d22 = [int(i) for i in d2]
    D2 = []
    calltrue = []
    for i in range(0,len(d22)):                                 # Convert Call Date to Julian
        if d22[i]<19321231:               # Here I should count forward by coupon periods to get past settle date
            D2.append(0)
            calltrue.append(False)
        else:
            d=dates.YMDtoJulian(d22[i])
            D2.append(d[0])
            calltrue.append(True)

    zz20['Maturity Date at time of Issue']=pd.Series(D1,index=zz20.index)

    zz20['First Call Date at Time of Issue']=pd.Series(D2,index=zz20.index)
    zz20['CallFlag'] = pd.Series(calltrue,index=zz20.index)

    if calltype == 1:
        zz20 = zz20[zz20['CallFlag'] == True]
    elif calltype == 2:
        zz20 = zz20[zz20['CallFlag'] == False]

    zz22 = zz20[['Coupon Rate (percent per annum)','Maturity Date at time of Issue',
                'Number of Interest Payments per Year','Redemption Amount',
                'Daycount','eom','First Call Date at Time of Issue',
                'CallFlag','Taxability of Interest']].copy()

    prices = ( np.array(zz20[["First Price, usually Bid"]]) + 
        np.array(zz20[["Second Price, usually Ask"]]) ) / 2.
    
    return zz22, prices


def filter_yield_to_worst_parms(quotedate, parms, yield_to_worst=False):
    
    quote = dates.YMDtoJulian(quotedate)[0]
        
    if yield_to_worst == True:
        # If current price < 100, leave maturity date as maturity date
        # If current price > 100, replace maturity date with first call data
        # If days to first call date < 365 then discard

        # Check conditions based on the current price and days to first call
        parms['days_to_first_call'] = parms['First Call Date at Time of Issue'] - quote
        parms = parms[(parms['CallFlag'] == False) | ((parms['CallFlag'] == True) & (parms['days_to_first_call'] >= 365))]
        parms.loc[(parms['Price'] >= 100) & (parms['CallFlag'] == True), 'Maturity Date at time of Issue'] = parms['First Call Date at Time of Issue']
        parms.drop('days_to_first_call', axis=1, inplace=True)
        parms = parms.reset_index(drop=True)
        parms['First Call Date at Time of Issue'] = 0.0
        parms['CallFlag'] = False
        
    
    # prices = np.array(parms[["Price"]])
        
    return parms


def create_weight(parms, wgttype=1, lam1=1, lam2=2):
    """Create weight for each bond for SSQ.

    Args:
        parms (pd.Dataframe): Bond parameters dataframe.
        wgttype (int, optional): The weight estimation method to adopt. 
                                wgttype = 0, 1, 2: 
                                0: no weight--weight=1 for all bond;
                                1: SSQ using only bid ask spread and duration;
                                2: SSQ using maximum likelihood. Defaults to 1. 
        lam1 (int, optional): lambda for adjusting DURATION. Defaults to 1.
        lam2 (int, optional): lambda for adjusting BWT = bid-ask spread/harmonic mean spread. Defaults to 2.

    Returns:
        parms: Bond parameters dataframe with weights and relevant parameters used for SSQ.
    """

    # Calculate harmonic mean by bill/bond using bid/ask spread for each month
    def harmonic_mean(s):
        s = s[s > 0]  # Exclude zeros to avoid division by zero
        if len(s) == 0:
            return 1  # Return 0 or np.nan if no non-zero elements are present
        return len(s) / np.sum(1.0 / s)
    
    harmonic_means = parms.groupby(['IBILL'])['Spread'].apply(harmonic_mean).reset_index()
    harmonic_means.rename(columns={'Spread': 'HMSpread'}, inplace=True)
    parms = pd.merge(parms, harmonic_means, on=['IBILL'], how='left')
    parms['BWT'] = parms.apply(lambda row: max(1, row['Spread']/row['HMSpread']), axis=1)
    
    if wgttype == 1:  # SSQ
        # Calculate weight: (Spread/HM)^lam2*(DUR/365)^lam1, as denominator for nominator squared price difference
        # SS = ((PRC(I) - PRICE(I))**2)/((BWT(I)**2)*(DUR(I)/365.)) + SS
        parms['Weight'] = parms.apply(lambda row: ((row['BWT']**lam2) * (row['DUR']/365)**lam1), axis=1)
    elif wgttype == 2:  # SSQML
        # Calculate weight: (Spread/HM)^lam2*(DUR/365)^lam1, as denominator for nominator squared price difference
        # SS2 for minimization 
        parms['Weight'] = parms.apply(lambda row: ((row['BWT']**lam2) * (row['DUR']/365)**lam1), axis=1)
        # nobs = len(parms)
        parms['SS2'] = parms.apply(lambda row: lam1 * np.log(row['DUR']/365) + lam2 * np.log(row['BWT']), axis=1)
    else:
        parms['Weight'] = 1
    
        

    return parms

