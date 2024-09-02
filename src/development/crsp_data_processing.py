##########################################
# This is to clean crsp data, create
# weights for analysis and calculations
##########################################

'''
Overview
-------------
1. Clean CRSP data, get rid off certain bonds
2. Create weights for each bond using different methods

Requirements
-------------
../../data/WRDS CRSP UST dataset in csv format

'''

import sys
import os
import numpy as np
import pandas as pd
from openpyxl import Workbook
import matplotlib.pyplot as plt
import plotly.express as px
import datetime
import importlib as imp
import scipy.optimize as so
import scipy.sparse as sp
import time as time
import matplotlib.pyplot as plt

sys.path.append('../../src/package')
sys.path.append('../../../BondsTable')
sys.path.append('../../tests')
sys.path.append('../../data')

import DateFunctions_1 as dates


# %% Clean up dataset (from CRSPBondsAnalysis.py)

def clean_crsp(filepath):
    """Clean up the original CRSP UST dataset from WRDS for yield calculations."""

    bonddata = pd.read_csv(filepath)
    bonddata.columns = bonddata.columns.str.upper()
    
    # Convert 'MCALDT' and 'TMATDT' to datetime and extract year and month for comparison
    bonddata = bonddata.dropna(subset=['MCALDT'])
    bonddata['MCALDT'] = bonddata['MCALDT'].astype('Int64')  # quote date
    bonddata['TMATDT'] = bonddata['TMATDT'].astype('Int64')  # maturity date
    bonddata['quote_date'] = pd.to_datetime(bonddata['MCALDT'].astype(str), format='%Y%m%d')
    bonddata['maturity_date'] = pd.to_datetime(bonddata['TMATDT'].astype(str), format='%Y%m%d')
    
    # Add callflag
    non_call = pd.isnull(bonddata['TFCALDT'])  # TRUE when call is nan
    bonddata.loc[non_call , 'TFCALDT'] = 0.0  # Replace nan s with 0s4
    bonddata['TFCALDT'] = bonddata['TFCALDT'].astype('Int64')
    bonddata['callflag'] = np.where(bonddata['TFCALDT'] == 0, 0, 1)
    
    # Idenfity bonds that has passed its first call date
    bonddata['passfirstcall'] = 0
    bonddata.loc[bonddata['TFCALDT'] < bonddata['MCALDT'], 'passfirstcall'] = 1
    
    ############################################################################
    # Cleaning and throwing out bonds
    ############################################################################
    # Extract numbers from CUSIPs
    # C in CUSIP represents that the security is exempt from SEC registration
    # B in CUSIP indicates that it is issued by a company that has been in bankruptcy proceedings
    bonddata['CRSPID_num'] = bonddata['CRSPID'].astype(str)
    bonddata['CRSPID_num'] = bonddata['CRSPID_num'].str.replace('C', '').str.replace('B', '')
    bonddata['CRSPID_num'] = bonddata['CRSPID_num'].astype(float)
    
    # Discard consol bonds
    bonddata = bonddata[bonddata['TMATDT'] != 20990401]
    
    # Extract quote year, month, day
    bonddata['YEAR'] = bonddata['quote_date'].dt.year
    bonddata['YEAR'] = bonddata['YEAR'].astype(int)
    bonddata['MONTH'] = bonddata['quote_date'].dt.month
    bonddata['DAY'] = bonddata['quote_date'].dt.day

    # Extract maturity year, month, day
    bonddata['CRSPID_date'] = bonddata['CRSPID_num'].astype(int)
    bonddata['JYEAR'] = bonddata['CRSPID_date'] // 10000
    bonddata['JMTH'] = (bonddata['CRSPID_date'] // 100) % 100
    bonddata['JDAY'] = bonddata['CRSPID_date'] % 100
    
    # Calculate days to maturity
    bonddata['days_to_maturity'] = (bonddata['maturity_date'] -
                                    bonddata['quote_date']).dt.days
 
    # TSC 1-jul-24. Exclude any ITYPE >=9. There are some ITYPE = 11 or 12 which are odd
    # Discard any ITYPE=8 or 9 or greater bonds. These are odd ones. They can be
    # identified because the fractional part of CRSPID will be GE 0.8
    # DO NOT EXCLUDE 19381015.904250
    bonddata = bonddata[~((bonddata['ITYPE'] == 8) | (bonddata['ITYPE'] >= 9)) |
                        (bonddata['CRSPID_num'] == 19381015.904250)]

    # Discard the Panama Canals of 16-36 and 18-38.  They are above
    # par and past call date for most of their life anyway.
    # tolerance = 1e-7  # Use a small numerical tolerance for floating point comparison
    # panama_canals_crspids = [19360801.5, 19381101.5]  # [19360801.502, 19381101.502]
    # bonddata = bonddata[~bonddata['CRSPID_num'].isin(panama_canals_crspids)]
    panama_canals_crspids = [19360801.502, 19381101.502]
    tolerance = 1e-7  # Use a small numerical tolerance for floating point comparison
    bonddata = bonddata[~bonddata['CRSPID_num'].apply(lambda x:
                                              any(abs(x - crspid) < tolerance for crspid in panama_canals_crspids))]

    # Discard this bond if maturity is less than three days
    bonddata = bonddata[bonddata['days_to_maturity'] >= 3]

    # ????? Multiply PCYLD (semi-annual yield) with 100
    # bonddata['PCYLD'] = bonddata['PCYLD'] * 100

    # Discard any bond ruled "BANK ELIGIBLE." up to 1955 or end of eligibility
    bonddata = bonddata[~((bonddata['YEAR'] < 1955) & (
        bonddata['TBANKDT'] >= bonddata['YEAR']))]

    # For all bonds after 1943 until now,
    # check if this is a flower bond. If it is, check it against the list
    # of flower bonds par excellance.  This is from a SUPERBOND run
    # of Larry Fisher's from 1976.  Exclude bonds according to the
    # date list below.
    # Starting in 1973, exclude all flower bonds.

    flower_bond_conditions = {19690615.5025: [(19560928, 19560928), (19570731, 19570731)], # 19690615.5025
                              19691215.5025: [(19560928, 19561031), (19570731, 19570831)], # 19691215.5025
                              19700315.5025: [(19550831, 19550831), (19560629, 19561031), # 19700315.5025
                                              (19570329, 19570830), (19580930, 19580930),
                                              (19590227, 19590227), (19590731, 19590831)],
                              19710315.5025: [(19550729, 19550930), (19560131, 19560131), # 19710315.5025
                                              (19560531, 19571031), (19580430, 19600630)],
                              19720615.5025: [(19550729, 19600831), (19601230, 19610228)], # 19720615.5025
                              19721215.5025: [(19550729, 19600831), (19601230, 19610228)], # 19721215.5025
                              19830615.50325: [(19670331, 19670331), (19680131, 19680131), # 19830615.50325
                                               (19680430, 19680531), (19710730, 19720229),
                                               (19720531, 19721229), (19730531, 19731130)],
                              19850515.10325: [(19670331, 19670331), (19671130, 19680930), # 19850515.10325
                                               (19690630, 19690630), (19691128, 19691231),
                                               (19700430, 19700430), (19700831, 19701030),
                                               (19710528, 19721229), (19730430, 19731130)],
                              19900215.1035: [(19670331, 19670331), (19671130, 19731231)], # 19900215.1035
                              19930215.504: [(19720731, 19720929), (19730831, 19730831)], # 19930215.504
                              19940515.50412: [(19720929, 19720929), (19721229, 19721229), # 19940515.50412
                                               (19730330, 19730330), (19730531, 19730531),
                                               (19701104, 19730831)],
                              19950215.103: [(19580131, 19580131), (19581031, 19590529), # 19950215.103
                                             (19591030, 19591030), (19591231, 19680430),
                                             (19680628, 19731231)],
                              19981115.1035: [(19660531, 19660531), (19670331, 19670331), # 19981115.1035
                                              (19671031, 19731231)]}
    

    # Function to check exclusion for a single bond
    def check_exclusion(row):
        if row['IFLWR'] != 1 and row['YEAR'] >= 1943:
            if row['YEAR'] > 1972:
                return True  # Exclude all flower bonds after 1973
            
            for crspid, date_ranges in flower_bond_conditions.items():
                if abs(row['CRSPID_num'] - crspid) < 1e-7:
                    for start_date, end_date in date_ranges:
                        if start_date <= row['MCALDT'] <= end_date:
                            return True  # Bond should be excluded based on date range
        return False  # Bond should not be excluded

    # Apply the check function to each row and store the result in a new column 'exclude'
    bonddata['exclude'] = bonddata.apply(check_exclusion, axis=1)

    # Filter the DataFrame to exclude the bonds based on the 'exclude' column
    bonddata = bonddata[~bonddata['exclude']]

    # Throw out any 1.5% coupon bond starting with the one maturing in 1975 (issued in 1970)
    # that matures on April 01 or October 01.
    # Convert coupon rate to a comparable scale if necessary
    # bonddata['COUPRT'] = bonddata['COUPRT'] / 100
    bonddata = bonddata[~((bonddata['JYEAR'] >= 1970) &
                        (bonddata['TCOUPRT'].between(1.5 - 0.000001, 1.5 + 0.000001, inclusive='both')) &
                        (((bonddata['JMTH'] == 4) & (bonddata['JDAY'] == 1)) |
                        ((bonddata['JMTH'] == 10) & (bonddata['JDAY'] == 1))))]

    # Exclude rows with negative prices in PRIC1R and PRIC2R (bid and ask)
    # Compute the price: 1 = BID, 2=ASK, 3=AVG OF BID/ASK

    # For rows where 'IPRICE' is 1, set 'PRIC2R' equal to 'PRIC1R'
    # bonddata.loc[bonddata['IPRICE'] == 1, 'PRIC2R'] = bonddata['PRIC1R']

    # For rows where 'IPRICE' is 2, set 'PRIC1R' equal to 'PRIC2R'
    # bonddata.loc[bonddata['IPRICE'] == 2, 'PRIC1R'] = bonddata['PRIC2R']

    # Exclude rows where both 'PRIC1R' and 'PRIC2R' are less than or equal to 0.01
    bonddata = bonddata[~((bonddata['TMBID'] <= 0) | (bonddata['TMASK'] <= 0))]
    bonddata = bonddata[~((bonddata['TMBID'] <= 0.01) & (bonddata['TMASK'] <= 0.01))]

    # Adjust 'PRIC1R' and 'PRIC2R' based on threshold
    bonddata.loc[bonddata['TMBID'] <= 0.01, 'TMBID'] = bonddata['TMASK']
    bonddata.loc[bonddata['TMASK'] <= 0.01, 'TMASK'] = bonddata['TMBID']

    # ??? 'PRICE' is very similar to 'TMNOMPRC' in the CRSP dataset
    # Compute the average price for the remaining rows, IPRICE = 3
    bonddata['PRICE'] = 0.5 * (bonddata['TMBID'] + bonddata['TMASK']) # need to indicate when IPRICE = 3

    # Convert 'TFCALDT' -- first call date to datetime
    bonddata['TFCALDT'] = bonddata['TFCALDT'].astype('Int64')
    bonddata['call_date'] = bonddata['TFCALDT']
    bonddata['call_date'] = pd.to_datetime(bonddata['call_date'], format='%Y%m%d', errors='coerce')

    # IF PRICE >= 100 SET MATUR = CALL DATE, IF DAYS TO CALL <= 365 DISCARD
    # Set 'MATUR' to days between call_date and quote_date where PRICE >= 100 and IDTCP is valid (IDTCP >= 1)
    # bonddata.loc[(bonddata['TMNOMPRC'] >= 100) & (bonddata['callflag'] != 0), 'MATUR'] = (bonddata['call_date'] -
    #                                                                                 bonddata['quote_date']).dt.days
    # # Exclude rows where DAYS TO CALL <= 365
    # bonddata = bonddata[~((bonddata['TMNOMPRC'] >= 100) & (bonddata['callflag'] != 0) & (bonddata['MATUR'] <= 365))]

    # Adjust decimals for prices
    bonddata['TMNOMPRC'] = bonddata['TMNOMPRC'].round(6)
    bonddata['TMBID'] = bonddata['TMBID'].round(6)
    bonddata['TMASK'] = bonddata['TMASK'].round(6)

    # Extract year, month, day from 'CALL_DATE' for further processing if called 
    bonddata.loc[(bonddata['TMNOMPRC'] >= 100) & (bonddata['callflag'] != 0), 'JYEAR'] = bonddata.loc[bonddata['TMNOMPRC'] >= 100, 'call_date'].dt.year
    bonddata.loc[(bonddata['TMNOMPRC'] >= 100) & (bonddata['callflag'] != 0), 'JMTH'] = bonddata.loc[bonddata['TMNOMPRC'] >= 100, 'call_date'].dt.month
    bonddata.loc[(bonddata['TMNOMPRC'] >= 100) & (bonddata['callflag'] != 0), 'JDAY'] = bonddata.loc[bonddata['TMNOMPRC'] >= 100, 'call_date'].dt.day

    # >> The below is commented out in FORTRAN
    # Exclude any coupon bond less than one year starting in 1980
    # This is done because Salomon's ask quotes are meaningless, starting in the early 1980's
    # bonddata = bonddata[~((bonddata['JYEAR'] >= 1980) & (bonddata['TCOUPRT'] > 0) &
    #                     (bonddata['MATUR'] <= 365))

    # Check for leap year in 'JYEAR' and set 'JK'
    bonddata['JK'] = 2
    bonddata.loc[bonddata['JYEAR'] % 4 == 0, 'JK'] = 1

    # Exclude rows where 'TMDURATN' is -1, -99, or NaN, all indicates missing durations
    bonddata = bonddata.query('TMDURATN != -1 & TMDURATN != -99 & TMDURATN == TMDURATN')

    # Calculate bid/ask spread for weight calculation
    bonddata['Spread'] = bonddata['TMASK'] - bonddata['TMBID']
    
    # Create IBILL column
    bonddata['IBILL'] = 0
    bonddata.loc[bonddata['ITYPE'] == 4, 'IBILL'] = 1

    return bonddata

