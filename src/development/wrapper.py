##########################################
# This is the wrapper program for        #
# cleaning data into parms, fitting fwd  #
# rates with and without                 #
# taxbility, produce predicted vs actual #
# price and yield. Store fwds as .csv    #
# and pickle.                            #
##########################################

'''
Overview
-------------
Wrapper function to
- Calculate and store forward rates dataframe - csv and pickle
- Produce and store predicted vs actual prices and yield table


Requirements
-------------


'''

#%% Import packages
# magic %reset resets by erasing variables, etc. 

import sys
import os
import numpy as np
import pandas as pd
import pickle
from openpyxl import Workbook  
import datetime
import importlib as imp
import scipy.optimize as so
import scipy.sparse as sp
from pylatex import Document, Section, Subsection, Command, Package
from tabulate import tabulate
from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
import time as time
# import cProfile
import time
# For "converter" to convert strings stored with .csv to numpy arrays
# https://stackoverflow.com/questions/42755214/how-to-keep-numpy-array-when-saving-pandas-dataframe-to-csv
import ast   # Abstract Syntax Trees - used in the "converter" for reading .csv

#%% Import py files

## Fixing Paths
sys.path.append('../../src/package')
sys.path.append('../../../BondsTable')
sys.path.append('../../tests')
sys.path.append('../../data')
sys.path.append('../../../FORTRAN2024/code')

# #os.chdir("C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024/curve_utils/src/development")
# #os.chdir("/Users/tcoleman/tom/yields/New2024/progs/curve_utils/src/development") # TSC added to change the working directory because he always forgets
# #BASE_PATH = "C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024"
# #BASE_PATH = "/Users/tcoleman/tom/yields/New2024/progs"
# paths = ['curve_utils/src/development', 'curve_utils/src/package', 'BondsTable',
#          'curve_utils/tests', 'curve_utils/data']
# for path in paths:
#         sys.path.append(os.path.join(BASE_PATH, path))
# OUTPUT_DIR = os.path.join(BASE_PATH, 'curve_utils/output')
# print(sys.path)
# sys.path.pop()    # removes last entry in sys.paths

import DateFunctions_1 as dates
import pvfn as pv
import pvcover as pvc
import discfact as discfact
import calculate_ratesprices as outputs
#import CRSPBondsAnalysis as analysis
import parzeroBond as pzb
import crsp_data_processing as data_processing
import produce_inputs as inputs
import util_fn as util
import output_to_latexpdf as output_to


imp.reload(dates)
imp.reload(pv)
imp.reload(pvc)
imp.reload(inputs)
imp.reload(outputs)
imp.reload(pzb)
imp.reload(util)
imp.reload(output_to)

#%% Wrapper Function

def produce_curve_wrapper(bonddata, curvetypes, start_date, end_date, base_breaks, estfile, output_dir, calltype=0,
                          wgttype=1, lam1=1, lam2=2, padj=False, padjparm=0, yield_to_worst=False, tax=False,
                          yvolsflg=False, yvols=0.2, opt_type="LnY"):
    """Loop through months from a given start and end date for cleaned crsp UST bond data to produce a df of a series of estimate parms 
    --forward rates, yvols, and a df of predicted versus actual price and yield. 

    Args:
        bonddata (pd.DataFrame): Cleaned CRSP bond dataset, after applying data_processing.clean_crsp and
                                 data_processing.create_weight to the WRDS CRSP monthly UST data.
        curvetypes (list): A list of curve type strings.
        start_date (int): Start date for the end result, in the format YYYYMMDD.
        end_date (int): End date for the end result, in the format YYYYMMDD.
        breaks (np.array): An array of date breaks, in years, e.g., np.array([0.0833, 0.5, 1., 2., 5., 10., 20., 30.])
        filename (str): The base filename for the output files.
        calltype (int, optional): Whether to filter callable bonds. 0=all, 1=callable only, 2=non-callable only.
                                  Defaults to 0.
        wgttype (int, optional): The weight type for SSQ. Defaults to 1.
        lam1 (float, optional): Lambda parameter 1 for weighting. Defaults to 1.
        lam2 (float, optional): Lambda parameter 2 for weighting. Defaults to 2.
        padj (bool, optional): Whether to apply price adjustment. Defaults to False.
        padjparm (float, optional): Parameter for price adjustment. Defaults to 0.
        yield_to_worst (bool, optional): Whether to use yield to worst for filtering. Defaults to False.
        tax (bool, optional): Whether to calculate and include tax-adjusted curves. Defaults to False.
        yvolsflg (bool, optional): Whether to include yield volatilities. Defaults to False.
        yvols (float, optional): Yield volatilities parameter. Defaults to 0.1.

    Returns:
        - final_curve_df: DataFrame with final curve/estimated data.
        - final_price_yield_df: DataFrame with final price and yield data.
    """

    # Convert start_date and end_date to datetime format if they are not already
    if not isinstance(start_date, datetime.datetime):
        start_date = pd.to_datetime(str(start_date), format='%Y%m%d')
    if not isinstance(end_date, datetime.datetime):
        end_date = pd.to_datetime(str(end_date), format='%Y%m%d')

    # Dictionary to store final DataFrames for each curve type
    #all_dfs = {}
    curve_list = []
    predyld_list = []
    wb = Workbook()
    # breaks_df['quote_date'] = breaks_df['quote_date'].astype(int)
    
    for curvetype in curvetypes:
        # curvetype = 'pwtf'
        curve_tax_list = []

        filtered_data = bonddata[(bonddata['quote_date'] > start_date) & (bonddata['quote_date'] < end_date)]
        quotedates = list(set(filtered_data['MCALDT']))

        for quotedate in quotedates:
            # quotedate=20000929
            parms = inputs.read_and_process_csvdata(filtered_data, quotedate, calltype)
            quotedate = int(quotedate)
            quotedate_Julian = dates.YMDtoJulian(quotedate)[0]
            
            # Count the number of bonds in each forward period
            def classify_ytm(ytm, breaks):
                for i in range(len(breaks) - 1):
                    if breaks[i] < ytm <= breaks[i + 1]:
                        return breaks[i]
                if ytm <= breaks[0]:
                    return 0
                if ytm > breaks[-1]:
                    return breaks[-1]
                return None
            
            parms['YTM'] = (parms['Maturity Date at time of Issue'] - quotedate_Julian)/365.25
            parms['YTM_bucket'] = parms['YTM'].apply(lambda x: classify_ytm(x, base_breaks))
            bucket_counts = parms['YTM_bucket'].value_counts().reindex(base_breaks, fill_value=0).sort_index()
            num_bonds = bucket_counts.tolist()
            breaks = [base_breaks[i] for i in range(len(base_breaks)) if num_bonds[i] != 0]
            breaks = np.array(breaks)

            # breaks = breaks_df.loc[breaks_df['quote_date'] == quotedate, 'breaks'].values[0]
            # breaks = np.array(breaks)
    
            parms = inputs.filter_yield_to_worst_parms(quotedate, parms, yield_to_worst)
            parms = inputs.create_weight(parms, wgttype, lam1=lam1, lam2=lam2)
            len_parms = len(parms)
            curve, prices, bondpv, stderr, yvol, mesg, ier = outputs.calc_rate_notax(parms, quotedate, breaks, curvetype, opt_type, wgttype, 
                                                                                     lam1, lam2, padj, padjparm, yvolsflg=yvolsflg, yvols=yvols)
            curve.append(stderr)  # add std errors
            curve.append(yvol)
            curve.append(mesg)
            curve.append(ier)
            num_bonds = [x for x in num_bonds if x != 0]
            curve.append(np.array(num_bonds))  # add number of bonds in each forward period
            
            predyld = outputs.get_predicted_actual_yieldprice_notax(parms, bondpv, prices, quotedate,
                curvetype, padj, yvolsflg=yvolsflg, yvols=yvol)
            predyld.insert(0, 'QuoteDate', quotedate)
            predyld.insert(0, 'type', curvetype)
            # Make multi-index
            predyld.set_index(['type','QuoteDate'],inplace=True, drop=False)
            
            # Aggregate curve and price_yield_df data
            # Change from Kathy's - build up list of full curves (not just rates)
            curve_list.append(curve)
            predyld_list.append(predyld)
            
            print(quotedate)
        
# NB - need to fix up tax
        if tax:
            tax_spd = np.array(["tax2_spd", "tax3_spd"])
            curve_tax = pd.DataFrame(curve_tax[3].reshape(1, -1), index=[quotedate], columns=np.append(breaks, tax_spd))
            curve_tax_list.append(curve_tax)
            curve_tax = outputs.cal_rate_with_tax(parms, quotedate, breaks, curvetype, wgttype, lam1, lam2, padj, padjparm)
            curve_tax_df = pd.concat(curve_tax_list)
            curve_tax_df = curve_tax_df.sort_index()
            all_dfs[curvetype]['curve_tax_df'] = curve_tax_df
        

    # Convert the accumulated list of curves into a df and sort by index 
    # The following is necessary because we want to have a multi-index (on 
    # quotedate and curve type) but the quote date is a np.ndarray and 
    # multi-index on that does not work. So:
        # Create a new column and name it 'quotedate_ind'
        # Convert to integer and then create index
        # Also create type_ind, so that we retain the original type and quotedate in the df
    curve_df = pd.DataFrame(curve_list)
    curve_df.columns = ['type','quotedate_jul','breaks','rates', 'stderr', 'yvols', 'mesg', 'ier', 'num_bonds']
    curve_df['ytw_flag']    = yield_to_worst
    curve_df['padj']        = padj
    curve_df['padjparm']    = padjparm
    curve_df['opt_type']    = opt_type
    curve_df['taxability']  = tax
    
    curve_df['quotedate_ymd'] = curve_df['quotedate_jul']
    curve_df['quotedate_ymd'] = curve_df['quotedate_ymd'].map(dates.JuliantoYMDint)
    curve_df['quotedate_ymd'] = curve_df['quotedate_ymd'].map(int)
    curve_df['type_ind'] =      curve_df['type']
    curve_df['quotedate_ind'] = curve_df['quotedate_ymd']
    curve_df.set_index(['type_ind','quotedate_ind'],inplace=True,drop=True)
    curve_df = curve_df.sort_index()

    # Converted accumulate list of actual vs predicted into df
    predyld_df = pd.concat(predyld_list)
    predyld_df = predyld_df.sort_values(by=[ 'MatYr', 'MatMth', 'MatDay'])
    predyld_df = predyld_df.sort_index()

    # Export to Excel on separate sheets
    # excel_file_path = os.path.join(output_dir, f'{estfile}_data.xlsx')
    # with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
    #     final_curve_df.to_excel(writer, sheet_name='Curve Data')
    #     if tax:
    #         final_curve_tax_df.to_excel(writer, sheet_name='Curve Tax Data')
    #     final_price_yield_df.to_excel(writer, sheet_name='Price Yield Data')

    # Create new folder to store results
    folder_path = os.path.join(output_dir, estfile)
    os.makedirs(folder_path, exist_ok=True)

    curve_df.to_csv(folder_path+'/'+estfile+'_curve.csv')
    # Also write to pickle because writing to csv does not work exactly right for numpy arrays
    curve_df.to_pickle(folder_path+'/'+estfile+'_curve.pkl')
    predyld_df.to_csv(folder_path+'/'+estfile+'_predyld.csv')
    # For actual vs predicted, use pickle because actual-vs-predicted will be large
    predyld_df.to_pickle(folder_path+'/'+estfile+'_predyld.pkl')

    return curve_df, predyld_df


