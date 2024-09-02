##########################################
# This is the wrapper program for        #
# cleaning data into parms, curve        #
# calculations with and without          #
# taxbility, produce predicted vs actual #
# price and yield, and plot curves       #
##########################################

'''
Overview
-------------

Requirements
-------------


'''

# magic %reset resets by erasing variables, etc. 


import sys
import os
import numpy as np
import pandas as pd
import pickle
from openpyxl import Workbook
import matplotlib.pyplot as plt
import plotly.express as px
import datetime
import importlib as imp
import scipy.optimize as so
import scipy.sparse as sp
import time as time
import matplotlib.pyplot as plt
import cProfile as cprofile
import pstats
from pstats import SortKey

sys.path.append('../../src/package')
sys.path.append('../../../BondsTable')
sys.path.append('../../tests')
sys.path.append('../../data')


# TSC added to change the working directory because he always forgets
# os.chdir('/Users/tcoleman/tom/yields/New2024/progs/curve_utils/src/development')
#print(sys.path)
#sys.path.pop()    # removes last entry in sys.path


#%%

import DateFunctions_1 as dates
import pvfn as pv
import pvcover as pvc
import discfact as df
import Curve_Plotting as plot
import CRSPBondsAnalysis as analysis

import crsp_data_processing as data_processing
import produce_inputs as inputs
import calculate_ratesprices as outputs
import plot_rates as plot_rates


imp.reload(dates)
imp.reload(pv)
imp.reload(pvc)
imp.reload(inputs)
imp.reload(outputs)
imp.reload(plot_rates)

OUTPUT_DIR = '../../output'

#%% Wrapper Function

def produce_curve_wrapper(bonddata, curvetypes, start_date, end_date, breaks, filename, calltype=0,
                          wgttype=1, lam1=1, lam2=2, padj=False, padjparm=0, yield_to_worst=False):
    """Loop through months from a given start and end date for cleaned crsp UST bond data to produce forward rates
    and predicted versus actual price and yield. Output to individual csv files and consolidated excel files with different tabs.

    Args:
        bonddata (pd.Dataframe): Cleaned CRSP bond dataset, after applying data_processing.clean_crsp and
                                 data_processing.create_weight to the WRDS CRSP monthly UST data.
        curvetypes (list): A list of curve type strings.
        start_date (int): Start date for end result.
        end_date (int): End date for end result.
        breaks (np.array): An array of date breaks, in years, e.g. np.array([0.0833, 0.5, 1.,2.,5.,10.,20.,30.])
        calltype (int, optional): Whether to filter callable bond. 0=all, 1=callable only, 2=non-callable only.
                                  Defaults to 0.
        weight_flag (int, optional): Whether to use weights for SSQ. Defaults to 0. >> could use weight_flag
                                     to determine weight method. 
    
    Returns:
        all_dfs: Dictionary containing final_curve_df and final_price_yield_df for each curvetype. Dataframes can be individually selected.
    """
    
    # curvetype = 'pwcf'
    
    # Convert start_date and end_date to datetime format if they are not already
    if not isinstance(start_date, datetime.datetime):
        start_date = pd.to_datetime(str(start_date), format='%Y%m%d')
    if not isinstance(end_date, datetime.datetime):
        end_date = pd.to_datetime(str(end_date), format='%Y%m%d')

    # Dictionary to store final DataFrames for each curve type
    all_dfs = {}

    wb = Workbook()
    
    for curvetype in curvetypes:
        
        curve_data_list = []
        curve_tax_data_list = []
        price_yield_data_list = []

        filtered_data = bonddata[(bonddata['quote_date'] > start_date) & (bonddata['quote_date'] < end_date)]
        quotedates = list(set(filtered_data['MCALDT']))

        for quotedate in quotedates:
            parms = inputs.read_and_process_csvdata(filtered_data, quotedate, calltype)
            quotedate = int(quotedate)
    
            parms = inputs.filter_yield_to_worst_parms(quotedate, parms, yield_to_worst)
            parms = inputs.create_weight(parms, wgttype, lam1=1, lam2=2)

            curve, prices, bondpv = outputs.calculate_rate_notax(parms, quotedate, breaks, curvetype, wgttype, lam1, lam2,
                                                                 padj, padjparm)
            curve_tax = outputs.calculate_rate_with_tax(parms, quotedate, breaks, curvetype, wgttype, lam1, lam2, padj, padjparm)

            price_yield_df = outputs.get_predicted_actual_yieldprice_notax(parms, bondpv, prices, quotedate, curvetype, padj)
            price_yield_df.insert(0, 'QuoteDate', quotedate)
            
            # Aggregate curve and price_yield_df data
            curve_df = pd.DataFrame(curve[3].reshape(1, -1), index=[quotedate], columns=breaks)
            tax_spd = np.array(["tax2_spd", "tax3_spd"])
            curve_tax_df = pd.DataFrame(curve_tax[3].reshape(1, -1), index=[quotedate], columns=np.append(breaks, tax_spd))
            curve_data_list.append(curve_df)
            curve_tax_data_list.append(curve_tax_df)
            price_yield_data_list.append(price_yield_df)
            
            print(quotedate)
        
        final_curve_df = pd.concat(curve_data_list)
        final_curve_df = final_curve_df.sort_index()
        final_curve_tax_df = pd.concat(curve_tax_data_list)
        final_curve_tax_df = final_curve_tax_df.sort_index()
        final_price_yield_df = pd.concat(price_yield_data_list)
        final_price_yield_df = final_price_yield_df.sort_values(by=['QuoteDate', 'MatYr', 'MatMth', 'MatDay'])
        
        all_dfs[curvetype] = {'curve_df': final_curve_df,
                              'curve_tax_df': final_curve_tax_df,
                              'price_yield_df': final_price_yield_df}
        
        # Export to CSV
        final_curve_df.to_csv(os.path.join(OUTPUT_DIR, f'curve_{curvetype}_{filename}.csv'))
        final_curve_tax_df.to_csv(os.path.join(OUTPUT_DIR, f'curve_tax_{curvetype}_{filename}.csv'))
        final_price_yield_df.to_csv(os.path.join(OUTPUT_DIR, f'price_yield_{curvetype}_{filename}.csv'))

        # Export to Excel on separate sheets
        excel_file_path = os.path.join(OUTPUT_DIR, f'{curvetype}_{filename}_data.xlsx')
        with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
            final_curve_df.to_excel(writer, sheet_name='Curve Data')
            final_curve_tax_df.to_excel(writer, sheet_name='Curve Tax Data')
            final_price_yield_df.to_excel(writer, sheet_name='Price Yield Data')
        
    return all_dfs



#%% Store dictionaries
def store_dict_output_pickle(data, filename):
    """Save dictionary of DataFrames to a Pickle file."""
    with open(os.path.join(OUTPUT_DIR, f'{filename}_dict'), 'wb') as file:
        pickle.dump(data, file)


def load_dfs_from_pickle(filename):
    """Load dictionary of DataFrames from a Pickle file."""
    with open(os.path.join(OUTPUT_DIR, f'{filename}_dict'), 'rb') as file:
        data = pickle.load(file)
    return data



# Attempt
# yield_to_worst = True
# settledate = 19460330
# filepath = '../../data/USTMonthly.csv'  # '../../data/1916to2024_YYYYMMDD.csv'
# bonddata = data_processing.clean_crsp(filepath)
# bonddata = data_processing.create_weight(bonddata, bill_weight=1)
# parms = inputs.read_and_process_csvdata(bonddata, settledate, 0)


    
OUTPUT_DIR = '../../output'

# Define user inputs
# quotedate = 20150130 # 19321231
calltype = 0  # 0 to keep all bonds, 1 for callable bonds, 2 for non-callable bonds
curvetypes = ['pwcf', 'pwlz', 'pwtf']
start_date = 19860701
start_date = 20000101
end_date = 20000201
breaks = np.array([7/365.25, 14/365.25, 21/365.25, 28/365.25, 35/365.25, 52/365.25, 92/365.25, 
                   184/365.25, 1, 2, 4, 8, 16, 24, 32])  # np.array([0.0833, 0.5, 1.,2.,5.,10.,20.,30.])
curve_points_yr = np.arange(.01,10,.01)

# New wrds data
filepath = '../../data/USTMonthly.csv'  # '../../data/1916to2024_YYYYMMDD.csv'
bonddata = data_processing.clean_crsp(filepath)

#profiles and saves results to disk (too long to display)
# cprofile.run('produce_curve_wrapper(bonddata, curvetypes, start_date, end_date, breaks, "1986topresent", calltype=calltype,wgttype=1, lam1=1, lam2=2, padj=False, padjparm=0, yield_to_worst=True)','prstats')

# See https://docs.python.org/3/library/profile.html
# These following are not very useful
p = pstats.Stats('prstats')
p.strip_dirs().sort_stats(-1).print_stats()

p.sort_stats(SortKey.NAME)
p.print_stats()

# These get to be useful
# sort according to time spent within each function, but this is for cumulative time 
#   including time spent in calls of subfunctions
p.sort_stats(SortKey.CUMULATIVE).print_stats(20)
# sort according to time spent within each function, only withing this function
#   excluding time spent in calls of subfunctions
p.sort_stats(SortKey.TIME).print_stats(20)

#store_dict_output_pickle(all_dfs_1986to2000, '1986to2000')
#all_dfs_1986to2000 = load_dfs_from_pickle('1986to2000')

# Plotly plot for January
#plot_rates.plot_rates_by_one_month(all_dfs_1986to2000, 'pwcf', 1, taxflag=True, taxability=1)


# filepath = "../../data/CRSP12-1932to3-1933.txt"
# fortran preprocessed
# parms, prices = read_and_process_fortrandata(filepath, calltype=0)

# Use uncleaned data
# bonddata_raw = analysis.clean_raw_crsp(filepath)
# # Add callflag
# non_call = pd.isnull(bonddata['TFCALDT'])     # TRUE when call is nan
# bonddata.loc[non_call , 'TFCALDT'] = 0.0      # Replace nan s with 0s
# bonddata['callflag'] = np.where(bonddata['TFCALDT'] == 0, 0, 1)
# parms, prices, weights = read_and_process_csvdata(bonddata, quotedate, calltype=0)
# curve, bondpv = calculate_rate_notax(parms, prices, weights, quotedate, breaks, curvetype, weight_flag=0)
# curvetax = calculate_rate_with_tax(parms, prices, quotedate, breaks, curvetype)
# price_yield_df = get_predicted_actual_yieldprice_notax(parms, bondpv, prices, quotedate, curvetype)
# plot_no_tax_curve(parms, prices, quotedate, breaks, curve_points_yr)
# plot_one_type_fwdcurve_tax(parms, prices, quotedate, breaks, curve_points_yr, curvetype)


