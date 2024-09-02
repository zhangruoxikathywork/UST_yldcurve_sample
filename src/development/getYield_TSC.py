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
#%% Import packages
# magic %reset resets by erasing variables, etc. 

import sys
import os
import numpy as np
import pandas as pd
import pickle
from openpyxl import Workbook
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
pio.renderers.default='svg'
import datetime
import importlib as imp
import scipy.optimize as so
import scipy.sparse as sp
import time as time
import matplotlib.pyplot as plt
import cProfile

# TSC added to change the working directory because he always forgets
# os.chdir('/Users/tcoleman/tom/yields/New2024/progs/curve_utils/src/development')
#print(sys.path)
#sys.path.pop()    # removes last entry in sys.path

# os.chdir("C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024/curve_utils/src/development") # KZ: for my vscode directory bug...

OUTPUT_DIR = '../../output'

#%% Import py files

# KZ: for my vscode directory bug...
# sys.path.append('C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024/curve_utils/src/package')
# sys.path.append('C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024/BondsTable')
# sys.path.append('C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024/curve_utils/tests')
# sys.path.append('C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024/curve_utils/data')
# OUTPUT_DIR = 'C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024/curve_utils/output'

sys.path.append('../../src/package')
sys.path.append('../../../BondsTable')
sys.path.append('../../tests')
sys.path.append('../../data')

import DateFunctions_1 as dates
import pvfn as pv
import pvcover as pvc
import discfact as discfact
import Curve_Plotting as plot
import CRSPBondsAnalysis as analysis

import crsp_data_processing as data_processing
import produce_inputs as inputs
import calculate_ratesprices as outputs
import plot_rates as plot_rates
import util_fn as util
import curve_utils.src.development.output_to_latexpdf as plotlatex

imp.reload(dates)
imp.reload(pv)
imp.reload(pvc)
imp.reload(inputs)
imp.reload(outputs)
imp.reload(plot_rates)
imp.reload(plot)
imp.reload(plotlatex)


#%% Wrapper Function

def produce_curve_wrapper(bonddata, curvetypes, start_date, end_date, breaks, filename, calltype=0,
                          wgttype=1, lam1=1, lam2=2, padj=False, padjparm=0, yield_to_worst=False, tax=False):
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
    #all_dfs = {}
    curve_data_list = []
    price_yield_data_list = []

    wb = Workbook()
    
    for curvetype in curvetypes:
        
        #curve_data_list = []
        curve_tax_data_list = []
        #price_yield_data_list = []

        filtered_data = bonddata[(bonddata['quote_date'] > start_date) & (bonddata['quote_date'] < end_date)]
        quotedates = list(set(filtered_data['MCALDT']))

        for quotedate in quotedates:
            parms = inputs.read_and_process_csvdata(filtered_data, quotedate, calltype)
            quotedate = int(quotedate)
    
            parms = inputs.filter_yield_to_worst_parms(quotedate, parms, yield_to_worst)
            parms = inputs.create_weight(parms, wgttype, lam1=1, lam2=2)

            curve, prices, bondpv = outputs.calculate_rate_notax(parms, quotedate, breaks, curvetype, wgttype, lam1, lam2,
                                                                 padj, padjparm)
            price_yield_df = outputs.get_predicted_actual_yieldprice_notax(parms, bondpv, prices, quotedate, curvetype, padj)
            price_yield_df.insert(0, 'QuoteDate', quotedate)
            price_yield_df.insert(0, 'type', curvetype)
            # Make multi-index
            price_yield_df.set_index(['type','QuoteDate'],inplace=True,drop=False)
            
            # Aggregate curve and price_yield_df data
            #curve_df = pd.DataFrame(curve[3].reshape(1, -1), index=[quotedate], columns=breaks)
            #curve_data_list.append(curve_df)
            # Change from Kathy's - build up list of full curves (not just rates)
            curve_data_list.append(curve)
    
            price_yield_data_list.append(price_yield_df)
            
            print(quotedate)
        
        #final_curve_df = pd.concat(curve_data_list)
        # Change from Kathy's - make list into dataframe, then define columns, sort
        #final_curve_df = final_curve_df.sort_index()
        
        #final_price_yield_df = pd.concat(price_yield_data_list)
        #final_price_yield_df = final_price_yield_df.sort_values(by=['QuoteDate', 'MatYr', 'MatMth', 'MatDay'])
        
        #all_dfs[curvetype] = {'curve_df': final_curve_df,
#                              'price_yield_df': final_price_yield_df}
        
# NB - need to fix up tax
        if tax:
            tax_spd = np.array(["tax2_spd", "tax3_spd"])
            curve_tax_df = pd.DataFrame(curve_tax[3].reshape(1, -1), index=[quotedate], columns=np.append(breaks, tax_spd))
            curve_tax_data_list.append(curve_tax_df)
            curve_tax = outputs.calculate_rate_with_tax(parms, quotedate, breaks, curvetype, wgttype, lam1, lam2, padj, padjparm)
            final_curve_tax_df = pd.concat(curve_tax_data_list)
            final_curve_tax_df = final_curve_tax_df.sort_index()
            all_dfs[curvetype]['curve_tax_df'] = final_curve_tax_df
        

    # Convert the accumulated list of curves into a df and sort by index 
    # The following is necessary because we want to have a multi-index (on 
    # quotedate and curve type) but the quote date is a np.ndarray and 
    # multi-index on that does not work. So:
        # Create a new column and name it 'quotedate_ind'
        # Convert to integer and then create index
        # Also create type_ind, so that we retain the original type and quotedate in the df
    final_curve_df = pd.DataFrame(curve_data_list)
    final_curve_df.columns = ['type','quotedate','breaks','rates']
    final_curve_df['quotedate_ind'] = final_curve_df['quotedate']
    final_curve_df['quotedate_ind'] = final_curve_df['quotedate_ind'].map(dates.JuliantoYMDint)
    final_curve_df['quotedate_ind'] = final_curve_df['quotedate_ind'].map(int)
    final_curve_df['type_ind'] = final_curve_df['type']
    final_curve_df.set_index(['type_ind','quotedate_ind'],inplace=True,drop=True)
    final_curve_df = final_curve_df.sort_index()

    # Converted accumulate list of actual vs predicted into df
    final_price_yield_df = pd.concat(price_yield_data_list)
    final_price_yield_df = final_price_yield_df.sort_values(by=[ 'MatYr', 'MatMth', 'MatDay'])
    final_price_yield_df = final_price_yield_df.sort_index()

    # Export to CSV
    final_curve_df.to_csv(os.path.join(OUTPUT_DIR, f'curve_{filename}.csv'))
    if tax:
        final_curve_tax_df.to_csv(os.path.join(OUTPUT_DIR, f'curve_tax_{filename}.csv'))
    final_price_yield_df.to_csv(os.path.join(OUTPUT_DIR, f'price_yield_{filename}.csv'))

    # Export to Excel on separate sheets
    excel_file_path = os.path.join(OUTPUT_DIR, f'{filename}_data.xlsx')
    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        final_curve_df.to_excel(writer, sheet_name='Curve Data')
        if tax:
            final_curve_tax_df.to_excel(writer, sheet_name='Curve Tax Data')
        final_price_yield_df.to_excel(writer, sheet_name='Price Yield Data')

        
#    return all_dfs
    return final_curve_df, final_price_yield_df


# Attempt
# yield_to_worst = True
# settledate = 19460330
# filepath = '../../data/USTMonthly.csv'  # '../../data/1916to2024_YYYYMMDD.csv'
# bonddata = data_processing.clean_crsp(filepath)
# bonddata = data_processing.create_weight(bonddata, bill_weight=1)
# parms = inputs.read_and_process_csvdata(bonddata, settledate, 0)


def main():
    
    OUTPUT_DIR = '../../output'
    # New wrds data
    filepath = '../../data/USTMonthly.csv'  # '../../data/1916to2024_YYYYMMDD.csv'
    # filepath = 'C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024/curve_utils/src/data/USTMonthly.csv'
    outfile = '1986t02000'
    outfile = 'test2000'
    
    # Define user inputs
    # quotedate = 20150130 # 19321231
    calltype = 0  # 0 to keep all bonds, 1 for callable bonds, 2 for non-callable bonds
    curvetypes = ['pwcf', 'pwlz', 'pwtf']
    start_date = 19860701
    start_date = 20000101
    end_date = 20000331
    breaks = np.array([7/365.25, 14/365.25, 21/365.25, 28/365.25, 35/365.25, 52/365.25, 92/365.25, 
                       184/365.25, 1, 2, 4, 8, 16, 24, 32])  # np.array([0.0833, 0.5, 1.,2.,5.,10.,20.,30.])
    curve_points_yr1 = np.arange(.01,4,.01)
    curve_points_yr2 = np.arange(4.01,32,.01)
    curve_points_yr3 = np.arange(0,32,.01)
    
    bonddata = data_processing.clean_crsp(filepath)

    df_curve, df_price_yield = produce_curve_wrapper(bonddata, curvetypes, start_date, end_date, breaks, outfile, calltype=calltype,
                                               wgttype=1, lam1=1, lam2=2, padj=False, padjparm=0, yield_to_worst=True, tax=False)

    # Store dfs - curves as .csv, predicted vs actual as pickle
    df_curve.to_csv(OUTPUT_DIR+'/'+outfile+'_curve.csv')
    df_price_yield.to_pickle(OUTPUT_DIR+'/'+outfile+'_predyld.pkl')

    # Read local curves from csv
    df_curve = pd.read_csv(OUTPUT_DIR+'/'+outfile+'_curve.csv',index_col=[0,1])

    # Plotly plot for January
    #plot_rates.plot_rates_by_one_month(all_dfs_1986to2000, 'pwcf', 1, taxflag=False, taxability=1)

    # Plot all types of curves and store to a selected folder per quotedate
    selected_breaks_1 = [7/365.25, 14/365.25, 21/365.25, 28/365.25, 35/365.25, 52/365.25, 92/365.25, 184/365.25, 1, 2, 4]
    selected_breaks_2 = [4, 8, 16, 24, 32]
    selected_breaks = [7/365.25, 14/365.25, 21/365.25, 28/365.25, 35/365.25, 52/365.25, 92/365.25, 184/365.25, 1, 2, 4, 8, 16, 24, 32]
    fwdcurve1 = plot_rates.plot_fwdrate_from_output(OUTPUT_DIR, outfile, '0-4yrs', outfile, 12, selected_breaks_1, curve_points_yr1, taxflag=False, taxability=1)
    fwdcurve2 = plot_rates.plot_fwdrate_from_output(OUTPUT_DIR, outfile, '4-32yrs', outfile, 12, selected_breaks_2, curve_points_yr2, taxflag=False, taxability=1)
    fwdcurve = plot_rates.plot_fwdrate_from_output(OUTPUT_DIR, outfile, '0-32yrs', outfile, 12, selected_breaks, curve_points_yr3, taxflag=False, taxability=1)

#    fwdcurve1.to_csv(os.path.join(OUTPUT_DIR, f'1986to2000fwdwbreaks_0-4yrs.csv'), index=False)
#    fwdcurve2.to_csv(os.path.join(OUTPUT_DIR, f'1986to2000fwdwbreaks_4-32yrs.csv'), index=False)
#    fwdcurve.to_csv(os.path.join(OUTPUT_DIR, f'1986to2000fwdwbreaks_0-32yrs.csv'), index=False)

    # Output plots to latex
    plotlatex.create_latex_with_images(OUTPUT_DIR, '1986to2000', '1986to2000fwdrate.tex')

if __name__ == "__main__":
    main()
