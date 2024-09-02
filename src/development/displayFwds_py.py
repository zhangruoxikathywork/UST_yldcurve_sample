##########################################
# This is the wrapper program for        #
# displaying forwards: plots and tables. #
# Read data in from stored .csv or pickel#
##########################################

'''
Overview
-------------
- Plot forward curves for all curve types
- Plot and output single curve graphs with actual yields as scatters
- Plot different estimations for comparison - yield to worst vs. option adjusted yield

Requirements
-------------
...
parzeroBond.py
plot_rates.py
results from calcFwds.py

'''

#%% Import python packages
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
#import scipy.optimize as so   # I think these not needed for plotting (when not fitting fwds) (TSC 7-jun-24)
import scipy.sparse as sp
import time as time
import cProfile
import re
# For "converter" to convert strings stored with .csv to numpy arrays
# https://stackoverflow.com/questions/42755214/how-to-keep-numpy-array-when-saving-pandas-dataframe-to-csv
import ast   # Abstract Syntax Trees - used in the "converter" for reading .csv

#%% Import py files

## Fixing Paths
sys.path.append('../../src/package')
sys.path.append('../../../BondsTable')
sys.path.append('../../tests')
sys.path.append('../../data')
OUTPUT_DIR = '../../output'

#os.chdir("C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024/curve_utils/src/development")
os.chdir("/Users/tcoleman/tom/yields/New2024/progs/curve_utils/src/development")
BASE_PATH = "C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024"
BASE_PATH = "/Users/tcoleman/tom/yields/New2024/progs"
paths = ['curve_utils/src/development', 'curve_utils/src/package', 'BondsTable',
         'curve_utils/tests', 'curve_utils/data']
for path in paths:
        sys.path.append(os.path.join(BASE_PATH, path))
OUTPUT_DIR = os.path.join(BASE_PATH, 'curve_utils/output')

# TSC added to change the working directory because he always forgets
# os.chdir('/Users/tcoleman/tom/yields/New2024/progs/curve_utils/src/development')
# print(sys.path)
# sys.path.pop()    # removes last entry in sys.path

import DateFunctions_1 as dates
import pvfn as pv
import pvcover as pvc
#import discfact as df
import Curve_Plotting as cp
import parzeroBond as pzb
import plot_rates as xplot
import util_fn as util
import output_to_latexpdf as plotlatex
#import curve_utils.src.development.output_to_latexpdf as plotlatex

imp.reload(dates)
imp.reload(pv)
imp.reload(pvc)
imp.reload(cp)
imp.reload(pzb)
imp.reload(plotlatex)
imp.reload(xplot)
imp.reload(util)


#%% Main script - Define user inputs and create plots

# Define paths
# BASE_PATH = "C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024"
# BASE_PATH = "/Users/tcoleman/tom/yields/New2024/progs"
# OUTPUT_DIR = os.path.join(BASE_PATH, 'curve_utils/output')
# OUTPUT_DIR = os.path.join(BASE_PATH, 'FORTRAN2024/results_15fwds')
OUTPUT_DIR = '../../output'

# Import fwd df
estfile = '1986to2000'
estfile = 'testshort'
estfile = 'test2000opt_030'
estfile = 'test2000opt_015'
estfile = 'test1990opt_vol_pwtf'
estfile = 'test2000opt_vol'
estfile = 'test2000opt_vol_old'
estfile = 'test2000opt_LnY_035'
estfile = 'testshort'
estfile = 'pycurve1925_193212' # .pkl
estfile = 'pycurve193001_194212' # .pkl
estfile = 'pycurve194012_198606' # .pkl
estfile = 'pycurve198606_present' # .pkl
estfile = 'testshort'

# Start and end date
start_date = 19860701
start_date = 19800101
end_date = 19830101
start_date = 19900101
end_date = 19950101
start_date = 20000101
end_date = 20030101
#start_date = 19800101
#end_date = 19851231

# Inputs
yvolsflg = True  # to estimate
yvols = 0.35
opt_type="LnY"  # The value of opt_type must be LnY, NormY, SqrtY or LnP
yield_to_worst = False # False - w/opt

tax = False
calltype = 0  # 0 to keep all bonds, 1 for callable bonds, 2 for non-callable bonds
curvetypes =  ['pwcf', 'pwlz', 'pwtf'] # ['pwtf']
wgttype = 1
lam1 = 1
lam2 = 2
sqrtscale = True
durscale = True
twostep = False
parmflag = True
padj = False
padjparm = 0
fortran = False

# Additional report inputs
ctype = 'pwcf'  # or False
tax = False
padjparm = None
date = None

# Define breaks
base_breaks = np.array([round(7/365.25,4), 0.0833, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]) # 1wk, 1mth, 3mth, 6mth, 1yr, 2, 3, 5, 7, 10, 20, 30
table_breaks_yr = np.array([0.0833, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])  

# Define plot points
curve_points_yr1 = np.arange(.01,4,.01)
curve_points_yr2 = np.arange(4.01,32,.01)
curve_points_yr3 = np.arange(0,32,.01)

plot_points_yr = np.arange(.01,30,.01)
plot_points_yr = np.arange(0,32,.01)  # np.arange(.01,4,.01), np.arange(4.01,32,.01)
plot_points_yr = np.round(np.arange(.01,30,.01)*365.25) / 365.25
plot_points_yr = np.concatenate((np.arange(0, 2, 1/365.25), 
                                 np.arange(2, 5, 0.01), 
                                 np.arange(5, 30, 0.02)))


#%%   Produce Tables

#####################################################################################
## Read dfs from /output
df_curve = pd.read_csv(OUTPUT_DIR+'/'+estfile+'/'+estfile+'_curve.csv',index_col=[0,1],
                   converters={'quotedate':util.from_np_array,'breaks': util.from_np_array,'rates': util.from_np_array})
# May be just easier to write & read pickle, but .csv is humanly readable and transferrable
df_curve = pd.read_pickle(OUTPUT_DIR+'/'+estfile+'/'+estfile+'_curve.pkl')
df_price_yield = pd.read_pickle(OUTPUT_DIR+'/'+estfile+'/'+estfile+'_predyld.pkl')

#####################################################################################
### Produce and output par bond, zero bond, and annuity rates and price tables
parbd_rateprice_df, zerobd_rateprice_df, annuity_rateprice_df, pbparms_df = pzb.pb_zb_anty_wrapper(
    df_curve, table_breaks_yr, estfile, OUTPUT_DIR, twostep=twostep, parmflag=parmflag, padj=padj)
parbd_rate, parbd_cprice, zerobd_rate, zerobd_cprice, annuity_rate, annuity_cprice = pzb.seperate_pb_zb_anty_wrapper(
    parbd_rateprice_df, zerobd_rateprice_df, annuity_rateprice_df)

#####################################################################################
### Produce monthly returns table
return_df = pzb.calc_par_total_ret(df_curve, table_breaks_yr, twostep, parmflag, padj)
total_ret, yld_ret, yld_excess = pzb.seperate_returns_wrapper(return_df)
# Save Returns df as csv and pkl
# return_df.to_csv(OUTPUT_DIR+'/'+estfile+'/'+estfile+'_return.csv')
# return_df.to_pickle(OUTPUT_DIR+'/'+estfile+'/'+estfile+'_return.pkl')

#####################################################################################
### Write individual dfs to csv
dfs = [parbd_rate, zerobd_rate, annuity_rate, total_ret, yld_ret, yld_excess]
names = ['parbd_rate', 'zerobd_rate', 'annuity_rate', 'total_ret', 'yld_ret', 'yld_excess']
dfs_100 = [util.multiply_numeric_entries_by_100(df) for df in dfs]

util.export_to_csv(dfs_100, names, os.path.join(OUTPUT_DIR, estfile), estfile)
util.export_to_pickle(dfs_100, names, os.path.join(OUTPUT_DIR, estfile), estfile)

#####################################################################################
### Write to txt and pdf
OUTPUT_DIR2 = f'{OUTPUT_DIR}/{estfile}'

plotlatex.write_df_to_txt(total_ret, OUTPUT_DIR2, estfile, 'total_ret', ctype)
plotlatex.write_txt_to_pdf(OUTPUT_DIR2, estfile, 'total_ret', ctype, tax, padjparm, date)

plotlatex.write_df_to_txt(yld_ret, OUTPUT_DIR2, estfile, 'yld_ret', ctype)
plotlatex.write_txt_to_pdf(OUTPUT_DIR2, estfile, 'yld_ret', ctype, tax, padjparm, date)

plotlatex.write_df_to_txt(yld_excess, OUTPUT_DIR2, estfile, 'yld_excess', ctype)
plotlatex.write_txt_to_pdf(OUTPUT_DIR2, estfile, 'yld_excess', ctype, tax, padjparm, date)

plotlatex.write_df_to_txt(parbd_rate, OUTPUT_DIR2, estfile, 'parbd_rate', ctype)
plotlatex.write_txt_to_pdf(OUTPUT_DIR2, estfile, 'parbd_rate', ctype, tax, padjparm, date)

plotlatex.write_df_to_txt(zerobd_rate, OUTPUT_DIR2, estfile, 'zerobd_rate', ctype)
plotlatex.write_txt_to_pdf(OUTPUT_DIR2, estfile, 'zerobd_rate', ctype, tax, padjparm, date)

plotlatex.write_df_to_txt(annuity_rate, OUTPUT_DIR2, estfile, 'annuity_rate', ctype)
plotlatex.write_txt_to_pdf(OUTPUT_DIR2, estfile, 'annuity_rate', ctype, tax, padjparm, date)


#%%   Produce Plots

#####################################################################################
### Plot forward curves for all curve types for the same month each year
plot_points_yr = np.arange(0,32,.01)  # np.arange(.01,4,.01), np.arange(4.01,32,.01)
selected_month = 12
taxflag = False
taxability = 1
sqrtscale = True
durscale = True
twostep = False
fwdcurve = xplot.plot_fwdrate_from_output(OUTPUT_DIR, estfile, estfile, selected_month, plot_points_yr,
                                         taxflag, taxability, sqrtscale)
#    fwdcurve.to_csv(os.path.join(OUTPUT_DIR, f'1986to2000fwdwbreaks_0-32yrs.csv'), index=False)
#xplot.plot_rates_by_one_month(all_dfs_1986to2000, 'pwcf', 12, taxflag=False, taxability=1)  # using Plotly
plotlatex.create_latex_with_images(OUTPUT_DIR, 'test2000', 'fwd_plots', 'test2000fwdrate.tex')  # Output plots to latex

#####################################################################################
### Plot and output single curve par bond curve with actual and predicted yields as scatters
### x-axis: yield to maturity vs duration

# Want to make these plot points into days (not fractional years) so that everything will work out
# OK when match the plot points against the par bond rates created from adding days
# So do this hack to round, then divide back by 365.25 (so that when multiply will give integer days)
plot_points_yr = np.round(np.arange(.01,30,.01)*365.25) / 365.25
crvtypes = ['pwtf']
rate_type = 'parbond'
durscale = True
yield_to_worst = False
yvols = 0
selected_month = 12
parmflag = True
padj = False

pzb.plot_act_pred_single_curve_wrapper(estfile, df_curve, df_price_yield, plot_points_yr, selected_month=selected_month,
                                   crvtype=crvtypes, rate_type=rate_type, opt_type=opt_type, yield_to_worst=yield_to_worst, yvols=yvols, 
                                   pbtwostep=twostep, parmflag=parmflag, padj=padj, sqrtscale=sqrtscale, durscale=False)
if durscale:
    pzb.plot_act_pred_single_curve_wrapper(output_dir=OUTPUT_DIR, estfile=estfile, curve_df=df_curve, df_price_yield=df_price_yield,
                                           plot_points_yr=plot_points_yr, selected_month=selected_month, crvtypes=crvtypes, 
                                           rate_type=rate_type, opt_type=opt_type, yield_to_worst=yield_to_worst, yvols=yvols, 
                                           pbtwostep=twostep, parmflag=parmflag, padj=padj, sqrtscale=sqrtscale, durscale=durscale)

#####################################################################################
## Plot different estimations for comparison - yield to worst vs. option adjusted yield
estim1 = 'test2000'  # yield to worst
estim2 = estfile # option adjusted yield
curve_df1 = pd.read_pickle(f'{OUTPUT_DIR}/{estim1}_curve.pkl')
curve_df2 = pd.read_pickle(f'{OUTPUT_DIR}/{estim2}_curve.pkl')
crvtypes = ['pwcf', 'pwtf']
labels = ['YldToWorst', 'OptAdjYld']
selected_month = 12 
taxflag = False
taxability = 1

pzb.plot_fwdrate_compare_wrapper(curve_df1, curve_df2, plot_points_yr, estfile, labels, selected_month=12,
                             taxflag=False, taxability=1, sqrtscale=sqrtscale)

#####################################################################################
# Create par bond yield curve animation

rate_type = 'parbond'
df_price_yield = df_price_yield
ctype = 'pwcf'  # or False
tax = False
padjparm = None
date = None

pzb.pb_animation_wrapper(OUTPUT_DIR, estfile, df_curve, df_price_yield, plot_points_yr, table_breaks_yr,
                         ['pwcf'], rate_type, opt_type, yield_to_worst, twostep, parmflag, padj,
                         sqrtscale, durscale=True, fortran=False)


#%%  Junk

# The parms_df that is returns has the updated par bond coupons (from the last curve
# run, usually pwtf)

print("--- %s overall time ---" % round(t1_all - t0_all,3))
print("--- %s calcfwd time ---" % round(t1_calcfwd - t0_calcfwd,3))


# if __name__ == "__main__":
#     main()
