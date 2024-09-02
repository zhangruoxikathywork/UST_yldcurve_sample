##########################################
# This is the wrapper program for        #
# cleaning data into parms, fitting fwd  #
# rates with and without                 #
# taxbility, produce predicted vs actual #
# price and yield. Store fwds as .csv    #
# and pickle. Plotting and writing       #
# tables moved to another script.        #
##########################################

'''
Overview
-------------
- Calculate and store forward rates dataframe - csv and pickle
- Produce and store predicted vs actual prices and yield table
- Produce and output par bond, zero bond, and annuity rates and price tables
- Produce and output returns tables


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
sys.path.append('../package')
sys.path.append('../../../BondsTable')
sys.path.append('../../tests')
sys.path.append('../../data')
sys.path.append('../../../FORTRAN2024/code')
OUTPUT_DIR = '../../output'
OUTPUT_DIR = '../../../FORTRAN2024/results_15fwds'

# #os.chdir("C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024/curve_utils/src/development")
#os.chdir("/Users/kristykwon/Documents/RAShip/2024_RAShip/UST-yieldcurves_2024/curve_utils/src/development")
os.chdir("/Users/tcoleman/tom/yields/New2024/progs/curve_utils/src/development") # TSC added to change the working directory because he always forgets
BASE_PATH = "/Users/kristykwon/Documents/RAShip/2024_RAShip/UST-yieldcurves_2024"
# #BASE_PATH = "C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024"
BASE_PATH = "/Users/tcoleman/tom/yields/New2024/progs"
paths = ['curve_utils/src/development', 'curve_utils/src/package', 'BondsTable',
         'curve_utils/tests', 'curve_utils/data']
for path in paths:
        sys.path.append(os.path.join(BASE_PATH, path))
OUTPUT_DIR = os.path.join(BASE_PATH, 'curve_utils/output')
# print(sys.path)
# sys.path.pop()    # removes last entry in sys.path

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
import wrapper as wrap
import plot_rates as xplot
import output_to_latexpdf as plotlatex


imp.reload(dates)
imp.reload(pv)
imp.reload(pvc)
imp.reload(inputs)
imp.reload(outputs)
imp.reload(pzb)
imp.reload(util)
imp.reload(wrap)
imp.reload(plotlatex)
imp.reload(xplot)


#%% Define inputs

# Define paths
# BASE_PATH = "C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024"
# BASE_PATH = "/Users/tcoleman/tom/yields/New2024/progs"
# OUTPUT_DIR = os.path.join(BASE_PATH, 'curve_utils/output')
# OUTPUT_DIR = os.path.join(BASE_PATH, 'FORTRAN2024/results_15fwds')
OUTPUT_DIR = '../../../FORTRAN2024/results_15fwds'

# Export file name
estfile = 'pycurve1925_193212' # .pkl       tax=2
estfile = 'pycurve193001_194212' # .pkl     tax=3
estfile = 'pycurve194012_198606' # .pkl     tax=1
estfile = 'pycurve198606_present' # .pkl    tax=1

# Inputs
yvolsflg = False # to estimate
yvols = 0
yield_to_worst = True # False - w/opt
tax = True
calltype = 0  # 0 to keep all bonds, 1 for callable bonds, 2 for non-callable bonds
wgttype = 1
lam1 = 1
lam2 = 2
crvtypes = ['pwcf']
rate_type = 'parbond'
sqrtscale = True
twostep = False
parmflag = True
yield_to_worst = True
durscale = False
fortran = True
opt_type='LnY'
padj = False  # padj = True
padjparm = 0


# Additional report inputs
estfile = 'pycurve1925_193212' # .pkl       tax=2
ctype = 'pwcf'  # or False
tax = 2
padj = False
padjparm = None  # like 0.1, or None
date = 20240706
start_date = 19250101
end_date = 19311231

estfile = 'pycurve193001_194212' # .pkl     tax=3
ctype = 'pwcf'  # or False
tax = 3
padj = False
padjparm = None  # like 0.1, or None
date = 20240706
start_date = 19320101
end_date = 19411231

estfile = 'pycurve194012_198606' # .pkl     tax=1
ctype = 'pwcf'  # or False
tax = 1
padj = True
padjparm = 0.1  # like 0.1, or None
date = 20240706
start_date = 19420101
end_date = 19860630

estfile = 'pycurve198606_present' # .pkl    tax=1
ctype = 'pwcf'  # or False
tax = 1
padj = True
padjparm = 0  # like 0.1, or None
date = 20240706
start_date = 19860701
end_date = 20990101


# Define breaks
base_breaks = np.array([round(7/365.25,4), 0.0833, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]) # 1wk, 1mth, 3mth, 6mth, 1yr, 2, 3, 5, 7, 10, 20, 30
table_breaks_yr = np.array([0.0833, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])  

# Define plot points
curve_points_yr1 = np.arange(.01,4,.01)
curve_points_yr2 = np.arange(4.01,32,.01)
curve_points_yr3 = np.arange(0,32,.01)

plot_points_yr = np.arange(.01,30,.01)
plot_points_yr = np.arange(0,32,.01)  # np.arange(.01,4,.01), np.arange(4.01,32,.01)
plot_points_yr = np.concatenate((np.arange(0, 2, 1/365.25), 
                                 np.arange(2, 5, 0.01), 
                                 np.arange(5, 30, 0.02)))

#%%   Produce Tables

#####################################################################################
## Read dfs from /output
df_curve = pd.read_pickle(OUTPUT_DIR+'/'+estfile+'_curve.pkl')

#####################################################################################
### Produce and output par bond, zero bond, and annuity rates and price tables
parbd_rateprice_df, zerobd_rateprice_df, annuity_rateprice_df, pbparms_df = pzb.pb_zb_anty_wrapper(
    df_curve, table_breaks_yr, estfile, OUTPUT_DIR, twostep=twostep, parmflag=parmflag, padj=padj)
parbd_rate, parbd_cprice, zerobd_rate, zerobd_cprice, annuity_rate, annuity_cprice = pzb.seperate_pb_zb_anty_wrapper(
    parbd_rateprice_df, zerobd_rateprice_df, annuity_rateprice_df)

parbd_rate.columns = [f'{yr}YR' for yr in table_breaks_yr] + ['Max', 'Min']
zerobd_rate.columns = [f'{yr}YR' for yr in table_breaks_yr] + ['Max', 'Min']
annuity_rate.columns = [f'{yr}YR' for yr in table_breaks_yr] + ['Max', 'Min']

#####################################################################################
### Produce monthly returns table
return_df = pzb.calc_par_total_ret(df_curve, table_breaks_yr, twostep, parmflag, padj)
total_ret, yld_ret, yld_excess = pzb.seperate_returns_wrapper(return_df)
# Save Returns df as csv and pkl
# return_df.to_csv(OUTPUT_DIR+'/'+estfile+'_return.csv')
# return_df.to_pickle(OUTPUT_DIR+'/'+estfile+'_return.pkl')

#####################################################################################
### Write individual dfs to csv
dfs = [parbd_rate, zerobd_rate, annuity_rate, total_ret, yld_ret, yld_excess]
names = ['parbd_rate', 'zerobd_rate', 'annuity_rate', 'total_ret', 'yld_ret', 'yld_excess']
dfs_100 = [util.multiply_numeric_entries_by_100(df) for df in dfs]
dfs_adj = [util.filter_by_quotedate(df, start_date, end_date) for df in dfs_100]

util.export_to_csv(dfs_adj, names, OUTPUT_DIR, estfile)
util.export_to_pickle(dfs_adj, names, OUTPUT_DIR, estfile)

#####################################################################################
### Write to txt and pdf

for df, name in zip(dfs_adj, names):
    plotlatex.write_df_to_txt(df, OUTPUT_DIR, estfile, name, ctype)
    plotlatex.write_txt_to_pdf(OUTPUT_DIR, estfile, name, ctype, tax, padjparm, date)

# plotlatex.write_df_to_txt(total_ret, OUTPUT_DIR, estfile, 'total_ret', ctype)
# plotlatex.write_txt_to_pdf(OUTPUT_DIR, estfile, 'total_ret', ctype, tax, padjparm, date)

# plotlatex.write_df_to_txt(yld_ret, OUTPUT_DIR, estfile, 'yld_ret', ctype)
# plotlatex.write_txt_to_pdf(OUTPUT_DIR, estfile, 'yld_ret', ctype, tax, padjparm, date)

# plotlatex.write_df_to_txt(yld_excess, OUTPUT_DIR, estfile, 'yld_excess', ctype)
# plotlatex.write_txt_to_pdf(OUTPUT_DIR, estfile, 'yld_excess', ctype, tax, padjparm, date)

# plotlatex.write_df_to_txt(parbd_rate, OUTPUT_DIR, estfile, 'parbd_rate', ctype)
# plotlatex.write_txt_to_pdf(OUTPUT_DIR, estfile, 'parbd_rate', ctype, tax, padjparm, date)

# plotlatex.write_df_to_txt(zerobd_rate, OUTPUT_DIR, estfile, 'zerobd_rate', ctype)
# plotlatex.write_txt_to_pdf(OUTPUT_DIR, estfile, 'zerobd_rate', ctype, tax, padjparm, date)

# plotlatex.write_df_to_txt(annuity_rate, OUTPUT_DIR, estfile, 'annuity_rate', ctype)
# plotlatex.write_txt_to_pdf(OUTPUT_DIR, estfile, 'annuity_rate', ctype, tax, padjparm, date)

#####################################################################################
### Combine pdfs
# RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results_1925_present')
plotlatex.combine_pdfs(OUTPUT_DIR, 'total_ret_pwcf')
plotlatex.combine_pdfs(OUTPUT_DIR, 'yld_ret_pwcf')
plotlatex.combine_pdfs(OUTPUT_DIR, 'yld_excess_pwcf')
plotlatex.combine_pdfs(OUTPUT_DIR, 'parbd_rate_pwcf')
plotlatex.combine_pdfs(OUTPUT_DIR, 'zerobd_rate_pwcf')
plotlatex.combine_pdfs(OUTPUT_DIR, 'annuity_rate_pwcf')

plotlatex.combine_csvs(OUTPUT_DIR, 'curve')
plotlatex.combine_csvs(OUTPUT_DIR, 'total_ret_pwcf')
plotlatex.combine_csvs(OUTPUT_DIR, 'yld_ret_pwcf')
plotlatex.combine_csvs(OUTPUT_DIR, 'yld_excess_pwcf')
plotlatex.combine_csvs(OUTPUT_DIR, 'parbd_rate_pwcf')
plotlatex.combine_csvs(OUTPUT_DIR, 'zerobd_rate_pwcf')
plotlatex.combine_csvs(OUTPUT_DIR, 'annuity_rate_pwcf')

#%%   Produce Plots

#####################################################################################
# Create par bond yield curve animation
pzb.pb_animation_wrapper(OUTPUT_DIR, estfile, df_curve, None, plot_points_yr, table_breaks_yr,
                         crvtypes, rate_type, opt_type, yield_to_worst, twostep, parmflag, padj,
                         sqrtscale, durscale, fortran, start_date, end_date, fps=1)  #, create_video=True, , output_format='gif'

crvtype = 'pwcf'
image_folder = os.path.join(OUTPUT_DIR, estfile, 'fwd_rates')

pzb.plot_fwdrate_wrapper(OUTPUT_DIR, estfile, df_curve, plot_points_yr, False, sqrtscale, yield_to_worst, start_date, end_date)
pzb.create_animation_from_images(image_folder, OUTPUT_DIR, estfile, file_type='fwd_rates', fps=1, output_format='gif')



# # Could also make a movie from all png files in a folder
# image_folder = os.path.join(OUTPUT_DIR, estfile)
# pzb.create_animation_from_images(image_folder, OUTPUT_DIR, estfile, crvtype='pwcf',
#                                  file_type='pbrate', fps=1, output_format='gif')

