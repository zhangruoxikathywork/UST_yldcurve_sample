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
sys.path.append('../../src/package')
sys.path.append('../../../BondsTable')
sys.path.append('../../tests')
sys.path.append('../../data')
sys.path.append('../../../FORTRAN2024/code')
OUTPUT_DIR = '../../output'

# #os.chdir("C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024/curve_utils/src/development")
#os.chdir("/Users/kristykwon/Documents/RAShip/2024_RAShip/UST-yieldcurves_2024/curve_utils/src/development")
os.chdir("D:/好好学习/UST_Project/UST-yieldcurves_2024/curve_utils/src/development")
# os.chdir("/Users/tcoleman/tom/yields/New2024/progs/curve_utils/src/development") # TSC added to change the working directory because he always forgets

# BASE_PATH = "/Users/kristykwon/Documents/RAShip/2024_RAShip/UST-yieldcurves_2024"
# #BASE_PATH = "C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024"
BASE_PATH = "D:/好好学习/UST_Project/UST-yieldcurves_2024"
# BASE_PATH = "/Users/tcoleman/tom/yields/New2024/progs"

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


#%% Define inputs

# Make this a script (so variables stay in workspace) instead of "main"
# To make back to main, just uncomment and then indent code below
#def main():

# Do some timing
t0_all = time.time()

# Define paths
# BASE_PATH = "C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024"
# BASE_PATH = "/Users/tcoleman/tom/yields/New2024/progs"
# BASE_PATH = "/Users/kristykwon/Documents/RAShip/2024_RAShip/UST-yieldcurves_2024"
# OUTPUT_DIR = os.path.join(BASE_PATH, 'curve_utils/output')
# OUTPUT_DIR = os.path.join(BASE_PATH, 'FORTRAN2024/results_15fwds')
OUTPUT_DIR = '../../../FORTRAN2024/results_15fwds'
OUTPUT_DIR = '../../output'

#####################################################################################
## Define user inputs

# wrds data import
filepath = '../../data/USTMonthly.csv'
# filepath = '/Users/kristykwon/Documents/RAShip/2024_RAShip/UST-yieldcurves_2024/curve_utils/data/USTMonthly.csv'
# filepath = 'C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024/curve_utils/data/USTMonthly.csv'

# Export file name
## Convention - need to showcase date, vol, opt_type, ytw, tax, etc.
estfile = '1986to2000'
estfile = 'testshort'
estfile = 'pycurve1925_193212' # .pkl
estfile = 'pycurve193001_194212' # .pkl
estfile = 'pycurve194012_198606' # .pkl
estfile = 'pycurve198606_present' # .pkl
estfile = 'testshort'
estfile = 'test2000opt_LnY_035'
# estfile = 'test2000opt_NormY_001'
# estfile = 'test2000opt_LnP_006'

# Start and end date
start_date = 19860701
start_date = 19800101
end_date = 19830101
start_date = 19900101
end_date = 19950101
start_date = 20000101
end_date = 20010101
#start_date = 19800101
#end_date = 19851231

# Inputs
yvolsflg = True  # True - to estimate vols
yvols = 0.35  # the reasonable start for LnY, NormY, LnP should be around 0.35, 0.01, 0.06. Too small will fail
opt_type="LnY"  # The value of opt_type must be LnY, NormY, SqrtY or LnP
yield_to_worst = False # False - w/opt

tax = False
calltype = 0  # 0 to keep all bonds, 1 for callable bonds, 2 for non-callable bonds
curvetypes =  ['pwcf', 'pwlz', 'pwtf']
# curvetypes = ['pwtf']
table_breaks_yr = np.array([0.0833, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])  
wgttype=1
lam1=1
lam2=2
sqrtscale=True
twostep = False
parmflag = True
padj = False
padjparm = 0

# TSC 27-apr-2024 Trying various sets of breaks - problems with too fine at short end
breaks = np.array([7/365.25, 14/365.25, 21/365.25, 28/365.25, 35/365.25, 52/365.25, 92/365.25, 
                   184/365.25, 1, 2, 4, 8, 16, 24, 32])  # np.array([0.0833, 0.5, 1.,2.,5.,10.,20.,30.])
breaks = np.array([14/365.25, 28/365.25, 52/365.25, 92/365.25, 
                   184/365.25, 1, 2, 4, 8, 16, 24, 32])  # np.array([0.0833, 0.5, 1.,2.,5.,10.,20.,30.])
breaks = np.array([28/365.25, 92/365.25, 
                   184/365.25, 1, 2, 4, 8, 16, 24, 32])  # np.array([0.0833, 0.5, 1.,2.,5.,10.,20.,30.])
breaks = np.array([round(7/365.25, 4), 30/365.25, 92/365.25, 
                   184/365.25, 1, 2, 3, 5, 7, 10, 20, 30])  # np.array([0.0833, 0.5, 1.,2.,5.,10.,20.,30.])
# Proposed base breaks: 1wk, 1mth, 3mth, 6mth, 1yr, 2, 3, 5, 7, 10, 20, 30
base_breaks = np.array([round(7/365.25,4), 0.0833, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])


#%% Script for running curves, producing pb, zb, annuity tables, returns tables

#####################################################################################
## Clean raw CRSP data
bonddata = data_processing.clean_crsp(filepath)
# breaks_df = pd.read_pickle(OUTPUT_DIR+'/'+'breaks_df.pkl')

#####################################################################################
### Produce forwards, actual vs predicted price & yield dfs
t0_calcfwd = time.time()
df_curve, df_price_yield = wrap.produce_curve_wrapper(bonddata, curvetypes, start_date, end_date, base_breaks, estfile,
                                                      OUTPUT_DIR, calltype=calltype, wgttype=wgttype, lam1=lam1, lam2=lam2,
                                                      padj=padj, padjparm=padjparm, yield_to_worst=yield_to_worst, 
                                                      tax=tax, yvolsflg=yvolsflg, yvols=yvols, opt_type=opt_type)
t1_calcfwd = time.time()

#####################################################################################
## Store dfs - curves as .csv, predicted vs actual as pickle - already store using produce_curve_wrapper
# Stupid, but need to change linewidth for numpy so that the .to_csv does not
# break lines into maximum width 75 chars. (When it does, cannot read back
# in properly)
# np.set_printoptions(linewidth=100000)
# df_curve.to_csv(OUTPUT_DIR+'/'+estfile+'_curve.csv')
# # Also write to pickle because writing to csv does not work exactly right for numpy arrays
# df_curve.to_pickle(OUTPUT_DIR+'/'+estfile+'_curve.pkl')
# # For actual vs predicted, use pickle because actual-vs-predicted will be large
# df_price_yield.to_pickle(OUTPUT_DIR+'/'+estfile+'_predyld.pkl')

#####################################################################################
## Read dfs from /output

df_curve = pd.read_csv(OUTPUT_DIR+'/'+estfile+'/'+estfile+'_curve.csv',index_col=[0,1],
                   converters={'quotedate':util.from_np_array,'breaks': util.from_np_array,'rates': util.from_np_array})
# May be just easier to write & read pickle, but .csv is humanly readable and transferrable
df_curve = pd.read_pickle(OUTPUT_DIR+'/'+estfile+'/'+estfile+'_curve.pkl')
df_price_yield = pd.read_pickle(OUTPUT_DIR+'/'+estfile+'/'+estfile+'_predyld.pkl')

t1_all = time.time()


# The parms_df that is returns has the updated par bond coupons (from the last curve
# run, usually pwtf)

print("--- %s overall time ---" % round(t1_all - t0_all,3))
print("--- %s calcfwd time ---" % round(t1_calcfwd - t0_calcfwd,3))
