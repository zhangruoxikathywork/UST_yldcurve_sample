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
OUTPUT_DIR = '../../output'

#####################################################################################
## Define user inputs

# wrds data import
filepath = '../../data/USTMonthly.csv'
# filepath = '/Users/kristykwon/Documents/RAShip/2024_RAShip/UST-yieldcurves_2024/curve_utils/data/USTMonthly.csv'
# filepath = 'C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024/curve_utils/data/USTMonthly.csv'

# Export file name

estfile = 'pycurve1925_193212_py' # .pkl
estfile = 'pycurve193001_194212_py' # .pkl
estfile = 'pycurve194012_198606_py' # .pkl
estfile = 'pycurve198606_present_py' # .pkl
estfile = 'test1950'

# Start and end date
start_date = 19860701
start_date = 19800101
end_date = 19830101
start_date = 19900101
end_date = 19950101
start_date = 19250101
end_date = 19330101
#start_date = 19800101
#end_date = 19851231

# Inputs
yvolsflg = False  # to estimate
yvols = 0
yield_to_worst = True # False - w/opt
tax = False
calltype = 0  # 0 to keep all bonds, 1 for callable bonds, 2 for non-callable bonds
curvetypes =  ['pwcf']
plot_points_yr = np.arange(0,32,.01)  # np.arange(.01,4,.01), np.arange(4.01,32,.01)
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

# 1wk, 1mth, 3mth, 6mth, 1yr, 2, 3, 5, 7, 10, 20, 30

#%% Script for running curves, producing pb, zb, annuity tables, returns tables

#####################################################################################
## Clean raw CRSP data
bonddata = data_processing.clean_crsp(filepath)

#####################################################################################
### Produce forwards, actual vs predicted price & yield dfs
t0_calcfwd = time.time()
df_curve, df_price_yield = wrap.produce_curve_wrapper(bonddata, curvetypes, start_date, end_date, breaks, estfile,
                                                    calltype=calltype, wgttype=wgttype, lam1=lam1, lam2=lam2,
                                                    padj=padj, padjparm=padjparm, yield_to_worst=yield_to_worst, 
                                                    tax=tax, yvolsflg=yvolsflg, yvols=yvols)
t1_calcfwd = time.time()


