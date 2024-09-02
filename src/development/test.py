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


#%% Test 27-apr-24

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

# https://stackoverflow.com/questions/42755214/how-to-keep-numpy-array-when-saving-pandas-dataframe-to-csv
import ast
def from_np_array(array_string):
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))

# Store dfs - curves as .csv, predicted vs actual as pickle
#df_curve.to_csv(OUTPUT_DIR+'/'+outfile+'_curve.csv')
#df_price_yield.to_pickle(OUTPUT_DIR+'/'+outfile+'_predyld.pkl')

# Read local curves from csv
df_curve = pd.read_csv(OUTPUT_DIR+'/'+outfile+'_curve.csv',index_col=[0,1],
                       converters={'quotedate':from_np_array,'breaks': from_np_array,'rates': from_np_array})
df_curve = pd.read_pickle(OUTPUT_DIR+'/'+outfile+'_curve.pkl')
df_price_yield = pd.read_pickle(OUTPUT_DIR+'/'+outfile+'_predyld.pkl')

df_curve.loc['pwcf',20000131]['rates']
(df_curve.loc['pwcf',20000131]['breaks'] - df_curve.loc['pwcf',20000131]['quotedate']) 
dates.JuliantoYMDint(df_curve.loc['pwcf',20000131]['breaks'])
x1 = df_price_yield.loc['pwcf',20000131]

#%%

OUTPUT_DIR = '../../output'
# New wrds data
filepath = '../../data/USTMonthly.csv'  # '../../data/1916to2024_YYYYMMDD.csv'
# filepath = 'C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024/curve_utils/src/data/USTMonthly.csv'
outfile = '1986t02000'
outfile = 'test2000'


all_dfs_1986to2000 = util.load_dfs_from_pickle(output_dir=OUTPUT_DIR, filename=outfile)

x1 = all_dfs_1986to2000['pwcf']['curve_df']  # pulls df from dictionary
x2 = all_dfs_1986to2000['pwcf']['price_yield_df']  # pulls df from dictionary

x2 = x1.loc[19860731]


quotedate=dates.YMDtoJulian(19321231)
breaks = np.array([0.0833, 0.5, 1.,2.,5.,10.,20.,30.])
breakdates = dates.CalAdd(quotedate,nyear=breaks)
rates = np.array([0.02] * len(breaks))
ratescc = 2.*np.log(1+rates/2.)
curvepwcf = ['pwcf',quotedate,breakdates,ratescc]
curvepwcf[1] = int(curvepwcf[1])
curvepwtf = ['pwtf',quotedate,breakdates,ratescc]
curvepwtf[1] = int(curvepwtf[1])


xdflist = []
xdflist.append(curvepwcf)
x1 = curvepwcf.copy()
x1[1] = x1[1] + 1
xdflist.append(x1)
xdflist.append(curvepwtf)
x2 = curvepwtf.copy()
x2[1] = x2[1] + 1
xdflist.append(x2)
x1df = pd.DataFrame(xdflist)

x2df = x1df.copy()
x2df.columns = ['type','quotedate_b','rates','breaks']
x2df['quotedate'] = x2df['quotedate_b']
x2df['quotedate'] = x2df['quotedate'].map(int)
x2df.set_index(['type','quotedate'],inplace=True,drop=False)
x2df.drop('quotedate',axis=1,inplace=True)
x2df.columns = ['type','quotedate','rates','breaks']


x1df.columns = ['type','quotedate','rates','breaks']
x1df.set_index('quotedate',inplace=True,drop=False)
x1df.set_index(['type','quotedate'],inplace=True,drop=False)
x1df.set_index('type',inplace=True,drop=False)
x1df = x1df.sort_index()
x1df.set_index('type',inplace=True,drop=False)
x1df.set_index(['quotedate','type'],inplace=True,drop=False)


x1df = pd.DataFrame(curvepwcf)

x2df = x1df.transpose()

x2df.columns = ['type','quotedate','rates','breaks']







