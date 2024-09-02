#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:00:39 2024

@author: tcoleman
"""

'''
Overview
-------------
- Produce and output par bond, zero bond, and annuity rates and price tables
- Plot and output single curve graphs with actual yields as scatters
- Plot different estimations for comparison - yield to worst vs. option adjusted yield

Requirements
-------------
...
parzeroBond.py
plot_rates.py

'''

#%% Import python packages

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import ast
import re

#%% Import py files

sys.path.append('../../src/package')
sys.path.append('../../../BondsTable')
sys.path.append('../../tests')
sys.path.append('../../data')
OUTPUT_DIR = '../../output'

## Fixing Paths
# KZ: for my vscode directory bug...
# os.chdir("C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024/curve_utils/src/development") # KZ: for my vscode directory bug...
# sys.path.append('C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024/curve_utils/src/package')
# sys.path.append('C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024/BondsTable')
# sys.path.append('C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024/curve_utils/tests')
# sys.path.append('C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024/curve_utils/data')
# OUTPUT_DIR = 'C:/Users/zhang/OneDrive/Documents/GitHub/UST-yieldcurves_2024/curve_utils/output'

import discfact as df
import pvfn as pv
import DateFunctions_1 as dates
import Curve_Plotting as cp
import parzeroBond as pzb
import plot_rates as plot
import util_fn as util

#%% Testing plotting

# a = parbond_rates(curvepwcf,parbond_breaks,twostep=False)
# cp.zero_curve_plot(fwdcurve_list,plot_points_yr)

# ft0 = time.perf_counter()
# cp.forward_curve_plot(fwdcurve_list,plot_points_yr)
# ft1 = time.perf_counter()

# pt0 = time.perf_counter()
# cp.parbond_curve_plot(fwdcurve_list,plot_points_yr,twostep=True)
# pt1 = time.perf_counter()
# cp.parbond_curve_plot(fwdcurve_list,plot_points_yr)
# pt2 = time.perf_counter()

# print('Forwards: ', ft1-ft0)
# print('two-step time: ', pt1-pt0, 'one-step time: ',pt2-pt1)

#cp.parbond_curve_plot_old(breakdates, rates, plot_points_yr, quotedate)


#%% Testing Tables, make rates and prices for table display

def pb_zb_anty_wrapper(curve_df, table_breaks_yr, infile, twostep=True, parmflag=True, padj=False):
    """Produce par bond, zero bond, annuity rates and prices tables, """

    # Produce pb, zb, annuity prices and rates tables
    parbd_rates_df, parbd_prices_df = pzb.produce_pb_zb_anty_dfs(curve_df, table_breaks_yr,
     'parbond', twostep, parmflag, padj)
    zerobd_rates_df, zerobd_prices_df = pzb.produce_pb_zb_anty_dfs(curve_df, table_breaks_yr,
     'zerobond', parmflag, padj)
    annuity_rates_df, annuity_prices_df = pzb.produce_pb_zb_anty_dfs(curve_df, table_breaks_yr,
     'annuity', parmflag, padj)

     # Export to CSV
    dataframes = [parbd_rates_df, parbd_prices_df, zerobd_rates_df, zerobd_prices_df, annuity_rates_df, annuity_prices_df]
    names = ['parbd_rates', 'parbd_prices', 'zerobd_rates', 'zerobd_prices', 'annuity_rates', 'annuity_prices']
    util.export_to_csv(dataframes, names, OUTPUT_DIR, infile)

    return parbd_rates_df, parbd_prices_df, zerobd_rates_df, zerobd_prices_df, annuity_rates_df, annuity_prices_df


#%% Testing single curve graphs, actual vs par/zero bond curves

def plot_act_pred_single_curve(curve_df, df_price_yield, plot_dates, crvtype, plot_points_yr, rate_type='parbond',
                      pbtwostep=False, parmflag=True, padj=False, sqrtscale=True):
    """Plot actual rates as scatters and fitted curve (usually par bond) for curve types for given single dates."""
    for crvtype in crvtypes:
        for date in plot_dates:
            xplot_curve = curve_df.xs(crvtype,level=0).loc[date]
            xdf_price_yield = df_price_yield.xs(crvtype,level=0).loc[date]
            pzb.plot_singlecurve(xplot_curve, plot_points_yr, rate_type=rate_type, df_price_yield=xdf_price_yield,
                                pbtwostep=False, parmflag=True, padj=False, sqrtscale=True)


# xplot_curve = curve_df.xs(crvtype,level=0).loc[plot_dates[0]]
# pzb.plot_singlecurve(xplot_curve, plot_points_yr, rate_type = 'parbond', pbtwostep=False, parmflag=True, padj=False, sqrtscale=True)
#xplot_curve = curve_df.xs(crvtype,level=0).loc[20011231]
#xdf_price_yield = df_price_yield.xs(crvtype,level=0).loc[20011231]
#pzb.plot_singlecurve(xplot_curve, plot_points_yr, rate_type = 'parbond', df_price_yield = xdf_price_yield, pbtwostep=False, parmflag=True, padj=False,sqrtscale=False)
#pzb.plot_singlecurve(xplot_curve, plot_points_yr, rate_type = 'parbond', df_price_yield = xdf_price_yield, pbtwostep=False, parmflag=True, padj=False,sqrtscale=True)


#%% Examining callables for 29-dec--2000

#xplot_curve = curve_df.xs(crvtype,level=0).loc[20001229]
#xdf_price_yield = df_price_yield.xs(crvtype,level=0).loc[20001229]

#xquote = xplot_curve.iloc[1]
#xdp1325_2014 = 146.006
#xmatjul = dates.YMDtoJulian(20140515)
#xparms1325_2009 = pd.DataFrame([13.25, dates.YMDtoJulian(20090515)[0],2,100.,'A/A','eomyes',dates.YMDtoJulian(20090515)[0],False,1]).T
#xparms1325_2014 = pd.DataFrame([13.25, dates.YMDtoJulian(20140515)[0],2,100.,'A/A','eomyes',dates.YMDtoJulian(20090515)[0],True,1]).T
#xnoptyld1325_2009 = pv.bondYieldFromPrice_parms(xdp1325_2014,settledate=xplot_curve.iloc[1],parms=xparms1325_2009)
#xoptyld1325_2014 = pv.bondYieldFromPrice_callable(xdp1325_2014,settledate=xplot_curve.iloc[1],vol=0.15,parms=xparms1325_2014,callflag=True)
#xnoptyld1325_2014 = pv.bondYieldFromPrice_parms(xdp1325_2014,settledate=xplot_curve.iloc[1],parms=xparms1325_2014)


#%%  Plot different estimations for comparison - yield to worst vs. option adjusted yield

def plot_fwdrate_compare_wrapper(curve_df1, curve_df2, plot_points_yr, outfile, labels, selected_month=12,
        taxflag=False, taxability=1, sqrtscale=True):
    """Plot and output forward rate graphs comparing yield to worst and option adjusted yield."""
    
    for crvtype in crvtypes:
        plot.plot_fwdrate_compare(
            df_curve1=curve_df1, df_curve2=curve_df2, output_dir=OUTPUT_DIR, plot_folder=outfile,
            outfile=outfile, selected_month=12, plot_points_yr=plot_points_yr,
            curve_type=crvtype, taxflag=False, taxability=1, labels=labels, sqrtscale=sqrtscale)


# outfile = 'test1990'
# selected_month = 12
# curve_df1 = pd.read_pickle(OUTPUT_DIR+'/'+estim1+'_curve.pkl')
# curve_df2 = pd.read_pickle(OUTPUT_DIR+'/'+estim2+'_curve.pkl')
# labels=['YldToWorst','OptAdjYld']
# sqrtscale = True
# curve_type = 'pwcf'
# plot_rates.plot_fwdrate_compare(df_curve1=curve_df1,df_curve2=curve_df2,output_dir=OUTPUT_DIR,
#                                 plot_folder=outfile, outfile=outfile, selected_month=12,
#                                 plot_points_yr=plot_points_yr, curve_type=curve_type, taxflag=False,
#                                  taxability=1, labels=labels, sqrtscale=sqrtscale)

# curve_type = 'pwtf'
# sqrtscale = True
# plot_rates.plot_fwdrate_compare(df_curve1=curve_df1, df_curve2=curve_df2, output_dir=OUTPUT_DIR,
#                                 plot_folder=outfile, outfile=outfile, selected_month=12,
#                                 plot_points_yr=plot_points_yr, curve_type=curve_type, taxflag=False,
#                                  taxability=1, labels=labels, sqrtscale=sqrtscale)



## Main script

## Produce and output par bond, zero bond, and annuity rates and price tables
infile = 'test2000'
curve_df = pd.read_pickle(OUTPUT_DIR+'/'+infile+'_curve.pkl')
df_price_yield = pd.read_pickle(OUTPUT_DIR+'/'+infile+'_predyld.pkl')
table_breaks_yr = np.array([0.0833, 0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30])
twostep = True
parmflag = True
padj = False

parbd_rates_df, parbd_prices_df, zerobd_rates_df, zerobd_prices_df, annuity_rates_df, annuity_prices_df = pb_zb_anty_wrapper(
    curve_df, table_breaks_yr, infile, twostep=True, parmflag=True, padj=False)


## Plot and output single curve graphs with actual yields as scatters
plot_points_yr = np.arange(.01,30,.01)
crvtypes = ['pwcf', 'pwlz', 'pwtf']
plot_dates = [19900131,19911231,19921231,19931231,19941230]
plot_dates = [20000131,20011231,20021231]
rate_type = 'parbond'

plot_act_pred_single_curve(curve_df, df_price_yield, plot_dates, crvtypes, plot_points_yr, rate_type,
                           pbtwostep=False, parmflag=True, padj=False, sqrtscale=True)


## Plot different estimations for comparison - yield to worst vs. option adjusted yield
estim1 = 'test2000'
estim2 = 'test2000opt'
curve_df1 = pd.read_pickle(f'{OUTPUT_DIR}/{estim1}_curve.pkl')
curve_df2 = pd.read_pickle(f'{OUTPUT_DIR}/{estim2}_curve.pkl')
crvtypes = ['pwcf', 'pwlz', 'pwtf']
labels = ['YldToWorst', 'OptAdjYld']
outfile = 'test2000'
selected_month=12 
taxflag=False
taxability=1
sqrtscale=True

plot_fwdrate_compare_wrapper(curve_df1, curve_df2, plot_points_yr, outfile, labels, selected_month=12,
                             taxflag=False, taxability=1, sqrtscale=True)
