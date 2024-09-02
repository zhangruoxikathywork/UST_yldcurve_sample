##########################################
# This is to generate plots with the
# output we calculated.
##########################################

'''
Overview
-------------
1. Generate plots
(1) 

Requirements
-------------
../../data/WRDS CRSP UST dataset in csv format
crsp_data_processing.py
produce_inputs.py
calculate_ratesprices.py

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
import cProfile

sys.path.append('../../src/package')
sys.path.append('../../../BondsTable')
sys.path.append('../../tests')
sys.path.append('../../data')

import DateFunctions_1 as dates
import pvfn as pv
import pvcover as pvc
import discfact as df
import Curve_Plotting as plot
import CRSPBondsAnalysis as analysis
import util_fn as util

imp.reload(dates)
imp.reload(pv)
imp.reload(pvc)

import crsp_data_processing as data_processing
import produce_inputs as inputs
import calculate_ratesprices as outputs


#%% Single Plot

def plot_no_tax_curve(parms, prices, quotedate, breaks, plot_points_yr, wgttype, lam1, lam2,padj, padjparm):
    
    
    curvepwcf, prices, bondpv = outputs.calculate_rate_notax(parms, prices, quotedate, breaks, 'pwcf',
                                                     wgttype, lam1, lam2, padj, padjparm)
    curvepwlz, prices, bondpv = outputs.calculate_rate_notax(parms, prices, quotedate, breaks, 'pwlz',
                                                     wgttype, lam1, lam2, padj, padjparm)
    curvepwtf, prices, bondpv = outputs.calculate_rate_notax(parms, prices, quotedate, breaks, 'pwtf',
                                                     wgttype, lam1, lam2, padj, padjparm)
    fwd_curve_notax_list = [curvepwcf, curvepwlz, curvepwtf]

    plot.zero_curve_plot(fwd_curve_notax_list,plot_points_yr)
    plot.forward_curve_plot(fwd_curve_notax_list,plot_points_yr)
    # curve_points = quotedate + plot_points_yr*365.25
    # plot.par_bond_curve_plot(breaks, rates, curve_points, quotedate)
    

def plot_one_type_fwdcurve_tax(parms, prices, quotedate, breaks, plot_points_yr, curvetype,
                               wgttype, lam1, lam2, padj, padjparm):
    
    curvetax = outputs.calculate_rate_with_tax(parms, prices, quotedate, breaks, curvetype,
                                               wgttype, lam1, lam2, padj, padjparm)

    fwd_curve_tax1 = curvetax[3][:-2]
    fwd_curve_tax2 = fwd_curve_tax1 + curvetax[3][-2]
    fwd_curve_tax3 = fwd_curve_tax1 + curvetax[3][-1]
    breakdates = dates.CalAdd(quotedate,nyear=breaks)
    fwd_curve_tax1_list = [curvetype,quotedate,breakdates,fwd_curve_tax1]
    fwd_curve_tax2_list = [curvetype,quotedate,breakdates,fwd_curve_tax2]
    fwd_curve_tax3_list = [curvetype,quotedate,breakdates,fwd_curve_tax3]

    fwd_curve_tax = [fwd_curve_tax1_list, fwd_curve_tax2_list, fwd_curve_tax3_list]
    
    colors = ['black', 'darkorange', 'purple']
    xquotedate = fwd_curve_tax[0][1]
    curve_points = xquotedate + plot_points_yr*365.25
    taxability = ['1 fully taxable', '2 partially exempt', '3 fully tax exempt']
    
    for i in range(len(fwd_curve_tax)):
        xcurve = fwd_curve_tax[i]
        forward_curve = -365*np.log((df.discFact((curve_points+1), xcurve))/(df.discFact(curve_points, xcurve)))
        plt.plot(plot_points_yr, forward_curve, color=colors[i], label=taxability[i])
    plt.legend()
    plt.title(f'Forward Curves with Taxability for {fwd_curve_tax[0][0]}')
    plt.savefig(f"../../output/forward_curves_w_taxability_for_{fwd_curve_tax[0][0]}.png")
    plt.show()


#%% Plot Rate For a Selected Month each Year in Plotly

def plot_rates_by_one_month(df, curvetype, selected_month, taxflag=False, taxability=1):
    """Plot forward rates based on the selected month given the dictionary generated from the wrapper program."""

    if taxflag:
        curve_tax_df = df[curvetype]['curve_tax_df']
#        curve_tax_df.index = pd.to_datetime(curve_tax_df.index.astype(str), format='%Y%m%d')
        curve_tax_df.index = pd.to_datetime(curve_tax_df.index.astype(str), format='ISO8601')
        filtered_data = curve_tax_df[curve_tax_df.index.month == selected_month]

        if taxability == 1:
            filtered_data = filtered_data.iloc[:, :-2]  # the last two columns are the tax spd
        else:
            # Get the tax spread column based on taxability
            tax_col = f'tax{taxability}_spd'
            # Subtract the tax spd from all rate columns
            for col in filtered_data.columns[:-2]:  # Exclude the last two tax spread columns
                filtered_data[col] = filtered_data[col] + filtered_data[tax_col]
            filtered_data = filtered_data.drop(columns=['tax2_spd', 'tax3_spd'])
        
    else:
        # If no tax flags are considered
        curve_df = df[curvetype]['curve_df']
#        curve_df.index = pd.to_datetime(curve_df.index.astype(str), format='%Y%m%d')
        curve_df.index = pd.to_datetime(curve_df.index.astype(str), format='ISO8601')
        filtered_data = curve_df[curve_df.index.month == selected_month]

    # Reshape DataFrame for plotting (only if not considering tax flags)
    filtered_data = filtered_data.melt(var_name='Maturity', value_name='Rate', ignore_index=False)
    filtered_data['Year'] = filtered_data.index.year

    # Plot forward rates
    fig = px.line(filtered_data, x='Maturity', y='Rate', color='Year',
                  labels={'Year': f'rate in Year {curvetype}', 'Rate': 'Forward Rate'})
    fig.update_layout(title_text=f'Forward Rates for Month {selected_month} (Curve: {curvetype}, Taxability: {taxability})')
    fig.show()


def plot_fwdrate_from_output(output_dir, plot_folder, estfile, selected_month, 
                      plot_points_yr, taxflag=False, taxability=1, sqrtscale=False):
        """Plot and export to png in a created folder multiple forward curve plots by curvetype."""

        df_curve = pd.read_pickle(output_dir+'/'+estfile+'_curve.pkl')
#        df = util.load_dfs_from_pickle(output_dir, dict_file)
        curve_points_day = plot_points_yr * 365.25

        # Create a new folder to store plots
        path = f"{output_dir}/{plot_folder}"
        try:
            os.makedirs(path, exist_ok=True)
            print(f"Directory '{path}' created")
        except OSError as error:
            print(f"Creation of the directory {path} failed due to: {error}")

        if not taxflag:
            df_curve['month'] = dates.JuliantoYMD(df_curve['quotedate_jul'])[1]  # get the months and put into df
            filtered_curve_df = df_curve[df_curve['month'] == selected_month]

            if (sqrtscale):
                sqrt_plot_points_yr = np.sqrt(plot_points_yr)

            # Loop through dates to create plots
            for date in (filtered_curve_df.index.get_level_values('quotedate_ind').unique()):  # This references the 'quotedate_ind' index. Maybe there's a more elgant way?
                julian_date = dates.YMDtoJulian(date)   # get the julian 
                plot_points = julian_date + curve_points_day
                curves_all = filtered_curve_df.xs(date,level=1)

                for curve_type in curves_all.index.get_level_values('type_ind'):  # loop over curve types
                    xcurvedf = curves_all.loc[curve_type]  # all the elements of the saved curve
                    curve = xcurvedf[0:4]  # select out the specific curve (this date)
                    yield_to_worst = xcurvedf['ytw_flag']   # 1-aug-24 set yield-to-worst flag and yield vols from saved curve
                    if not(yield_to_worst):
                        yvols = np.round(xcurvedf['yvols'],decimals=4)
                    term1 = df.discFact(plot_points + 1, curve)
                    term2 = df.discFact(plot_points, curve)
                    result = -365 * np.log(term1 / term2)
                    if (sqrtscale):
                        plt.plot(sqrt_plot_points_yr, 100*result,label=f'{curve_type} - {date}')
                    else:
                        plt.plot(plot_points_yr, 100*result,label=f'{curve_type} - {date}')

            # sqrt root ticks and labeling
                if (sqrtscale):
                    x1 = max(plot_points_yr)
                    if (x1 > 20):
                        plt.xticks(ticks=np.sqrt([0.25,1,2,5,10,20,30]).tolist(),labels=['0.25','1','2','5','10','20','30'])
                    elif (x1 > 10):
                        plt.xticks(ticks=np.sqrt([0.25,1,2,5,10,20]).tolist(),labels=['0.25','1','2','5','10','20'])
                    elif (x1 >5):
                        plt.xticks(ticks=np.sqrt([0.25,1,2,5,7,10]).tolist(),labels=['0.25','1','2','5','7','10'])
                    else :
                        plt.xticks(ticks=np.sqrt([0.25,1,2,3,5]).tolist(),labels=['0.25','1','2','3','5'])
                    plt.xlabel('Maturity (Years, SqrRt Scale)')            
                else:
                    plt.xlabel('Maturity (Years)')            

                
                plt.title(f'Forward Rates for {date}')
                if yield_to_worst==False:
                    plt.title(f'Forward Rates for {date}, vol={yvols}')
                else:
                    plt.title(f'Forward Rates for {date}, Yield-to-Worst')
                plt.ylabel('Rate')
                plt.legend()
                plt.grid(True)
                full_path = f'{output_dir}/{plot_folder}/fwd_plots'
                os.makedirs(full_path, exist_ok=True)
                plt.savefig(f'{full_path}/{date}_fwd_plot.png')
                # plt.savefig(f'{output_dir}/{estfile}/{date}ForwardRate.png')
                plt.show()
                plt.close()
            

            
            return filtered_curve_df
        


def plot_fwdrate_from_output_old(output_dir, plot_folder, plot_name, estfile, selected_month, 
                      plot_points_yr, taxflag=False, taxability=1):
        """Plot and export to png in a created folder multiple forward curve plots by curvetype."""

        def calculate_forward_curve(row):
            date_as_int = row['quotedate_ymd']
            julian_date = row['quotedate_jul']
            plot_points = julian_date + curve_points_day
#            breaks = np.array(selected_breaks)*365 + julian_date
            curve = [row['type'], row['quotedate_jul'], row['breaks'], row['rates']]
            term1 = df.discFact(plot_points + 1, curve)
            term2 = df.discFact(plot_points, curve)
            result = -365 * np.log(term1 / term2)
            return result

        df_curve = pd.read_pickle(output_dir+'/'+estfile+'_curve.pkl')
#        df = util.load_dfs_from_pickle(output_dir, dict_file)
        curve_points_day = plot_points_yr * 365.25

        # Create a new folder to store plots
        path = f"{output_dir}/{plot_folder}"
        try:
            os.makedirs(path, exist_ok=True)
            print(f"Directory '{path}' created")
        except OSError as error:
            print(f"Creation of the directory {path} failed due to: {error}")

        if not taxflag:
            df_curve['month'] = dates.JuliantoYMD(df_curve['quotedate_jul'])[1]  # get the months and put into df
            filtered_curve_df = df_curve[df_curve['month'] == selected_month]
            result_df = filtered_curve_df.apply(calculate_forward_curve, axis=1)
            result_df = result_df.apply(pd.Series)
            result_df.columns = plot_points_yr

            filtered_curve_df = result_df.melt(var_name='Maturity', value_name='Rate', ignore_index=False)
            filtered_curve_df['Year'] = filtered_curve_df.index.get_level_values('quotedate_ind')
#                fwdcurve_alltypes[curve_type] = filtered_curve_df

#            fwdcurve_alltypes = pd.concat(fwdcurve_alltypes.values(), keys=fwdcurve_alltypes.
#                                        keys()).reset_index(level=0).rename(columns={'level_0': 'Curve Type'})

            # Loop through dates to create plots
            for date in (filtered_curve_df.index.get_level_values('quotedate_ind').unique()):  # This references the 'quotedate_ind' index. Maybe there's a more elgant way?
                plt.figure(figsize=(10, 6))
                data_for_date = filtered_curve_df[filtered_curve_df.index.get_level_values('quotedate_ind') == date]

                # Loop through each curve type to plot
                for curve_type in data_for_date.index.get_level_values('type_ind').unique():
                    subset = data_for_date[data_for_date.index.get_level_values('type_ind') == curve_type]
                    plt.plot(subset['Maturity'], subset['Rate'], label=f'{curve_type} - {date}')

                plt.title(f'Forward Rates for {date}')
                plt.xlabel('Maturity')
                plt.ylabel('Rate')
                plt.legend()
                plt.grid(True)
                plt.savefig(f'{output_dir}/{estfile}/{date}ForwardRate-{plot_name}.png')
                plt.show()
                plt.close()
            
            return filtered_curve_df
        

def plot_fwdrate_compare(df_curve1,df_curve2,output_dir,plot_folder,  estfile, selected_month, 
                      plot_points_yr, curve_type = 'pwcf', taxflag=False, taxability=1,
                      labels =['_x','_y'], sqrtscale=False):
        """Plot and export to png in a created folder multiple forward curve plots by curvetype."""

        curve_points_day = plot_points_yr * 365.25

        # Create a new folder to store plots
        path = f"{output_dir}/{plot_folder}"
        try:
            os.makedirs(path, exist_ok=True)
            print(f"Directory '{path}' created")
        except OSError as error:
            print(f"Creation of the directory {path} failed due to: {error}")

        if not taxflag:
            df_curve1['month'] = dates.JuliantoYMD(df_curve1['quotedate_jul'])[1]  # get the months and put into df
            filtered_curve_df1 = df_curve1[df_curve1['month'] == selected_month]
            df_curve2['month'] = dates.JuliantoYMD(df_curve2['quotedate_jul'])[1]  # get the months and put into df
            filtered_curve_df2 = df_curve2[df_curve2['month'] == selected_month]
            filtered_curve_df = pd.merge(filtered_curve_df1,filtered_curve_df2,how='inner',
                                      on=['type_ind','quotedate_ind'],suffixes=labels)
            filtered_curve_df = filtered_curve_df.xs(curve_type,level=0)  # Selects out the one curve type, reduces df to single index

            # Loop through dates to create plots
            for date in (filtered_curve_df.index.get_level_values('quotedate_ind')):  # This references the 'quotedate_ind' index. Maybe there's a more elgant way?
                julian_date = dates.YMDtoJulian(date)   # get the julian 
                plot_points = julian_date + curve_points_day

                xcurvedf = filtered_curve_df.loc[date]   # select out the first curve (this date) full df
                curve = xcurvedf[0:4]     # select out the curve parms only 
                yvols = np.round(xcurvedf['yvols'],decimals=4)
                term1 = df.discFact(plot_points + 1, curve)
                term2 = df.discFact(plot_points, curve)
                result = -365 * np.log(term1 / term2)
#                plt.plot(plot_points,100*result,label=f'{curve_type} - {labels[0]} - {date}',color='red')
            # sqrt root 
                if (sqrtscale):
                    sqrt_plot_points_yr = np.sqrt(plot_points_yr)
                    plt.plot(sqrt_plot_points_yr, 100*result,label=f'{curve_type} - {labels[0]} - {date}',color='red')
                    x1 = max(plot_points_yr)
                    if (x1 > 20):
                        plt.xticks(ticks=np.sqrt([0.25,1,2,5,10,20,30]).tolist(),labels=['0.25','1','2','5','10','20','30'])
                    elif (x1 > 10):
                        plt.xticks(ticks=[np.sqrt(0.25,1,2,5,10,20)],labels=['0.25','1','2','5','10','20'])
                    elif (x1 >5):
                        plt.xticks(ticks=[np.sqrt(0.25,1,2,5,7,10)],labels=['0.25','1','2','5','7','10'])
                    else :
                        plt.xticks(ticks=[np.sqrt(0.25,1,2,3,5)],labels=['0.25','1','2','3','5'])
                    plt.xlabel('Maturity (Years, SqrRt Scale)')            
                else:
                    plt.plot(plot_points_yr, 100*result,label=f'{curve_type} - {labels[0]}',color='red')
                    plt.xlabel('Maturity (Years)')            

                curve = filtered_curve_df.loc[date][6:10]  # select out the second curve (this date)
                term1 = df.discFact(plot_points + 1, curve)
                term2 = df.discFact(plot_points, curve)
                result = -365 * np.log(term1 / term2)
            # sqrt root 
                if (sqrtscale):
                    plt.plot(sqrt_plot_points_yr, 100*result,label=f'{curve_type} - {labels[1]} - yvol {yvols}',color='blue')
                else:
                    plt.plot(plot_points_yr, 100*result,label=f'{curve_type} - {labels[1]} - yvol {yvols}',color='blue')
#                plt.plot(plot_points,100*result,label=f'{curve_type} - {labels[1]} - {date}',color='blue')
                
                plt.title(f'Forward Rates for {date}_comp')
                plt.ylabel('Rate')
                plt.legend()
                plt.grid(True)
                # plt.savefig(f'{output_dir}/{estfile}/Fwd_comp/{date}ForwardRate_{curve_type}_comp.png')
                # Create directory if it does not exist
                full_path = f'{output_dir}/{estfile}/fwd_comp'
                os.makedirs(full_path, exist_ok=True)
                plt.savefig(f'{full_path}/{date}ForwardRate_{curve_type}_comp.png')
                plt.show()
                plt.close()
            
            return filtered_curve_df


def plot_fwdrate_compare_old(df_curve1,df_curve2,output_dir,plot_folder,  estfile, selected_month, 
                      plot_points_yr, curve_type = 'pwcf', taxflag=False, taxability=1,labels =['_x','_y']):
        """Plot and export to png in a created folder multiple forward curve plots by curvetype."""

        def calculate_forward_curve(row):
            date_as_int = row['quotedate_ymd']
            julian_date = row['quotedate_jul']
            plot_points = julian_date + curve_points_day
#            breaks = np.array(selected_breaks)*365 + julian_date
            curve = [row['type'], row['quotedate_jul'], row['breaks'], row['rates']]
            term1 = df.discFact(plot_points + 1, curve)
            term2 = df.discFact(plot_points, curve)
            result = -365 * np.log(term1 / term2)
            return result

#        df_curve = pd.read_pickle(output_dir+'/'+estfile+'_curve.pkl')
#        df = util.load_dfs_from_pickle(output_dir, dict_file)
        curve_points_day = plot_points_yr * 365.25

        # Create a new folder to store plots
        path = f"{output_dir}/{plot_folder}"
        try:
            os.makedirs(path, exist_ok=True)
            print(f"Directory '{path}' created")
        except OSError as error:
            print(f"Creation of the directory {path} failed due to: {error}")

        if not taxflag:
            df_curve1['month'] = dates.JuliantoYMD(df_curve1['quotedate_jul'])[1]  # get the months and put into df
            filtered_curve_df1 = df_curve1[df_curve1['month'] == selected_month]
            result_df = filtered_curve_df1.apply(calculate_forward_curve, axis=1)
            result_df = result_df.apply(pd.Series)
            result_df.columns = plot_points_yr
            filtered_curve_df1 = result_df.melt(var_name='Maturity', value_name='Rate', ignore_index=False)
            filtered_curve_df1['Year'] = filtered_curve_df1.index.get_level_values('quotedate_ind')
            filtered_curve_df1['Estim'] = labels[0]

            df_curve2['month'] = dates.JuliantoYMD(df_curve2['quotedate_jul'])[1]  # get the months and put into df
            filtered_curve_df2 = df_curve2[df_curve2['month'] == selected_month]
            result_df = filtered_curve_df2.apply(calculate_forward_curve, axis=1)
            result_df = result_df.apply(pd.Series)
            result_df.columns = plot_points_yr
            filtered_curve_df2 = result_df.melt(var_name='Maturity', value_name='Rate', ignore_index=False)
            filtered_curve_df2['Year'] = filtered_curve_df2.index.get_level_values('quotedate_ind')
            filtered_curve_df2['Estim'] = labels[1]

            filtered_curve_df = pd.concat([filtered_curve_df1,filtered_curve_df2])
#                fwdcurve_alltypes[curve_type] = filtered_curve_df

#            fwdcurve_alltypes = pd.concat(fwdcurve_alltypes.values(), keys=fwdcurve_alltypes.
#                                        keys()).reset_index(level=0).rename(columns={'level_0': 'Curve Type'})

            # Loop through dates to create plots
            for date in (filtered_curve_df.index.get_level_values('quotedate_ind').unique()):  # This references the 'quotedate_ind' index. Maybe there's a more elgant way?
                plt.figure(figsize=(10, 6))
                data_for_date = filtered_curve_df[filtered_curve_df.index.get_level_values('quotedate_ind') == date]

                # Loop through the two curves 
                for label in labels:
                    subset = data_for_date[data_for_date.index.get_level_values('type_ind') == curve_type] 
                    subset = subset.loc['Estim' == label]
                    plt.plot(subset['Maturity'], subset['Rate'], label=f'{curve_type} - {date}')

                plt.title(f'Forward Rates for {date}')
                plt.xlabel('Maturity')
                plt.ylabel('Rate')
                plt.legend()
                plt.grid(True)
                plt.savefig(f'{output_dir}/{estfile}/{date}ForwardRate.png')
                plt.show()
                plt.close()
            
            return filtered_curve_df

def plot_fwdrate_from_output_old(output_dir, plot_folder, plot_name, dict_file, selected_month, selected_breaks, 
                      plot_points_yr, taxflag=False, taxability=1):
        """Plot and export to png in a created folder multiple forward curve plots by curvetype."""

        def calculate_forward_curve(row):
            date_as_int = int(row.name.strftime('%Y%m%d'))
            julian_date = dates.YMDtoJulian(date_as_int)
            curve_points = julian_date + curve_points_day
            breaks = np.array(selected_breaks)*365 + julian_date
            curve = [curve_type, julian_date, breaks, np.array(row)]
            term1 = df.discFact(curve_points + 1, curve)
            term2 = df.discFact(curve_points, curve)
            result = -365 * np.log(term1 / term2)
            return result

        df = util.load_dfs_from_pickle(output_dir, dict_file)
        curve_points_day = plot_points_yr * 365.25

        # Create a new folder to store plots
        path = f"{output_dir}/{plot_folder}"
        try:
            os.makedirs(path, exist_ok=True)
            print(f"Directory '{path}' created")
        except OSError as error:
            print(f"Creation of the directory {path} failed due to: {error}")

        if not taxflag:
            fwdcurve_alltypes = {}
            for curve_type, data in df.items():
                curve_df = data['curve_df'].copy()
                curve_df.index = pd.to_datetime(curve_df.index.astype(str), format='ISO8601')
                filtered_curve_df = curve_df[curve_df.index.month == selected_month]
                filtered_curve_df = filtered_curve_df.loc[:, filtered_curve_df.columns.isin(selected_breaks)]

                result_df = filtered_curve_df.apply(calculate_forward_curve, axis=1)
                result_df = result_df.apply(pd.Series)
                result_df.columns = plot_points_yr

                # a = ['pwcf', 15160, np.array(selected_breaks), (filtered_curve_df.iloc[0]).tolist()]
                # forward_curve = -365*np.log((df.discFact((curve_points+1),
                #                                           a))/(df.discFact(curve_points, a)))

                filtered_curve_df = result_df.melt(var_name='Maturity', value_name='Rate', ignore_index=False)
                filtered_curve_df['Year'] = filtered_curve_df.index.year
                fwdcurve_alltypes[curve_type] = filtered_curve_df

            fwdcurve_alltypes = pd.concat(fwdcurve_alltypes.values(), keys=fwdcurve_alltypes.
                                        keys()).reset_index(level=0).rename(columns={'level_0': 'Curve Type'})

            # Loop through dates to create plots
            for date in (fwdcurve_alltypes.index.unique()):
                plt.figure(figsize=(10, 6))
                data_for_date = fwdcurve_alltypes[fwdcurve_alltypes.index == date]

                # Loop through each curve type to plot
                for curve_type in data_for_date['Curve Type'].unique():
                    subset = data_for_date[data_for_date['Curve Type'] == curve_type]
                    plt.plot(subset['Maturity'], subset['Rate'], label=f'{curve_type} - {date.year}')

                plt.title(f'Forward Rates for {date.strftime("%Y-%m-%d")}')
                plt.xlabel('Maturity')
                plt.ylabel('Rate')
                plt.legend()
                plt.grid(True)
                plt.savefig(f'{output_dir}/{dict_file}/{date.strftime("%Y-%m-%d")}ForwardRate-{plot_name}.png')
                plt.show()
                plt.close()
            
            return fwdcurve_alltypes