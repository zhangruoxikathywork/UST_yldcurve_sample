#%%
import sys
import os
import numpy as np
#import numpy_financial
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import matplotlib.animation as animation
import time
import ast
import re


#%%

sys.path.append('../../src/package')
sys.path.append('../../../BondsTable')
sys.path.append('../../tests')
sys.path.append('../../data')

import discfact as df
import pvfn as pv
import DateFunctions_1 as dates
import Curve_Plotting as cp
import util_fn as util


#########################################################################################################
#########################################################################################################
#%% Function to create parbond, annuity, and zero bond dataframes


def create_parbnd_df(curve_df, table_breaks_yr,addmths=True):
    """
    Take in an input vector of a curve and years to maturity, and create parms for par, zero, and annuity bonds 
      Loop through the curve dates
      Create a dataframe indexed by quotedate & by type (par, annuity, zero), which holds parms    
      If arg 'addmths' is True then round the breaks to months and use CallAdd
        - For making table of 5yr, 10yr etc par bonds we want exact maturity (by months)
        - For graphing we want frequent (roughly every 3 day) points, so add by day
    """
    if not isinstance(table_breaks_yr, np.ndarray):
        table_breaks_yr = np.array(table_breaks_yr)
    if addmths:  # add by months, otherwise by days
        parbnd_breaks = np.round(table_breaks_yr*12)   # get number of months by rounding
    else:
        parbnd_breaks = np.round(table_breaks_yr*365.25)   # get number of days by rounding

    nrows = len(parbnd_breaks)
    ncols = 8
    parms_zero = pd.DataFrame(np.zeros([nrows,ncols+2]))   # make empty df (extra col for type & qd)
    parms_zero.iloc[:,2] = 2        
    parms_zero.iloc[:,4] = "A/A"
    parms_zero.iloc[:,5] = 'eomyes'
    parms_zero.iloc[:,7] = False
    parms_zero.iloc[:,8] = 'zero'
    parms_annuity = parms_zero.copy()    # make copy for annuity, populate those two
    parms_annuity.iloc[:,8] = 'annuity'
    parms_zero.iloc[:,3] = 100.
    parms_annuity.iloc[:,0] = 1.
    parms_par = parms_zero.copy()  # Make a copy which will be for the par bonds. But cannot put in coupon until calculate rates
    parms_par.iloc[:,8] = 'parbond'

    ncurve = len(curve_df)    # check if this is only one date
    if ncurve == 1:
        curve_dates = curve_df.index
    else:
        curve_dates = curve_df.index.get_level_values(1).unique().tolist()
    curve_dates_julian = dates.YMDtoJulian(curve_dates)

  # I think this looping and concat'ing is very inefficient
  # But I cannot figure out another way to do it
    parms_df = pd.DataFrame()
    for qdate_jul,qdate_ymd in zip(curve_dates_julian,curve_dates):
        if addmths:    # add by months, otherwise by days
            xmaturity = dates.CalAdd(qdate_jul,nmonth=parbnd_breaks)   # add months to quote date to get maturity dates
        else:
            xmaturity = dates.CalAdd(qdate_jul, nday=parbnd_breaks) # add days to quote date to get maturity dates
        parms_zero.iloc[:,1] = xmaturity   # start populating with date - first common to zero and annuity
        parms_zero.iloc[:,9] = int(qdate_ymd)
        parms_annuity.iloc[:,1] = xmaturity
        parms_annuity.iloc[:,9] = qdate_ymd
        parms_par.iloc[:,1] = xmaturity
        parms_par.iloc[:,9] = int(qdate_ymd)
        parms_df = pd.concat([parms_df,parms_zero,parms_annuity,parms_par])
    parms_df.columns = ['coup','maturity','freq','FV','A/A','eom','calldate','callflag','type','quotedate']
    parms_df.set_index(['type','quotedate'],inplace=True,drop=True)
    parms_df = parms_df.sort_index()

    return parms_df


def parbond_rates(curve, parms_df, twostep=False, parmflag=True, padj=False):
    """Take in an input curve (breaks, rates) and parms df for calculating par bond rates and prices. 
        The input df should be for one month, and should include the par, zero, and annuity parms
       this function takes in an input curve (breaks, rates) and parms df for calculating
       par bond rates. It uses the standard 
         coupon = (100 - PVZero) / PVAnnuity
       which is based on PV(par bond) = 100. 
       This gives the right answer on coupon dates, but because accrued interest is linear
       (rather than exponential) it does not quite give the right answer in between coupons
       Instead, we want to use the definition for a par bond:
         coupon = yield 
       We can use a two-step procedure to get very close to the answer:
       1) Calculate the coupon using the PV = 100 formula above
       2) Calculate the yield for a bond with that coupon
       Because the coupon from step 1 will be within a few bp of the correct answer, 
       the yield for a bond with that coupon (close to the par bond coupon) will now
       be even closer to the par bond coupon. For all practical pursposes this will
       be close enough. (A quick test for 2999 plot points from .01 to 30 by .01, with a curve:
           quotedate = 31-dec-1932
           breaks = 1mth, 6mth, 1yr, 2yr, 5yr, 10yr, 20yr, 30yr
           rates = 2%, 2.5%, 3%, 3.5%, 4%, 4.5%, 5%, 6%
       shows errors of roughly 2bp at the very short end (a few days) and a mean square
       error of 0.28bp. For the second step, errors at the short end are on the order of
       E-10bp, and mse of 0.001bp)
    """
    xquotedate = curve[1]

    parms_zero = parms_df.loc['zero']
    parms_annuity = parms_df.loc['annuity']
    parms_par = parms_df.loc['parbond']
    df_maturity = pv.pvBondFromCurve(curve, settledate = xquotedate, parms = parms_zero)[:,1]
    xx = pv.pvBondFromCurve(curve, settledate = xquotedate, parms = parms_annuity)
    pv_annuity = xx[:,1]
    accrued_interest = xx[:,1]-xx[:,0]
    # This calculates the coupon rate for a clean price of 100
    # This uses "Price = 100" but this is a good defintion of "par bond" only on an exact coupon date
    # Otherwise, I think we should use "Coupon = Yield"
    # The coupon calculated in this way will almost be the right coupon, but not quite
    # We can do a quick fix by calculating the yield for this coupon, because
    # this will be almost the coupon that makes yield = coupon
    coupon_rates = (100-df_maturity)/(pv_annuity - accrued_interest)
    parms_par.loc[:,'coup'] = coupon_rates        # Put initial coupon estimates into parms_df
    dirtyprice = coupon_rates * pv_annuity + df_maturity
    cleanprice = dirtyprice - accrued_interest
    pbrates = coupon_rates.copy()/100.
    if twostep:
        # Now loop through bonds to calculate yields
        for j in range(len(pbrates)):
            pbrates[j] = pv.bondYieldFromPrice_callable(dirtyprice[j], 
                            parms=parms_par.iloc[j:(j+1)], 
                            settledate=xquotedate, parmflag=True, padj=False)
#        parms_par.loc[:,'coup'] = pbrates * 100         # Update par bond coupons with fixed values
    return pbrates, cleanprice, dirtyprice


def zerobond_rates(curve, parms_df, parmflag=True, padj=False):
    """
        Take in an input curve (breaks, rates) and maturities for calculating zero bond rates and prices. 
        Works for one single month
    """

    xquotedate = curve[1]
#    plot_points = np.round(xquotedate + zerobd_breaks_yr*365.25,0)
#    parms_maturity= []
#    for i in plot_points:   # Make the dummy bonds for zero and annuity
#        parms_maturity_single = [0,i,2,100.,"A/A","eomyes",0, False]
#        parms_maturity.append(parms_maturity_single)
#    parms_maturity = pd.DataFrame(parms_maturity)
    parms_zero = parms_df.loc['zero']
    df_maturity = pv.pvBondFromCurve(curve, settledate = xquotedate, parms = parms_zero)[:,1]
    # Which of the following to use? Shape looks the sames, but different scales
    # zbrates = df_maturity.copy()/100.
    # for j in range(len(zbrates)):
    #     zbrates[j] = pv.bondYieldFromPrice_callable(df_maturity[j], parms=parms_maturity.loc[j:j],
    #                                                  parmflag=True, padj=False)
#    discount_curve = df.discFact(plot_points,curve)
    parm_points = (parms_zero['maturity'] - xquotedate) / 365.25
    zbrates = np.log(df_maturity)/(-parm_points)

    return zbrates, df_maturity, df_maturity


def annuity_rates(curve, parms_df, parmflag=True, padj=False):
    """
        Take in an input curve (breaks, rates) and maturities for calculating annuity rates and prices. 
        Works for one single month
    """

    xquotedate = curve[1]
#    plot_points = np.round(xquotedate + annuity_breaks_yr*365.25,0)

 #   parms_annuity = []
 #   for i in plot_points:
 #       parms_annuity_single = [1.,i,2,0.,"A/A","eomyes",0, False]
 #       parms_annuity.append(parms_annuity_single)
 #   parms_annuity = pd.DataFrame(parms_annuity)
    parms_annuity = parms_df.loc['annuity']
    xx = pv.pvBondFromCurve(curve, settledate = xquotedate, parms = parms_annuity)
    cpv_annuity = xx[:,0] # clean price
    dpv_annuity = xx[:,1] # dirty price
    annuityrates = dpv_annuity.copy()/100.
    for j in range(len(annuityrates)):
        annuityrates[j] = pv.bondYieldFromPrice_callable(
            dpv_annuity[j], settledate=xquotedate, parms=parms_annuity.iloc[j:(j+1)], parmflag=True, padj=False)

    return annuityrates, cpv_annuity, dpv_annuity


def produce_pb_zb_anty_dfs(curve_df, parms_df, rate_type, twostep=False,
                            parmflag=True, padj=False):
    """
    Generate dataframes for par bond, zero bond, or annuity rates and prices from a given yield curve dataframe.

    Args:
        curve_df (pd.DataFrame): DataFrame containing the curve information with columns ['type', 'quotedate', 'breaks', 'rates'].
        rate_breaks_yr (list): List of maturities (in years) for which the rates are to be calculated.
        rate_type (str): Type of rate to calculate. Must be one of 'parbond', 'zerobond', or 'annuity'.
        twostep (bool, optional): If True, use a two-step procedure for calculating par bond rates. Defaults to False.
        parmflag (bool, optional): Parameter flag for rate calculations. Defaults to True.
        padj (bool, optional): Adjustment parameter for rate calculations. Defaults to False.

    Returns (new):
        rateprice_df (pd.DataFrame): DataFrame containing the calculated rates with columns ['ctype', 'quotedate', 'rtype'] + [f'{yr}YR' for yr in rate_breaks_yr].
            ctype = curve type (eg pwcf, pwlz, pwtf)
            rtype = rate type = 'rate' or 'cprice' or 'dprice' 

    Raises:
        ValueError: If rate_type is not one of 'parbond', 'zerobond', or 'annuity'.
    """

#    parms_df = create_parbnd_df(curve_df, table_breaks_yr)
    rates = []
    cprices = []
    dprices = []
    xparms_ind = parms_df.index
    for index, row in curve_df.iterrows():
        curve = [row['type'], row['quotedate_jul'], row['breaks'], row['rates']]
        xqd = int(dates.JuliantoYMDint(row['quotedate_jul'])[0])
        xparms_df = parms_df.xs(xqd,level="quotedate")
#        xparms_df = parms_df.loc[row['quotedate_jul']]
        if rate_type == 'parbond': # For parbonds also get back the updated parms with updated coupons
            rate, cprice,dprice = parbond_rates(curve, xparms_df, twostep, parmflag, padj)
#  Insert the par bond rates into xparms_ind
            xparbool = xparms_ind == ('parbond',xqd)
            parms_df.loc[xparbool,'coup'] = rate * 100      
        elif rate_type == 'zerobond':
            rate, cprice,dprice = zerobond_rates(curve, xparms_df, parmflag, padj)
        elif rate_type == 'annuity':
            rate, cprice,dprice = annuity_rates(curve, xparms_df, parmflag, padj)
        else:
            raise ValueError("rate_type must be one of 'parbond', 'zerobond', or 'annuity'")

        rates.append([curve[0], dates.JuliantoYMDint(curve[1])[0], 'rate'] + list(rate)) # can't round here - do it later [np.round(x, 3) for x in rate]) # 
        cprices.append([curve[0], dates.JuliantoYMDint(curve[1])[0], 'cprice'] + list(cprice)) # [np.round(x, 2) for x in cprice])
        dprices.append([curve[0], dates.JuliantoYMDint(curve[1])[0], 'dprice'] + list(dprice)) #[np.round(x, 2) for x in dprice])

    xqdj = curve_df.iloc[0,1]
    xqdi = dates.JuliantoYMDint(xqdj)[0]           # this should be the first quote date - any will work
    xparms_df = parms_df.xs(xqdi,level="quotedate")
    xparms_df = xparms_df.loc['zero'] # pick out the zero for first qd
    rate_breaks_yr = np.round((xparms_df['maturity'] - xqdj[0])/365.25,decimals=2)
    columns = ['ctype', 'quotedate', 'rtype'] + [f'{yr}YR' for yr in rate_breaks_yr]
    rates_df = pd.DataFrame(rates, columns=columns)
    cprices_df = pd.DataFrame(cprices, columns=columns)
    dprices_df = pd.DataFrame(dprices, columns=columns)
    rateprice_df = pd.concat([rates_df,cprices_df,dprices_df])
    rateprice_df.set_index(['ctype','quotedate','rtype'],inplace=True,drop=True)
    rateprice_df = rateprice_df.sort_index()

    return rateprice_df, parms_df    # return back the parms_df because for parbonds it will be populated with coupons


def pb_zb_anty_wrapper(curve_df, table_breaks_yr, estfile, output_dir, twostep=True, parmflag=True, padj=False):
    """Produce par bond, zero bond, annuity rates and prices tables. """

# Create df of bond parameters
    parms_df = create_parbnd_df(curve_df, table_breaks_yr, addmths=True)

    # Produce pb, zb, annuity prices and rates tables
    # For the par bonds take back the updated parms_df which has the par bond coupons inserted
    # For zero and annuity just throw away the returned parms_df because it is unchanged
    parbd_rateprice_df, pbparms_df = produce_pb_zb_anty_dfs(curve_df, parms_df,
     'parbond', twostep, parmflag, padj)
    zerobd_rateprice_df, x1 = produce_pb_zb_anty_dfs(curve_df, parms_df,
     'zerobond', parmflag, padj)
    annuity_rateprice_df, x1 = produce_pb_zb_anty_dfs(curve_df, parms_df,
     'annuity', parmflag, padj)

# Put par bond coupons into the parms_df
     # Export to CSV
    dataframes = [parbd_rateprice_df, zerobd_rateprice_df, annuity_rateprice_df]
    names = ['parbd_rates_prices', 'zerobd_rates_prices', 'annuity_rates_prices']
    util.export_to_csv(dataframes, names, output_dir, estfile)

    return parbd_rateprice_df, zerobd_rateprice_df, annuity_rateprice_df, pbparms_df


def seperate_pb_zb_anty_wrapper(parbd_rateprice_df, zerobd_rateprice_df, annuity_rateprice_df):
    """Seperate pb, zb, annuity prices and rates into individual tables."""

    parbd_rate = parbd_rateprice_df.xs('rate', level='rtype')
    parbd_cprice = parbd_rateprice_df.xs('cprice', level='rtype')
    zerobd_rate = zerobd_rateprice_df.xs('rate', level='rtype')
    zerobd_cprice = zerobd_rateprice_df.xs('cprice', level='rtype')
    annuity_rate = annuity_rateprice_df.xs('rate', level='rtype')
    annuity_cprice = annuity_rateprice_df.xs('cprice', level='rtype')

    # dataframes = [parbd_rate, parbd_cprice, zerobd_rate, zerobd_cprice, annuity_rate, annuity_cprice]
    # names = ['parbd_rate', 'parbd_cprice', 'zerobd_rate', 'zerobd_cprice', 'annuity_rate', 'annuity_cprice']
    # util.export_to_csv(dataframes, names, output_dir, estfile)

    return parbd_rate, parbd_cprice, zerobd_rate, zerobd_cprice, annuity_rate, annuity_cprice


def find_max_min_pb_rate(pzb_rate):
    
    pzb_rate = pzb_rate.reset_index()
    pzb_rate['quotedate'] = pzb_rate['quotedate'].astype(int).astype(str)
    pzb_rate['year'] = pd.to_datetime(pzb_rate['quotedate'], format='%Y%m%d').dt.year
    pzb_rate['5_year_bin'] = (pzb_rate['year'] // 5) * 5

    numeric_cols = pzb_rate.select_dtypes(include=[np.number]).columns

    # Group by ctype and 5-year bin, then find the overall max and min among all rates
    max_min_per_5yr = pzb_rate.groupby(['ctype', '5_year_bin']).apply(lambda df: pd.Series({
        'max_5yr': df[numeric_cols].drop(columns=['year', '5_year_bin']).max().max(),
        'min_5yr': df[numeric_cols].drop(columns=['year', '5_year_bin']).min().min()
    })).reset_index()

    pzb_rate = pzb_rate.merge(max_min_per_5yr, on=['ctype', '5_year_bin'], how='left')

    pzb_rate.set_index(['ctype','quotedate'], inplace=True, drop=True)

    return pzb_rate


#########################################################################################################
#########################################################################################################
#%% Calculate Total Return, Yield Return, Return in Excess of Yield
## (Income Returns, Capital Gain Returns)

def calc_par_total_ret(curve_df, table_breaks_yr, twostep=False, parmflag=True, padj=False):
    """Calcualte monthly return using par bond yield."""

    table_breaks_yr = [value for value in table_breaks_yr if value >= 1/3]
    # Produce last month's parms
    parms_df = create_parbnd_df(curve_df, table_breaks_yr, addmths=True)
    parbd_rateprice_df, parms_df = produce_pb_zb_anty_dfs(curve_df, parms_df,
    'parbond', twostep, parmflag, padj)
    quotedates_prev = parms_df.reset_index()['quotedate'].unique().tolist()[:-1]
    quotedates = parms_df.reset_index()['quotedate'].unique().tolist()[1:]
    # Filter out the rows where the quotedate index matches the last_quotedate
    parms_df_pre = parms_df[parms_df.index.get_level_values('quotedate') != int(quotedates[-1])]
    curve_df_lag = curve_df[curve_df.index.get_level_values('quotedate_ind') != quotedates_prev[0]]

    total_ret_list = []

    for index, row in curve_df_lag.iterrows():
        curve = [row['type'], row['quotedate_jul'], row['breaks'], row['rates']]
        curve_type = row['type']
        qd = int(dates.JuliantoYMDint(row['quotedate_jul'])[0])
        qd_pre = int(quotedates_prev[quotedates.index(qd)])
        
        # qd_pre = dates.JuliantoYMDint(dates.CalAdd(row['quotedate_jul'],add="sub",nyear=0,nmonth=1,nday=0, eom="eomyes"))
        parms_df_pre_ind = parms_df_pre[(parms_df_pre.index.get_level_values('quotedate') == qd_pre)]
        pbrates, cleanprice, dirtyprice = parbond_rates(curve, parms_df_pre_ind, twostep=twostep, parmflag=parmflag, padj=padj)
        total_ret = dirtyprice/100-1

        row_r = [curve[0], dates.JuliantoYMDint(curve[1])[0], 'total_return']
        row_r += total_ret.tolist()
        total_ret_list.append(row_r)

    columns = ['ctype', 'quotedate', 'return_type'] + [f'{yr}YR' for yr in table_breaks_yr]
    total_ret_prc_df = pd.DataFrame(total_ret_list, columns=columns)
    total_ret_prc_df.set_index(['ctype','quotedate','return_type'], inplace=True, drop=True)
    xtrindex = total_ret_prc_df.index.get_level_values

# Now yield return
    parbd_rate_df = parbd_rateprice_df[parbd_rateprice_df.index.get_level_values('rtype') == 'rate']

    def income_return(x, AD):
        return (1 + x / 2) ** (AD / (365.25 / 2)) - 1


    # Calculate actual days for each quotedate
    quotedate_ind = parbd_rate_df.index.get_level_values('quotedate').unique()
    quotedate_ind = dates.YMDtoJulian(quotedate_ind)     # convert to Julian dates
    AD_values = quotedate_ind[1:] - quotedate_ind[0:-1]  # Take difference, dropping the first
#    AD_values = np.array([calculate_ad(qd) for qd in quotedate_ind])

    breaks_cols = parbd_rate_df.columns.difference(['AD'])
    income_ret_df = parbd_rate_df.copy()
    income_ret_df = income_ret_df[income_ret_df.index.get_level_values('quotedate') != int(quotedates[-1])]
    ctype_ind = income_ret_df.index.get_level_values('ctype').unique()
    income_ret_df['AD'] = np.tile(AD_values,len(ctype_ind))

    for column in breaks_cols:
        income_ret_df[column] = income_ret_df.apply(lambda row: income_return(row[column], row['AD']), axis=1)
    income_ret_df.drop(columns='AD', inplace=True)
    income_ret_df.rename_axis(index={'rtype': 'return_type'}, inplace=True)
    income_ret_df = income_ret_df.reset_index()
    income_ret_df['return_type'] = 'yield_return'
    income_ret_df['quotedate'] = xtrindex('quotedate')
    income_ret_df.columns = columns      # To make sure names consistent
    income_ret_df.set_index(['ctype','quotedate','return_type'], inplace=True, drop=True)

    x1 = np.asarray(total_ret_prc_df.values.tolist())   # array of total returns
    x2 = np.asarray(income_ret_df.values.tolist())   # array of yield returns
    x3 = (1 + x1) / (1 + x2) - 1.0            # return in excess of yield
    columns1 = [f'{yr}YR' for yr in table_breaks_yr]
    cg_return_df = pd.DataFrame(x3.tolist(),columns=columns1)
    cg_return_df.insert(0,'return_type','yield_excess')
    cg_return_df.insert(0,'quotedate',xtrindex('quotedate'))
    cg_return_df.insert(0,'ctype',xtrindex('ctype'))
    cg_return_df.set_index(['ctype','quotedate','return_type'], inplace=True, drop=True)
    return_df = pd.concat([total_ret_prc_df,income_ret_df,cg_return_df])
    return_df = return_df.sort_index()

    return return_df


def seperate_returns_wrapper(return_df):

    total_ret = return_df.xs('total_return', level='return_type')
    yld_ret = return_df.xs('yield_return', level='return_type')  # income_ret
    yld_excess = return_df.xs('yield_excess', level='return_type')  # capgain_ret

    return total_ret, yld_ret, yld_excess


################################################################################################################
################################################################################################################
#%% Plot par bond, zero bond, annuity yield curves

def plot_pb_zb_anty_curve(curve_df, plot_points_yr, rate_type, estfile, output_dir, 
                          twostep=False, parmflag=True, padj=False, log_scale=False):
    """
    This function takes in input curves (breaks, rates) and plots par bond curves 
    at a quarterly frequency. It plots 3 types of curves all at once.

    Args:
        curve_df (pd.DataFrame): DataFrame containing the curves data.
        plot_points_yr (list): List of maturities (in years) for which the rates are to be calculated.
        quote_date(float): Quote date, in the form of julian date, e.g., quote_date = 36555.0
        rate_type (str): Type of rate to calculate. Must be one of 'parbond', 'zerobond', or 'annuity'.
        twostep (bool, optional): If True, use a two-step procedure for calculating par bond rates. Defaults to False.
        parmflag (bool, optional): Parameter flag for rate calculations. Defaults to True.
        padj (bool, optional): Adjustment parameter for rate calculations. Defaults to False.
    
    Example:
        plot_curve = curve_df.xs(20000131,level=1)
        plot_pb_zb_anty_curve(plot_curve, plot_points_yr, 'parbond', log_scale=False)
    """
    quote_date = curve_df.iloc[0]['quotedate_jul']
    quote_date_ymd = dates.JuliantoYMDint(quote_date)
    colors = {'pwcf': 'red', 'pwlz': 'green', 'pwtf': 'blue'}
#    curve_df_qd = curve_df.loc[curve_df['quotedate'] == quote_date]
    rateprice_df, parms_df = produce_pb_zb_anty_dfs(curve_df, plot_points_yr, rate_type, twostep, parmflag, padj)
    rate_df = rateprice_df.xs('rate',level=2)

    for curvetype, color in colors.items():
        curve_row = rate_df[rate_df['type'] == curvetype]
        if curve_row.empty:
            continue
        rates = curve_row.iloc[0, 2:].values
        plt.plot(plot_points_yr, 100*rates, color=color, label=curvetype)

    plt.legend()
    plt.title(f'{rate_type} Curves, {quote_date_ymd}')
    plt.xlabel('Maturity (Years)')
    plt.ylabel('Rates')
    if log_scale == True:
        plt.xscale('log')
    # plt.savefig(f"../../output/{rate_type}_curves_{quote_date}.png")
    plt.savefig(f'{output_dir}/{estfile}/{quote_date_ymd}ForwardRate.png')
    plt.show()


#%% Plot par bond, zero bond, annuity yield curves

def plot_singlecurve(output_dir, estfile, curve, plot_points_yr, rate_type = 'parbond',
                     df_price_yield=None, pbtwostep=False, parmflag=True, padj=False, sqrtscale=False):
    """
    This function takes in input curve (type, quotedate, breaks, rates) and plots single curve 

    Args:
        output_dir (str): Output directory.
        estfile (str): Output folder for the graphs.
        curve containing curves data (type, quotedate_jul, breaks, rates).
        plot_points_yr (list): List of maturities (in years) for which the rates are to be plotted.
        quote_date(float): Quote date, in the form of julian date, e.g., quote_date = 36555.0
        rate_type (str): Type of rate to calculate. Must be one of 'parbond', 'zerobond', or 'annuity'.
        pbtwostep (bool, optional): If True, use a two-step procedure for calculating par bond rates. Defaults to False.
        parmflag (bool, optional): Parameter flag for rate calculations. Defaults to True.
        padj (bool, optional): Adjustment parameter for rate calculations. Defaults to False.
    """
    quote_date = curve.iloc[1]
    quote_date_ymd = dates.JuliantoYMDint(quote_date)
    curvetype = curve.iloc[0]
    curve_df = curve.to_frame().T     # Do this because the function 'produce...' wants df
#    rate_df, price_df = produce_pb_zb_anty_dfs(curve_df, plot_points_yr, rate_type, pbtwostep, parmflag, padj)
# Create df of bond parameters
    parms_df = create_parbnd_df(curve_df, plot_points_yr,addmths=False)  # add by days
    rateprice_df, parms_df = produce_pb_zb_anty_dfs(curve_df, parms_df, rate_type, pbtwostep, parmflag, padj)
    rate_df = rateprice_df.xs('rate',level=2)

    if not(df_price_yield is None) :
        x4 = df_price_yield[['MatYr','MatMth','MatDay']].to_numpy().tolist()
        x5 = dates.YMDtoJulian(x4)
        x6 = dates.YMDtoJulian(df_price_yield.iloc[0,1].tolist())
        actpred_yr = (x5 - x6)/365.25
        xact_yield = df_price_yield[['actualyield']].to_numpy()

    # sqrt root in the short-term
    if (sqrtscale):
        sqrt_plot_points_yr = np.sqrt(plot_points_yr)
        plt.plot(sqrt_plot_points_yr, 100*rate_df.iloc[0],  label=curvetype)
        if not(df_price_yield is None) :
            sqrt_actpred_yr = np.sqrt(actpred_yr)
            plt.plot(sqrt_actpred_yr, 100*xact_yield,'.')            
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
        plt.plot(plot_points_yr, 100*rate_df.iloc[0],  label=curvetype)
        if not(df_price_yield is None) :
            plt.plot(actpred_yr, 100*xact_yield,'.')            
        plt.xlabel('Maturity (Years)')


    plt.legend()
    plt.title(f'{rate_type.capitalize()} Curve, {int(quote_date_ymd)}')
    plt.ylabel('Rates')
    # Create directory if it does not exist
    full_path = f'{output_dir}/{estfile}/act_pred_{rate_type}'
    os.makedirs(full_path, exist_ok=True)
    plt.savefig(f'{full_path}/act_pred_{rate_type}_{int(quote_date_ymd)}.png')
    # plt.savefig(f'{output_dir}/{estfile}/act_pred_{rate_type}/act_pred_{rate_type}_{quote_date}.png')
    plt.show()



def plot_singlecurve_act_pred(output_dir, estfile, curve, plot_points_yr, rate_type='parbond', yield_to_worst=True,
                              yvols=0, df_price_yield=None, pbtwostep=False, parmflag=True, padj=False, sqrtscale=False,
                              durscale=False):
    """
    This function takes in input curve (type, quotedate, breaks, rates) and plots single curve with actual and predicted scatters.

    Args:
        output_dir (str): Output directory.
        estfile (str): Output folder for the graphs.
        curve containing curves data (type, quotedate_jul, breaks, rates).
        plot_points_yr (list): List of maturities (in years) for which the rates are to be plotted.
        quote_date(float): Quote date, in the form of julian date, e.g., quote_date = 36555.0
        rate_type (str): Type of rate to calculate. Must be one of 'parbond', 'zerobond', or 'annuity'.
        pbtwostep (bool, optional): If True, use a two-step procedure for calculating par bond rates. Defaults to False.
        parmflag (bool, optional): Parameter flag for rate calculations. Defaults to True.
        padj (bool, optional): Adjustment parameter for rate calculations. Defaults to False.
        sqrtscale: if True then plots on square root scale
        durscale: if True then uses duration (but not square root - either or)
    """
    quote_date = curve.iloc[1]
    quote_date_ymd = dates.JuliantoYMDint(quote_date)
    curvetype = curve.iloc[0]
    curve_df = curve.to_frame().T     # Do this because the function 'produce...' wants df
#    rate_df, cprice_df, dprice_df = produce_pb_zb_anty_dfs(curve_df, plot_points_yr, rate_type, pbtwostep, parmflag, padj)
# Create df of bond parameters
    parms_df = create_parbnd_df(curve_df, plot_points_yr,addmths=False)  # add by days
    rateprice_df, parms_df = produce_pb_zb_anty_dfs(curve_df, parms_df, rate_type, pbtwostep, parmflag, padj)
    rate_df = rateprice_df.xs('rate',level=2)
#    rate_df = rateprice_df.loc['rate']
#    rate_df, cprice_df = produce_pb_zb_anty_dfs(curve_df, plot_points_yr, rate_type, pbtwostep, parmflag, padj)
    if durscale:   # need to get duration of par bonds, but this is the PV of an annuity which we can easily calculate
#        arate_df, acprice_df, adprice_df = produce_pb_zb_anty_dfs(curve_df, plot_points_yr, 'annuity', pbtwostep, parmflag, padj)
        arateprice_df, parms_df = produce_pb_zb_anty_dfs(curve_df, parms_df, 'annuity', pbtwostep, parmflag, padj)
        arate_df = arateprice_df.xs('rate',level=2)
        acprice_df = arateprice_df.xs('cprice',level=2)
        x1 = np.array(arate_df.iloc[0])
        x2 = np.array(acprice_df.iloc[0])
        xplot_points_yr = x2 * (1 + x1 / 2)  #  This is correct formula, but as of 29-jul-24 annuity rates are wrong
#        xplot_points_yr = acprice_df.iloc[0, 2:]   #  This is not correct formula, but as of 29-jul-24 annuity rates are wrong
        xplot_points_yr = xplot_points_yr.tolist()  # This converts from list of objects to list of numbers ??
    else:
        xplot_points_yr = plot_points_yr
        

    if not(df_price_yield is None) :
        x4 = df_price_yield[['MatYr','MatMth','MatDay']].to_numpy().tolist()
        x5 = dates.YMDtoJulian(x4)
        x6 = dates.YMDtoJulian(df_price_yield.iloc[0,1].tolist())
        actpred_yr = (x5 - x6)/365.25
        xact_yield = df_price_yield[['actualyield']].to_numpy()
        xpre_yield = df_price_yield[['predictedyield']].to_numpy()
        call_flags = df_price_yield['CallFlag'].astype(bool).to_numpy()
        if durscale:
            actpred_yr = df_price_yield[['MacDur']].to_numpy()


    # sqrt root in the short-term
    if sqrtscale and not(durscale):
        sqrt_plot_points_yr = np.sqrt(xplot_points_yr)
        plt.plot(sqrt_plot_points_yr, 100 * rate_df.iloc[0], label=curvetype)
        if df_price_yield is not None:
            sqrt_actpred_yr = np.sqrt(actpred_yr)
            for i in range(len(call_flags)):
                if call_flags[i]:
                    plt.scatter(sqrt_actpred_yr[i], 100 * xact_yield[i], color='red', label='Callable Actual' if 'Callable Actual' not in plt.gca().get_legend_handles_labels()[1] else "")
                    plt.scatter(sqrt_actpred_yr[i], 100 * xpre_yield[i], color='purple', label='Callable Predicted' if 'Callable Predicted' not in plt.gca().get_legend_handles_labels()[1] else "")
                else:
                    plt.scatter(sqrt_actpred_yr[i], 100 * xact_yield[i], color='orange', label='Non-Call Actual' if 'Non-Call Actual' not in plt.gca().get_legend_handles_labels()[1] else "")
                    plt.scatter(sqrt_actpred_yr[i], 100 * xpre_yield[i], color='lightblue', label='Non-Call Predicted' if 'Non-Call Predicted' not in plt.gca().get_legend_handles_labels()[1] else "")
        x1 = max(xplot_points_yr)
        if x1 > 20:
            plt.xticks(ticks=np.sqrt([0.25, 1, 2, 5, 10, 20, 30]), labels=['0.25', '1', '2', '5', '10', '20', '30'])
        elif x1 > 10:
            plt.xticks(ticks=np.sqrt([0.25, 1, 2, 5, 10, 20]), labels=['0.25', '1', '2', '5', '10', '20'])
        elif x1 > 5:
            plt.xticks(ticks=np.sqrt([0.25, 1, 2, 5, 7, 10]), labels=['0.25', '1', '2', '5', '7', '10'])
        else:
            plt.xticks(ticks=np.sqrt([0.25, 1, 2, 3, 5]), labels=['0.25', '1', '2', '3', '5'])
        if durscale:
            plt.xlabel('Duration (Years, SqrRt Scale)')
        else:
            plt.xlabel('Maturity (Years, SqrRt Scale)')
    else:
        plt.plot(xplot_points_yr, 100 * rate_df.iloc[0], label=curvetype)
        if df_price_yield is not None:
            for i in range(len(call_flags)):
                if call_flags[i]:
                    plt.scatter(actpred_yr[i], 100 * xact_yield[i], color='red', label='Callable Actual' if 'Callable Actual' not in plt.gca().get_legend_handles_labels()[1] else "")
                    plt.scatter(actpred_yr[i], 100 * xpre_yield[i], color='purple', label='Callable Predicted' if 'Callable Predicted' not in plt.gca().get_legend_handles_labels()[1] else "")
                else:
                    plt.scatter(actpred_yr[i], 100 * xact_yield[i], color='orange', label='Non-Call Actual' if 'Non-Call Actual' not in plt.gca().get_legend_handles_labels()[1] else "")
                    plt.scatter(actpred_yr[i], 100 * xpre_yield[i], color='lightblue', label='Non-Call Predicted' if 'Non-Call Predicted' not in plt.gca().get_legend_handles_labels()[1] else "")
        if durscale:
            plt.xlabel('Duration (Years)')
        else:
            plt.xlabel('Maturity (Years)')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    if not yield_to_worst:
        plt.title(f'{rate_type.capitalize()} Curve, {int(quote_date_ymd)}, vol={yvols}')
    else:
        plt.title(f'{rate_type.capitalize()} Curve, {int(quote_date_ymd)}, Yield-to-Worst')
    plt.ylabel('Rates')

    # Create directory if it does not exist
    if durscale:
        full_path = f'{output_dir}/{estfile}/act_pred_{rate_type}_dur'
    else:
        full_path = f'{output_dir}/{estfile}/act_pred_{rate_type}'
    os.makedirs(full_path, exist_ok=True)
    plt.savefig(f'{full_path}/act_pred_{rate_type}_{int(quote_date_ymd)}.png')
    plt.show()








#%%


def plot_pbcurve(output_dir, estfile, curve, plot_points_yr, rate_type='parbond', yield_to_worst=True,
                 yvols=0, df_price_yield=None, pbtwostep=False, parmflag=True, padj=False, sqrtscale=False,
                 durscale=False, fortran=True, y_min=None, y_max=None):
    
    quote_date = curve.iloc[1]
    quote_date_ymd = dates.JuliantoYMDint(quote_date)
    curvetype = curve.iloc[0]
    curve_df = curve.to_frame().T     # Do this because the function 'produce...' wants df

# Create df of bond parameters
    parms_df = create_parbnd_df(curve_df, plot_points_yr,addmths=False)  # add by days
    rateprice_df, parms_df = produce_pb_zb_anty_dfs(curve_df, parms_df, rate_type, pbtwostep, parmflag, padj)
    rate_df = rateprice_df.xs('rate', level=2)
    xplot_points_yr = plot_points_yr
    
    plt.clf()  # Clear previous plots

    if df_price_yield is not None :
        x4 = df_price_yield[['MatYr','MatMth','MatDay']].to_numpy().tolist()
        x5 = dates.YMDtoJulian(x4)
        x6 = dates.YMDtoJulian(df_price_yield.iloc[0,1].tolist())
        actpred_yr = (x5 - x6)/365.25
        xact_yield = df_price_yield[['actualyield']].to_numpy()
        xpre_yield = df_price_yield[['predictedyield']].to_numpy()
        call_flags = df_price_yield['CallFlag'].astype(bool).to_numpy()
        if durscale:
            actpred_yr = df_price_yield[['MacDur']].to_numpy()

    # sqrt root in the short-term
    if sqrtscale and not(durscale):
        sqrt_plot_points_yr = np.sqrt(xplot_points_yr)
        plt.plot(sqrt_plot_points_yr, 100 * rate_df.iloc[0], label=curvetype)
        if df_price_yield is not None:
            sqrt_actpred_yr = np.sqrt(actpred_yr)
            for i in range(len(call_flags)):
                if call_flags[i]:
                    plt.scatter(sqrt_actpred_yr[i], 100 * xact_yield[i], color='red', label='Callable Actual' if 'Callable Actual' not in plt.gca().get_legend_handles_labels()[1] else "")
                    plt.scatter(sqrt_actpred_yr[i], 100 * xpre_yield[i], color='purple', label='Callable Predicted' if 'Callable Predicted' not in plt.gca().get_legend_handles_labels()[1] else "")
                else:
                    plt.scatter(sqrt_actpred_yr[i], 100 * xact_yield[i], color='orange', label='Non-Call Actual' if 'Non-Call Actual' not in plt.gca().get_legend_handles_labels()[1] else "")
                    plt.scatter(sqrt_actpred_yr[i], 100 * xpre_yield[i], color='lightblue', label='Non-Call Predicted' if 'Non-Call Predicted' not in plt.gca().get_legend_handles_labels()[1] else "")
        x1 = max(xplot_points_yr)
        if x1 > 20:
            plt.xticks(ticks=np.sqrt([0.25, 1, 2, 5, 10, 20, 30]), labels=['0.25', '1', '2', '5', '10', '20', '30'])
        elif x1 > 10:
            plt.xticks(ticks=np.sqrt([0.25, 1, 2, 5, 10, 20]), labels=['0.25', '1', '2', '5', '10', '20'])
        elif x1 > 5:
            plt.xticks(ticks=np.sqrt([0.25, 1, 2, 5, 7, 10]), labels=['0.25', '1', '2', '5', '7', '10'])
        else:
            plt.xticks(ticks=np.sqrt([0.25, 1, 2, 3, 5]), labels=['0.25', '1', '2', '3', '5'])
        if durscale:
            plt.xlabel('Duration (Years, SqrRt Scale)')
        else:
            plt.xlabel('Maturity (Years, SqrRt Scale)')
    else:
        plt.plot(xplot_points_yr, 100 * rate_df.iloc[0], label=curvetype)
        if df_price_yield is not None:
            for i in range(len(call_flags)):
                if call_flags[i]:
                    plt.scatter(actpred_yr[i], 100 * xact_yield[i], color='red', label='Callable Actual' if 'Callable Actual' not in plt.gca().get_legend_handles_labels()[1] else "")
                    plt.scatter(actpred_yr[i], 100 * xpre_yield[i], color='purple', label='Callable Predicted' if 'Callable Predicted' not in plt.gca().get_legend_handles_labels()[1] else "")
                else:
                    plt.scatter(actpred_yr[i], 100 * xact_yield[i], color='orange', label='Non-Call Actual' if 'Non-Call Actual' not in plt.gca().get_legend_handles_labels()[1] else "")
                    plt.scatter(actpred_yr[i], 100 * xpre_yield[i], color='lightblue', label='Non-Call Predicted' if 'Non-Call Predicted' not in plt.gca().get_legend_handles_labels()[1] else "")
        if durscale:
            plt.xlabel('Duration (Years)')
        else:
            plt.xlabel('Maturity (Years)')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.ylabel('Rates')

    if isinstance(quote_date_ymd, (np.ndarray, pd.Series)):
        quote_date_ymd = quote_date_ymd.item()

    date_format = pd.to_datetime(str(int(quote_date_ymd)), format='%Y%m%d').strftime('%Y-%m-%d')

    if not fortran:
        if not yield_to_worst:
            plt.title(f'{rate_type.capitalize()} Curve, {date_format}, vol={yvols} (in bp)')
        else:
            plt.title(f'{rate_type.capitalize()} Curve, {date_format}, Yield-to-Worst (in bp)')
    else:
        plt.title(f'{rate_type.capitalize()} Curve, {date_format} (in bp)')


    plt.ylim(y_min*100 - 0.1 * abs(y_min*100), y_max*100 + 0.1 * abs(y_max*100))

    plt.legend()
    
    # Create output directory if it doesn't exist
    full_path = os.path.join(output_dir, estfile)
    os.makedirs(full_path, exist_ok=True)
    
    # Save the figure
    plt.savefig(f'{full_path}/act_pred_{rate_type}_{quote_date}.png')



# def plot_pbcurve(output_dir, estfile, curve, plot_points_yr, rate_type='parbond', yield_to_worst=True,
#                               yvols=0, df_price_yield=None, pbtwostep=False, parmflag=True, padj=False, sqrtscale=False,
#                               durscale=False, fortran=True, y_min=None, y_max=None):

#     quote_date = curve.iloc[1]
#     quote_date_ymd = dates.JuliantoYMDint(quote_date)
#     curvetype = curve.iloc[0]
#     curve_df = curve.to_frame().T     # Do this because the function 'produce...' wants df

# # Create df of bond parameters
#     parms_df = create_parbnd_df(curve_df, plot_points_yr,addmths=False)  # add by days
#     rateprice_df, parms_df = produce_pb_zb_anty_dfs(curve_df, parms_df, rate_type, pbtwostep, parmflag, padj)
#     rate_df = rateprice_df.xs('rate',level=2)
#     xplot_points_yr = plot_points_yr
        
#     if not(df_price_yield is None) :
#         x4 = df_price_yield[['MatYr','MatMth','MatDay']].to_numpy().tolist()
#         x5 = dates.YMDtoJulian(x4)
#         x6 = dates.YMDtoJulian(df_price_yield.iloc[0,1].tolist())
#         actpred_yr = (x5 - x6)/365.25
#         xact_yield = df_price_yield[['actualyield']].to_numpy()
#         xpre_yield = df_price_yield[['predictedyield']].to_numpy()
#         call_flags = df_price_yield['CallFlag'].astype(bool).to_numpy()
#         if durscale:
#             actpred_yr = df_price_yield[['MacDur']].to_numpy()

#     # sqrt root in the short-term
#     if sqrtscale and not(durscale):
#         sqrt_plot_points_yr = np.sqrt(xplot_points_yr)
#         plt.plot(sqrt_plot_points_yr, 100 * rate_df.iloc[0], label=curvetype)
#         if df_price_yield is not None:
#             sqrt_actpred_yr = np.sqrt(actpred_yr)
#             for i in range(len(call_flags)):
#                 if call_flags[i]:
#                     plt.scatter(sqrt_actpred_yr[i], 100 * xact_yield[i], color='red', label='Callable Actual' if 'Callable Actual' not in plt.gca().get_legend_handles_labels()[1] else "")
#                     plt.scatter(sqrt_actpred_yr[i], 100 * xpre_yield[i], color='purple', label='Callable Predicted' if 'Callable Predicted' not in plt.gca().get_legend_handles_labels()[1] else "")
#                 else:
#                     plt.scatter(sqrt_actpred_yr[i], 100 * xact_yield[i], color='orange', label='Non-Call Actual' if 'Non-Call Actual' not in plt.gca().get_legend_handles_labels()[1] else "")
#                     plt.scatter(sqrt_actpred_yr[i], 100 * xpre_yield[i], color='lightblue', label='Non-Call Predicted' if 'Non-Call Predicted' not in plt.gca().get_legend_handles_labels()[1] else "")
#         x1 = max(xplot_points_yr)
#         if x1 > 20:
#             plt.xticks(ticks=np.sqrt([0.25, 1, 2, 5, 10, 20, 30]), labels=['0.25', '1', '2', '5', '10', '20', '30'])
#         elif x1 > 10:
#             plt.xticks(ticks=np.sqrt([0.25, 1, 2, 5, 10, 20]), labels=['0.25', '1', '2', '5', '10', '20'])
#         elif x1 > 5:
#             plt.xticks(ticks=np.sqrt([0.25, 1, 2, 5, 7, 10]), labels=['0.25', '1', '2', '5', '7', '10'])
#         else:
#             plt.xticks(ticks=np.sqrt([0.25, 1, 2, 3, 5]), labels=['0.25', '1', '2', '3', '5'])
#         if durscale:
#             plt.xlabel('Duration (Years, SqrRt Scale)')
#         else:
#             plt.xlabel('Maturity (Years, SqrRt Scale)')
#     else:
#         plt.plot(xplot_points_yr, 100 * rate_df.iloc[0], label=curvetype)
#         if df_price_yield is not None:
#             for i in range(len(call_flags)):
#                 if call_flags[i]:
#                     plt.scatter(actpred_yr[i], 100 * xact_yield[i], color='red', label='Callable Actual' if 'Callable Actual' not in plt.gca().get_legend_handles_labels()[1] else "")
#                     plt.scatter(actpred_yr[i], 100 * xpre_yield[i], color='purple', label='Callable Predicted' if 'Callable Predicted' not in plt.gca().get_legend_handles_labels()[1] else "")
#                 else:
#                     plt.scatter(actpred_yr[i], 100 * xact_yield[i], color='orange', label='Non-Call Actual' if 'Non-Call Actual' not in plt.gca().get_legend_handles_labels()[1] else "")
#                     plt.scatter(actpred_yr[i], 100 * xpre_yield[i], color='lightblue', label='Non-Call Predicted' if 'Non-Call Predicted' not in plt.gca().get_legend_handles_labels()[1] else "")
#         if durscale:
#             plt.xlabel('Duration (Years)')
#         else:
#             plt.xlabel('Maturity (Years)')

#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     plt.legend(by_label.values(), by_label.keys())

#     date_format = pd.to_datetime(str(int(quote_date_ymd)), format='%Y%m%d').strftime('%Y-%m-%d')

#     if not fortran:
#         if not yield_to_worst:
#             plt.title(f'{rate_type.capitalize()} Curve, {date_format}, vol={yvols} (in bp)')
#         else:
#             plt.title(f'{rate_type.capitalize()} Curve, {date_format}, Yield-to-Worst (in bp)')
#     else:
#         plt.title(f'{rate_type.capitalize()} Curve, {date_format} (in bp)')

#     plt.ylabel('Rates')
#     plt.ylim(y_min*100 - 0.1 * abs(y_min*100), y_max*100 + 0.1 * abs(y_max*100))

#     # Create directory if it does not exist
#     if durscale:
#         full_path = f'{output_dir}/{estfile}/act_pred_{rate_type}_dur'
#     else:
#         full_path = f'{output_dir}/{estfile}/act_pred_{rate_type}'
#     os.makedirs(full_path, exist_ok=True)
#     plt.savefig(f'{full_path}/act_pred_{rate_type}_{int(quote_date_ymd)}.png')
#     # plt.show()



def pb_animation_wrapper(output_dir, estfile, curve_df, df_price_yield, plot_points_yr, table_breaks_yr,
                         crvtypes=['pwtf'], rate_type='parbond', yield_to_worst=True,
                         pbtwostep=False, parmflag=True, padj=False, sqrtscale=False,
                         durscale=False, fortran=True):
    """
    Create an animation of par bond curves over time, looping through all quotedates.
    """

    def animate(date):
        xplot_curve = curve_df.xs(crvtype, level=0).loc[date]
        if df_price_yield is not None:
            xdf_price_yield = df_price_yield.xs(crvtype, level=0).loc[date]
        else:
            xdf_price_yield = None
    
        # Calculate the appropriate y-axis limits using 5-year max and min
        y_min = parbd_rate.loc[(crvtype, str(int(date))), 'min_5yr']
        y_max = parbd_rate.loc[(crvtype, str(int(date))), 'max_5yr']
        
        plt.clf()
        plot_pbcurve(output_dir=output_dir, estfile=estfile, curve=xplot_curve, plot_points_yr=plot_points_yr, 
                     rate_type=rate_type, yield_to_worst=yield_to_worst, yvols=0, df_price_yield=xdf_price_yield, 
                     pbtwostep=pbtwostep, parmflag=parmflag, padj=padj, sqrtscale=sqrtscale, durscale=durscale, 
                     fortran=fortran, y_min=y_min, y_max=y_max)


    # Generate min/max values for the 5-year periods
    parbd_rateprice_df, zerobd_rateprice_df, annuity_rateprice_df, pbparms_df = pb_zb_anty_wrapper(
        curve_df, table_breaks_yr, estfile, output_dir, twostep=pbtwostep, parmflag=parmflag, padj=padj)
    parbd_rate, parbd_cprice, zerobd_rate, zerobd_cprice, annuity_rate, annuity_cprice = seperate_pb_zb_anty_wrapper(
        parbd_rateprice_df, zerobd_rateprice_df, annuity_rateprice_df)
    parbd_rate = find_max_min_pb_rate(parbd_rate)


    for crvtype in crvtypes:
        plot_dates = curve_df.index.get_level_values(1).unique().tolist()
        
        fig = plt.figure()
        
        # Create the animation
        anim = FuncAnimation(fig, animate, frames=plot_dates, repeat=False)
        
        # Save the animation as a GIF
        gif_filename = os.path.join(output_dir, f'pbrate_animation_{estfile}_{crvtype}.gif')
        anim.save(gif_filename, writer=PillowWriter(fps=10))



#     for crvtype in crvtypes:
#         plot_dates = curve_df.index.get_level_values(1).unique().tolist()
#         fig = plt.figure()

#         # Create the animation
#         anim = FuncAnimation(plt.figure(), animate, frames=plot_dates, repeat=False)
        
#         # Use PillowWriter to save as GIF
#         # writer = PillowWriter(fps=10)  # Adjust fps if needed
#         # anim.save(f'{output_dir}/pbrate_animation_{estfile}_{crvtype}.gif', writer=writer)
#         plt.show()
#         anim.save(f'{output_dir}/pbrate_animation_{estfile}_{crvtype}.gif', writer='pillow', fps=10)

# #     for crvtype in crvtypes:
# #         plot_dates = curve_df.index.get_level_values(1).unique().tolist()

# #         # Create the animation
# #         anim = FuncAnimation(plt.figure(), animate, frames=plot_dates, repeat=False)
# #         # plt.show()
# #         writer = FFMpegWriter(fps=60) 
# #         anim.save(f'{output_dir}/pbrate_animation_{estfile}_{crvtype}.mp4', writer = writer, fps = 30) 
#         # try:
#         #     writer = FFMpegWriter(fps=2, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
#         # except FileNotFoundError:
#         #     print("FFMpeg not found, falling back to PillowWriter.")
#         #     writer = PillowWriter(fps=2)
#         # plt.show()
#         #anim.save(f'{output_dir}/pbrate_animation_{estfile}_{crvtype}.mp4', writer=writer)






















#%% Testing single curve graphs, actual vs predicted and par bond curves

def plot_act_single_curve_wrapper(output_dir, estfile, curve_df, df_price_yield, plot_points_yr, selected_month=12, crvtypes=['pwtf'], 
                                  rate_type='parbond', pbtwostep=False, parmflag=True, padj=False, sqrtscale=True):
    """
    Plot actual rates as scatters and fitted curve (typically par bond) for curve types (typically pwtf) 
    for a given month each year.

    Parameters
    ----------
    estfile: str
        Output folder for the graphs.
    curve_df : pandas.DataFrame
        DataFrame containing curve data with multi-index (crvtype, quotedate_ind).
    df_price_yield : pandas.DataFrame
        DataFrame containing price and yield data with multi-index (crvtype, quotedate_ind).
    plot_points_yr : array-like
        Array of points for plotting the curve.
    selected_month : int, optional
        Month for which to plot the curves, by default 12.
    crvtype : list, optional
        List of curve types to plot, by default ['pwtf'].
    rate_type : str, optional
        Type of rate to plot, by default 'parbond'.
    pbtwostep : bool, optional
        Flag for using two-step par bond, by default False.
    parmflag : bool, optional
        Flag for parameter adjustment, by default True.
    padj : bool, optional
        Flag for padding adjustment, by default False.
    sqrtscale : bool, optional
        Flag for using square root scale, by default True.

    Returns
    -------
    None
        The function creates fitted par bond curve plots with actual yields for display and output to the specific output folder.
    
    """

    for crvtype in crvtypes:
        curve_df['month'] = dates.JuliantoYMD(curve_df['quotedate_jul'])[1]
        filtered_curve_df = curve_df[curve_df['month'] == selected_month]
        plot_dates = filtered_curve_df.index.get_level_values(1).unique().tolist()
        
        for date in plot_dates:
            xplot_curve = curve_df.xs(crvtype,level=0).loc[date]
            if df_price_yield is not None:
                xdf_price_yield = df_price_yield.xs(crvtype,level=0).loc[date]
            else:
                xdf_price_yield = None
            plot_singlecurve(output_dir=output_dir, estfile=estfile, curve=xplot_curve, plot_points_yr=plot_points_yr,
                                 rate_type=rate_type, df_price_yield=xdf_price_yield,
                                 pbtwostep=pbtwostep, parmflag=parmflag, padj=padj, sqrtscale=sqrtscale)



def plot_act_pred_single_curve_wrapper(output_dir, estfile, curve_df, df_price_yield, plot_points_yr, selected_month=12,
                                       crvtypes=['pwtf'], rate_type='parbond', yield_to_worst=True, yvols=0, pbtwostep=False,
                                       parmflag=True, padj=False, sqrtscale=True, durscale=False):
    """
    Plot actual rates and predicted rates as scatters by callability and fitted curve (typically par bond)
    for curve types (typically pwtf) for a given month each year.

    Parameters
    ----------
    estfile: str
        Output folder for the graphs.
    curve_df : pandas.DataFrame
        DataFrame containing curve data with multi-index (crvtype, quotedate_ind).
    df_price_yield : pandas.DataFrame
        DataFrame containing price and yield data with multi-index (crvtype, quotedate_ind).
    plot_points_yr : array-like
        Array of points for plotting the curve.
    selected_month : int, optional
        Month for which to plot the curves, by default 12.
    crvtype : list, optional
        List of curve types to plot, by default ['pwtf'].
    rate_type : str, optional
        Type of rate to plot, by default 'parbond'.
    pbtwostep : bool, optional
        Flag for using two-step par bond, by default False.
    parmflag : bool, optional
        Flag for parameter adjustment, by default True.
    padj : bool, optional
        Flag for padding adjustment, by default False.
    sqrtscale : bool, optional
        Flag for using square root scale, by default True.

    Returns
    -------
    None
        The function creates fitted par bond curve plots with actual yields for display and output to the specific output folder.
    
    """

    for crvtype in crvtypes:
        curve_df['month'] = dates.JuliantoYMD(curve_df['quotedate_jul'])[1]
        filtered_curve_df = curve_df[curve_df['month'] == selected_month]
        plot_dates = filtered_curve_df.index.get_level_values(1).unique().tolist()
        
        for date in plot_dates:
            xplot_curve = curve_df.xs(crvtype,level=0).loc[date]
            yield_to_worst = xplot_curve['ytw_flag']   # 1-aug-24 set yield-to-worst flag and yield vols from saved curve
            if not(yield_to_worst):
                yvols = round(xplot_curve['yvols'],4)
            xdf_price_yield = df_price_yield.xs(crvtype,level=0).loc[date]
            plot_singlecurve_act_pred(output_dir=OUTPUT_DIR, estfile=estfile, curve=xplot_curve, plot_points_yr=plot_points_yr,
                                 rate_type=rate_type, yield_to_worst=yield_to_worst, yvols=yvols, df_price_yield=xdf_price_yield,
                                 pbtwostep=pbtwostep, parmflag=parmflag, padj=padj, sqrtscale=sqrtscale,durscale=durscale)

        # for date in plot_dates:
        #     xplot_curve = curve_df.xs(crvtype, level=0).loc[[date]]
        #     if isinstance(xplot_curve, pd.DataFrame) and 'ytw_flag' in xplot_curve.columns:
        #         yield_to_worst = xplot_curve['ytw_flag']
        #     elif isinstance(xplot_curve, pd.Series) and 'ytw_flag' in xplot_curve.index:
        #         yield_to_worst = xplot_curve['ytw_flag']
        #     else:
        #         # xplot_curve['ytw_flag'] = yield_to_worst
        #         yield_to_worst = yield_to_worst
        #     if not(yield_to_worst):
        #         if isinstance(xplot_curve, pd.DataFrame) and 'yvols' in xplot_curve.columns:
        #             yvols = xplot_curve['yvols']
        #             yvols = round(xplot_curve['yvols'], 4)
        #         elif isinstance(xplot_curve, pd.Series) and 'yvols' in xplot_curve.index:
        #             yvols = xplot_curve['yvols']
        #             yvols = round(xplot_curve['yvols'], 4)
        #         else:
        #             yvols = yvols
        #             # xplot_curve['yvols'] = yvols

        #     if df_price_yield is not None:
        #         xdf_price_yield = df_price_yield.xs(crvtype,level=0).loc[date]
        #     else:
        #         xdf_price_yield = None
            
        #     plot_singlecurve_act_pred(output_dir=output_dir, estfile=estfile, curve=xplot_curve, plot_points_yr=plot_points_yr,
        #                             rate_type=rate_type, yield_to_worst=yield_to_worst, yvols=yvols, df_price_yield=xdf_price_yield,
        #                             pbtwostep=pbtwostep, parmflag=parmflag, padj=padj, sqrtscale=sqrtscale, durscale=durscale)




#%%  Plot different estimations for comparison - yield to worst vs. option adjusted yield

def plot_fwdrate_compare_wrapper(output_dir, curve_df1, curve_df2, plot_points_yr, estfile, labels, selected_month=12,
                                 crvtypes=['pwcf', 'pwtf'], taxflag=False, taxability=1, sqrtscale=True):
    """Plot and output forward rate graphs comparing yield to worst and option adjusted yield."""
    
    for crvtype in crvtypes:
        xplot.plot_fwdrate_compare(
            df_curve1=curve_df1, df_curve2=curve_df2, output_dir=output_dir, plot_folder=estfile,
            estfile=estfile, selected_month=selected_month, plot_points_yr=plot_points_yr,
            curve_type=crvtype, taxflag=False, taxability=1, labels=labels, sqrtscale=sqrtscale)

