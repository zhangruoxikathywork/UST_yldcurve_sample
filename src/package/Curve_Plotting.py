#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:19:33 2017

@author: dantongma

"""

#%%
import numpy as np
import discfact as df
import pandas as pd
import pvfn as pv
import matplotlib.pyplot as plt

#%% Plot zero curves
# breaks_test = np.array([0.5,1,2,5,10,])
# rates_test = np.array([0.01,0.02,0.025,0.037,0.042])
# plot_points_test = np.arange(.25,10,0.25)

# def zero_curve_plot_old (breaks,rates,plot_points):
# ## this function takes in input curves (breaks, rates), and plot zero curves at a quarterly 
# ## frequency (plot_points' step is 0.25). it plots 3 types of curve all at once

#     curve_types = ['pwcf','pwlz','pwtf']
#     colors = ['red', 'yellow', 'blue']
#     ## pwcf = red, pwlz = yellow, pwtf = blue. this will be consistent for all curve plotting
#     for i, color in zip(curve_types, colors):
#         input_curve = [i,0.0,breaks,rates]
#         discount_curve = df.discFact(plot_points,input_curve)
#         zero_curve = np.log(discount_curve)/(-plot_points)
#         plt.plot(plot_points, zero_curve, color=color)
#     plt.show()



def zero_curve_plot_old (breaks,rates,plot_points):
## this function takes in input curves (breaks, rates), and plot zero curves at a quarterly 
## frequency (plot_points' step is 0.25). it plots 3 types of curve all at once

    curve_types = ['pwcf','pwlz','pwtf']
    colors = ['red', 'green', 'blue']
    ## pwcf = red, pwlz = green, pwtf = blue. this will be consistent for all curve plotting
    for ctype, color in zip(curve_types, colors):
        stored_curve = [ctype,0.0,breaks,rates]
        discount_curve = df.discFact(plot_points,stored_curve)
        zero_curve = np.log(discount_curve)/(-plot_points)
        plt.plot(plot_points, zero_curve, color=color, label=ctype)
    plt.legend()
    plt.title('Zero Curves')
    plt.show()


# zero_curve_plot_old(breaks_test,rates_test,plot_points_test)


def zero_curve_plot(curves,plot_points_yr):
## this function takes in input curves (breaks, rates), and plot zero curves at a quarterly 
## frequency (plot_points' step is 0.25). it plots 3 types of curve all at once

    colors = ['red', 'green', 'blue']
    ## pwcf = red, pwlz = green, pwtf = blue. this will be consistent for all curve plotting
    xquotedate = curves[0][1]
    plot_points = xquotedate + plot_points_yr*365.25
    for i in range(len(curves)):
#        stored_curve = [ctype,0.0,breaks,rates]
        xcurve = curves[i]
        discount_curve = df.discFact(plot_points,xcurve)
        zero_curve = np.log(discount_curve)/(-plot_points_yr)
        plt.plot(plot_points_yr, zero_curve, color=colors[i], label=xcurve[0])
    plt.legend()
    plt.title('Zero Curves')
    plt.savefig("../../output/zero_curves.png")
    plt.show()  # Show the plot


#%% Plot 3 forward curves
# breaks_test = np.array([0.5,1,2,5,10,])
# rates_test = np.array([0.01,0.02,0.025,0.037,0.042])
# plot_points_test = np.arange(.25,10,0.25)


# def forward_curve_plot_old (breaks,rates,plot_points):
# ## this function takes in input curves (breaks, rates), and plot forward curves at a quarterly 
# ## frequency (plot_points' step is 0.25). it plots 3 types of curve all at once

#     curve_types = ['pwcf','pwlz','pwtf']
#     colors = ['red','yellow','blue']
#     ## pwcf = red, pwlz = yellow, pwtf = blue. this will be consistent for all curve plotting

#     for i,color in zip(curve_types, colors):
#         stored_curve = [i,0.0,breaks,rates]
#         forward_curve = -365*np.log((df.discFact((plot_points+1/365), stored_curve))/(df.discFact(plot_points, stored_curve)))
#         plt.plot(plot_points, forward_curve, color = color)
#     plt.show()

# forward_curve_plot_old (breaks_test,rates_test,plot_points_test)


def forward_curve_plot_old (breaks,rates,plot_points):
## this function takes in input curves (breaks, rates), and plot forward curves at a quarterly 
## frequency (plot_points' step is 0.25). it plots 3 types of curve all at once

    curve_types = ['pwcf','pwlz','pwtf']
    colors = ['red','green','blue']
    ## pwcf = red, pwlz = green, pwtf = blue. this will be consistent for all curve plotting

    for ctype,color in zip(curve_types, colors):
        stored_curve = [ctype,0.0,breaks,rates]
        forward_curve = -365*np.log((df.discFact((plot_points+1/365), stored_curve))/(df.discFact(plot_points, stored_curve)))
        plt.plot(plot_points, forward_curve, color = color,label=ctype)
    plt.legend()
    plt.title("Forward Curves")
    plt.show()


# curve_list = [curve_est_pwcf,curve_est_pwlz,curve_est_pwtf]
# zero_curve_plot (curve_list,plot_points_yr)


def forward_curve_plot (curves,plot_points_yr):
## this function takes in input curves (breaks, rates), and plot forward curves at a quarterly 
## frequency (plot_points' step is 0.25). it plots 3 types of curve all at once

    colors = ['red','green','blue']
    ## pwcf = red, pwlz = green, pwtf = blue. this will be consistent for all curve plotting
    xquotedate = curves[0][1]
    plot_points = xquotedate + plot_points_yr*365.25

    for i in range(len(curves)):
        xcurve = curves[i]
        forward_curve = -365*np.log((df.discFact((plot_points+1), xcurve))/(df.discFact(plot_points, xcurve)))
        plt.plot(plot_points_yr, forward_curve, color = colors[i],label=xcurve[0])
    plt.legend()
    plt.title("Forward Curves")
    plt.savefig("../../output/forward_curves.png")
    plt.show()


#%% Plot 3 par bond curves


def parbond_curve_plot_old(breaks, rates, plot_points, quotedate):
    
    curve_types = ['pwcf','pwlz','pwtf']
    colors = ['red','green','blue']
    ## pwcf = red, pwlz = green, pwtf = blue. this will be consistent for all curve plotting
    xquotedate = quotedate[0]

    for ctype,color in zip(curve_types, colors):
        stored_curve = [ctype,quotedate,breaks,rates]
        parms_maturity= []
        
        for i in plot_points:
            parms_maturity_single = [0,xquotedate + 365.25*i,2,100.,"A/A","eomyes",0, False]
            parms_maturity.append(parms_maturity_single)
        parms_maturity = pd.DataFrame(parms_maturity)
        df_maturity = pv.pvBondFromCurve(stored_curve, settledate = quotedate, parms = parms_maturity)[:,1]
        
        
        parms_annuity = []
        for i in plot_points:
            parms_annuity_single = [1.,xquotedate + 365.25*i,2,0.,"A/A","eomyes",0, False]
            parms_annuity.append(parms_annuity_single)
        parms_annuity = pd.DataFrame(parms_annuity)
        
        
        xx = pv.pvBondFromCurve(stored_curve, settledate = quotedate, parms = parms_annuity)
        pv_annuity = xx[:,1]
        accrued_interest = xx[:,1]-xx[:,0]
        coupon_rates = (100-df_maturity)/(pv_annuity - accrued_interest)
        plt.plot(plot_points, coupon_rates, color = color,label=ctype)
    plt.legend()
    plt.title('Par Bond Curves')
    plt.savefig("../../output/par_bond_curves.png")
    plt.show()


def parbond_rates(curve,plot_points_yr,twostep=False):
## this function takes in an input curve (breaks, rates) and maturities for calculating
## par bond rates. It uses the standard 
##   coupon = (100 - PVZero) / PVAnnuity
## which is based on PV(par bond) = 100. 
## This gives the right answer on coupon dates, but because accrued interest is linear
## (rather than exponential) it does not quite give the right answer in between coupons
## Instead, we want to use the definition for a par bond:
##   coupon = yield 
## We can use a two-step procedure to get very close to the answer:
## 1) Calculate the coupon using the PV = 100 formula above
## 2) Calculate the yield for a bond with that coupon
## Because the coupon from step 1 will be within a few bp of the correct answer, 
## the yield for a bond with that coupon (close to the par bond coupon) will now
## be even closer to the par bond coupon. For all practical pursposes this will
## be close enough. (A quick test for 2999 plot points from .01 to 30 by .01, with a curve:
##     quotedate = 31-dec-1932
##     breaks = 1mth, 6mth, 1yr, 2yr, 5yr, 10yr, 20yr, 30yr
##     rates = 2%, 2.5%, 3%, 3.5%, 4%, 4.5%, 5%, 6%
## shows errors of roughly 2bp at the very short end (a few days) and a mean square
## error of 0.28bp. For the second step, errors at the short end are on the order of
## E-10bp, and mse of 0.001bp)
    xquotedate = curve[1]
    plot_points = np.round(xquotedate + plot_points_yr*365.25,0)

    parms_maturity= []
    parms_annuity = []
    for i in plot_points:   # Make the dummy bonds for zero and annuity
        parms_maturity_single = [0,i,2,100.,"A/A","eomyes",0, False]
        parms_maturity.append(parms_maturity_single)
        parms_annuity_single = [1.,i,2,0.,"A/A","eomyes",0, False]
        parms_annuity.append(parms_annuity_single)
    parms_maturity = pd.DataFrame(parms_maturity)
    parms_annuity = pd.DataFrame(parms_annuity)
    parms_pb = parms_maturity.copy()  # Make a copy which will be for the par bonds
    
    df_maturity = pv.pvBondFromCurve(curve, settledate = xquotedate, parms = parms_maturity)[:,1]
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
    parms_pb[0] = coupon_rates
    dirtyprice = coupon_rates * pv_annuity + df_maturity
    pbrates = coupon_rates.copy()/100.
    if twostep:
        # Now loop through bonds to calculate yields
        for j in range(len(pbrates)):
            pbrates[j] = pv.bondYieldFromPrice_parms(dirtyprice[j], 
                            parms=parms_pb.loc[j:j], 
                            settledate=xquotedate, parmflag=True, padj=False)
    return(pbrates)


def parbond_curve_plot(curves,plot_points_yr,twostep=False):
## this function takes in input curves (breaks, rates), and plot forward curves at a quarterly 
## frequency (plot_points' step is 0.25). it plots 3 types of curve all at once
    
    colors = ['red','green','blue']
    ## pwcf = red, pwlz = green, pwtf = blue. this will be consistent for all curve plotting
    
    for i in range(len(curves)):
        xcurve = curves[i]
        pbrates = parbond_rates(xcurve,plot_points_yr,twostep)
        plt.plot(plot_points_yr, pbrates, color = colors[i],label=xcurve[0])

    plt.legend()
    plt.title('Par Bond Curves')
    plt.savefig("../../output/par_bond_curves.png")
    plt.show()




def parbond_curve_plot_alt(curves,plot_points_yr,twostep=False):
## this function takes in input curves (breaks, rates), and plot forward curves at a quarterly 
## frequency (plot_points' step is 0.25). it plots 3 types of curve all at once
    
    colors = ['red','green','blue']
    ## pwcf = red, pwlz = green, pwtf = blue. this will be consistent for all curve plotting
    xquotedate = curves[0][1]
    plot_points = np.round(xquotedate + plot_points_yr*365.25,0)

    parms_maturity= []
    parms_annuity = []
    for i in plot_points:
        parms_maturity_single = [0,i,2,100.,"A/A","eomyes",0, False]
        parms_maturity.append(parms_maturity_single)
        parms_annuity_single = [1.,i,2,0.,"A/A","eomyes",0, False]
        parms_annuity.append(parms_annuity_single)
    parms_maturity = pd.DataFrame(parms_maturity)
    parms_annuity = pd.DataFrame(parms_annuity)
    parms_pb = parms_maturity.copy()
    
    for i in range(len(curves)):
        xcurve = curves[i]
        df_maturity = pv.pvBondFromCurve(xcurve, settledate = xquotedate, parms = parms_maturity)[:,1]
        xx = pv.pvBondFromCurve(xcurve, settledate = xquotedate, parms = parms_annuity)
        pv_annuity = xx[:,1]
        accrued_interest = xx[:,1]-xx[:,0]
        # This calculates the coupon rate for a clean price of 100
        # This uses "Price = 100" but this is a good defintion of "par bond" only on an exact coupon date
        # Otherwise, I think we should use "Coupon = Yield"
        # The coupon calculated in this way will almost be the right coupon, but not quite
        # We can do a quick fix by calculating the yield for this coupon, because
        # this will be almost the coupon that makes yield = coupon
        coupon_rates = (100-df_maturity)/(pv_annuity - accrued_interest)
        parms_pb[0] = coupon_rates
        dirtyprice = coupon_rates * pv_annuity + df_maturity
        pbrates = coupon_rates.copy()/100.
        if twostep:
            # Now loop through bonds to calculate yields
            for j in range(len(pbrates)):
                pbrates[j] = pv.bondYieldFromPrice_parms(dirtyprice[j], 
                                parms=parms_pb.loc[j:j], 
                                settledate=xquotedate, parmflag=True, padj=False)
        plt.plot(plot_points_yr, pbrates, color = colors[i],label=xcurve[0])

    plt.legend()
    plt.title('Par Bond Curves')
    plt.savefig("../../output/par_bond_curves.png")
    plt.show()



#%% Plot forward curves with taxability

# xres_pwcf_tax_test = np.array([0.01, 0.02, 0.025, 0.037, 0.042, 0.02, 0.03])  # append optimized tax spread
# quote_date = dates.YMDtoJulian(20171012)
# break_dates_test = dates.CalAdd(quote_date,nyear=breaks_test)
# plot_points_yr = np.arange(.25, 10, 0.25)


def get_fwd_rates_w_tax(xres_tax, curvetype, quotedate, breakdates):
    """
    Take in minimization result with tax spread for one type of curve and
    return a list of params for each taxability type.
    Args
    ----
    xres_tax: solver result storing variable
        The variable that stores the optimized forward result, not the foward curve points.
    curvetype: string
        One of the three types: 'pwcf','pwlz','pwtf'.
    quotedate: double or int
        Julian date obtained from dates.YMDtoJulian(), e.g., dates.YMDtoJulian(20171012).
    breakdates: numpy array
        Obtained from dates.CalAdd(quotedate, nyear=breaks_test), where break_test is breaks in years,
        e.g, np.array([0.5,1,2,5,10,]).

    Returns
    -------
    fwd_curve:
        A list of three lists of forward curves params--curve type, quote dates, break dates,
        forward curves--for each taxability.
    
    """
    
    fwd_curve_tax1 = xres_tax['x'][:-2]
    fwd_curve_tax2 = fwd_curve_tax1 + xres_tax['x'][-2]
    fwd_curve_tax3 = fwd_curve_tax1 + xres_tax['x'][-1]
    
    fwd_curve_tax1_list = [curvetype,quotedate,breakdates,fwd_curve_tax1]
    fwd_curve_tax2_list = [curvetype,quotedate,breakdates,fwd_curve_tax2]
    fwd_curve_tax3_list = [curvetype,quotedate,breakdates,fwd_curve_tax3]
    
    fwd_curve = [fwd_curve_tax1_list, fwd_curve_tax2_list, fwd_curve_tax3_list]
    return fwd_curve


# pwcf_fwd_curve_tax = get_fwd_rates_w_tax(xres_pwcf_tax_test, 'pwcf', quote_date, break_dates_test)


def forward_curve_plot_w_taxability(curves, plot_points_yr):
    """Plot 3 taxability types for one type of forward curve.
    Args
    ----
    curves: list
        A list of forward curve params for 3 taxability, derived from get_fwd_rates_w_tax.
    plot_points_yr: numpy array
        An array of curve points for plotting,
        e.g., np.arange(.25, 10, 0.25) >> year 0-10 by quarter.

    Returns
    -------
    A plot for one type of curve with curves of three taxabilities.
    """
    colors = ['black', 'darkorange', 'purple']
    xquotedate = curves[0][1]
    plot_points = xquotedate + plot_points_yr*365.25
    taxability = ['1 fully taxable', '2 partially exempt', '3 fully tax exempt']
    
    for i in range(len(curves)):
        xcurve = curves[i]
        forward_curve = -365*np.log((df.discFact((plot_points+1), xcurve))/(df.discFact(plot_points, xcurve)))
        plt.plot(plot_points_yr, forward_curve, color=colors[i], label=taxability[i])
    plt.legend()
    plt.title(f'Forward Curves with Taxability for {curves[0][0]}')
    plt.show()
    

#forward_curve_plot_w_taxability(pwcf_fwd_curve_tax, plot_points_yr)
