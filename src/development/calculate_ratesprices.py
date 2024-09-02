##########################################
# This is to produce the forward
# rates and predicted vs actual price and
# yield
##########################################

'''
Overview
-------------
1. Produce outputs
(1) forward rate without taxability
(2) forward rate with taxability
(3) predicted vs actual price and yield without taxability
(4) predicted vs actual price and yield with taxability

Requirements
-------------
../../data/WRDS CRSP UST dataset in csv format
crsp_data_processing.py
produce_inputs.py

'''

import sys
import os
import numpy as np
import pandas as pd
import datetime
import importlib as imp
import scipy.optimize as so
import scipy.sparse as sp
import time as time

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
import crsp_data_processing as data_processing
import produce_inputs as inputs

imp.reload(dates)
imp.reload(pv)
imp.reload(pvc)

# Debugger setup
# yield_to_worst = True
# settledate = 19321231
# #settledate = dt.YMDtoJulian(settledate)[0]
# filepath = '../../data/USTMonthly.csv'  # '../../data/1916to2024_YYYYMMDD.csv'
# bonddata = data_processing.clean_crsp(filepath)
# # bonddata = data_processing.create_weight(bonddata, bill_weight=1)

# parms = inputs.read_and_process_csvdata(bonddata, settledate, 0)
# parms = inputs.filter_yield_to_worst_parms(settledate, parms, yield_to_worst=True)
# parms = inputs.create_weight(parms, wgttype=1, lam1=1, lam2=2)
# calltype = 0  # 0 to keep all bonds, 1 for callable bonds, 2 for non-callable bonds
# curvetype = 'pwcf'
# breaks = np.array([7/365.25, 14/365.25, 21/365.25, 28/365.25, 35/365.25, 52/365.25, 92/365.25, 
#                        184/365.25, 1, 2, 4, 8, 16, 24, 32])  # np.array([0.0833, 0.5, 1.,2.,5.,10.,20.,30.])
# curve_points_yr = np.arange(.01,10,.01)
# quotedate = settledate
# wgttype = 1
# lam1=1
# lam2=2

# %% Calculate rates

def calc_rate_notax(parms, quotedate, breaks, curvetype, opt_type, wgttype=1,
                    lam1=1, lam2=2, padj=False, padjparm=0, yvolsflg=False, yvols=0.1):
    """Calculate forward rates without taxability.
    """

    len_parms = len(parms)

    quotedate = dates.YMDtoJulian(quotedate)
    breakdates = dates.CalAdd(quotedate,nyear=breaks)
    rates = np.array([0.02] * len(breaks))
    ratescc = 2.*np.log(1+rates/2.) # [curve+tax+padj+lambda1,lambda2]

    if yvolsflg:
        vols = np.array([yvols])    # Set the starting value to the input "yvols"
        est = np.concatenate([ratescc, vols])  # Optimize rates and vols
    else:
        yvols = np.full(len(parms), yvols)
        est = ratescc  # Only optimize rates

    curve = [curvetype,quotedate,breakdates,est]  
    prices =  np.array(parms['Price'])
    weights = np.array(parms['Weight'])
    ss2 = 0
    
    if wgttype == 2:
        ss2 = parms['SS2']

    sparsenc = pv.bondParmsToMatrix(quotedate, parms, padj)

    # Now for callable bonds： Set the calldate for non-calls beyond the maturity date (20991231 works even for consol)
    # and the function bondParmsToMatrix will skip that bond - setting the whole row to zero in sparse matrix
    calldate = parms['First Call Date at Time of Issue'].copy()
    x1 = parms['CallFlag']
    calldate[parms['CallFlag'] == 0] = dates.YMDtoJulian(20991231)[0]
    sparseyc = pv.bondParmsToMatrix(calldate, parms, padj)

    # Curve optimization without taxability
# TSC 3-jul-2024. If wgttype = 1 or 0, then this is NLLS, and use so.leastsq which
# requires the Act - Pred (not the sum of squares)    
    if wgttype in [1, 0]:
# Faster than "BFGS" (about 2/3 the time)
        xres = so.leastsq(pvc.pvCallable_Cover, est, args=(prices, weights, list(curve), yvols, opt_type, len_parms, sparsenc, sparseyc,
                                                           yvolsflg, padj, padjparm, wgttype, lam1, lam2, ss2), full_output=True)
        # xres = so.minimize(pvc.pvCallable_lnYCover, est, args=(prices, weights, list(curve), yvols, len_parms, sparsenc, sparseyc,
        #                                                        yvolsflg, padj, padjparm, wgttype, lam1, lam2, ss2), method="BFGS", jac=False)
        if yvolsflg:
            est = xres[0]
            rates = est[:len(breaks)]
            vols = est[-1]
            yvols = np.array([vols] * len(parms))
            curve[3] = rates
        else:
            curve[3] = xres[0]
            
        ssq = pvc.pvCallable_Cover(curve[3], prices, weights, list(curve), yvols, opt_type, len_parms, sparsenc, sparseyc, yvolsflg, 
                                   padj, padjparm, wgttype, lam1, lam2, ss2)
        ssq = sum(ssq**2)

        # Calculate stderr from the Hessian matrix
        # hessian_inv = xres.hess_inv
        # if hessian_inv is not None:
        #     stderr = np.sqrt(np.diag(hessian_inv) * ssq / (len(prices) - 1))
        # else:
        #     stderr = None
        #     print("Warning: Covariance matrix could not be computed. Optimization might not have converged properly.")

        if xres[1] is not None:
            stderr = np.sqrt(np.diag(xres[1]) * ssq / (len(prices) - 1))
        else:
            stderr = None
            print("Warning: Covariance matrix could not be computed. Optimization might not have converged properly.")

        mesg = xres[3]
        ier = xres[4]
        # mesg = xres.message
        # ier = xres.success
        # xres = so.leastsq(pvc.pvCallable_lnYCover, ratescc, args=(prices, weights, list(curve), yvols, sparsenc, sparseyc,
        #                                                        padj, padjparm, wgttype, lam1, lam2, ss2),full_output=True)
        # curve[3] = xres[0]
        # ssq = pvc.pvCallable_lnYCover(curve[3],prices, weights, list(curve), yvols, sparsenc, sparseyc,
        #                                                        padj, padjparm, wgttype, lam1, lam2, ss2)
        # ssq = sum(ssq**2)
        # stderr = np.sqrt(np.diag(xres[1]) * ssq / (len(prices)-1))
    
# If wgttype= 2 then ML and use so.optimize and return likelihood function
    elif wgttype == 2:

        # xres = so.minimize(pvc.pvCallable_lnYCover, est, args=(prices, weights, list(curve), yvols, len_parms,
        #  sparsenc, sparseyc, yvolsflg, padj, padjparm, wgttype, lam1, lam2, ss2), method="BFGS", jac=False)
        xres = so.leastsq(pvc.pvCallable_Cover, est, args=(prices, weights, list(curve), yvols, opt_type, len_parms, sparsenc, sparseyc,
                                                           yvolsflg, padj, padjparm, wgttype, lam1, lam2, ss2), full_output=True)

        
        if yvolsflg:
            est = xres[0]
            rates = est[:len(breaks)]
            vols = est[-1]
            yvols = np.array([vols] * len(parms))
            curve[3] = rates
        else:
            curve[3] = xres[0]
            
        ssq = pvc.pvCallable_Cover(curve[3], prices, weights, list(curve), yvols, opt_type, len_parms, sparsenc, sparseyc, yvolsflg, 
                                   padj, padjparm, wgttype, lam1, lam2, ss2)
        ssq = sum(ssq**2)
        
        # if yvolsflg:
        #     est = xres['x']
        #     rates = est[:len(breaks)]
        #     vols = est[-1]
        #     yvols = np.array([vols] * len(parms))
        #     curve[3] = rates
        # else:
        #     curve[3] = xres['x']
        
        mesg = xres[3]
        ier = xres[4]
        
        # mesg = xres.message
        # ier = xres.success

    bondpv = pv.pvCallable(curve, yvols, opt_type=opt_type, sparsenc=sparsenc, sparseyc=sparseyc, padj=padj, padjparm=padjparm)
    # errbondpv = bondpv[:,0] - prices[:,0]
    vol_single = yvols[0] if not yvolsflg else vols
    
    return curve, prices, bondpv, stderr, vol_single, mesg, ier
    
# # If wgttype= 2 then ML and use so.optimize and return likelihood function
#     elif wgttype == 2:
#         xres = so.minimize(pvc.pvCallable_lnYCover, est, args=(prices, weights, list(curve), yvols,
#          len_parms, sparsenc, sparseyc, yvolsflg, padj, padjparm, wgttype, lam1, lam2, ss2), method="BFGS", jac=False)
        
#         if yvolsflg:
#             curve[3] = xres['x']
#         else:
#             est = xres['x']
#             rates = est[:len(breaks)]
#             yvols = est[-1]
#             curve[3] = rates
        
        # xres = so.minimize(pvc.pvCallable_lnYCover, ratescc_yvols, args=(prices, weights, list(curve), yvols, sparsenc, sparseyc,
        #                                                        padj, padjparm, wgttype, lam1, lam2, ss2), 
        #                method="BFGS", jac=False)
        # curve[3] = xres['x']




def cal_rate_with_tax(parms, quotedate, breaks, curvetype, opt_type, wgttype=1, lam1=1, lam2=2,
                            padj=False, padjparm=0):
    """Calculate forward rates with taxability.
    """

    quotedate = dates.YMDtoJulian(quotedate)  # 19321231
    breakdates = dates.CalAdd(quotedate,nyear=breaks)
    rates = np.array([0.02] * len(breaks))
    ratescc = 2.*np.log(1+rates/2.)
    curve = [curvetype,quotedate,breakdates,ratescc]  # 'pwcf'
    yvols = np.tile(.10,len(parms))
    prices = parms['Price']
    weights = parms['Weight']
    ss2 = 0

    if wgttype == 2:
        ss2 = parms['SS2']

    sparsenc = pv.bondParmsToMatrix(quotedate, parms, padj)

    # Now for callable bonds.
    # Set the calldate for non-calls beyond the maturity date (20991231 works even for consol)
    # and the function bondParmsToMatrix will skip that bond - setting the whole row to zero in sparse matrix
    calldate = parms['First Call Date at Time of Issue'].copy()
    x1 = parms['CallFlag']
    calldate[parms['CallFlag'] == False] = dates.YMDtoJulian(20991231)[0]

    sparseyc = pv.bondParmsToMatrix(calldate, parms, padj)

    calldate = np.array(parms.iloc[:,6])                 # Assume that dates are already Julian
    callflag = np.array(parms.iloc[:,7])
    xcalldate = calldate.copy()
    # if (callflag.any()) :                                # Process callables if any
    #     xcalldate = calldate.copy()
    #     xcalldate[~callflag] = dates.YMDtoJulian(20991231)   # For non-callable bonds set call date beyond any maturity

    taxtype = parms['Taxability of Interest']  # parms.iloc[:,8] 
    x1 = np.array(taxtype == 1)
    sparsenc1 = pv.bondParmsToMatrix(quotedate,parms[x1],padj)
    sparseyc1 = pv.bondParmsToMatrix(xcalldate[x1],parms[x1],padj)

    x2 = np.array(taxtype == 2)
    sparsenc2 = pv.bondParmsToMatrix(quotedate,parms[x2],padj)
    sparseyc2 = pv.bondParmsToMatrix(xcalldate[x2],parms[x2],padj)
    x3 = np.array(taxtype == 3)
    sparsenc3 = pv.bondParmsToMatrix(quotedate,parms[x3],padj)
    sparseyc3 = pv.bondParmsToMatrix(xcalldate[x3],parms[x3],padj)
    
    ratestax = curve[3].tolist()
    ratestax.extend([.0,.0])
    ratestax = np.array(ratestax)
    
    # Curve optimization with taxability
    xssq = pvc.pvCallable_Cover_tax(ratestax, curve, opt_type, padj=padj, padjparm=padjparm, wgttype=wgttype, lam1=lam1, lam2=lam2, ss2=ss2,
                                prices1=prices[x1], weights1=weights[x1], vols1=yvols[x1], sparsenc1 = sparsenc1, sparseyc1 = sparseyc1,
                                prices2=prices[x2], weights2=weights[x2], vols2=yvols[x2], sparsenc2 = sparsenc2, sparseyc2 = sparseyc2,
                                prices3=prices[x3], weights3=weights[x3], vols3=yvols[x3], sparsenc3 = sparsenc3, sparseyc3 = sparseyc3) 
    
    # Curve optimization with and without taxability
    xres_tax = so.minimize(pvc.pvCallable_Cover_tax, ratestax, args=(curve, opt_type, padj, padjparm, wgttype, lam1, lam2, ss2, 
                                                                        prices[x1], weights[x1], yvols[x1], sparsenc1, sparseyc1,
                                                                        prices[x2], weights[x2], yvols[x2], sparsenc2, sparseyc2,
                                                                        prices[x3], weights[x3], yvols[x3], sparsenc3, sparseyc3), 
                           method="BFGS", jac=False)
    curve[3] = xres_tax['x']
    
    # bondpv = pv.pvCallable(curve, yvols, sparsenc = sparsenc, sparseyc = sparseyc)
    # errbondpv = bondpv[:,0] - prices[:,0]
    
    return curve


#%% Actual vs predicted price and yield without taxability

def get_predicted_actual_yieldprice_notax(parms, bondpv, prices, quotedate, curvetype,
                                          padj=False, yvolsflg=False, yvols=0.1):
    """Get predicted and actual yield and price without taxability.

    Args:
        parms (pd.Dataframe): Bond parameters dataframe for a quote date
        bondpv (array): Predicted dirty and clean price
        prices (array): Bond price, most often the bid ask average
        quotedate (int): Quote date, in the form of YYYYMMDD, e.g. 19321231
        curvetype (str): Curve type

    Returns:
        price_yield_df: Dataframe comparing predicted and actual price and yield
    """

    quotedate = dates.YMDtoJulian(quotedate)

    # Calculate predicted price and clean actual price
    predicted_price = bondpv.copy()
    predicted_price = predicted_price[:, 1].tolist()
    x1 = prices + (bondpv[:,1] - bondpv[:,0])  # Need actual dirty price, so ge AI from predicted
    actual_price = x1.flatten().tolist()
    # If put in a single vol, use it for all vols
    yvols = np.full(len(parms), yvols) 

    yvols_df = pd.DataFrame(yvols)   # In following "apply" could not figure out how to index except via df

    # Temp variable with maturities
    x1 = dates.JuliantoYMD(parms['Maturity Date at time of Issue']).T
    # Set up price yield df
    price_yield_df = pd.DataFrame({'MatYr':x1[:,0],'MatMth':x1[:,1],'MatDay':x1[:,2],
                                            'Coup':parms['Coupon Rate (percent per annum)'],
                                            'predictedprice': predicted_price, 'actualprice': actual_price,
                                            'predictedyield': np.nan, 'actualyield': np.nan})
    def MacDur(dirtyprice, settledate, xyield, parms, padj, vol  ):
        priceup = pv.bondPriceFromYield_callable(xyield + .0005, 
            settledate=settledate,parms=parms, vol=vol,parmflag=True, padj=padj)
        pricedn = pv.bondPriceFromYield_callable(xyield - .0005, 
            settledate=settledate,parms=parms, vol=vol,parmflag=True, padj=padj)
        bpv = (pricedn - priceup) / 0.10
        macdur = 100 * (bpv / dirtyprice) * (1 + xyield/2)
        return float(macdur)



    price_yield_df['predictedyield'] = price_yield_df.apply(
        lambda row: pv.bondYieldFromPrice_callable(
            dirtyprice=price_yield_df.loc[row.name, 'predictedprice'],
            settledate=quotedate,
            parms=parms.loc[[row.name]],
            padj=padj,vol=yvols_df.loc[row.name]
        ), axis=1)
    
    price_yield_df['actualyield'] = price_yield_df.apply(
        lambda row: pv.bondYieldFromPrice_callable(
            dirtyprice=price_yield_df.loc[row.name, 'actualprice'],
            settledate=quotedate,
            parms=parms.loc[[row.name]],
            padj=padj,vol=yvols_df.loc[row.name]
        ), axis=1)
    
    price_yield_df['MacDur'] = price_yield_df.apply(
        lambda row: MacDur(
            dirtyprice=price_yield_df.loc[row.name, 'actualprice'],
            settledate=quotedate,
            xyield = price_yield_df.loc[row.name,'actualyield'],
            parms=parms.loc[[row.name]],
            padj=padj,vol=yvols_df.loc[row.name]
        ), axis=1)


    price_yield_df['CallFlag'] = parms['CallFlag']
    # output_path = f'../../output/price_yield_df_{curvetype}.csv'
    # price_yield_df.to_csv(output_path, index=False)

    return price_yield_df
