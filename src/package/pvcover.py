#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:39:19 2017

@author: tcoleman
"""

#%%
# Contains the cover functions for optimization. 
#   TSC 3-aug-17 - The first (pvCallable_lnYCover) is a duplicate from file pvfn.py and 
#       the copy there should be deleted at some point

#%%

import numpy as np
import pandas as pd
import scipy.stats as ss
import scipy.sparse as sp
import scipy.optimize as so

#%%

import pvfn as pv
#import discfact as df
#import DateFunctions_1 as dates



#%% ------ Cover functions for new ln Yield callable bond functions, to modify curve ------

def pvCallable_Cover(est, prices, weights, curve, vols, opt_type, len_parms, sparsenc, sparseyc, yvolsflg=False, padj=False, 
                     padjparm=0, wgttype=1, lam1=1, lam2=2, ss2=0) :  # estparms insread of rates, new arg estindex # cuvve, vols, 
    """Cover function for callable bond function, that inserts curve parameters."""
# Need to read in "curve" as well as "rates" because we need to other parameters of the curve
# (type, breaks, etc.)

    num_breaks = len(curve[2])
    rates = est[:num_breaks]
    curve[3] = rates

    if yvolsflg:
        vols = est[-1]
        vols = np.full(len_parms, vols)

    bondpv = pv.pvCallable(curve, vols, opt_type=opt_type, sparsenc=sparsenc, sparseyc=sparseyc, padj=padj, padjparm=padjparm)

    if wgttype in [1, 0]:
        SS = (bondpv[:, 0] - prices) / np.sqrt(weights)
    elif wgttype == 2:
        SS1i = ((bondpv[:, 0] - prices) ** 2) / weights
        SS1 = sum(SS1i)
        SS2 = sum(ss2)
        nobs = len(prices)
        SS = (nobs * np.log(SS1) + SS2) / 2
    
    # SS = sum(SS**2)  # delete if using leastsq

    return (SS)


#def pvCallable_lnYCover_tax_old(rates, prices, settledate, tparms, tcurve, yvols) :
#    "Cover function for callable bond function, that inserts curve parameters"
## This version assumes spreads for type-2 and type-3 tax status
#
#    tcurve[4:6] = rates[-2:]       # Inserts last 2 elements of "rates" (the spreads) into 
#                                  # the appropriate spots in "curve"
#    tcurve[3] = rates[:-2]         # Inserts the other elements into "rates"
#    bondpv = pvCallable(settledate,tparms,tcurve,yvols)
#    diffs = bondpv[:,0] - prices[:,0]
#    diffs = sum(diffs ** 2)
#
#    return(diffs)

#        xcurve1 = tcurve[0:4]                # Always need base (fully taxable) curve
#        if ((len(tcurve) == 4) or (parms.shape[1] == 6)) :              # Only fully taxable
#            taxcurve = False
#        elif (len(tcurve) == 5) :         # Only one spread - make xcurve2 & xcurve3 same
#            xcurve2 = list(xcurve1)
#            xcurve2[3] = xcurve1[3] + tcurve[4]
#            xcurve3 = list(xcurve2)
#        elif (len(tcurve) == 6) :
#            xcurve2 = list(xcurve1)
#            xcurve2[3] = xcurve1[3] + tcurve[4]
#            xcurve3 = list(xcurve1)
#            xcurve3[3] = xcurve1[3] + tcurve[5]
#        else :
#            return ("Error in PVBondFromCurve_vec with lenght of tcurve")



def pvCallable_Cover_tax(rates, tcurve, opt_type, padj=False, padjparm=0, wgttype=1, lam1=1, lam2=2, ss2=0,
                         prices1=0, weights1=0, vols1=0, sparsenc1=0, sparseyc1=0,
                         prices2=0, weights2=0, vols2=0, sparsenc2=0, sparseyc2=0,
                         prices3=0, weights3=0, vols3=0, sparsenc3=0, sparseyc3=0) :
    "Cover function for callable bond function, that inserts curve parameters"
# Need to read in "curve" as well as "rates" because we need to other parameters of the curve
# (type, breaks, etc.)
# TSC 3-aug-17: for taxability, assume that the last two elements of "rates" are the spreads
#  2nd-last is type-2 spread, 1st-last is type-3 spread

#    xcurve1 = tcurve.copy()
    tcurve[3] = rates[:-2]
    xcurve2 = tcurve.copy()          # need to make copies so that we don't change the original when 
    xcurve3 = tcurve.copy()          # we change the rates in these two (spread) curves
    xcurve2[3] = tcurve[3] + rates[-2]
    xcurve3[3] = tcurve[3] + rates[-1]
    diffs = 0.0

    if (not(type(sparsenc1) == int)) :
        bondpv = pv.pvCallable(tcurve, vols1, opt_type=opt_type, sparsenc=sparsenc1, sparseyc=sparseyc1, padj=padj, padjparm=padjparm)
        # diffs = diffs + sum( (bondpv[:,0] - prices1[:,0]) ** 2 )
        if ((wgttype == 1) | (wgttype == 0)):
            SSi = ((bondpv[:,0] - prices1)**2) / weights1
            SS = sum(SSi)
        elif wgttype == 2:
            SS1i = ((bondpv[:,0] - prices1)**2) / weights1
            SS1 = sum(SS1i)
            SS2 = sum(ss2)
            nobs = len(prices1)
            SS = (nobs * np.log(SS1) + SS2) / 2
        diffs = diffs + SS

    if (not(type(sparsenc2) == int)) :
        bondpv = pv.pvCallable(xcurve2, vols2, opt_type=opt_type, sparsenc=sparsenc2, sparseyc=sparseyc2, padj=padj, padjparm=padjparm)
        # diffs = diffs + sum( (bondpv[:,0] - prices2[:,0]) ** 2 )
        if ((wgttype == 1) | (wgttype == 0)):
            SSi = ((bondpv[:,0] - prices2)**2) / weights2
            SS = sum(SSi)
        elif wgttype == 2:
            SS1i = ((bondpv[:,0] - prices2)**2) / weights2
            SS1 = sum(SS1i)
            SS2 = sum(ss2)
            nobs = len(prices2)
            SS = (nobs * np.log(SS1) + SS2) / 2
        diffs = diffs + SS

    if (not(type(sparsenc3) == int)) :
        bondpv = pv.pvCallable(xcurve3, vols3, opt_type=opt_type, sparsenc=sparsenc3, sparseyc=sparseyc3, padj=padj, padjparm=padjparm)
        # diffs = diffs + sum( (bondpv[:,0] - prices3[:,0]) ** 2 )
        if ((wgttype == 1) | (wgttype == 0)):
            SSi = ((bondpv[:,0] - prices3)**2) / weights3
            SS = sum(SSi)
        elif wgttype == 2:
            SS1i = ((bondpv[:,0] - prices3)**2) / weights3
            SS1 = sum(SS1i)
            SS2 = sum(ss2)
            nobs = len(prices3)
            SS = (nobs * np.log(SS1) + SS2) / 2
        diffs = diffs + SS

    return(diffs)



def pvCallable_Cover_tax_weight(rates, tcurve, opt_type, padj=False, padjparm=0, wgttype=1, lam1=1, lam2=2, ss2=0,
                                prices1=0, vols1=0, sparsenc1=0, sparseyc1=0, weights1=0,
                                prices2=0, vols2=0,sparsenc2=0,sparseyc2=0, weights2=0,
                                prices3=0,vols3=0,sparsenc3=0,sparseyc3=0, weights3=0) :
    "Cover function for callable bond function, that inserts curve parameters"
# Need to read in "curve" as well as "rates" because we need to other parameters of the curve
# (type, breaks, etc.)
# TSC 3-aug-17: for taxability, assume that the last two elements of "rates" are the spreads
#  2nd-last is type-2 spread, 1st-last is type-3 spread

#    xcurve1 = tcurve.copy()
    tcurve[3] = rates[:-2]
    xcurve2 = tcurve.copy()          # need to make copies so that we don't change the original when 
    xcurve3 = tcurve.copy()          # we change the rates in these two (spread) curves
    xcurve2[3] = tcurve[3] + rates[-2]
    xcurve3[3] = tcurve[3] + rates[-1]
    diffs = 0.0
    if (not(type(sparsenc1) == int)) :
        bondpv = pv.pvCallable(tcurve, vols1, opt_type=opt_type, sparsenc = sparsenc1, sparseyc = sparseyc1)
        diffs = diffs + sum( ((bondpv[:,0] - prices1) ** 2) * weights1 )
    if (not(type(sparsenc2) == int)) :
        bondpv = pv.pvCallable(xcurve2, vols2, opt_type=opt_type, sparsenc = sparsenc2, sparseyc = sparseyc2)
        diffs = diffs + sum( ((bondpv[:,0] - prices2) ** 2) * weights2 )
    if (not(type(sparsenc3) == int)) :
        bondpv = pv.pvCallable(xcurve3, vols3, opt_type=opt_type, sparsenc = sparsenc3, sparseyc = sparseyc3)
        diffs = diffs + sum( ((bondpv[:,0] - prices3) ** 2) * weights3 )

    return(diffs)