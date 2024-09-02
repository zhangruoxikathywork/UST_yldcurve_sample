# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 14:07:04 2016

@author: Chenjinzi, modified by Coleman
"""

import numpy as np
import scipy.stats as ss
import discfact as df
import pvfn as pv

def PVCallable_sequential(settledate,parms,curve):
    PV=[]
    Price=[]
    xlen=len(parms)

    for i in range(0,xlen):
        if parms[i][6]==0:
            PVCallable=pv.PVBondFromCurve_vec(settledate,[parms[i]], curve)
            PV.append(PVCallable[1])
            Price.append(PVCallable[0])
        else:
            PNonCall=pv.PVBondFromCurve_vec(settledate,[parms[i]],curve)
            PVNonCall=PNonCall[1]
            PriceNonCall=PNonCall[0]
            dfcall=df.discFact(parms[i][6],curve)   #discount factor to discount call option value to taday
            pfwd=pv.PVBondFromCurve_vec(parms[i][6],[parms[i]],curve)  # discount coupouns after calldate back to today
            pvfwd=pfwd[1]
            Pfwd=pvfwd/dfcall        #forward price 
            T=(parms[i][6]-settledate)/365.25
            sigma=0.1                  # volatility of bond price
            K=100                     # exercise price
            d1=(np.log(Pfwd/K)+(1/2)*(sigma**2)*T)/(sigma*np.sqrt(T))
            d2=d1-sigma*np.sqrt(T)
            call=dfcall*(ss.norm.cdf(d1)*Pfwd-ss.norm.cdf(d2)*K)
            pvcallable=PVNonCall-call
            pricecallable=PriceNonCall-call
            PV.append(pvcallable)
            Price.append(pricecallable)
    return([np.array(Price),np.array(PV)])


def pvCallable(settledate,parms,curve):
    bondpv=[]
    bondprice=[]

    x1 = [len(x) for x in parms]          # list of length of parms for each bond
    x2 = range(len(parms))                # no. of bonds 
    x3 = np.array(x2)[(np.array(x1) > 6)] # indexes of bonds with parms with 7 elements (call dates)
    x4 = [parms[i] for i in x3]           # parms for bonds with call dates
    x5 = [x4[i][6] for i in range(0,len(x4))]  # the 7th (6th) element - call date
    x6 = x3[np.array(x5) > 0]                 # indexes of bonds where 7th element is actually a date

    bondpv = pv.PVBondFromCurve_vec(settledate,parms,curve)     # calculate the PV of all bonds to the final maturity date

    if (len(x6) > 0) :
        xparms = [parms[i] for i in x6]
        PNonCall=pv.PVBondFromCurve_vec(settledate,xparms,curve)
        PVNonCall=PNonCall[1]
#        PriceNonCall=PNonCall[0]
        calldate = np.array([parms[i][6] for i in x6])  # With numpy array this would be x[,0] but doesn't work
        dfcall=df.discFact(calldate,curve)   #discount factor to discount call option value to taday
        pfwd=pv.PVBondFromCurve_vec(calldate,xparms,curve)  # discount coupouns after calldate back to today
        pvfwd=pfwd[1]
        Pfwd=pvfwd/dfcall        #forward price 
        T=(calldate-settledate)/365.25
        sigma=0.1                  # volatility of bond price
        K=100                     # exercise price
        d1=(np.log(Pfwd/K)+(1/2)*(sigma**2)*T)/(sigma*np.sqrt(T))
        d2=d1-sigma*np.sqrt(T)
        call=dfcall*(ss.norm.cdf(d1)*Pfwd-ss.norm.cdf(d2)*K)
        aicallable = bondpv[1,x6] - bondpv[0,x6]
        pvcall=PVNonCall-call
        bondpv[1,x6] = pvcall
        bondpv[0,x6] = pvcall - aicallable


    return(bondpv)
