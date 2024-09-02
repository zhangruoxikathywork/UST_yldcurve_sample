# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:41:39 2016

@author: Chenjinzi
"""

import sys
import numpy as np
import pandas as pd
import importlib as imp
import scipy.optimize as so

sys.path.append('../src/package')

#%%

import DateFunctions_1 as dates
import pvfn as pv

imp.reload(dates)
imp.reload(pv)


#%% Read in the data (from flat file) into Pandas DataFrame

# Simple to read a .csv file into a datafrme and index by multiple variables - here we need to do by CRSP ID, Coupon, Uniqueness number, and Quote Date
#xxtemp = pd.read_csv("C:/Users/Chenjinzi/Documents/Python Scripts/CRSP12-1932to3-1933.txt",sep='\t',
#                     index_col=['Quote Date','CRSP Issue Identification Number','Coupon Rate (percent per annum)','Uniqueness Number'])
xxtemp = pd.read_csv("/Users/tcoleman/tom/yields/New2015/misc/CRSP12-1932to3-1933.txt",sep='\t',
                     index_col=['Quote Date','CRSP Issue Identification Number','Coupon Rate (percent per annum)','Uniqueness Number'])

# Simple to read a .csv file into a datafrme and index by multiple variables - here we need to do by CRSP ID, Coupon, Uniqueness number, and Quote Date
#xxtemp = pd.read_csv("/Users/tcoleman/tom/yields/New2015/misc/temp.txt",sep='\t',
#                     index_col=['Quote Date','CRSP Issue Identification Number','Coupon Rate (percent per annum)','Uniqueness Number'])

#xxtemp2 = pd.read_csv("/Users/tcoleman/tom/yields/New2015/misc/temp2.txt",sep='\t',
#                     index_col=['Quote Date','CRSP Issue Identification Number','Coupon Rate (percent per annum)','Uniqueness Number'])

# Convert currency ($ & , separator) to float
# This is faster than using the converters in pd.read_csv, apparently
#xxtemp2['Face Value Outstanding'] = (xxtemp2['Face Value Outstanding']
#                              .str.replace(r'[^-+\d.]', '').astype(float))
xxtemp['Face Value Outstanding'] = (xxtemp['Face Value Outstanding']
                              .str.replace(r'[^-+\d.]', '').astype(float))
xxtemp['Publicly Hel Face Value Outstanding'] = (xxtemp['Publicly Held Face Value Outstanding']
                              .str.replace(r'[^-+\d.]', '').astype(float))


#%% Select out two relevant months (last month and this month)
# Dataframe is indexed by
# 0   Quote Date
# 1   CRSP ID
# 2   Coupon
# 3   Uniqueness


#To "index" into different parts of datafrme we can use indexes. For levels other than 0 have to specify level
# These will be "series"
xx1 = xxtemp.xs(19321231)['Number of Interest Payments per Year']

xx2 = xxtemp.xs(19330131)[['Number of Interest Payments per Year','Maturity Date at time of Issue']]

xx3 = xxtemp.xs(19321231)['First Price, usually Bid']

xx4 = xxtemp.xs(19340502.2,level=1)['First Price, usually Bid']

# To get all data for a specific quote date
# This will remain "DataFrame"
xx3212 = xxtemp.xs(19321231)



# To index multiple columns
xx6 = xx3212[['First Price, usually Bid','Face Value Outstanding']]

xx7 = xxtemp.xs(19340502.2,level=1)[['First Price, usually Bid','Face Value Outstanding']]

# removes the indexes - applied to xxtemp this is the same as not reading in with indexes
# But here it's applied to one quote date
yy3212 = xx3212.reset_index()

#zz1 = pd.concat([xx3212,xx3301],axis=1)

#%%

zz1 = yy3212[['Coupon Rate (percent per annum)','Maturity Date at time of Issue',
    'Number of Interest Payments per Year','First Call Date at Time of Issue',
    "First Price, usually Bid","Second Price, usually Ask"]]
    
x1 = pd.isnull(zz1['First Call Date at Time of Issue'])     # TRUE when call is nan

zz1.loc[x1,'First Call Date at Time of Issue'] = 0.0      # Replace nan s with 0 s

x1 = zz1['Number of Interest Payments per Year'] == 0       # Check for no interest payments (Bills)

x2 = zz1["Second Price, usually Ask"] < 0.
zz1.loc[x2,"Second Price, usually Ask"] = zz1.loc[x2,"First Price, usually Bid"]

zz2 = zz1.loc[np.logical_not(x1)]                           # Get rid of Bills


zz2['Redemption Amount'] = pd.Series(100.0, index=zz2.index)  # Append a new column (using original index)

zz3 = np.array(zz2)                                         # Convert from Pandas DataFrame to NumPy Array
                              # But cannot convert to NumPy Array once we put in string variables. 

zz2['Daycount'] = pd.Series("A/A", index=zz2.index)
zz2['eom'] = pd.Series("eomyes", index=zz2.index)
zz20 = zz2.reset_index()
#zz21 = zz20[['Coupon Rate (percent per annum)','Number of Interest Payments per Year', 
#    'Redemption Amount','Daycount','eom']]

d1 = zz20['Maturity Date at time of Issue']
d1 = d1.tolist()                                              # Convert Maturity Date to Julian
D1 = dates.YMDtoJulian(d1)

d2 = zz20['First Call Date at Time of Issue']   
d2 = d2.tolist()
d22 = [int(i) for i in d2]
D2 = []
for i in range(0,len(d22)):                                 # Convert Call Date to Julian
    if d22[i]<19321231:
        D2.append(0)
    else:
        d=dates.YMDtoJulian(d22[i])
        D2.append(d[0])

zz20['Maturity Date at time of Issue']=pd.Series(D1,index=zz20.index)

zz20['First Call Date at Time of Issue']=pd.Series(D2,index=zz20.index)

zz22 = zz20[['Coupon Rate (percent per annum)','Maturity Date at time of Issue','Number of Interest Payments per Year','Redemption Amount','Daycount','eom','First Call Date at Time of Issue']]
zz4 = zz22.values.tolist() 

zzPrices = ( np.array(zz20[["First Price, usually Bid"]]) + 
    np.array(zz20[["Second Price, usually Ask"]]) ) / 2.


#%%  build pwcf curve

parms=zz4
settledate=dates.YMDtoJulian(19321231)
breaks = np.array([1.,2.,5.,10.,40.])
#breakdates = []
#for i in range(0,len(breaks)):
#    bd = dates.CalAdd(settledate,nyear=breaks[i])
#    breakdates.append(bd)

#breakdates = np.array(breakdates)
breakdates = dates.CalAdd(settledate,nyear=breaks)
rates = np.array([0.02,0.02,0.02,0.02,0.02])
ratescc = 2.*np.log(1+rates/2.)
curve = ['pwcf',settledate,breakdates,ratescc]

#%%
# 30-jun-17 This does not work - old version with lists, needs to be updated to pandas dataframe

pvols = np.tile(.03,len(parms))
yvols = np.tile(.10,len(parms))

bondpv = pv.PVBondFromCurve_vec(settledate,parms,curve)     # calculate the PV of all bonds to the final maturity date


xxpvcLNP = pv.pvCallable_lnP(settledate,parms,curve,pvols)
xxpvcLNY = pv.pvCallable_lnY(settledate,parms,curve,yvols)

curve_est = list(curve)

ssq = pv.pvCallable_lnYCover(ratescc, zzPrices, settledate, parms, curve_est, yvols)

#%% Try to optimize
# 30-jun-17 This does not work - old version with lists, needs to be updated to pandas dataframe

xres_pwcf = so.minimize(pv.pvCallable_lnYCover, ratescc, args=(zzPrices, settledate,parms,curve_est,yvols), 
        method="Nelder-Mead", jac=False)
#        hess=None, hessp=None, bounds=None, 
#        constraints=(), tol=None, callback=None, options=None)

curve_est = list(curve)
curve_est[0] = 'pwtf'
xres_pwcf = so.minimize(pv.pvCallable_lnYCover, ratescc, args=(zzPrices, settledate,parms,curve_est,yvols), 
        method="Nelder-Mead", jac=False)
#        hess=None, hessp=None, bounds=None, 
#        constraints=(), tol=None, callback=None, options=None)



