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
import scipy.sparse as sp
import time as time
import cProfile

sys.path.append('../../src/package')
sys.path.append('../../tests')

#%%

import DateFunctions_1 as dates
import pvfn as pv
import pvcover as pvc


imp.reload(dates)
imp.reload(pv)
imp.reload(pvc)


#%% Read in the data (from flat file) into Pandas DataFrame

# Simple to read a .csv file into a datafrme and index by multiple variables - here we need to do by CRSP ID, Coupon, Uniqueness number, and Quote Date
#xxtemp = pd.read_csv("C:/Users/Chenjinzi/Documents/Python Scripts/CRSP12-1932to3-1933.txt",sep='\t',
#                     index_col=['Quote Date','CRSP Issue Identification Number','Coupon Rate (percent per annum)','Uniqueness Number'])
xxtemp = pd.read_csv("../../data/CRSP12-1932to3-1933.txt",sep='\t',
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
                              .str.replace(r'[^-+\d.]', '',regex=True).astype(float))
xxtemp['Publicly Held Face Value Outstanding'] = (xxtemp['Publicly Held Face Value Outstanding']
                              .str.replace(r'[^-+\d.]', '',regex=True).astype(float))


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

#%% Manipulating the pandas dataframes 

# Want to do a copy so that it does not refer back to the original data frame
zz1 = yy3212[['Coupon Rate (percent per annum)','Maturity Date at time of Issue',
    'Number of Interest Payments per Year','First Call Date at Time of Issue',
    "First Price, usually Bid","Second Price, usually Ask","Taxability of Interest"]].copy()
    
x1 = pd.isnull(zz1['First Call Date at Time of Issue'])     # TRUE when call is nan

zz1.loc[x1,'First Call Date at Time of Issue'] = 0.0      # Replace nan s with 0 s

x1 = zz1['Number of Interest Payments per Year'] == 0       # Check for no interest payments (Bills)

x2 = zz1["Second Price, usually Ask"] < 0.
zz1.loc[x2,"Second Price, usually Ask"] = zz1.loc[x2,"First Price, usually Bid"]

zz2 = zz1.loc[np.logical_not(x1)].copy()                           # Get rid of Bills
zz2 = zz1.copy()                                                   # Keep Bills

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
calltrue = []
for i in range(0,len(d22)):                                 # Convert Call Date to Julian
    if d22[i]<19321231:               # Here I should count forward by coupon periods to get past settle date
        D2.append(0)
        calltrue.append(False)
    else:
        d=dates.YMDtoJulian(d22[i])
        D2.append(d[0])
        calltrue.append(True)

zz20['Maturity Date at time of Issue']=pd.Series(D1,index=zz20.index)

zz20['First Call Date at Time of Issue']=pd.Series(D2,index=zz20.index)
zz20['CallFlag'] = pd.Series(calltrue,index=zz20.index)

zz22 = zz20[['Coupon Rate (percent per annum)','Maturity Date at time of Issue',
             'Number of Interest Payments per Year','Redemption Amount',
             'Daycount','eom','First Call Date at Time of Issue',
             'CallFlag','Taxability of Interest']].copy()
zz4 = zz22.values.tolist() 

prices = ( np.array(zz20[["First Price, usually Bid"]]) + 
    np.array(zz20[["Second Price, usually Ask"]]) ) / 2.


#%%  build pwcf curve

#parms=zz22.copy()
#parms = parms.iloc[0:40,:]       # remove the Consol (maturity 2099)
#prices = prices[0:40,]
quotedate=dates.YMDtoJulian(19321231)
breaks = np.array([0.0833, 0.5, 1.,2.,5.,10.,20.,30.])
#breaks = np.array([1.,2.,5.,10.,20.,30.])
#breaks = np.array([1.,2.,5.,10.,30.])

#breakdates = []
#for i in range(0,len(breaks)):
#    bd = dates.CalAdd(quotedate,nyear=breaks[i])
#    breakdates.append(bd)

#breakdates = np.array(breakdates)
breakdates = dates.CalAdd(quotedate,nyear=breaks)
rates = np.array([0.02] * len(breaks))
ratescc = 2.*np.log(1+rates/2.)
ratescc_sprd = np.append(ratescc,[-.0005,-.0005])
curvepwcf = ['pwcf',quotedate,breakdates,ratescc]
tcurvepwcf = ['pwcf',quotedate,breakdates,ratescc,-0.0005,-0.0005]   # Version with spreads for taxability (type 2 and type 3)

#%% Test converting bond parameters to (sparse) arrays

sparsenc = pv.bondParmsToMatrix(quotedate,parms)


cfsparse = sparsenc[0]          # This should already be written as a sparse array
cfdata = sparsenc[1]
datedata = sparsenc[2]       # This is vector of data (to feed into discFact())
aivector = sparsenc[3]      
cfindex = sparsenc[4]
cfptr = sparsenc[5]
bondcount = sparsenc[6]
maxcol = sparsenc[7]
maturity = sparsenc[8]    
finalpmt = sparsenc[9]
coupon = sparsenc[10]
frequency = sparsenc[11]
calldate = sparsenc[12]
callflag = sparsenc[13]

# Now for callable bonds.
# Set the calldate for non-calls beyond the maturity date (20991231 works even for consol)
# and the function bondParmsToMatrix will skip that bond - setting the whole row to zero in sparse matrix
calldate = parms['First Call Date at Time of Issue'].copy()
x1 = parms['CallFlag']
calldate[~parms['CallFlag']] = dates.YMDtoJulian(20991231)[0]

sparseyc = pv.bondParmsToMatrix(calldate,parms)

cfsparse = sparseyc[0]          # This should already be written as a sparse array
cfdata = sparseyc[1]
datedata = sparseyc[2]       # This is vector of data (to feed into discFact())
aivector = sparseyc[3]      
cfindex = sparseyc[4]
cfptr = sparseyc[5]
bondcount = sparseyc[6]
maxcol = sparseyc[7]
maturity = sparseyc[8]    
finalpmt = sparseyc[9]
coupon = sparseyc[10]
frequency = sparseyc[11]
calldate = sparseyc[12]
callflag = sparseyc[13]


#%% Test PVBondFromCurve without & with taxability

yvols = np.tile(.10,len(parms))

bondpv = pv.pvBondFromCurve(curvepwcf,sparsenc)     # calculate the PV of all bonds to the final maturity date
#bondpvt = pv.PVBondFromCurve_vec(quotedate,parms,tcurve)     # calculate the PV of all bonds to the final maturity date


#xxpvcLNP = pv.pvCallable_lnP(quotedate,parms,curvepwcf,pvols)
bondpv_call = pv.pvCallable(curvepwcf, yvols, sparsenc = sparsenc, sparseyc = sparseyc)



calldate = np.array(parms.iloc[:,6])                 # Assume that dates are already Julian
callflag = np.array(parms.iloc[:,7])
if (callflag.any()) :                                # Process callables if any
    xcalldate = calldate.copy()
    xcalldate[~callflag] = dates.YMDtoJulian(20991231)   # For non-callable bonds set call date beyond any maturity

taxtype = parms.iloc[:,8]
x1 = np.array(taxtype == 1)
sparsenc1 = pv.bondParmsToMatrix(quotedate,parms[x1])
sparseyc1 = pv.bondParmsToMatrix(xcalldate[x1],parms[x1])
x2 = np.array(taxtype == 2)
sparsenc2 = pv.bondParmsToMatrix(quotedate,parms[x2])
sparseyc2 = pv.bondParmsToMatrix(xcalldate[x2],parms[x2])
x3 = np.array(taxtype == 3)
sparsenc3 = pv.bondParmsToMatrix(quotedate,parms[x3])
sparseyc3 = pv.bondParmsToMatrix(xcalldate[x3],parms[x3])

#%% Test pvCallable_lnYCover_tax

ratestax = curvepwcf[3].tolist()
ratestax.extend([.0,.0])
ratestax = np.array(ratestax)

xssq = pvc.pvCallable_lnYCover_tax(ratestax, curvepwcf, prices1=prices[x1],vols1=yvols[x1], sparsenc1 = sparsenc1, sparseyc1 = sparseyc1,
                            prices2=prices[x2],vols2=yvols[x2],sparsenc2 = sparsenc2,sparseyc2 = sparseyc2,
                            prices3=prices[x3],vols3=yvols[x3],sparsenc3 = sparsenc3,sparseyc3 = sparseyc3) 


#%% PWCF optimization with and without taxability

t0 = time.perf_counter()
xres_pwcf = so.minimize(pvc.pvCallable_lnYCover, ratescc, args=(prices, list(curvepwcf), yvols, sparsenc, sparseyc), 
        method="Nelder-Mead", jac=False)
#        hess=None, hessp=None, bounds=None, 
#        constraints=(), tol=None, callback=None, options=None)
t1 = time.perf_counter()
print("%.5f seconds (optimize new)" % (t1-t0))


t2 = time.perf_counter()
xres_pwcf_tax = so.minimize(pvc.pvCallable_lnYCover_tax, ratestax, args=(list(curvepwcf), prices[x1],yvols[x1], sparsenc1, sparseyc1,
                            prices[x2],yvols[x2],sparsenc2,sparseyc2,
                            prices[x3],yvols[x3],sparsenc3,sparseyc3), 
        method="Nelder-Mead", jac=False)
#        hess=None, hessp=None, bounds=None, 
#        constraints=(), tol=None, callback=None, options=None)
t3 = time.perf_counter()
print("%.5f seconds (optimize new, tax)" % (t3-t2))
print("%.5f ratio (tax vs notax)" % ((t3-t2)/(t1-t0)))

curvepwcf[3] = xres_pwcf['x']
bondpv_pwcf = pv.pvCallable(curvepwcf, yvols, sparsenc = sparsenc, sparseyc = sparseyc)
errbondpv = bondpv_pwcf[:,0] - prices[:,0]


#%% PWCF actual vs predicted price and yield

# Calculate predicted price and clean actual price
predicted_bondpv = pv.pvCallable(list(curvepwcf), yvols, sparsenc=sparsenc, sparseyc=sparseyc)
predicted_dirty_bondpv = predicted_bondpv[:, 0].tolist()
actual_bondpv = prices.flatten().tolist()

# Temp variable with maturities
x1 = dates.JuliantoYMD(parms['Maturity Date at time of Issue']).T
# Set up price yield df
price_yield_df = pd.DataFrame({'MatYr':x1[:,0],'MatMth':x1[:,1],'MatDay':x1[:,2],
                                            'Coup':parms['Coupon Rate (percent per annum)'],
                                            'predictedprice': predicted_dirty_bondpv, 'actualprice': actual_bondpv,
                                            'predictedyield': np.nan, 'actualyield': np.nan})

# Single attempts
# dirtyprice = price_yield_df['predictedprice'][2]
# parms1 = parms.iloc[[2]]
# settledate = quotedate

# pv.bondYieldFromPrice_parms(dirtyprice=dirtyprice, settledate=quotedate, parms=parms1)
# pv.bondYieldFromPrice_callable(dirtyprice=dirtyprice, settledate=quotedate, parms=parms1)

price_yield_df['predictedyield'] = price_yield_df.apply(
        lambda row: pv.bondYieldFromPrice_callable(
            dirtyprice=price_yield_df.loc[row.name, 'predictedprice'],
            settledate=quotedate,
            parms=parms.loc[[row.name]]
        ), axis=1)
    
price_yield_df['actualyield'] = price_yield_df.apply(
        lambda row: pv.bondYieldFromPrice_callable(
            dirtyprice=price_yield_df.loc[row.name, 'actualprice'],
            settledate=quotedate,
            parms=parms.loc[[row.name]]
        ), axis=1)

# Single attempts
# dirtyprice = price_yield_df['predictedprice'][2]
# parms1 = parms.iloc[[2]]
# settledate = quotedate

# pv.bondYieldFromPrice_parms(dirtyprice=dirtyprice, settledate=quotedate, parms=parms1)
# pv.bondYieldFromPrice_callable(dirtyprice=dirtyprice, settledate=quotedate, parms=parms1)


# def apply_yield(row, parms_df, price_df, pricecol):
#     """Apply bondYieldFromPrice_parms and bondYieldFromPrice_callable to each row based on CallFlag.

#     This function iterates over each row of a DataFrame, using the 'CallFlag' field to determine
#     the appropriate yield calculation method. For non-callable bonds (CallFlag == False),
#     'bondYieldFromPrice_parms' is used. For callable bonds, 'bondYieldFromPrice_callable' is applied.

#     Args:
#         row (pd.series): A row from a dataFrame representing bond data
#         parms_df (pd.datafame_): parms datafame
#         price_df (pd.datafame_): price dataframe
#         pricecol (string): column name of price input

#     Returns:
#        float: The calculated yield for the bond represented by `row`
    
#     Example:
#         >>> parms.apply(apply_yield, axis=1, parms_df=parms, price_df=price_yield_df, pricecol='predictedprice')
#     """
# # Actually, the function "bondYieldFromPrice_callable" should work for either callable or non-callable, 
# #    so should not need the "apply_yield" function
# #    if row['CallFlag'] == False:
# #        return pv.bondYieldFromPrice_parms(price_df.loc[row.name, pricecol], settledate=quotedate, parms=parms_df.loc[[row.name]])
# #    else:
# #        return pv.bondYieldFromPrice_callable(price_df.loc[row.name, pricecol], settledate=quotedate, parms=parms_df.loc[[row.name]])
#     return pv.bondYieldFromPrice_callable(price_df.loc[row.name, pricecol], settledate=quotedate, parms=parms_df.loc[[row.name]])


# price_yield_df['predictedyield'] = parms.apply(apply_yield, axis=1, parms_df=parms, price_df=price_yield_df, pricecol='predictedprice')
# price_yield_df['actualyield'] = parms.apply(apply_yield, axis=1, parms_df=parms, price_df=price_yield_df, pricecol='actualprice')

#%%  Testing bond 23 - a callable bond that has trouble with OAY

# Quotedate = 31-dec-1932
# Bond 23 (call 15-oct-1933, maturity 15-oct-1938, price 103.672) does not work right for callable YTM
# Actual price 103.672 -> y=0.716%, predicted 104.365 -> negative yield which doesn't work for lny model

xPycsab_001 = pv.bondPriceFromYield_callable(xyield=.001,  settledate=quotedate,parms=parms[23:24], 
                               freq=2, parmflag=True,callflag=True,vol=0.2)
xPycsab_02 = pv.bondPriceFromYield_callable(xyield=.02,  settledate=quotedate,parms=parms[23:24], 
                               freq=2, parmflag=True,callflag=True,vol=0.2)
xPycsab_m01 = pv.bondPriceFromYield_callable(xyield=-0.01,  settledate=quotedate,parms=parms[23:24], 
                               freq=2, parmflag=True,callflag=True,vol=0.2)
xYycsab = pv.bondYieldFromPrice_callable(dirtyprice=103.672,settledate=quotedate,parms=parms[23:24], 
                               freq=2, parmflag=True,callflag=True,vol=0.2)




xPnc_50 = pv.bondPriceFromYield_parms(0.5,  settledate=quotedate,parms=parms[23:24], freq=2, parmflag=True)
xPnc_m3 = pv.bondPriceFromYield_parms(-0.03,  settledate=quotedate,parms=parms[23:24], freq=2, parmflag=True)

xYnc_mat = pv.bondYieldFromPrice_parms(103.672,  settledate=quotedate,parms=parms[23:24], freq=2, parmflag=True)
x1 = parms[23:24].copy()
x1.iloc[0,1] = dates.YMDtoJulian(np.array([1933,10,15]))
xYnc_call = pv.bondYieldFromPrice_parms(103.672,  settledate=quotedate,parms=x1, freq=2, parmflag=True)

print(xPycsab_001,xPycsab_02,xPycsab_m01,xYycsab,xYnc_mat,xYnc_call)

# This is a summary:
#   The bond only has a short time to call (10 months)
#   It is trading at a significant premium (DP=103.672, CP=102.77, AI=0.899)
#   Y to Mat = 3.713%
#   Y to Call = 0.7162%
#   OAY should be lower than 0.7162, and indeed 
#   OAY = 0.7157%

# Other bonds with similar maturity have YTM of around 2.5-3%. 
#   So initially this seems like a puzzle as to why this bond is priced so high (such low OAY)
#   But other bonds with maturity similar to first call have yields about 0.7%
#   So this looks right


# But question why it doesn't work with PWLZ


#%% PWLZ optimization with and without taxability

curvepwlz = curvepwcf.copy()
curvepwlz[0] = 'pwlz'

t0 = time.perf_counter()
xres_pwlz = so.minimize(pvc.pvCallable_lnYCover, ratescc, args=(prices, list(curvepwlz), yvols, sparsenc, sparseyc), 
        method="Nelder-Mead", jac=False)

xres_pwcf = so.minimize(pvc.pvCallable_lnYCover, ratescc, args=(prices, list(curvepwcf), yvols, sparsenc, sparseyc), 
        method="Nelder-Mead", jac=False)

#        hess=None, hessp=None, bounds=None, 
#        constraints=(), tol=None, callback=None, options=None)
t1 = time.perf_counter()
print("%.5f seconds (optimize new)" % (t1-t0))


t2 = time.perf_counter()
xres_pwlz_tax = so.minimize(pvc.pvCallable_lnYCover_tax, ratestax, args=(list(curvepwlz), prices[x1],yvols[x1], sparsenc1, sparseyc1,
                            prices[x2],yvols[x2],sparsenc2,sparseyc2,
                            prices[x3],yvols[x3],sparsenc3,sparseyc3), 
        method="Nelder-Mead", jac=False)

#
#        hess=None, hessp=None, bounds=None, 
#        constraints=(), tol=None, callback=None, options=None)
t3 = time.perf_counter()
print("%.5f seconds (optimize new, tax)" % (t3-t2))
print("%.5f ratio (tax vs notax)" % ((t3-t2)/(t1-t0)))


curvepwlz[3] = xres_pwlz['x']
bondpv_pwlz = pv.pvCallable(curvepwlz, yvols, sparsenc = sparsenc, sparseyc = sparseyc)
errbondpv = np.vstack((errbondpv,bondpv_pwlz[:,0] - prices[:,0]))


#%% PWTF optimization with and without taxability

curvepwtf = curvepwcf.copy()
curvepwtf[0] = 'pwtf'

t0 = time.perf_counter()
xres_pwtf = so.minimize(pvc.pvCallable_lnYCover, ratescc, args=(prices, list(curvepwtf), yvols, sparsenc, sparseyc), 
        method="Nelder-Mead", jac=False)
#        hess=None, hessp=None, bounds=None, 
#        constraints=(), tol=None, callback=None, options=None)
t1 = time.perf_counter()
print("%.5f seconds (optimize new)" % (t1-t0))


t2 = time.perf_counter()
xres_pwtf_tax = so.minimize(pvc.pvCallable_lnYCover_tax, ratestax, args=(list(curvepwtf), prices[x1],yvols[x1], sparsenc1, sparseyc1,
                            prices[x2],yvols[x2],sparsenc2,sparseyc2,
                            prices[x3],yvols[x3],sparsenc3,sparseyc3), 
        method="Nelder-Mead", jac=False)
#        hess=None, hessp=None, bounds=None, 
#        constraints=(), tol=None, callback=None, options=None)
t3 = time.perf_counter()
print("%.5f seconds (optimize new, tax)" % (t3-t2))
print("%.5f ratio (tax vs notax)" % ((t3-t2)/(t1-t0)))


curvepwtf[3] = xres_pwtf['x']
bondpv_pwtf = pv.pvCallable(curvepwtf, yvols, sparsenc = sparsenc, sparseyc = sparseyc)
errbondpv = np.transpose(np.vstack((errbondpv,bondpv_pwtf[:,0] - prices[:,0])))


#%% Compare curves and fitted values





#%% Profile an optimize run

#xprof = cProfile.run('so.minimize(pv.pvCallable_lnYCover, ratescc, args=(prices, curve_est, yvols, sparsenc, sparseyc),method="Nelder-Mead", jac=False)', 
#                    sort='tottime')
#xprof = cProfile.run('so.minimize(pv.pvCallable_lnYCover, ratescc, args=(prices, curve_est, yvols, sparsenc, sparseyc),method="Nelder-Mead", jac=False)', 
#                    'profile_minimize.out')
#xprof = cProfile.run('so.minimize(pv.pvCallable_lnYCover_tax, ratescc_sprd, args=(prices, quotedate,parms,curve_est,yvols),method="Nelder-Mead", jac=False)',
#                     'profile_minimize.out')
#
#import pstats
#p = pstats.Stats('profile_minimize.out')
#p.strip_dirs()
#p.sort_stats('cumulative').print_stats(30)
#p.sort_stats('time').print_stats(30)


# as of 26-jul-17, For tottime, the top are:
#    bondPriceFromYield_matrix 
#       - maybe reducing calculations inside by pre-calculating dates as yrs from settle?
#         But I tried that and it makes minimal difference in timing. 
#    dfpwcf (naturally) Would also mean other discount factor functions
#    pvCallable_lny
#       - This has quit a bit of calculations internally. Maybe try to clean them up
#         in python? But I think it may require cython. 



#%% Plot
import test_curvefit2510 as curvefit
import discfact as df

quotedate=dates.YMDtoJulian(19321231)
#breaks = np.array([1.,2.,5.,10.,40.])
breakdates = dates.CalAdd(quotedate,nyear=breaks)
#curve_points_yr = np.arange(.1,40,.1)
curve_points_yr = np.arange(.01,10,.01)

pwcf_fwd_curve_notax = xres_pwcf['x']
pwcf_fwd_curve_notax_list = ['pwcf',quotedate,breakdates,pwcf_fwd_curve_notax]

pwlz_fwd_curve_notax = xres_pwlz['x']
pwlz_fwd_curve_notax_list = ['pwlz',quotedate,breakdates,pwlz_fwd_curve_notax]

pwtf_fwd_curve_notax = xres_pwtf['x']
pwtf_fwd_curve_notax_list = ['pwtf',quotedate,breakdates,pwtf_fwd_curve_notax]

fwd_curve_notax_list = [pwcf_fwd_curve_notax_list, pwlz_fwd_curve_notax_list, pwtf_fwd_curve_notax_list]

curvefit.zero_curve_plot(fwd_curve_notax_list,curve_points_yr)
curvefit.forward_curve_plot(fwd_curve_notax_list,curve_points_yr)


#%% Plot with taxability

import matplotlib.pyplot as plt


pwcf_fwd_curve_tax1 = xres_pwcf_tax['x'][:-2]
pwcf_fwd_curve_tax2 = pwcf_fwd_curve_tax1 + xres_pwcf_tax['x'][-2]
pwcf_fwd_curve_tax3 = pwcf_fwd_curve_tax1 + xres_pwcf_tax['x'][-1]

pwcf_fwd_curve_tax1_list = ['pwcf',quotedate,breakdates,pwcf_fwd_curve_tax1]
pwcf_fwd_curve_tax2_list = ['pwcf',quotedate,breakdates,pwcf_fwd_curve_tax2]
pwcf_fwd_curve_tax3_list = ['pwcf',quotedate,breakdates,pwcf_fwd_curve_tax3]

pwcf_fwd_curve_tax = [pwcf_fwd_curve_tax1_list, pwcf_fwd_curve_tax2_list, pwcf_fwd_curve_tax3_list]


def get_fwd_curves_w_tax(xres_tax, curvetype, quotedate, breakdates):
    
    fwd_curve_tax1 = xres_tax['x'][:-2]
    fwd_curve_tax2 = fwd_curve_tax1 + xres_tax['x'][-2]
    fwd_curve_tax3 = fwd_curve_tax1 + xres_tax['x'][-1]
    
    fwd_curve_tax1_list = [curvetype,quotedate,breakdates,fwd_curve_tax1]
    fwd_curve_tax2_list = [curvetype,quotedate,breakdates,fwd_curve_tax2]
    fwd_curve_tax3_list = [curvetype,quotedate,breakdates,fwd_curve_tax3]
    
    fwd_curve = [fwd_curve_tax1_list, fwd_curve_tax2_list, fwd_curve_tax3_list]
    return fwd_curve


pwcf_fwd_curve_tax = get_fwd_curves_w_tax(xres_pwcf_tax, 'pwcf', quotedate, breakdates)
pwlz_fwd_curve_tax = get_fwd_curves_w_tax(xres_pwlz_tax, 'pwlz', quotedate, breakdates)
pwtf_fwd_curve_tax = get_fwd_curves_w_tax(xres_pwtf_tax, 'pwtf', quotedate, breakdates)


def forward_curve_plot_w_taxability(curves, curve_points_yr):

    colors = ['black', 'darkorange', 'purple']
    xquotedate = curves[0][1]
    curve_points = xquotedate + curve_points_yr*365.25
    taxability = ['1 fully taxable', '2 partially exempt', '3 fully tax exempt']
    
    for i in range(len(curves)):
        xcurve = curves[i]
        forward_curve = -365*np.log((df.discFact((curve_points+1), xcurve))/(df.discFact(curve_points, xcurve)))
        plt.plot(curve_points_yr, forward_curve, color=colors[i], label=taxability[i])
    plt.legend()
    plt.title(f'Forward Curves with Taxability for {curves[0][0]}')
    plt.show()
    

forward_curve_plot_w_taxability(pwcf_fwd_curve_tax, curve_points_yr)
forward_curve_plot_w_taxability(pwlz_fwd_curve_tax, curve_points_yr)
forward_curve_plot_w_taxability(pwtf_fwd_curve_tax, curve_points_yr)


