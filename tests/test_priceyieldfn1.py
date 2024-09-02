# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 21:04:28 2016

@author: tcoleman"""

import sys
import numpy as np
import importlib as imp
import pandas as pd

sys.path.append('../src/package')

#%%

import discfact as df
import priceyieldfn1 as py

imp.reload(py)
imp.reload(df)

#%% ------ Sample data for curves ------
# Sample maturity dates for testing dates and bonds
mat = np.array([0.25,0.5,0.75,0.9,0.99,1.0,1.2])
# Rates
ratessab = np.array([0.01,0.02005,0.02512,0.03724])
ratescc = 2.*np.log(1+ratessab/2.)
# Create curve lists - type, anchor point (quote date), break points, rates
curvepwcf = ['pwcf',0.0,np.array([0.5,1.,2.,5.]),ratescc]
curvepwlz = ['pwlz',0.0,np.array([0.5,1.,2.,5.]),ratescc]
curvepwtf = ['pwtf',0.0,np.array([0.5,1.,2.,5.]),ratescc]

xxpwcf = df.dfpwcf(mat,0.,curvepwcf[2],curvepwcf[3])
xxpwlz = df.dfpwlz(mat,0.,curvepwlz[2],curvepwlz[3])     # From some simple hand checks, seems to work
xxpwtf = df.dfpwtf(mat,0.,curvepwtf[2],curvepwtf[3])
xx2pwcf = df.discFact(mat,curvepwcf)
xx2pwlz = df.discFact(mat,curvepwlz)
xx2pwtf = df.discFact(mat,curvepwtf)
#
#

#%% ------ Sample data for calculating bond PV ------

i = np.array([5,5,5,5,5,5,5])
pmt = np.array([6,6,6,6,6,6,6])
fv = np.array([100,100,100,100,100,100,100])
freq = np.array([2,2,2,2,2,2,2])
xx1PVBondYld = py.PVBondStubFromYld_vec(mat,i,pmt,fv,freq) 
xx2PVBondYld = py.PVBondStubFromCurve_vec(mat,i,pmt,fv,freq) 
# The results for this should be: (mat, i, pmt, fv, freq, -> PV)
#  0.25, 5%, 6%, 100, 2 -> 100.25455
#  0.5, 5%, 6%, 100, 2 -> 100.487805
#  0.75, 5%, 6%, 100, 2 -> 100.736373
#  0.9, 5%, 6%, 100, 2, -> 0.4yr stub, 1 half-yr period, 100.875293
#  0.99, 5%, 6%, 100, 2, -> 2 per exact, 100.963712
#  1.01, 5%, 6%, 100, 2, -> 2 per exact, 100.963712
#  1.2, 5%, 6%, 100, 2, -> .4 per stub, 2 per exact, 101.159603
xxPVBondCurvepwcf = py.PVBondStubFromCurve_vec(mat,curvepwcf,pmt,fv,freq) 
xxPVBondCurvepwlz = py.PVBondStubFromCurve_vec(mat,curvepwlz,pmt,fv,freq) 
xxPVBondCurvepwtf = py.PVBondStubFromCurve_vec(mat,curvepwtf,pmt,fv,freq) 
# For pwcf curve I think the answers should be:
# 0.25, 101.24347
# 0.5, 102.48756
# 0.75, 103.47021
# 0.9, 104.04677
# 0.99 (1), 104.4554
# 1.01 (1), 104.4554
# 1.2, 105.1334

xxPVBondCurvepwcf2 = py.PVBondApproxFromCurve_vec(mat,curvepwcf,pmt,fv,freq) 
# I think these differ from PVBondStubFromCurve because of accruied interest? But I'm not sure (10-aug-17)

#%% Test (old) price from yield and yield from price

#%% Create parameter vectors

#    eight bonds, 
# 0      2.25% of 15-nov-2025 non-call       20 half-yrs + 1 maturity = 21          
# 1      2.5% of 15-feb-2026 non-call        20 half-yrs + 1 maturity = 21
# 2      2.25% of 15-nov-2025 callable 15-nov-2020  20 half-year + 1 maturity = 21
# 3      2.25% of 15-nov-2025 callable 15-nov-2020  20 half-year + 1 maturity = 21 (duplicate)
# 4      5% of 15-nov-2025 non-call          20 half-yrs + 1 maturity = 21
# 5      5% of 15-nov-2025 callable 15-nov-2020  20 half-yrs + 1 maturity = 21
# 6      4% of 15-nov-2022                   14 half-yrs + 1 maturity = 15
# 7      3% of 15-feb-2020                   8 half-yrs + 1 maturity = 9
# 8      0% of 15-may-2016 (bill)            no coupons + 1 maturity = 1

# First version of parms does not have tax status. 2nd version has tax status. Make 4th & 5th bond partially tax-exempt
parms = [[2.25,45975,2,100.,"A/A","eomyes",0,False],[2.5,46067,2,100.,"A/A","eomyes",0,False],
         [2.25,45975,2,100.,"A/A","eomyes",44149,True],[2.25,45975,2,100.,"A/A","eomyes",44149,True],
	[5.,45975,2,100.,"A/A","eomyes",0,False],[5.,45975,2,100.,"A/A","eomyes",44149,True],
    [5.,44879,2,100.,"A/A","eomyes",0,False],[5.,43875,2,100.,"A/A","eomyes",0,False],
    [0.,42504,0,100.,"A/A","eomyes",0,False]]
parms = pd.DataFrame(parms)
# Quote 19-feb-2016
quotedate = 42418


xpricei = py.BondPriceFromYield(0.05,0.0, quotedate, parms.iloc[0,0], parms.iloc[0,1], parms.iloc[0,2], parms.iloc[0,3], parms.iloc[0,4], parms.iloc[0,5])  # Default dirtyprice=0 used
price = xpricei
xyieldi = py.BondYieldFromPrice(price, quotedate, parms.iloc[0,0], parms.iloc[0,1], parms.iloc[0,2], parms.iloc[0,3], parms.iloc[0,4], parms.iloc[0,5])
