# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 21:57:04 2016

@author: tcoleman
"""

import sys
import numpy as np
import importlib as imp

sys.path.append('../src/package')

#%%

import DateFunctions_1 as dates

imp.reload(dates)

#%%
# Test the following:
# - Convert single date-number 19960215 to Julian (35109)
# - Convert list of date-numbers list[19960215, 19960315] to Julian (35109, 35138)
# - Convert list of date-lists [[1996, 2, 15],[1996, 3, 15]] to Julian (35109, 35138)
# - Convert np.array of date-numbers [19960215, 19960315] to Julian (35109, 35138)
# - Check that invalid date 29-feb-2018 is rejected as invalde (raise ValueError)

# The result should print "True" if all four pass

#testarray = np.array([False,False,False,False])
testarray = []
def test1():
# single integer
    c0 = 19960215 
    j0 = dates.YMDtoJulian(c0)
    if (j0 == 35109.):
        testarray.append(True)
    else:
        testarray.append(False)
# list of integers
    c1 = [19960215,19960315]   
    j1 = dates.YMDtoJulian(c1)
    if all(j1 == np.array([35109.,35138.])):
        testarray.append(True)
    else:
        testarray.append(False)
# array of integers
    c1 = np.array(c1)             
    j2 = dates.YMDtoJulian(c1)
    if all(j2 == np.array([35109.,35138.])):
        testarray.append(True)
    else:
        testarray.append(False)
# single tuple list
    c2 = [1996, 2, 15]         
    j2 = dates.YMDtoJulian(c2)
    if (j2 == 35109.):
        testarray.append(True)
    else:
        testarray.append(False)
# single tuple array
    j2 = dates.YMDtoJulian(np.array(c2))
    if (j2 == 35109.):
        testarray.append(True)
    else:
        testarray.append(False)
# list of tuples
    c2 = [[1996, 2, 15],[1996, 3, 15]]         
    j2 = dates.YMDtoJulian(c2)
    if all(j2 == np.array([35109.,35138.])):
        testarray.append(True)
    else:
        testarray.append(False)
# array of tuples
    c2 = np.array(c2)                        
    j2 = dates.YMDtoJulian(c2)
    if all(j2 == np.array([35109.,35138.])):
        testarray.append(True)
    else:
        testarray.append(False)
# Need to do this one a little different because I do a "try" and only write the True if it is caught
    testarray.append(False)
    try:
        c3 = [20180228,20180229]    # Invalid ate
        j3 = dates.YMDtoJulian(c3)
    except ValueError:
        print('Caught the bad date (29-feb-2018)')
        testarray[-1] = True
test1()
print('Testing conversion of YMD to Julian. True means all tests pass:',all(testarray))
#
#x0 = [1996,2,15]
#
#if (type(x0) == int):
#    x0 = [x0]
#
#if (type(x0) == list):
#    yout = np.array(x0)
#    if (yout.ndim == 1 and yout[0] > 20000) : # The checking that first element < 20,000 handles the case of single date like [1996,2,15]
#        yyy = np.floor(yout/10000.)
#        ymm = np.floor((yout - yyy*10000.)/100.)
#        ydd = (yout - yyy*10000. - ymm*100.)
#        yout = np.array([yyy,ymm,ydd])
#    else :
#        yout = yout.T

   
#%% -------------- Sample date for testing date function ----------------
# Tests the eom functionality by taking differences between two verctors of dates, and
# displaying differences in YRS, MTHS, DAYS
# Julian dates are days from 31-dec-1899 
jdate1 = np.array((35109,35109,35109,35109,35109,35109,36584,36218,36279,36279,36279,12053,35383))  #
#  15-feb-1996  35,109 0
#  15-feb-1996  35,109 1
#  15-feb-1996  35,109 2
#  15-feb-1996  35,109 3
#  15-feb-1996  35,109 4
#  15-feb-1996  35,109 5
#  29-feb-2000  36,584 6
#  28-feb-1999  36,218 7
#  30-apr-1999  36,279 8
#  30-apr-1999  36,279 9
#  30-apr-1999  36,279 10
#  31-dec-1932  12,053 11
#  15-nov-1996  35,383 12


jdate2 = np.array((35109,35138,35476,35504,36572,36601,36768,36402,36463,36462,36461,16802,36529))  #

#                         Diff    eom=y eom=n
#                          y m d   d     d
#  15-feb-1996  35,109 0   0 0 0
#  15-mar-1996  35,138 1   0 1 0 
#  16-feb-1997  35,476 2   1 0 1
#  16-mar-1997  35,504 3   1 1 1
#  17-feb-2000  36,572 4   4 0 2
#  17-mar-2000  36,601 5   4 1 2
#  31-aug-2000  36,768 6   0 6     0     2
#  31-aug-1999  36,402 7   0 6     0     3
#  31-oct-1999  36,463 8   0 6     0     1
#  30-oct-1999  36,462 9   0 6     0     0
#  29-oct-1999  36,461 10  0 6    -1    -1
#  1-jan-1946   16,802 11  14 -11 -30
#  5-jan-2000   36,529 12  4 -10 -10

ymd1 = dates.JuliantoYMD(jdate1)
ymd2 = dates.JuliantoYMD(jdate2)

# This calculates date difference (y m d) WITHOUT eom convention. It should match the "eom=n" above
diff = ymd2 - ymd1
# This calculates date differene WITH eom convention. It should match the "eom=y" above
eomdiff = dates.DateDiff(jdate1,jdate2)

mthdiff = 12.*eomdiff[0] + eomdiff[1] + eomdiff[2]/30.

np.remainder(mthdiff,6)


#%% ------ Testing DateDiff -----------
# Want to go from eom to 1st of the next month
#                                 yr  mth  dy
# 0 31-dec-1996 (-> 1-jan-2000)   4  -11  -30   -30day of 31-day 36.032mths
# 1 30-jun-1996 (-> 1-jan-2000)   4   -5  -29   -29days of 30    42.033mths
# 2 31-jan-1996 (-> 1-feb-2000)   4   1   -30  4yrs+1day         48.032mths
# 3 31-jul-1996 (-> 1-feb-2000)   4  -5   -30                    42.032mths
# 4 28-feb-1996 (-> 1-mar-2000)   4   1   -27  4yrs+1day         48.069mths
# 5 31-aug-1996 (-> 1-mar-2000)   4   -5  -30  3yrs, 6mths, 1day 42.032
# 6 16-jan-1996 (->15-feb-2000)   4  1   -1   4yrs, 0mths, just under a month     48.968mths
# 7 14-jan-1996 (->15-feb-2000)   4  1    1    4yrs, 1mth, 1 day  49.0322
date1 = [19961231,19960630,19960131,19960731,19960228,19960831,19960116,19960114]
jdate1 = dates.YMDtoJulian(date1)
ymd1 = dates.JuliantoYMD(jdate1)

date2 = [20000101,20000101,20000201,20000201,20000301,20000301,20000215,20000215]
jdate2 = dates.YMDtoJulian(date2)
ymd2 = dates.JuliantoYMD(jdate2)

diff = ymd2 - ymd1
eomdiff = dates.DateDiff(jdate1,jdate2)

# The following seems to give the right number of months

leap = dates.IsALeapYear(ymd1[0])
leap = leap.astype(int)

iin = np.array( ((31,28,31,30,31,30,31,31,30,31,30,31),
            (31,29,31,30,31,30,31,31,30,31,30,31)) )

daysinmth = iin[leap,(ymd1[1].astype(int)-1)]

mthdiff = 12.*eomdiff[0] + eomdiff[1] + eomdiff[2] / daysinmth


#%% Testing CalAdd

JulianDate = np.array((40250,40250,40250,40250,40250,40250,40250,40250))
xdates = np.array((40000,40100,40000,40100,40200,40389,39871,39505))
xymd1 = dates.JuliantoYMD(xdates)
nyear = np.array((0,0,0,1,2,1,1,0))
nmonth = np.array((0,0,0,6,6,6,6,6))
nday = np.array((0,5,100,0,0,0,0,0))
x1 = np.size(nyear)
x2 = np.size(nmonth)
x3 = np.size(nday)
y1 = max([x1,x2,x3])

xymds = dates.JuliantoYMD(JulianDate)
xleap = dates.IsALeapYear(xymds[0])
dd = xymds[2]
mm = xymds[1]
xleap = dates.IsALeapYear(xymd1[0])
dd = xymd1[2]
mm = xymd1[1]

xdd = dd - nday

#%% Sample date for testing calladd
# Julian dates are days from 31-dec-1899 
#  15-feb-1996  35,109 0
#  15-feb-1996  35,109 1
#  29-feb-2000  36,584 2
#  29-feb-2000  36,584 3
#  27-feb-1999  36,217 4
#  28-feb-1999  36,218 5
#  28-feb-1999  36,218 6
#  28-feb-1999  36,218 7
#  30-apr-1999  36,279 8
#  30-apr-1999  36,279 9
#  30-apr-1999  36,279 10
#  31-oct-1999  36,463 11
#  30-oct-1999  36,462 12
jdate1 = np.array((35109,
                   35109,
                   36584,
                   36584,
                   36217,
                   36218,
                   36218,
                   36218,
                   36279,
                   36279,
                   36279,
                   36463,
                   36462))  #

nday = np.array((23,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0))
nmonth = np.array((0,
                   6,
                   6,
                   6,
                   6,
                   6,
                   12,
                   0,
                   6,
                   12,
                   18,
                   6,
                   6))

nyear = np.array((0,
                  1,
                  0,
                  1,
                  0,
                  0,
                  0,
                  1,
                  0,
                  0,
                  0,
                  0,
                  0))

xx1 = dates.CalAdd(jdate1,"add",nyear=nyear,nmonth=nmonth,nday=nday)

# Results should be  
#  9-mar-1996   35,132 0
#  15-aug-1997  35,656 1
#  31-aug-2000  36,768 2
#  31-aug-2001  37,133 3
#  27-aug-1999  36,398 4
#  31-aug-1999  36,402 5
#  29-feb-2000  36,584 6
#  29-feb-2000  36,584 7
#  31-oct-1999  36,463 8
#  30-apr-2000  36,645 9
#  31-oct-2000  36,829 10
#  30-apr-2000  36,645 11
#  30-apr-2000  36,645 12

xx2 = dates.JuliantoYMD(xx1)
#%% Now test CalAdd and should fail
nday = np.array((23,
                   23,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0,
                   0))

dates.CalAdd(jdate1,"add",nyear=nyear,nmonth=nmonth,nday=nday)
# Now should fail because adding days & months
