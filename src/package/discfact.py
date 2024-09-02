# -*- coding: utf-8 -*-
"""
Created on Mon October 12 10:22:56 2015
Modified October 10 2015

@author: tcoleman
"""

import numpy as np
import numba        # To use numba JIT compiling.
                    # Put the "decorator" @numba.njit just above each function to be JIT compiled
    
# -----------------
#%% ------ Curve Functions (discfact) ------
# This version will work on dates - calculates offsets internally
# Object (list) "curve" is composed as:
#  1. crvtype - (string) name to identify the type of curve
#       pwcf - piece-wise constant forward (flat forwards, continusuously-compounded)
#       pwlz - piece-wise linear zero (cc)
#       pwtf - piece-wise twisted forward (linear between breaks but not connected at breaks) 
#              New 2/2024, with twist at end, and extending at flat fwd beyond last break
#       pwtflf - piece-wise twisted forward last flat (linear between breaks but not connected at breaks) 
#              New 2/2024, last period flat at end, and extending at flat fwd beyond last break
#       pwtfold - piece-wise twisted forward (linear between breaks but not connected at breaks)
#              Original version with last period flat fwd and extending at flat fwd beyond last break
#  2. valuedate - curve value date. Will be zero (in which case use offsets) or date (as days or date) 
#  3. For pwcf, pwlz, pwtf the rates (numeric vector) - numbers like 0.025 for 2.5% - assumed cc at 365.25 annual rate
#     For dailydf a zoo object of dfs (dates, offsets, dfs)
#  4. For pwcf, pwlz, pwtf the breaks (vector, same length as rates, of either dates or years from today, 
#     based on whether valuedate is yrs (zero) or date)

def discFact(dfdate,curve): 
    crvtype = curve[0]
    if crvtype == "pwcf":
        df = dfpwcf(dfdate,curve[1],curve[2],curve[3])
        return df
    if crvtype == "pwlz":
#        df = dfpwlz(dfdate,curve[1],curve[2],curve[3])
        df = dfpwlz(dfdate,curve[1],curve[2],curve[3])
        return df
    if (crvtype == "pwtf" or crvtype == "pwtflf"):
        df = dfpwtf(dfdate,curve[1],curve[2],curve[3],crvtype)
        return df
    if crvtype == "pwtfold":
        df = dfpwtfold(dfdate,curve[1],curve[2],curve[3])
        return df
    return "error in discFact - not valid curve type"
    
#@numba.njit
def dfpwcf(dfday,valuedate,breaks,rates):     # This is vectorized, and will work when dfday is a vector
#    x1 = np.hstack((valuedate,breaks))       # Put value date onto beginning (since breaks and rates are same length)
    x1 = dfday.size     
    if (x1 == 0):
        return(dfday)
    xday = 1.0                                # Rates are in decimal (0.05 is 5%) but allow dates (dfday, breaks) to be either
    if (valuedate > 0.0) :                    # 1) year offsets, with valuedate = 0, so don't need to divide by 365.25
        xday = 365.25                          # 2) days (Julian dates) , with valuedate being > 0, then divide by 365.25
#    xbreaks = np.tile((breaks-valuedate)/xday,(x1,1))   # Array with breaks along cols, each row for one dfday entry
    xx1 = (breaks-valuedate)/xday
    xbreaks = xx1.repeat(x1).reshape(( -1, x1))  # "reshape((-1,x1))" does the transpose of np.tile(x1,1) 
                             # "reshape((x1,-1),order='F') replicates np.tile(x1,1) but won't work with numba
    xbreaks = xbreaks.T
                                              # b1 b2 b3 b4
                                              # b1 b2 b3 b4
    x1 = breaks.size
#    xdfday = np.tile((dfday-valuedate)/xday,(x1,1))
#    xdfday = xdfday.T                         # Array with dfday along rows, each col the same
    xx1 = (dfday-valuedate)/xday
    xdfday = xx1.repeat(x1).reshape((-1, x1))  # "reshape((-1,x1))" does the transpose of np.tile(x1,1) 
                                              # dfd1 dfd1 dfd1 dfd1
                                              # dfd2 dfd2 dfd2 dfd2
    xbreaks[:,-1] = np.maximum(xdfday[:,-1],xbreaks[:,-1])  # TSC 7-feb-2024. This fixes problem of a CF extending
                                              # beyond last break. For pwcf we can simply extend the break and this
                                              # will extend the constant forward out to the end of the dfdate
    x2 = np.minimum(xdfday,xbreaks)           # Each row is a different dfday - the entry in the col is either the break for that col, 
                                              # or the dfday, whichever is smaller
                                              # b1 b2 dfd1 dfd1
                                              # b1 b2 b3 dfd2 
    x1 = np.shape(x2)
    x4 = np.zeros(x1)                       # empty holder, into which we will put x2 shifted one col to the left
    x1 = x1[1] - 1
    x4[:,1:] = x2[:,0:x1]                     # x4 is now the one-earlier break (or dfday, whichever is less)
                                              # 0 b1 b2 dfd1
                                              # 0 b1 b2 b3 
    x5 = x2 - x4                              # This gives the length of the period (either the full break, or from last break to dfday)
                                              # b1 b2-b1 dfd1-b2 0
                                              # b1 b2-b1 b3-b2 dfd2-b3 
    df = np.dot(x5,rates)                     # Now just do dot product - this gives the piece-wise constant part
    return(np.exp(-df))

#@numba.njit
def dfpwlz(dfday, valuedate, breaks, rates):   # This is vectorized, and will work when dfday is a vector
    xday = 1.0                                # Rates are in decimal (0.05 is 5%) but allow dates (dfday, breaks) to be either
    if (valuedate > 0.0) :                    # 1) year offsets, with valuedate = 0, so don't need to divide by 365.25
        xday = 365.25                          # 2) days (Julian dates) , with valuedate being > 0, then divide by 365.25
    xrates = np.hstack((rates[0:1],rates))              # Tack the first rates onto beginning, so interpolation in 1st period will be flat
                      # TSC 7-jun-24 For numba JIT compile, need to have the two elements for hstack the same
                      # Here that is both ndarray. the "rates[0:1]" gives a one-element array, while "rates[0]" gives a float
    x1 = dfday.size
    if (x1 == 0):
        return(dfday)
#    xbreaks = np.tile(np.hstack((0.,breaks)),(x1,1))          # Array with breaks along cols, each row for one dfday entry
    xx1 = (breaks-valuedate)/xday
#    xbreaks = np.insert(xx1,0,0.)
    xbreaks = np.hstack((np.array([0]),xx1))    # Array with breaks along cols, each row for one dfday entry
                                                           # tack a 0. onto beginning
    x1 = breaks.size + 1
#    xdfday = np.tile((dfday-valuedate)/xday,(x1,1))
#    xdfday = xdfday.T                         # Array with dfday along rows, each col the same
    xx1 = (dfday-valuedate)/xday
    xdfday = xx1.repeat(x1).reshape((-1, x1))  # "reshape((-1,x1))" does the transpose of np.tile(x1,1) 
    x2 = np.minimum(xdfday,xbreaks)           # Each row is a different dfday - the entry in the col is either the break for that col, or the dfday, whichever is smaller
    x3 = np.argmax(x2,axis=1)                 # Index of the next larger break
    df = xrates[x3] * (xdfday[:,1] - xbreaks[x3-1]) + xrates[x3-1] * (xbreaks[x3] - xdfday[:,1])  # xrates[x3] is zero at next break, xrates[x3-1] is zero at last breaks
    df = df / (xbreaks[x3] - xbreaks[x3-1])   # Interpolated zero rate
    df[(xbreaks[-1] - xdfday[:,-1])<0.0] = rates[-1]  # TSC 7-feb-2024. This replaces the zero for any dfdate beyond
                                              # the last break with the last zero rate. This forces flat zero beyond
                                              # last break
    df = df * xdfday[:,1]                           # multiply zero rate by maturity
    return(np.exp(-df))
    
#@numba.njit
def dfpwtf(dfday,valuedate,breaks,rates,crvtype="pwtf"):     # This is vectorized, and will work when dfday is a vector
# First, calculate the piece-wise constant part. Because the slopes are done relative to the mid-point, the linear
# term (as opposed to constant term) only contributes in the last forward section (the slope integrates out for
# all the earlier sections)
# NOTE (20-jan-24) THE LAST BREAK MUST BE BEYOND LAST BOND

    xday = 1.0                                # Rates are in decimal (0.05 is 5%) but allow dates (dfday, breaks) to be either
    if (valuedate > 0.0) :                    # 1) year offsets, with valuedate = 0, so don't need to divide by 365.25
        xday = 365.25                          # 2) days (Julian dates) , with valuedate being > 0, then divide by 365.25
    xlendfday = dfday.size  
    if xlendfday > 0:
        x1 = np.max(dfday)
    else:
        return(dfday)
    xtendFlag = False
    if (x1 > breaks[-1]) :                  # If there is a date beyond last break then add on that to breaks and duplicate last rate
        breaks = np.append(breaks,(x1+1))
        rates = np.append(rates,rates[-1])
        xtendFlag = True                      # Also set a flag because we will have to treat the last period differently if extended
    xbreaks2 = (breaks-valuedate)/xday
#    xbreaks = np.tile(xbreaks2,(xlendfday,1))  # Array with breaks along cols, each row for one dfday entry
    xbreaks = xbreaks2.repeat(xlendfday).reshape((-1,xlendfday))  # "reshape((-1,x1))" does the transpose of np.tile(x1,1) 
    xbreaks = xbreaks.T
#    xbreaks = xbreaks2.repeat(xlendfday).reshape((xlendfday,-1),order='F')  # "reshape((-1,x1))" does the transpose of np.tile(x1,1) 
                                              # b1 b2 b3 b4
                                              # b1 b2 b3 b4
    xlenbreaks = xbreaks2.size
#    xdfday = np.tile((dfday-valuedate)/xday,(xlenbreaks,1))
#    xdfday = xdfday.T                         # Array with dfday along rows, each col the same
# np.tile does not work with numba, so try this instead (don't need to do the transpose)
    xx1 = (dfday-valuedate)/xday
    xdfday = xx1.repeat(xlenbreaks).reshape((-1, xlenbreaks))
                                              # dfd1 dfd1 dfd1 dfd1
                                              # dfd2 dfd2 dfd2 dfd2
#    xbreaks[:,-1] = np.maximum(xdfday[:,-1],xbreaks[:,-1]) + 0.1  # TSC 7-feb-2024. This fixes problem of a CF extending
                                              # beyond last break. For pwcf we can simply extend the break and this
                                              # will extend the constant forward out to the end of the dfdate
    xbreaks2[-1] = np.max(xbreaks[:,-1])      # Fixes problem with slopes ??
    x2 = np.minimum(xdfday,xbreaks)           # Each row is a different dfday - the entry in the col is either the break for that col, 
                                              # or the dfday, whichever is smaller
                                              # b1 b2 dfd1 dfd1
                                              # b1 b2 b3 dfd2 
    x1 = np.shape(x2)
    x4 = np.zeros(x1)                       # empty holder, into which we will put x2 shifted one col to the left
    x1 = x1[1] - 1
    x4[:,1:] = x2[:,0:x1]                     # x4 is now the one-earlier break (or dfday, whichever is less)
                                              # 0 b1 b2 dfd1
                                              # 0 b1 b2 b3 
    x5 = x2 - x4                              # This gives the length of the period (either the full break, or from last break to dfday)
                                              # b1 b2-b1 dfd1-b2 0
                                              # b1 b2-b1 b3-b2 dfd2-b3 
    df = np.dot(x5,rates)                     # Now just do dot product - this gives the piece-wise constant part
# Now calculate the slope part. The slope part only applies to the forward period in which dfday falls. It is
#    1/2 * Si *(dfd-bi-1)^2 - 1/2 * Si *(dfd-bi-1)*(bi+bi-1)
# This is the (dfd-bi-1) and (bi+bi-1) part
    xindex = np.argmax((xbreaks - xdfday) > 0.0 , axis=1)  # Index of first fwd period where break > dfday (end-period)
                                                       # First gets Boolean which is TRUE where break > dfday, then Index of 1st occurrence
#    x7 = x5[np.arange(xlendfday),xindex]          # This is vector of (dfday - break i-1)
    x7 = np.take_along_axis(x5, xindex[:,None], axis=1)  # TSC 7-jun-24 accomplishes same as line above, but works with numba??
    x7 = x7[:,0]
    x8 = np.copy(xbreaks2)
    x8[1:xlenbreaks] = x8[1:xlenbreaks] + xbreaks2[0:(xlenbreaks-1)] # Vector of break i-1 + break 1
    x9 = x8[xindex]                               # Vector of the (bi + bi-1) that corresponds to the longest break for that dfday
                                              # Extend both breaks and rates vectors
# TSC 7-feb-2024
# This code just below works for final slope = 0
#    xrext = np.hstack((np.array((0,rates[0])),rates,rates[-2]))  # Put -1=0, 0=f1, rates , fn+1=f-2. 
                                              # Putting fn+1=f-2 forces last slope 0, and then put bn+1=(bn)+1 so not divide by zero
                                              # Putting 0=f1 and then for xbext put -1=b1, 0=0 makes first slope (f2-f1)/((b2-b0)/2)
#    xbext = np.hstack((np.array((xbreaks2[0],0)),xbreaks2,(xbreaks2[-1]+1))) # -1=b1, 0=0, breaks, bn+1=(bn)+1
# This code just below works for final slope = (fk - fk-1) / ((bk - bk-2)/2)
    if xtendFlag :                            # If yes extended, then need to make last slope zero, and 2nd-last avg of only last two rates                                              
        xrext = np.hstack((np.array((0,rates[0])),rates,np.array([rates[-2]])))  # Put -1=0, 0=f1, rates , fn+1=f-2. 
                                              # Putting fn+1=f-2 forces last slope 0, and then put bn+1=(bn)+1 so not divide by zero
                                              # Putting 0=f1 and then for xbext put -1=b1, 0=0 makes first slope (f2-f1)/((b2-b0)/2)
        xrext[-2] = rates[-2]                 # Also need to put fn=f-2 and then for xbext put bn = (bn-2) will 
                                              #     calculate slope as (fk - fk-1) / ((bk - bk-2)/2)
        xbext = np.hstack((np.array((xbreaks2[0],0)),xbreaks2,np.array([xbreaks2[-1]+1]))) # -1=b1, 0=0, breaks, bn+1=(bn)+1
        xbext[-2] = xbreaks2[-2]
    else :                                    # This is code for no extension (no dfday longer than breaks)
        xrext = np.hstack((np.array((0,rates[0])),rates,np.array([rates[-1]])))  # Put -1=0, 0=f1, rates , fn+1=f-1. 
                                              # Putting fn+1=f-1 and then for xbext put bn+1 = (bn-1) will 
                                              #     calculate slope as (fk - fk-1) / ((bk - bk-2)/2)
                                              # Putting 0=f1 and then for xbext put -1=b1, 0=0 makes first slope (f2-f1)/((b2-b0)/2)
        xbext = np.hstack((np.array((xbreaks2[0],0)),xbreaks2,np.array([xbreaks2[-2]]))) # -1=b1, 0=0, breaks, bn+1=(bn)+1
    x10 = xdfday[:,1]*xdfday[:,1] - xbext[(xindex+1)]*xbext[(xindex+1)] - x7*x9     # This is the dfd^2 - bi-1^2 - (dfd-bi-1)*(bi+bi-1) part
# Now calculate the slopes
    xslope = np.zeros(xlendfday)             # Array (vector) to hold slopes
                                              # The general expression for the slope is:
                                              #   [fk+1 - fk-1] / [(bk - bk-2)/2 + (bk+1 - bk-1)/2]
                                              # for 1st we want
                                              #   [f2 - f1] / [(b2 - b0)/2]   (indexing from 1, so for python indexing index lower by 1
                                              # for last we want 0
                                              # Tacking to the front and back of breaks and reates will get the right answers:
                                              #    breaks: (b1 0) to the front and (bn) to the end
                                              #    rates:  (0 f1) to the front and fn-1 to the end 
    xslope = xrext[(xindex+3)] - xrext[(xindex+1)]  # This is [fk+1 - fk-1] - index has to be incremented by 2 (because 2 tacked on to front)
    xslope = xslope / (((xbext[(xindex+2)]-xbext[xindex])/2.0) + ((xbext[(xindex+3)]-xbext[(xindex+1)])/2.0))
# Now combine everything
    df = df + xslope * x10 / 2.   
    return(np.exp(-df))


def dfpwtfold(dfday,valuedate,breaks,rates):     # This is vectorized, and will work when dfday is a vector
# First, calculate the piece-wise constant part. Because the slopes are done relative to the mid-point, the linear
# term (as opposed to constant term) only contributes in the last forward section (the slope integrates out for
# all the earlier sections)
# NOTE (20-jan-24) THE LAST BREAK MUST BE BEYOND LAST BOND

    xday = 1.0                                # Rates are in decimal (0.05 is 5%) but allow dates (dfday, breaks) to be either
    if (valuedate > 0.0) :                    # 1) year offsets, with valuedate = 0, so don't need to divide by 365.25
        xday = 365.25                          # 2) days (Julian dates) , with valuedate being > 0, then divide by 365.25
    xlendfday = dfday.size     
    xbreaks2 = (breaks-valuedate)/xday
#    xbreaks = np.tile(xbreaks2,(xlendfday,1))  # Array with breaks along cols, each row for one dfday entry
    xbreaks = xbreaks2.repeat(xlendfday).reshape((-1,xlendfday))  # "reshape((-1,x1))" does the transpose of np.tile(x1,1) 
    xbreaks = xbreaks.T
#    xbreaks = xbreaks2.repeat(xlendfday).reshape((xlendfday,-1),order='F')  # "reshape((-1,x1))" does the transpose of np.tile(x1,1) 
                                              # b1 b2 b3 b4
                                              # b1 b2 b3 b4
    xlenbreaks = breaks.size
#    xdfday = np.tile((dfday-valuedate)/xday,(xlenbreaks,1))
#    xdfday = xdfday.T                         # Array with dfday along rows, each col the same
    xx1 = (dfday-valuedate)/xday
    xdfday = xx1.repeat(xlenbreaks).reshape((-1, xlenbreaks))  # "reshape((-1,x1))" does the transpose of np.tile(x1,1) 
                                              # dfd1 dfd1 dfd1 dfd1
                                              # dfd2 dfd2 dfd2 dfd2
    xbreaks[:,-1] = np.maximum(xdfday[:,-1],xbreaks[:,-1]) + 0.1  # TSC 7-feb-2024. This fixes problem of a CF extending
                                              # beyond last break. For pwcf we can simply extend the break and this
                                              # will extend the constant forward out to the end of the dfdate
    xbreaks2[-1] = np.max(xbreaks[:,-1])      # Fixes problem with slopes ??
    x2 = np.minimum(xdfday,xbreaks)           # Each row is a different dfday - the entry in the col is either the break for that col, 
                                              # or the dfday, whichever is smaller
                                              # b1 b2 dfd1 dfd1
                                              # b1 b2 b3 dfd2 
    x1 = np.shape(x2)
    x4 = np.zeros(x1)                       # empty holder, into which we will put x2 shifted one col to the left
    x1 = x1[1] - 1
    x4[:,1:] = x2[:,0:x1]                     # x4 is now the one-earlier break (or dfday, whichever is less)
                                              # 0 b1 b2 dfd1
                                              # 0 b1 b2 b3 
    x5 = x2 - x4                              # This gives the length of the period (either the full break, or from last break to dfday)
                                              # b1 b2-b1 dfd1-b2 0
                                              # b1 b2-b1 b3-b2 dfd2-b3 
    df = np.dot(x5,rates)                     # Now just do dot product - this gives the piece-wise constant part
# Now calculate the slope part. The slope part only applies to the forward period in which dfday falls. It is
#    1/2 * Si *(dfd-bi-1)^2 - 1/2 * Si *(dfd-bi-1)*(bi+bi-1)
# This is the (dfd-bi-1) and (bi+bi-1) part
    xindex = np.argmax((xbreaks - xdfday) > 0.0 , axis=1)  # Index of first fwd period where break > dfday (end-period)
                                                       # First gets Boolean which is TRUE where break > dfday, then Index of 1st occurrence
    x7 = x5[np.arange(xlendfday),xindex]          # This is vector of (dfday - break i-1)
    x8 = np.copy(xbreaks2)
    x8[1:xlenbreaks] = x8[1:xlenbreaks] + xbreaks2[0:(xlenbreaks-1)] # Vector of break i-1 + break 1
    x9 = x8[xindex]                               # Vector of the (bi + bi-1) that corresponds to the longest break for that dfday
                                              # Extend both breaks and rates vectors
    xrext = np.hstack((np.array((0,rates[0])),rates,rates[-2]))  # Put -1=0, 0=f1, rates , fn+1=f-2
    xbext = np.hstack((np.array((xbreaks2[0],0)),xbreaks2,(xbreaks2[-1]+1))) # -1=b1, 0=0, breaks, bn+1=(bn)+1
    x10 = xdfday[:,1]*xdfday[:,1] - xbext[(xindex+1)]*xbext[(xindex+1)] - x7*x9     # This is the dfd^2 - bi-1^2 - (dfd-bi-1)*(bi+bi-1) part
# Now calculate the slopes
    xslope = np.zeros(xlendfday)             # Array (vector) to hold slopes
                                              # The general expression for the slope is:
                                              #   [fk+1 - fk-1] / [(bk - bk-2)/2 + (bk+1 - bk-1)/2]
                                              # for 1st we want
                                              #   [f2 - f1] / [(b2 - b0)/2]   (indexing from 1, so for python indexing index lower by 1
                                              # for last we want 0
                                              # Tacking to the front and back of breaks and reates will get the right answers:
                                              #    breaks: (b1 0) to the front and (bn) to the end
                                              #    rates:  (0 f1) to the front and fn-1 to the end 
    xslope = xrext[(xindex+3)] - xrext[(xindex+1)]  # This is [fk+1 - fk-1] - index has to be incremented by 2 (because 2 tacked on to front)
    xslope = xslope / (((xbext[(xindex+2)]-xbext[xindex])/2.0) + ((xbext[(xindex+3)]-xbext[(xindex+1)])/2.0))
# Now combine everything
    df = df + xslope * x10 / 2.   
    return(np.exp(-df))


