# Functions to PV instruments, intended for time-series (say monthly observations on swap rates)

#  Tom Coleman, 10/15
#  Converted from the r code, from C code, derived from earlier Fortran code

# 12-oct-15 
#    still r code
#    maybe use python module "dateutil" which extends "datetime" - rather than coding
#    everything myself? But does it do different bases (A/A, A/360, 30-360?) Do I care?
#    Probably - care about eom convention - go from 28-feb to 31-aug to 29-feb to 31-aug to 28-feb

# This function will extend as.Date to work putting in just a double - but I'm not sure
# how to properly override the as.Date function in base.


import numpy as np

#%%
#///////////////////////////////////////////////////////////////////////////
# IsALeapYear- Returns True if specified year is a leap year else False
# If the year is divisible by 4 and 1900 < yyyy < 2099 then it is a leap year.
#////////////////////////////////////////////////////////////////////////////

def IsALeapYear(year):
#    xyear = np.array(year)        # When year is a Pandas series the statment below does not work
    leap = np.all((np.remainder(year,4) == 0, year != 1900, year > 0),axis=0)  # Python doesn't seem to have
                                 # a logical and (&&) so use np.all which checks that all conditions 
                                 # hold. This works for arrays. (NB - this correctly excludes 1900 as leap year)
    return(leap)


#%%
#////////////////////////////////////////////////////////////////////////////
# JuliantoYMD-  Extracts the yyyy, mm, dd from a Julian
# JuliantoYMD() should be replaced by @nalyst function
#  t_dmdy(DATE date, int W *month, int W *day, int W *year, 
#      int W *status)
#////////////////////////////////////////////////////////////////////////////

# OLD - This counts from 1-jan-1970 to be consistent with R dates
# Current - Counts from 31-dec-1899 (to be consistent with old Excel from 1-mar-1900 forward)

def JuliantoYMD(JulianDate): 

# iprior contains the number of days preceding each month  */
   iprior =  np.array(( (0,31,59,90,120,151,181,212,243,273,304,334,365),
      (0,31,60,91,121,152,182,213,244,274,305,335,366) )) 

# JulianDate is relative to 1-jan-1900 (Excel thinks it is 31-dec-1899 because Excel
# thinks 1900 is a leap year when it was not. Only .

   xy = np.floor(JulianDate * 100. / 36525)
   y = xy + 1900.
   y = y.astype(int)

#     /* If a leap Year */
   leap = IsALeapYear(y)  
   leap = leap.astype(int)           
   d = np.floor(JulianDate - 365. * xy - np.floor(xy/4.)) + leap
   x1 = np.tile(d,(13,1))
   x1 = x1.T
   x2 = np.minimum(iprior[leap],x1)
   m = x2.argmax(1)
   d = d - iprior[leap,m-1]
   ret = np.array((y,m,d))
   

   return(ret)
 

#%%
#////////////////////////////////////////////////////////////////////////////
# JuliantoYMDint-  Extracts the yyyy, mm, dd from a Julian
# JuliantoYMD() should be replaced by @nalyst function
#  t_dmdy(DATE date, int W *month, int W *day, int W *year, 
#      int W *status)
#////////////////////////////////////////////////////////////////////////////

# OLD - This counts from 1-jan-1970 to be consistent with R dates
# Current - Counts from 31-dec-1899 (to be consistent with old Excel from 1-mar-1900 forward)

def JuliantoYMDint(JulianDate): 

# iprior contains the number of days preceding each month  */
   iprior =  np.array(( (0,31,59,90,120,151,181,212,243,273,304,334,365),
      (0,31,60,91,121,152,182,213,244,274,305,335,366) )) 

# JulianDate is relative to 1-jan-1900 (Excel thinks it is 31-dec-1899 because Excel
# thinks 1900 is a leap year when it was not. Only .

   xy = np.floor(JulianDate * 100. / 36525)
   y = xy + 1900.

#     /* If a leap Year */
   leap = IsALeapYear(y)  
   leap = leap.astype(int)           
   d = np.floor(JulianDate - 365. * xy - np.floor(xy/4.)) + leap
   x1 = np.tile(d,(13,1))
   x1 = x1.T
   x2 = np.minimum(iprior[leap],x1)
   m = x2.argmax(1)
   d = d - iprior[leap,m-1]
#   ret = np.array((y,m,d))
   ret = 10000*y + 100*m + d
   

   return(ret)
 

#%%
#///////////////////////////////////////////////////////////////////////////
# YMDtoJulian-  Subroutine to convert Calendar Dates to Linear (Julian) Dates
# Return 1 is an error return
#///////////////////////////////////////////////////////////////////////////

def YMDtoJulian(ymdarray):
  "Convert YMD to Julian: dates may be int, list, or array: 19960215, [1996, 2, 15], np.array([1996,2,15]). Single or vector."

# Should work with multiple input formats (as long as all the same in one call):
# 1) Integers like 19960215
# 2) Lists like [1996, 02, 15]
# 3) numpy array like (1996, 2, 15)
# Should work for both single dates and multiple (vector) of dates


# iin contains the number of days in the months */
  iin = np.array( ((31,28,31,30,31,30,31,31,30,31,30,31),
            (31,29,31,30,31,30,31,31,30,31,30,31)) )
# iprior contains the number of days preceding each month  */
  iprior =  np.array(( (0,31,59,90,120,151,181,212,243,273,304,334,365),
                       (0,31,60,91,121,152,182,213,244,274,305,335,366) )) 

# Old - This counts from 1-jan-1970 to be consistent with R dates
# Current - Counts from 31-dec-1899 (to be consistent with old Excel from 1-mar-1900 forward)


  if (type(ymdarray) == int):
      ymdarray = [ymdarray]

  if (type(ymdarray) == list):     # Convert to np.array if list
      ymdarray = np.array(ymdarray)
      if (ymdarray.ndim > 1):      # If this comes in as a list like [[1996,2,15],[1996, 3, 15]] then we need to transpose
         ymdarray = ymdarray.T     # TSC 13-jan-24 The second if statement I think should have been under the first, not on its own

  if (ymdarray.ndim == 1):
      if (ymdarray[0] > 20000) : # The checking that first element > 20,000 handles the case of single date like [1996,2,15]
          yyy = np.floor(ymdarray/10000.)
          ymm = np.floor((ymdarray - yyy*10000.)/100.)
          ydd = (ymdarray - yyy*10000. - ymm*100.)
          ymdarray = np.array([yyy,ymm,ydd])
      else:              # If we get here, must be like [1996, 02, 15] and need to make another dimension
          ymdarray = np.reshape(ymdarray,(np.size(ymdarray),1))   # This esentiall transposes (for 1-dim)

      

  yyyy = np.where( (ymdarray[0] >= 1900),ymdarray[0]-1900, ymdarray[0])
#    // Year should be fall within range [1900,2099]
#    if ((yyyy > 199) || (yyyy < 0)) {
#        TmgError (ERR_WARNING, ER_DATE_RANGE,
#            "%s: year %d should fall within range [1900,2099]\n",
#            "YMDtoJulian", yyyy, NULL)
#        return ((DATE) ER_DATE_RANGE)
#    }
#    // Check for sensible number of Months
#    */
#    if ((MM > 12) || (MM < 0))  {
#        TmgError (ERR_WARNING, ER_DATE_NUM,
#            "%s: bad month number %d\n",
#            "YMDtoJulian", MM, NULL)
#        return ((DATE) ER_DATE_NUM)
#    }
#    // Adjust for leap Years
  leap = IsALeapYear(yyyy+1900)
  leap = leap.astype(int)

  if (any(ymdarray[2,:] < 1) or any(ymdarray[2,:] > iin[leap, ymdarray[1,:].astype(int)-1])) :
    raise ValueError("error in YMDtoJulian - date is not valid",ymdarray)




#    // Check for the correct number of Days in Month
#    if ((DD < 1) || (DD > iin[leap*12+MM-1]))   {
#        TmgError (ERR_WARNING, ER_DATE_NONEXISTENT,
#            "%s: non-existent day %d/%d/%d\n",
#            "YMDtoJulian", yyyy, MM, DD, NULL)
#        return ((DATE) ER_DATE_NONEXISTENT)
#    }

#    // Calendar Date is Valid, Convert to Julian

  JulianDate = 365. * yyyy + np.floor(np.maximum(0,yyyy-1.)/4.) + iprior[leap,(ymdarray[1].astype(int)-1)] + ymdarray[2] 
    
  return (np.array(JulianDate))

#%%
#////////////////////////////////////////////////////////////////////////////
# DateDiff - Difference between two (Julian) dates expressed as (yr, mth, day)
# Takes two Julian Dates (or vectors or Julian Dates) and expresses the 
# difference in years, months, days, BUT correctly handles month-end. 
#
# This would be simple except for the month-end problem:
#   Without month-end:
#      Convert each date to MDY
#      Take difference
# But when both dates are month-end, (and month-end flag on) we want day-diff = zero. 
#   30-apr to 31-oct should be 0 days (+ 6 months)
#   29-apr to 31-oct should be 2 days (+ 6 months)
#   31-oct to 30-apr should be 0 days (+ 6 months)

# Use this function to see whether we are on exact half-year

def DateDiff(jdate1,jdate2,eom="eomyes"):

# iin contains the number of days in the months */
    iin = np.array( ((31,28,31,30,31,30,31,31,30,31,30,31),
            (31,29,31,30,31,30,31,31,30,31,30,31)) )

    ymd1 = JuliantoYMD(jdate1)
    ymd2 = JuliantoYMD(jdate2)


    flag = eom=="eomyes" or eom=="yes" or eom=="YES"

    diff = ymd2 - ymd1

    if (not(flag)) :
      return(diff)
    else :
#      ymd1 = ymd1.T     # Transpose so we can access the dates using first dimension for both single and 
#      ymd2 = ymd2.T     # vector dates
      dd1 = ymd1[2]     # day of month
      dd2 = ymd2[2]
      leapy1 = IsALeapYear(ymd1[0])      # Check for leap year
      leapy2 = IsALeapYear(ymd2[0])
      leapy1 = leapy1.astype(int)
      leapy2 = leapy2.astype(int)
      dd1 = np.tile(dd1,(12,1))          # This will be days, replicated 12 times (to match iin above)
      dd1 = dd1.T                        # Transpose to get months along 2nd dim
      dd1 = dd1 == iin[leapy1]           # Check if dates match any month-end (iin[leapy1] is re-cast)
      dd1 = np.any(dd1,axis=1)           # Do "or" along appropriate dimension
      dd2 = np.tile(dd2,(12,1))
      dd2 = dd2.T                        # Transpose to get months along 2nd dim
      dd2 = dd2 == iin[leapy1]           # Check if dates match any month-end (iin[leapy1] is re-cast)
      dd2 = np.any(dd2,axis=1)           # Do "or" along appropriate dimension
      eomtoeom = np.logical_not(np.logical_and(dd1,dd2))
#      eomtoeom = np.logical_not(dd1,dd2) # This will be TRUE for not eom-to-eom. When converted to integer
                                         # then we can use this to multiply by the change in days - 1 will 
                                         # leave change, 0 will zero it out
      eomtoeom = eomtoeom.astype(int)
#      diff = diff.T                        # Need to transpose so works with single and vector
      diff[2] = diff[2] * eomtoeom
      return(diff)


#%%
#////////////////////////////////////////////////////////////////////////////
# CalAdd-  Takes a Date in and adds or subtracts NYear years, NMonth months,
# and NDay days from the date and returns the new date in Julian format.
#
# If add = "sub", calendar subtraction is performed 
# else if add = "add", calendar addition is performed
#
# The operations are done in the order of Day > Month > Year.
# If the Days in the end of the month before operations on month all 
# subsequent days will be set to end of the month after the month operation.
# (N.B. - This the appropriate convention for U.S. Treasuries, 
# but not for e.g., U.K. Gilts)
#
# If EOM = "eomyes", end-of-month conventions are used, anything else non-eom
#////////////////////////////////////////////////////////////////////////////

# 2/2016, T Coleman: vectorized to handle a vector of dates. There can be a few cases
#    Arguments must be comformable. This means:
#    1) The "addands" (nyear, nmonth, nday) must either all be same length,
#       but one or two may be zero (i.e. nmonth and nday may be vectors 4 long,
#       and nyear zero)
#    2) The start date (JulianDate) must either be scalar (single date) or
#       multiple dates, vector same length as addands
#       - Single date: all addands get added to that single date
#       - Multiple dates: addands 1 added to date 1, addands 2 to date 2, ...


def CalAdd(JulianDate,add="add",nyear=0,nmonth=0,nday=0, eom="eomyes"):

  iin = np.array( ((31,28,31,30,31,30,31,31,30,31,30,31),
          (31,29,31,30,31,30,31,31,30,31,30,31)) )


  xeomflag = eom == "eomyes"       # Set True when eomyes and we are at end of month
  xnoerror = False          # Set by checking on conformability of inputs
  
#   /* Set Cal_Add to return an error value by default */
#   At the moment this will not fail gracefully
  xreturn = "Error in CallAdd" 
  x1 = np.size(nyear)
  x2 = np.size(nmonth)
  x3 = np.size(nday)
  x4 = np.size(JulianDate)
  x5 = np.size(eom)
  y1 = max([x1,x2,x3,x4,x5])
  xdayflag = False                 # Used to check that we don't add both days AND (nmonth, nyear)
  xmyflag = False
  if np.array_equal(nyear,0) :     # Now, for addands that are zero (not used) we replace
    x1 = y1                        # length by max, and make vector of zeros
    nyear = np.zeros((y1),float)
  else :
    xmyflag = nyear > 0            # check which elements of nyear > 0
  if np.array_equal(nmonth,0) :
    x2 = y1
    nmonth = np.zeros((y1),float)
  else :
    xmyflag = xmyflag | (nmonth > 0)
  if np.array_equal(nday,0) :
    x3 = y1
    nday = np.zeros((y1),float)
  else:
    xdayflag = nday > 0            # We need to check each element (some additions may have nday > 0)
  if x4 == 1 :   # For startdate (JulianDate) check on size (will return 1 if scalar or
    x4 = y1                       # length 1 array) and if length 1 then repeat
    JulianDate = np.array(JulianDate)
    JulianDate = JulianDate.repeat(y1)
  if x5 == 1 :   # For startdate (JulianDate) check on size (will return 1 if scalar or
    x5 = y1                       # length 1 array) and if length 1 then repeat
    eom = np.array(eom)
    eom = eom.repeat(y1)
  xnoerror = not(np.any(xdayflag & xmyflag))     # This checks that we do not have any additions for which BOTH nday>0 AND (nmonth or nyear > 0)
                                            # We want to allow EITHER adding days only (easier to do just adding & subtracting Julian Dates
                                            # but allow here for conevenience) OR adding months & years. But not BOTH days & (month/year).
                                            # (For example when add 28 days to 31-jan-2000 we want to go to 28-feb-2000, not to eom 29-feb-2000. 
                                            # But when adding 1 month we want to go 31-jan-2000 -> 29-feb-2000)
                                            # Because all arguments (except "add" or "subtract") can be vectors, need to go through
                                            # all this checking on size, expanding scalar 0 to vector 0, etc. 
  xnoerror = (x1 == y1) and (x2 == y1) and (x3 == y1) and (x4 == y1) and (x5 == y1) and xnoerror  # Checks that they are all same #%% Checking some of the internal code for CalAdd

  if not(xnoerror) :      # If error the exit badly
    return(xreturn)


#    /* Convert Julian into component dates. */
  xymd = JuliantoYMD(JulianDate)
      
#   /* If a leap Year */
  xleap = IsALeapYear(xymd[0])

  mm = xymd[1]
  yy = xymd[0]
  dd = xymd[2]
  xeomflag = ((dd >= iin[xleap.astype(int),(mm.astype(int)-1)]) & xeomflag)  # TSC 3/16 - need to have this here to check whether start
                                                                                   # date is eom. This wasn't in original C code - I think
                                                                                   # a mistake, inherited from original FORTRAN code
                                                                                   # But also, impose check that EITHER adding days (nmonth & nyear=0)
                                                                                   # OR adding years & months (nday=0). It does not really make
                                                                                   # sense to 
#    /* Julian Date Subtraction */
  if (add=="sub") :
    dd = dd - nday
    bdd = dd <= 0                     # We will need the comparisons both for the while, and for the add/subtract
    while any(bdd) :               # Need to check if that has taken us into "negative days" and then adjust
      mm = mm - 1*bdd              # Decrement month by 1, but only for elements with negative days
      bmm = mm <= 0
      if any(bmm) :
          mm = mm + 12*bmm
          yy = yy - 1*bmm 
      xleap = IsALeapYear(yy)
      dd = dd + bdd*iin[xleap.astype(int),(mm.astype(int)-1)]
      bdd = dd <= 0              # Check if still negative days

#        /* Save the day If it is the end of the month we will 
#        // always make the new date also end of the month.*/
# It belongs here (rather than prior to nday addition) so that to eom-to-eom only for adding months & years. (For example
# when add 28 days to 31-jan-2000 we want to go to 28-feb-2000, not to eom 29-feb-2000. But when adding 1 month we want to go 31-jan-2000 -> 29-feb-2000)
# Because there is now a check to disallow nday > 0 & (nmonth >0, nyear >0) it is ok to have it here. 
    xeomflag = ((dd >= iin[xleap.astype(int),(mm.astype(int)-1)]) & (eom == "eomyes"))
    mm = mm - nmonth
    bmm = mm <= 0         # Check if gone past a year boundary for any of our subtracts. If yes, decrement a year
    while any(bmm):
      yy = yy - 1*bmm      # This only decrements for elements where mm <= 0
      mm = mm + 12*bmm
      bmm = mm <= 0
    
    yy = yy - nyear
    xleap = IsALeapYear(yy)
        
#       /* Adjust for the end of the month */
    dd = xeomflag * iin[xleap.astype(int),(mm.astype(int)-1)] + (1-xeomflag)* np.minimum(dd,iin[xleap.astype(int),(mm.astype(int)-1)])
                   #/* end subtraction */

#   /* Do Addition of Dates */
  else:
    dd = dd + nday
    xdd = iin[xleap.astype(int),(mm.astype(int)-1)]
    bdd = dd > xdd                     # We will need the comparisons both for the while, and for the add/subtract
  # I think don't need this here        xleap = IsALeapYear(yy)

    while any(bdd) : 
      xleap = IsALeapYear(yy)
      dd = dd - bdd*iin[xleap.astype(int),(mm.astype(int)-1)]
      mm = mm + 1*bdd
      bmm = mm > 12           # Check if we've gone past year-end for any of our adds. If yes, add a year
      if any(bmm) :           
        mm = mm - 12*bmm      # This only decrements for elements where mm > 12
        yy = yy + 1*bmm
      bdd = dd > xdd          # Check if still past month-end. 
  
  #        /* Save the day If it is the end of the month we will 
  #        // always make the new date also end of the month.*/
    xeomflag = ((dd >= iin[xleap.astype(int),(mm.astype(int)-1)]) & xeomflag)
    mm = mm + nmonth
        
    bmm = mm > 12
    while any(bmm) :
      mm = mm - 12*bmm
      yy = yy + 1*bmm
      bmm = mm > 12    
        
    yy = yy + nyear
    xleap = IsALeapYear(yy)
        
  #    Adjust for the end of the month - if originally eom (and input eom="eomyes") then set to end-of-month
  #    If not, then minimum of day or end-of-month (e.g. add 1 month to 30-jan-1999 and go to 28-feb-99, 1 month to 
  #    30-jan-2000 go to 29-feb-99)
    dd = xeomflag * iin[xleap.astype(int),(mm.astype(int)-1)] + (1-xeomflag)* np.minimum(dd,iin[xleap.astype(int),(mm.astype(int)-1)])
         #/* end addition */
  
  Cal_Add = YMDtoJulian(np.array([yy,mm,dd]))

  return(Cal_Add)                     
#/* CalAdd */

# #////////////////////////////////////////////////////////////////////////////
# # DaysBetween- Returns the number of days between two julian dates 
# #////////////////////////////////////////////////////////////////////////////

# DaysBetween = function(JulianStart, JulianEnd, Basis="DC_30_360")
# {
# #    /* Initialize days */
#     days = JulianEnd - JulianStart + 1

#     switch(Basis,
#         DC_30_360 = {                    #/* 30/360 (ISDA Convention) */
#             ymds = JuliantoYMD(JulianStart)
#             ymde = JuliantoYMD(JulianEnd)
# #        /* Tom Coleman's Methodology */
#             z = ymds[3]
#             if (ymds[3] == 31) z = 30
#             days1 = 360 * ymds[1] + 30 * ymds[2] + z
#             if ((ymde[3] == 31) && (z == 30)) z = 30
#             if ((ymde[3] == 31) && (ymds[3] < 30)) z = ymde[3]
#             if (ymde[3] < 31) z = ymde[3]
#             days2 = 360 * ymde[1] + 30 * ymde[2] + z
#             days2 - days1
#         },
#         DC_30E_360 = {          #/* 30E/360 */
#             ymds = JuliantoYMD(JulianStart)
#             ymde = JuliantoYMD(JulianEnd)
#             z = ymds[3]
#             if (ymds[3] == 31) z = 30
#             days1 = 360 * ymds[1] + 30 * ymds[2] + z
#             if (ymde[3] == 31) z = 30
#             if (ymde[3] != 31) z = ymde[3]
#             if ((ymde[2] == 2) && (ymde[3] == 28 + IsALeapYear(ymde[1]))) z = 30
#             days2 = 360 * ymde[1] + 30 * ymde[2] + z            
#             days2 - days1
#         },
#         Actual = JulianEnd - JulianStart,
#         ACTUAL = JulianEnd - JulianStart,
#     # This is all others (actual days)
#         JulianEnd - JulianStart )
# } #/* DaysBetween */


# #////////////////////////////////////////////////////////////////////////////
# # DayFraction-  Returns the fractional portion of a year from JulianStart
# # to JulianEnd, based on the input FractBasis
# #////////////////////////////////////////////////////////////////////////////

# #   - FractBasis - day fraction basis, string, most important are:
# #        "DC_30_360" - 30/360 ISDA convention (US swaps and, I think, most others) - days between 
# #               30/360, divisor 360
# #        "DC_ACT_360" - days between actual, divisor 360
# #        "DC_ACT_365F" - days between actual, divisor 365 (for all years)
# #        "DC_ACT_ACT" - days between actual, divisor is a little complicated:
# #                   if begin & end date in same year, either 365 or 366 depending on whether leap year
# #                   if not then calculate separately for the two years, using 365 or 366 for each
# #                      year (implicitely assumes days between is not more than a year - OK for swaps)
# #        "DC_ACT_365" - same as DC_ACT_ACT


# DayFraction = function(JulianStart, JulianEnd, FractBasis="DC_30_360") {

# #    /* Ensure dates have been entered in correct order [start,end] */
#     if (JulianStart > JulianEnd) return (-1.0)
        
#     switch (FractBasis,
#         DC_ACT_360 = {
#             divisor = 360.0
#             (DaysBetween( JulianStart, JulianEnd, FractBasis)) / divisor
#         },
#         DC_30_360 = {
#             divisor = 360.0
#             (DaysBetween(JulianStart, JulianEnd, FractBasis)) / divisor
#         },
#         DC_30E_360 = {
#             divisor = 360.0
#             (DaysBetween(JulianStart, JulianEnd, FractBasis)) / divisor
#         },
#         DC_30E_360U = {
#             fractbasis = "DC_30E_360"
#             divisor = 360.0
#             (DaysBetween(JulianStart, JulianEnd, fractbasis)) / divisor
#         },
#         DC_30_360U = {
#             fractbasis = "DC_30_360"
#             divisor = 360.0
#             (DaysBetween(JulianStart, JulianEnd, fractbasis)) / divisor
#         },
#         DC_ACT_365F = {
#             divisor = 365.0
#             (DaysBetween(JulianStart, JulianEnd, FractBasis)) / divisor
#         },
#         DC_365_365 = {
#             divisor = 365.25
#             (DaysBetween(JulianStart, JulianEnd, FractBasis)) / divisor
#         },
#         DC_FIXED = 1.0,
#             # finally the default: /* DC_ACT_ACT equiv. DC_ACT_365 */
#         {                  
#             ymds = JuliantoYMD(JulianStart)
#             ymde = JuliantoYMD(JulianEnd)
#             #/* If the two years are the same, then simply 
#             #// determine whether this is a 
#             #// leap year, and user 366 if it is. */
#             if (ymds[1] == ymde[1])
#             {
#                 leap = IsALeapYear(ymds[1])
#                 divisor = 365.0 + leap
#                 (DaysBetween(JulianStart, JulianEnd, FractBasis))/ divisor
#             }
#             else
#             {
#                 tdate = YMDtoJulian(ymds[1],12,31)
#                 days = DaysBetween(JulianStart, tdate,
#                         FractBasis)
#                 leap = IsALeapYear(ymds[1])
#                 DayFrac = days / (365.0 + leap)
#                 tdate = YMDtoJulian(ymde[1]-1,12,31)
#                 days = DaysBetween(tdate,JulianEnd,FractBasis)
#                 leap = IsALeapYear(ymde[1])
#                 DayFrac + (ymde[1] - ymds[1] - 1) +  days / (365.0 + leap)
#             }
#         }
#     )

# } #/* DayFraction */

