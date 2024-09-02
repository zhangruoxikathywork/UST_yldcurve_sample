# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 21:04:28 2016

@author: tcoleman"""

from importlib import reload
import sys
import numpy as np

sys.path.append('../src/package')

#%%

import discfact as df

reload(df)

#%% ------ Sample data for curves ------
mat = np.array([0.25,0.5,0.75,0.9,0.99,1.0,1.2])
ratessab = np.array([0.01,0.02005,0.02512,0.03724])
ratescc = 2.*np.log(1+ratessab/2.)
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
#%% 

