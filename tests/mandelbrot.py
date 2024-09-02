###################################
# Testing out cython and numba    #
# with Mandlebrot set example     #
# from https://python.plainenglish.io/advanced-python-programming-writing-efficient-code-with-cython-and-numba-edf67122f7ce
# Also look at f2py which wraps   #
# Fortran or C code               #
###################################


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:44:39 2024

@author: tcoleman
"""

#%% Standard python code (not optimized)

import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(c, maxiter=100):
    z = c
    for n in range(maxiter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return maxiter
def mandelbrot_set(xmin, xmax, ymin, ymax, width=1024, height=1024, maxiter=100):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    pixels = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            pixels[i, j] = mandelbrot(x[j] + y[i]*1j, maxiter)
    return pixels
pixels = mandelbrot_set(-2, 0.5, -1.25, 1.25)
plt.imshow(pixels, cmap="viridis")
plt.axis("off")
plt.show()


#%% Calling cythonized code

# First, create a new file with function definition TYPING ALL VARIABLES
# File is mandelbrot.pyx

# Then create a "setup" file (mandelbrot_setup.py) which will be called from
# command line by:
#    python mandelbrot_setup.py build_ext --inplace
# Creates new file "mandelbrot.cpython-*.so" and can import this and use new
# function "mandelbrot_set_cython" which is defined in "mandelbroth.pyx"

# May need to install cython:
#    conda install conda-forge::cython

from mandelbrot import mandelbrot_set_cython

pixels = mandelbrot_set_cython(-2, 0.5, -1.25, 1.25)
plt.imshow(pixels, cmap="viridis")
plt.axis("off")
plt.show()


#%% numba version

import numba

@numba.njit
def mandelbrot_numba(c, maxiter=100):
    z = c
    for n in range(maxiter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return maxiter
@numba.njit(parallel=False)   # Setting parallel=True speeds up by about another factor of 3
def mandelbrot_set_numba(xmin, xmax, ymin, ymax, width=1024, height=1024, maxiter=100):
    pixels = np.zeros((height, width), dtype=np.int32)
    for i in numba.prange(height):
        y = ymin + i*(ymax-ymin)/(height-1)
        for j in range(width):
            x = xmin + j*(xmax-xmin)/(width-1)
            pixels[i, j] = mandelbrot_numba(x + y*1j, maxiter)
    return pixels


pixels = mandelbrot_set_numba(-2, 0.5, -1.25, 1.25)
plt.imshow(pixels, cmap="viridis")
plt.axis("off")
plt.show()


#%% Running timeit


import timeit

print("Python:", timeit.timeit("mandelbrot_set(-2, 0.5, -1.25, 1.25)", globals=globals(), number=10))
print("Cython:", timeit.timeit("mandelbrot_set_cython(-2, 0.5, -1.25, 1.25)", globals=globals(), number=10))
print("Numba:", timeit.timeit("mandelbrot_set_numba(-2, 0.5, -1.25, 1.25)", globals=globals(), number=10))

# Seems that cython and numba are similar speed-up. Factor of about 35 for both. 
# numba parallel=True speeds up by another factor of 3

