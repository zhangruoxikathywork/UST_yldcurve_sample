###################################
# Testing out cython with         #
# Mandlebrot set example          #
# from https://python.plainenglish.io/advanced-python-programming-writing-efficient-code-with-cython-and-numba-edf67122f7ce
###################################


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 17:23:06 2024

@author: tcoleman
"""
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("mandelbrot.pyx"),
    include_dirs=[numpy.get_include()]
)