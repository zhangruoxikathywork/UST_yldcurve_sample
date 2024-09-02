###################################
# Testing out cython with         #
# Mandlebrot set example          #
# from https://python.plainenglish.io/advanced-python-programming-writing-efficient-code-with-cython-and-numba-edf67122f7ce
###################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
cimport numpy as np

cpdef int mandelbrot_cython(double complex c, int maxiter):
    cdef double complex z = c
    cdef int n
    for n in range(maxiter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return maxiter
cpdef np.ndarray[int, ndim=2] mandelbrot_set_cython(double xmin, double xmax, double ymin, double ymax, int width=1024, int height=1024, int maxiter=100):
    cdef np.ndarray[int, ndim=2] pixels = np.zeros((height, width), dtype=np.int32)
    cdef int i, j
    cdef double x, y
    for i in range(height):
        y = ymin + i*(ymax-ymin)/(height-1)
        for j in range(width):
            x = xmin + j*(xmax-xmin)/(width-1)
            pixels[i, j] = mandelbrot_cython(x + y*1j, maxiter)
    return pixels