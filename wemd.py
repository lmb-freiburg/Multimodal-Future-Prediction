from numpy.ctypeslib import ndpointer
import ctypes
import numpy as np
import os

lib = ctypes.cdll.LoadLibrary('wemd/lib/libwemd.so')

c_WEMD_1D = lib.WEMD_1D
c_WEMD_1D.restype = ctypes.c_float
c_WEMD_1D.argtypes = [ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                      ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                      ctypes.c_int]


c_WEMD_2D = lib.WEMD_2D
c_WEMD_2D.restype = ctypes.c_float
c_WEMD_2D.argtypes = [ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                      ndpointer(ctypes.c_float, flags='C_CONTIGUOUS'),
                      ctypes.c_int,
                      ctypes.c_int]

def computeWEMD(hist1, hist2):
    cont_hist1 = np.ascontiguousarray(hist1, np.float32)
    cont_hist2 = np.ascontiguousarray(hist2, np.float32)
    if len(hist1.shape) == 1:
        distance = c_WEMD_1D(cont_hist1, cont_hist2, hist1.shape[0])
    else:
        distance = c_WEMD_2D(cont_hist1, cont_hist2, hist1.shape[0], hist1.shape[1])

    return distance




