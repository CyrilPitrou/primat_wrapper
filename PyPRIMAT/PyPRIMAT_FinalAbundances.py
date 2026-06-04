# -*- coding: utf-8 -*-
from PyPR import PyPRclass


def compute_abundances(omegabh2=0.022425, DeltaNeff=0.0,
                       fEDE=0., zcEDE=1e8, wnEDE=1.0,
                       smallnet_flag=True, verbose_flag=False):
    """
    Compute BBN abundances using PyPRIMAT.

    Returns the raw result dict from PyPRclass.solve():
      YPBBN, YPCMB, DoH, He3oH, Li7oH, Neff, ...
    """
    params = {
        "Omegabh2":          omegabh2,
        "DeltaNeff":         DeltaNeff,
        "fEDE":              fEDE,
        "zcEDE":             zcEDE,
        "wnEDE":             wnEDE,
        "smallnet_flag":     smallnet_flag,
        "verbose_flag":      verbose_flag,
        "compute_nTOp_flag": False,  # use pre-tabulated rates for speed
        "save_nTOp_flag":    False,
    }
    return PyPRclass(params).solve()
