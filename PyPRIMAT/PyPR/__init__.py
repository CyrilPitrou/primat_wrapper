# -*- coding: utf-8 -*-
"""
PyPR — core package for PyPRIMAT.

Public API::

    from PyPR import PyPRclass
    result = PyPRclass({"Omegabh2": 0.022425}).solve()
"""

from .PyPR_main import PyPRclass

__all__ = ["PyPRclass"]
