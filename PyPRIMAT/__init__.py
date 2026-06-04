# -*- coding: utf-8 -*-
"""
PyPRIMAT — Python implementation of PRIMAT for Big Bang Nucleosynthesis.

Quick start::

    from PyPR import PyPRclass
    result = PyPRclass({"Omegabh2": 0.022425}).solve()
    print(result["YPBBN"], result["DoHx1e5"])
"""

from PyPR import PyPRclass

__version__ = "0.1.0"
__all__ = ["PyPRclass"]
