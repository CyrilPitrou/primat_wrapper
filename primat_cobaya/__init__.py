"""
primat_cobaya
==============
A Cobaya theory+likelihood package wrapping the PyPRIMAT Big Bang
Nucleosynthesis solver.

The two Cobaya components are exposed here for convenience, so they can be
imported as ``from primat_cobaya import PrimatTheory, PrimatLikelihood``.
In Cobaya YAML files, reference them by their fully-qualified dotted path:

    theory:
      primat_cobaya.primat_theory.PrimatTheory:
    likelihood:
      primat_cobaya.primat_likelihood.PrimatLikelihood:
"""

from .primat_theory import PrimatTheory
from .primat_likelihood import PrimatLikelihood

__all__ = ["PrimatTheory", "PrimatLikelihood"]
__version__ = "0.1.0"
