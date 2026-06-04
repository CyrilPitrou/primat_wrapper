"""
primat_wrapper
==============
A Cobaya theory+likelihood package wrapping the PRIMAT / PyPRIMAT Big Bang
Nucleosynthesis solvers.

The two Cobaya components are exposed here for convenience, so they can be
imported as ``from primat_wrapper import PrimatTheory, PrimatLikelihood``.
In Cobaya YAML files, reference them by their fully-qualified dotted path:

    theory:
      primat_wrapper.primat_theory.PrimatTheory:
    likelihood:
      primat_wrapper.primat_likelihood.PrimatLikelihood:
"""

from .primat_theory import PrimatTheory
from .primat_likelihood import PrimatLikelihood

__all__ = ["PrimatTheory", "PrimatLikelihood"]
__version__ = "0.1.0"
