"""
primat_likelihood.py
--------------------
Cobaya Likelihood that computes a Gaussian log-probability for
primordial He-4 and D/H abundances provided by PrimatTheory.

How the data flow works
-----------------------
PrimatTheory.calculate() stores YHe and DH in state["derived"].
The likelihood declares them in get_requirements() so Cobaya builds
the dependency graph (likelihood → theory).  Cobaya then injects them
as keyword arguments to logp().
"""

from cobaya.likelihood import Likelihood
import numpy as np
from scipy.stats import norm


class PrimatLikelihood(Likelihood):
    """
    Gaussian likelihood for primordial nucleosynthesis abundances.

    YHe and DH are declared as requirements and arrive as keyword
    arguments to logp().  This class does no PRIMAT I/O whatsoever.
    """

    # Observed He-4 mass fraction (Yp)
    YHe_mean:         float = 0.2458
    YHe_sigma:        float = 0.0013
    YHe_PRIMAT_sigma: float = 0.0001091146   # PRIMAT theoretical uncertainty

    # Observed D/H ratio
    DH_mean:          float = 2.527e-5
    DH_sigma:         float = 0.030e-5
    DH_PRIMAT_sigma:  float = 2.754096e-7    # PRIMAT theoretical uncertainty

    # ------------------------------------------------------------------ #
    # Cobaya interface                                                     #
    # ------------------------------------------------------------------ #

    def initialize(self):
        """Pre-build the Gaussian distributions (obs ⊕ theory error in quadrature)."""
        YHe_total_sigma = np.sqrt(self.YHe_sigma**2 + self.YHe_PRIMAT_sigma**2)
        DH_total_sigma  = np.sqrt(self.DH_sigma**2  + self.DH_PRIMAT_sigma**2)

        self._YHe_norm = norm(loc=self.YHe_mean, scale=YHe_total_sigma)
        self._DH_norm  = norm(loc=self.DH_mean,  scale=DH_total_sigma)

        self.log.info(
            f"PrimatLikelihood initialised:\n"
            f"  YHe = {self.YHe_mean} ± {YHe_total_sigma:.6f} (obs ⊕ theory)\n"
            f"  D/H = {self.DH_mean:.4e} ± {DH_total_sigma:.4e} (obs ⊕ theory)"
        )

    def get_requirements(self):
        """
        Declare YHe and DH as requirements so Cobaya can build the
        dependency graph: likelihood → PrimatTheory.

        This does NOT create a circular dependency because PrimatTheory
        declares them via the class-level `output_params` attribute, which
        Cobaya's _assign_params resolves before _set_dependencies_and_providers
        runs.  By that point YHe and DH are known outputs of PrimatTheory,
        so requesting them here simply wires the correct edge in the graph.

        Cobaya then injects them as keyword arguments into logp().
        """
        return {"YHe": None, "DH": None}

    def logp(self, YHe, DH, **params_values):
        """
        YHe and DH are injected by Cobaya from PrimatTheory's derived output.
        Returns -inf if the theory signalled failure via NaN.
        """
        if np.isnan(YHe) or np.isnan(DH):
            return -np.inf

        logp_YHe = self._YHe_norm.logpdf(YHe)
        logp_DH  = self._DH_norm.logpdf(DH)
        total    = logp_YHe + logp_DH

        self.log.debug(
            f"YHe={YHe:.8f}  D/H={DH:.6e}  "
            f"logp_YHe={logp_YHe:.3f}  logp_DH={logp_DH:.3f}  total={total:.3f}"
        )
        return total
