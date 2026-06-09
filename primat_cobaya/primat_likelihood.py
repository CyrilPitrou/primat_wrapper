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
from scipy.interpolate import RegularGridInterpolator
import importlib.resources


class PrimatLikelihood(Likelihood):
    """
    Gaussian likelihood for primordial nucleosynthesis abundances.

    YHe and DH are declared as requirements and arrive as keyword
    arguments to logp().  This class does no PRIMAT I/O whatsoever.
    """

    # Observed He-4 mass fraction (Yp)
    YHe_mean:         float = 0.2458
    YHe_sigma:        float = 0.0013
    YHe_PRIMAT_sigma: float = 0.0001091146   # PRIMAT theoretical uncertainty (uniform mode)

    # Observed D/H ratio
    DH_mean:          float = 2.527e-5
    DH_sigma:         float = 0.030e-5
    DH_PRIMAT_sigma:  float = 2.754096e-7    # PRIMAT theoretical uncertainty (uniform mode)

    # When True, the theoretical uncertainty is interpolated from the BBN error table
    # as a function of omegabh2 and DeltaNeff instead of using the constants above.
    tabulated_BBN_error: bool = True

    # Used in non-uniform mode when DeltaNeff is fixed (not varied by the sampler).
    # Set to None to have it arrive via get_requirements() / logp() like omegabh2.
    DeltaNeff: float = 0.0

    # ------------------------------------------------------------------ #
    # Cobaya interface                                                     #
    # ------------------------------------------------------------------ #

    def initialize(self):
        """Pre-build Gaussian distributions or 2-D interpolators depending on mode."""
        if self.tabulated_BBN_error:
            self._init_interpolated()
        else:
            self._init_uniform()

    def _init_uniform(self):
        """Pre-build fixed Gaussian distributions (obs ⊕ constant theory error)."""
        YHe_total_sigma = np.sqrt(self.YHe_sigma**2 + self.YHe_PRIMAT_sigma**2)
        DH_total_sigma  = np.sqrt(self.DH_sigma**2  + self.DH_PRIMAT_sigma**2)

        self._YHe_norm = norm(loc=self.YHe_mean, scale=YHe_total_sigma)
        self._DH_norm  = norm(loc=self.DH_mean,  scale=DH_total_sigma)

        self.log.info(
            f"PrimatLikelihood initialised (uniform uncertainty):\n"
            f"  YHe = {self.YHe_mean} ± {YHe_total_sigma:.6f} (obs ⊕ theory)\n"
            f"  D/H = {self.DH_mean:.4e} ± {DH_total_sigma:.4e} (obs ⊕ theory)"
        )

    def _init_interpolated(self):
        """Load the BBN error table and build 2-D interpolators over (DeltaN, omegabh2)."""
        data_path = importlib.resources.files("primat_cobaya") / "data" / "PRIMAT_Yp_DH_ErrorMC_100_2024.dat"
        data = np.loadtxt(data_path, comments="#")
        # columns: Ombh2(0) eta10(1) DeltaN(2) Yp(3) Yp^BBN(4) sig(Yp^BBN)(5) D/H(6) sig(D/H)(7)

        deltaN_vals   = np.unique(data[:, 2])   # 25 values, outer loop
        omegabh2_vals = np.unique(data[:, 0])   # 52 values, inner loop

        n_delta = len(deltaN_vals)
        n_omega = len(omegabh2_vals)

        sig_YHe_grid = data[:, 5].reshape(n_delta, n_omega)
        sig_DH_grid  = data[:, 7].reshape(n_delta, n_omega)

        # Store bounds for clamping in logp() — points outside the grid use
        # the nearest edge value rather than linear extrapolation.
        self._deltaN_bounds   = (deltaN_vals[0],   deltaN_vals[-1])
        self._omegabh2_bounds = (omegabh2_vals[0], omegabh2_vals[-1])

        self._sig_YHe_interp = RegularGridInterpolator(
            (deltaN_vals, omegabh2_vals), sig_YHe_grid,
            method="linear", bounds_error=True,
        )
        self._sig_DH_interp = RegularGridInterpolator(
            (deltaN_vals, omegabh2_vals), sig_DH_grid,
            method="linear", bounds_error=True,
        )

        self.log.info(
            f"PrimatLikelihood initialised (interpolated uncertainty):\n"
            f"  Table: {n_delta} DeltaN values × {n_omega} omegabh2 values\n"
            f"  DeltaN range : {deltaN_vals[0]} – {deltaN_vals[-1]}\n"
            f"  omegabh2 range: {omegabh2_vals[0]} – {omegabh2_vals[-1]}\n"
            f"  YHe obs sigma = {self.YHe_sigma}\n"
            f"  D/H obs sigma = {self.DH_sigma:.4e}"
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

        In tabulated mode, omegabh2 and (optionally) DeltaNeff are also
        requested so their current values are available for interpolation.
        """
        reqs = {"YHe": None, "DH": None}
        if self.tabulated_BBN_error:
            reqs["omegabh2"] = None
            if self.DeltaNeff is None:
                reqs["DeltaNeff"] = None
        return reqs

    def logp(self, YHe, DH, **params_values):
        """
        YHe and DH are injected by Cobaya from PrimatTheory's derived output.
        Returns -inf if the theory signalled failure via NaN.
        """
        if np.isnan(YHe) or np.isnan(DH):
            return -np.inf

        if self.tabulated_BBN_error:
            omegabh2  = params_values["omegabh2"]
            DeltaNeff = self.DeltaNeff if self.DeltaNeff is not None else params_values["DeltaNeff"]

            DeltaNeff_c = np.clip(DeltaNeff, *self._deltaN_bounds)
            omegabh2_c  = np.clip(omegabh2,  *self._omegabh2_bounds)
            pt = [[DeltaNeff_c, omegabh2_c]]
            sig_YHe_theory = self._sig_YHe_interp(pt)[0]
            sig_DH_theory  = self._sig_DH_interp(pt)[0]

            YHe_total_sigma = np.sqrt(self.YHe_sigma**2 + sig_YHe_theory**2)
            DH_total_sigma  = np.sqrt(self.DH_sigma**2  + sig_DH_theory**2)

            logp_YHe = norm.logpdf(YHe, loc=self.YHe_mean, scale=YHe_total_sigma)
            logp_DH  = norm.logpdf(DH,  loc=self.DH_mean,  scale=DH_total_sigma)
        else:
            logp_YHe = self._YHe_norm.logpdf(YHe)
            logp_DH  = self._DH_norm.logpdf(DH)

        total = logp_YHe + logp_DH

        self.log.debug(
            f"YHe={YHe:.8f}  D/H={DH:.6e}  "
            f"logp_YHe={logp_YHe:.3f}  logp_DH={logp_DH:.3f}  total={total:.3f}"
        )
        return total
