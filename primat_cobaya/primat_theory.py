"""
primat_theory.py
----------------
Cobaya Theory class that runs PRIMAT and stores primordial nucleosynthesis
abundances as derived params.

Two helium conventions
----------------------
PRIMAT computes He-4 in two conventions that differ slightly due to nuclear
mass corrections:

  YpBBN  = 4 · Y(He4)   — "BBN convention"; this is what spectroscopic
                           He-4 abundance measurements report.
                           → consumed by PrimatLikelihood.

  YHe    = YPCMB         — "CMB convention"; the actual mass fraction of
                           helium that enters recombination physics.
                           → fed into CLASS (or CAMB) via the 'YHe' parameter.

For typical BBN values YHe is ~0.07 % smaller than YpBBN. PRIMAT returns
both conventions directly.

Data flow
---------
  calculate() runs PRIMAT and writes:
      state["derived"] = {"YpBBN": ..., "YHe": ..., "DH": ...}

  All three are declared via the class-level `output_params` attribute so
  Cobaya's _assign_params knows which component owns them.

  PrimatLikelihood receives YpBBN and DH as keyword arguments to logp().
  CLASS receives YHe because 'YHe' is a recognised CLASS parameter name and
  the run YAML declares it with 'derived: False' in the classy params block.
"""

from cobaya.theory import Theory
import numpy as np
import os
import time


class PrimatTheory(Theory):
    """
    Theory class that calls PRIMAT and provides primordial abundances
    YHe (Yp mass fraction) and DH (D/H ratio) as Cobaya derived parameters.
    """

    # ------------------------------------------------------------------ #
    # Declare output params at CLASS level so Cobaya's _assign_params     #
    # knows which component owns these derived quantities.               #
    # YpBBN → PrimatLikelihood; YHe (=YPCMB) → CLASS; DH → both.       #
    # ------------------------------------------------------------------ #
    output_params = ["YpBBN", "YHe", "DH"]

    # ------------------------------------------------------------------ #
    # Class-level attributes — overridden by YAML values automatically   #
    # ------------------------------------------------------------------ #
    PRIMAT_PATH: str = ""
    ReducedNetwork: bool = True
    DeltaNeff: float = 0.0        # set to None to vary via sampler
    Verbose: bool = False

    # Early Dark Energy
    fEDE:  float = 0.0           # set to None to vary via sampler
    zcEDE: float = 1.0e8         # set to None to vary via sampler
    wnEDE: float = 1.0           # set to None to vary via sampler

    # ------------------------------------------------------------------ #
    # Cobaya interface                                                     #
    # ------------------------------------------------------------------ #

    def initialize(self):
        # base = the primat_tools repo root (parent of the primat_cobaya/ package).
        # Relative PRIMAT_PATH values from the YAML are resolved against it.
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._init_primat(base)

    def _init_primat(self, base):
        """Verify that primat is importable. Prefers the installed package; falls back to PRIMAT_PATH."""
        import sys
        try:
            import primat  # noqa: F401 — just checking it's importable
            from primat.backend import HAS_C_BACKEND
            backend = "C" if HAS_C_BACKEND else "pure-Python"
            self.log.info(f"PrimatTheory initialised with primat (installed package, {backend} backend available).")
            return
        except ImportError:
            pass

        # Not installed — try path-based fallback (prefer PRIMAT, fall back to PyPRIMAT)
        if not self.PRIMAT_PATH:
            primat_dir = os.path.normpath(os.path.join(base, "..", "PRIMAT"))
            pyprimat_dir = os.path.normpath(os.path.join(base, "..", "PyPRIMAT"))

            if os.path.isdir(primat_dir):
                self.PRIMAT_PATH = primat_dir
            elif os.path.isdir(pyprimat_dir):
                self.PRIMAT_PATH = pyprimat_dir
            else:
                raise FileNotFoundError(
                    f"primat is not installed and neither PRIMAT nor PyPRIMAT directory found. "
                    "Install PRIMAT with: pip install -e /path/to/PRIMAT"
                )
        elif not os.path.isabs(self.PRIMAT_PATH):
            self.PRIMAT_PATH = os.path.join(base, self.PRIMAT_PATH)

        if not os.path.isdir(self.PRIMAT_PATH):
            raise FileNotFoundError(
                f"PRIMAT directory not found at {self.PRIMAT_PATH}. "
                "Install PRIMAT with: pip install -e /path/to/PRIMAT"
            )
        if self.PRIMAT_PATH not in sys.path:
            sys.path.insert(0, self.PRIMAT_PATH)
        self.log.info("PrimatTheory initialised with primat (path fallback).")
        self.log.info(f"  PRIMAT path : {self.PRIMAT_PATH}")

    def get_requirements(self):
        """
        Declare which sampler parameters this theory reads.
        omegabh2 is always required. EDE / DeltaNeff are only requested
        when they are set to None in the YAML (i.e. varied by the sampler).
        """
        reqs = {"omegabh2": None}
        if self.fEDE is None:
            reqs["fEDE"] = None
        if self.zcEDE is None:
            reqs["zcEDE"] = None
        if self.DeltaNeff is None:
            reqs["DeltaNeff"] = None
        if self.wnEDE is None:
            reqs["wnEDE"] = None
        return reqs

    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Run PRIMAT for the current parameter point and store abundances
        in state["derived"] so Cobaya routes them to the likelihood and
        writes them to the chain.
        """
        start_time = time.time()

        omegabh2  = self.provider.get_param("omegabh2")
        fEDE      = self.fEDE      if self.fEDE      is not None else self.provider.get_param("fEDE")
        zcEDE     = self.zcEDE     if self.zcEDE     is not None else self.provider.get_param("zcEDE")
        DeltaNeff = self.DeltaNeff if self.DeltaNeff is not None else self.provider.get_param("DeltaNeff")
        wnEDE     = self.wnEDE     if self.wnEDE     is not None else self.provider.get_param("wnEDE")

        self.log.info(
            f"--- Running PRIMAT ---  "
            f"omegabh2={omegabh2}  DeltaNeff={DeltaNeff}  "
            f"fEDE={fEDE}  zcEDE={zcEDE}  wnEDE={wnEDE}"
        )

        results = self._run_bbn_primat(omegabh2, DeltaNeff=DeltaNeff,
                                         fEDE=fEDE, zcEDE=zcEDE, wnEDE=wnEDE)

        # Always populate state["derived"] — use NaN on failure so the
        # likelihood can detect it and return -inf cleanly.
        if results is None:
            state["derived"] = {"YpBBN": np.nan, "YHe": np.nan, "DH": np.nan}
            return False

        YpBBN = results.get("YpBBN")
        YHe   = results.get("YHe")   # CMB convention
        DH    = results.get("DH")

        if YpBBN is None or YHe is None or DH is None:
            self.log.error(
                f"Could not extract YpBBN/YHe/DH from BBN output. "
                f"Available keys: {list(results.keys())}"
            )
            state["derived"] = {"YpBBN": np.nan, "YHe": np.nan, "DH": np.nan}
            return False

        state["derived"] = {"YpBBN": YpBBN, "YHe": YHe, "DH": DH}

        self.log.info(
            f"--- PRIMAT done in {time.time()-start_time:.1f}s ---  "
            f"YpBBN={YpBBN:.8f}  YHe(CMB)={YHe:.8f}  D/H={DH:.6e}"
        )
        return True

    # ------------------------------------------------------------------ #
    # BBN runner                                                           #
    # ------------------------------------------------------------------ #

    def _run_bbn_primat(self, omegabh2, DeltaNeff=0.0, fEDE=0.0, zcEDE=1e8, wnEDE=1.0):
        """Invoke PRIMAT and return a dict of abundances, or None on failure.

        Uses the primat.backend.run_bbn dispatcher, which automatically selects
        the C backend (if available) or falls back to pure-Python.
        """
        try:
            from primat.backend import run_bbn

            params = {
                "Omegabh2":  omegabh2,
                "DeltaNeff": DeltaNeff,
                "fEDE":      fEDE,
                "zcEDE":     zcEDE,
                "wnEDE":     wnEDE,
                "verbose":   self.Verbose,
            }

            if self.ReducedNetwork:
                params["network"] = "small"
            else:
                # Old "medium" network is now "large" with amax=8
                params["network"] = "large"
                params["amax"] = 8

            results = run_bbn(params)

            return {
                "YpBBN": results['YPBBN'],
                "YHe":   results['YPCMB'],  # CMB convention; fed into CLASS
                "DH":    results['DoH'],
            }
        except Exception as e:
            self.log.error(f"PRIMAT failed: {e}")
            return None
