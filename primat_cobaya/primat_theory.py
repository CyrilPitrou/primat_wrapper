"""
primat_theory.py
----------------
Cobaya Theory class that runs PyPRIMAT and stores primordial nucleosynthesis
abundances as derived params.

Two helium conventions
----------------------
PyPRIMAT computes He-4 in two conventions that differ slightly due to nuclear
mass corrections:

  YpBBN  = 4 · Y(He4)   — "BBN convention"; this is what spectroscopic
                           He-4 abundance measurements report.
                           → consumed by PrimatLikelihood.

  YHe    = YPCMB         — "CMB convention"; the actual mass fraction of
                           helium that enters recombination physics.
                           → fed into CLASS (or CAMB) via the 'YHe' parameter.

For typical BBN values YHe is ~0.07 % smaller than YpBBN. PyPRIMAT returns
both conventions directly.

Data flow
---------
  calculate() runs PyPRIMAT and writes:
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
    Theory class that calls PyPRIMAT and provides primordial abundances
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
    PyPRIMAT_PATH: str = ""
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
        # Relative PyPRIMAT_PATH values from the YAML are resolved against it.
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._init_pyprimat(base)

    def _init_pyprimat(self, base):
        """Verify that pyprimat is importable. Prefers the installed package; falls back to PyPRIMAT_PATH."""
        import sys
        try:
            import pyprimat  # noqa: F401 — just checking it's importable
            self.log.info("PrimatTheory initialised with PyPRIMAT (installed package).")
            return
        except ImportError:
            pass

        # Not installed — try path-based fallback
        if not self.PyPRIMAT_PATH:
            self.PyPRIMAT_PATH = os.path.normpath(os.path.join(base, "..", "PyPRIMAT"))
        elif not os.path.isabs(self.PyPRIMAT_PATH):
            self.PyPRIMAT_PATH = os.path.join(base, self.PyPRIMAT_PATH)

        if not os.path.isdir(self.PyPRIMAT_PATH):
            raise FileNotFoundError(
                f"pyprimat is not installed and PyPRIMAT directory not found at {self.PyPRIMAT_PATH}. "
                "Install PyPRIMAT with: pip install -e /path/to/PyPRIMAT"
            )
        if self.PyPRIMAT_PATH not in sys.path:
            sys.path.insert(0, self.PyPRIMAT_PATH)
        self.log.info("PrimatTheory initialised with PyPRIMAT (path fallback).")
        self.log.info(f"  PyPRIMAT path : {self.PyPRIMAT_PATH}")

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
        Run PyPRIMAT for the current parameter point and store abundances
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
            f"--- Running PyPRIMAT ---  "
            f"omegabh2={omegabh2}  DeltaNeff={DeltaNeff}  "
            f"fEDE={fEDE}  zcEDE={zcEDE}  wnEDE={wnEDE}"
        )

        results = self._run_bbn_pyprimat(omegabh2, DeltaNeff=DeltaNeff,
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
            f"--- PyPRIMAT done in {time.time()-start_time:.1f}s ---  "
            f"YpBBN={YpBBN:.8f}  YHe(CMB)={YHe:.8f}  D/H={DH:.6e}"
        )
        return True

    # ------------------------------------------------------------------ #
    # BBN runner                                                           #
    # ------------------------------------------------------------------ #

    def _run_bbn_pyprimat(self, omegabh2, DeltaNeff=0.0, fEDE=0.0, zcEDE=1e8, wnEDE=1.0):
        """Invoke PyPRIMAT directly and return a dict of abundances, or None on failure."""
        try:
            from pyprimat import PyPR
            results = PyPR({
                "Omegabh2":  omegabh2,
                "DeltaNeff": DeltaNeff,
                "fEDE":      fEDE,
                "zcEDE":     zcEDE,
                "wnEDE":     wnEDE,
                "network":   "small" if self.ReducedNetwork else "medium",
                "verbose":   self.Verbose,
            }).solve()
            # PyPRIMAT provides both conventions directly; use them as-is.
            return {
                "YpBBN": results['YPBBN'],
                "YHe":   results['YPCMB'],  # CMB convention; fed into CLASS
                "DH":    results['DoH'],
            }
        except Exception as e:
            self.log.error(f"PyPRIMAT failed: {e}")
            return None
