"""
primat_theory.py
----------------
Cobaya Theory class that runs a BBN code (PRIMAT via Mathematica, or PyPRIMAT
via Python) and stores primordial nucleosynthesis abundances YHe (Yp) and DH
(D/H) as derived parameters.

Data flow
---------
  calculate() runs the BBN code and writes:
      state["derived"] = {"YHe": ..., "DH": ...}

  YHe and DH are declared via the class-level `output_params` attribute,
  which is how Cobaya's _assign_params discovers at config-time which
  component owns these derived quantities.

  PrimatLikelihood receives them as keyword arguments to logp() because
  they are also listed in the global params block of the run YAML with
  no prior (= derived params).
"""

from cobaya.theory import Theory
import numpy as np
import os
import subprocess
import csv
import tempfile
import time
import shutil

# Standard locations to search for a valid MathKernel executable.
MATHKERNEL_CANDIDATES = [
    "/Applications/Wolfram.app/Contents/MacOS/MathKernel",
    "/Applications/Mathematica.app/Contents/MacOS/MathKernel",
    "/usr/local/Wolfram/Mathematica/12.0/Executables/MathKernel",
    "math13",
    "MathKernel",
]


class PrimatTheory(Theory):
    """
    Theory class that calls PRIMAT (via Mathematica) or PyPRIMAT (pure Python)
    and provides primordial abundances YHe (Yp mass fraction) and DH (D/H ratio)
    as Cobaya derived parameters.
    """

    # ------------------------------------------------------------------ #
    # Declare output params at CLASS level so Cobaya's _assign_params     #
    # knows which component owns YHe and DH.                             #
    # ------------------------------------------------------------------ #
    output_params = ["YHe", "DH"]

    # ------------------------------------------------------------------ #
    # Class-level attributes — overridden by YAML values automatically   #
    # ------------------------------------------------------------------ #
    PRIMAT_PATH: str = ""
    PyPRIMAT_PATH: str = ""
    BBN_solver: str = "PyPRIMAT"  # "PyPRIMAT" or "PRIMAT"
    MathKernelCommand: str = ""  # auto-detected if empty
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
        # base = the primat_tools repo root (parent of the primat_wrapper/ package).
        # Relative PRIMAT_PATH values from the YAML are resolved against it.
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        if self.BBN_solver == "PRIMAT":
            self.log.info("Attempting to use PRIMAT (Mathematica) as BBN solver.")
            self._init_primat(base)

        # _init_primat may fall back to "PyPRIMAT" if MathKernel is unavailable
        if self.BBN_solver == "PyPRIMAT":
            self._init_pyprimat(base)

    def _find_mathkernel(self):
        """
        Search for a valid MathKernel executable.

        If MathKernelCommand is set by the user, try it first.  If it is
        not valid, fall through and search MATHKERNEL_CANDIDATES in order,
        skipping the user-supplied command (already tried).  Return the
        first valid command found, or None if none works.
        """
        user_cmd = self.MathKernelCommand

        # Build the search list: user command first (if given), then candidates,
        # excluding the user command from the tail to avoid testing it twice.
        search = []
        if user_cmd:
            search.append(user_cmd)
        search += [c for c in MATHKERNEL_CANDIDATES if c != user_cmd]

        for candidate in search:
            resolved = shutil.which(candidate) or candidate
            if self._is_valid_mathkernel(resolved):
                if candidate != user_cmd:
                    self.log.info(
                        f"MathKernelCommand '{user_cmd}' not valid; "
                        f"found working kernel at '{resolved}'."
                    )
                return resolved

        return None

    def _init_primat(self, base):
        """Set up PRIMAT paths and find a valid MathKernel. Falls back to PyPRIMAT on failure."""

        # Resolve PRIMAT_PATH: explicit YAML value → $PRIMAT_DIR env var → sibling ../PRIMAT
        if not self.PRIMAT_PATH:
            env_dir = os.environ.get("PRIMAT_DIR", "")
            if env_dir:
                self.PRIMAT_PATH = env_dir
            else:
                self.PRIMAT_PATH = os.path.normpath(os.path.join(base, "..", "PRIMAT"))
        elif not os.path.isabs(self.PRIMAT_PATH):
            self.PRIMAT_PATH = os.path.join(base, self.PRIMAT_PATH)

        self.primat_script = os.path.join(
            self.PRIMAT_PATH, "PythonInterface", "PyPRIMAT_FinalAbundances.m"
        )
        if not os.path.exists(self.primat_script):
            self.log.warning(
                f"PRIMAT script not found at {self.primat_script}. "
                "Falling back to PyPRIMAT."
            )
            self.BBN_solver = "PyPRIMAT"
            return

        # Find a working MathKernel (tries user command first, then candidates)
        found = self._find_mathkernel()
        if not found:
            self.log.warning(
                "No valid MathKernel found in any of the standard locations. "
                "Falling back to PyPRIMAT."
            )
            self.MathKernelCommand = ""
            self.BBN_solver = "PyPRIMAT"
            return

        self.MathKernelCommand = found
        self.log.info("PrimatTheory initialised with PRIMAT.")
        self.log.info(f"  PRIMAT script : {self.primat_script}")
        self.log.info(f"  MathKernel    : {self.MathKernelCommand}")

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

    @staticmethod
    def _is_valid_mathkernel(cmd):
        """Return True if cmd resolves to a working MathKernel executable."""
        resolved = shutil.which(cmd) or cmd
        if not os.path.isfile(resolved) or not os.access(resolved, os.X_OK):
            return False
        try:
            result = subprocess.run(
                [resolved, "-version"],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

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
        Run the BBN code for the current parameter point and store abundances
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
            f"--- Running {self.BBN_solver} ---  "
            f"omegabh2={omegabh2}  DeltaNeff={DeltaNeff}  "
            f"fEDE={fEDE}  zcEDE={zcEDE}  wnEDE={wnEDE}"
        )

        if self.BBN_solver == "PRIMAT":
            results = self._run_bbn_primat(omegabh2, DeltaNeff=DeltaNeff,
                                           fEDE=fEDE, zcEDE=zcEDE, wnEDE=wnEDE)
        else:
            results = self._run_bbn_pyprimat(omegabh2, DeltaNeff=DeltaNeff,
                                             fEDE=fEDE, zcEDE=zcEDE, wnEDE=wnEDE)

        # Always populate state["derived"] — use NaN on failure so the
        # likelihood can detect it and return -inf cleanly.
        if results is None:
            state["derived"] = {"YHe": np.nan, "DH": np.nan}
            return False

        YHe = self._extract_YHe(results)
        DH  = self._extract_DH(results)

        if YHe is None or DH is None:
            self.log.error(
                f"Could not extract YHe or DH from BBN output. "
                f"Available keys: {list(results.keys())}"
            )
            state["derived"] = {"YHe": np.nan, "DH": np.nan}
            return False

        state["derived"] = {"YHe": YHe, "DH": DH}

        self.log.info(
            f"--- {self.BBN_solver} done in {time.time()-start_time:.1f}s ---  "
            f"YHe={YHe:.8f}  D/H={DH:.6e}"
        )
        return True

    # ------------------------------------------------------------------ #
    # BBN runners                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_mathematica(value):
        """Format a Python value as a Mathematica literal."""
        if isinstance(value, bool):
            return "True" if value else "False"
        return str(value)

    def _run_bbn_primat(self, omegabh2, DeltaNeff=0.0, fEDE=0.0, zcEDE=1e8, wnEDE=1.0):
        """Invoke PRIMAT via MathKernel and return a dict of abundances, or None on failure."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            output_file = tmp.name

        try:
            use_EDE = fEDE > 0.0
            args = {
                "$ReducedNetwork":      self.ReducedNetwork,
                r"h2\[CapitalOmega]b0": omegabh2,
                "Nrelat":               DeltaNeff,  # Mathematica variable name
                "$Verbose":             self.Verbose,
                "$EDEBool":             use_EDE,
            }
            if use_EDE:
                args.update({"fEDE": fEDE, "zcEDE": zcEDE, "wnEDE": wnEDE})

            extra = "; ".join(f"{k}={self._to_mathematica(v)}" for k, v in args.items()) + ";"
            option_output = f'$Outputfile="{output_file}"'

            if self.Verbose:
                self.log.info(f"PRIMAT args : {extra}")

            # Run MathKernel from the script's directory (PRIMAT resolves its
            # own relative paths there) via subprocess's cwd= rather than a
            # global os.chdir, so we never mutate this process's working dir.
            proc = subprocess.run(
                [self.MathKernelCommand, "-initfile",
                 os.path.basename(self.primat_script), extra, option_output],
                capture_output=True, text=True, timeout=300,
                cwd=os.path.dirname(self.primat_script),
            )

            if self.Verbose:
                self.log.info(f"MathKernel exit code: {proc.returncode}")
                for line in (proc.stdout or "").splitlines()[:50]:
                    if line.strip():
                        self.log.info(f"  stdout: {line}")
                if proc.stderr:
                    self.log.info(f"  stderr: {proc.stderr}")

            if proc.returncode != 0:
                self.log.error(f"PRIMAT failed (exit {proc.returncode}): {proc.stderr}")
                return None

            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                self.log.error("PRIMAT produced no output file.")
                return None

            results = {}
            with open(output_file, newline="") as f:
                for row in csv.reader(f):
                    if len(row) >= 2:
                        try:
                            results[row[0]] = float(row[1])
                        except ValueError:
                            pass

            if self.Verbose:
                self.log.info(f"PRIMAT keys: {list(results.keys())[:15]}")
            return results

        finally:
            if os.path.exists(output_file):
                os.remove(output_file)

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
            return {"YHe": results['YPBBN'], "DH": results['DoH']}
        except Exception as e:
            self.log.error(f"PyPRIMAT failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Abundance extractors                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_YHe(results):
        """Return Yp (He-4 mass fraction) from BBN output dict."""
        for key in ["YP", "Yp", "YHe4", "Y_p", "YHe", "yp"]:
            if key in results:
                return results[key]
        return None

    @staticmethod
    def _extract_DH(results):
        """Return D/H ratio from BBN output dict."""
        for key in ["D/H", "DH", "YD/YH", "D_H"]:
            if key in results:
                return results[key]
        YD = next((results[k] for k in ["YD", "Y_D", "D", "yD"] if k in results), None)
        YH = next((results[k] for k in ["YH", "Y_H", "H", "yH", "YH1", "Y_H1"] if k in results), None)
        if YD is not None and YH and YH > 0:
            return YD / YH
        return None
