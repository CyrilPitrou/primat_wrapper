"""
primat_theory.py
----------------
Cobaya Theory class that runs the Mathematica PRIMAT code and stores
primordial nucleosynthesis abundances YHe (Yp) and DH (D/H) as derived
parameters.

Data flow
---------
  calculate() runs PRIMAT and writes:
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


class PrimatTheory(Theory):
    """
    Theory class that calls PRIMAT (via Mathematica) and provides
    primordial abundances YHe (Yp mass fraction) and DH (D/H ratio)
    as Cobaya derived parameters.
    """

    # ------------------------------------------------------------------ #
    # Declare output params at CLASS level.                               #
    # This is what Cobaya's _assign_params reads at config time to know  #
    # which component owns YHe and DH.                                   #
    # ------------------------------------------------------------------ #
    output_params = ["YHe", "DH"]

    # ------------------------------------------------------------------ #
    # Class-level attributes — overridden by YAML values automatically   #
    # ------------------------------------------------------------------ #
    PRIMAT_PATH: str = ""
    MathKernelCommand: str = ""
    ReducedNetwork: bool = True
    Nrelat: float = 0.0      # set to None to vary via sampler
    Verbose: bool = False

    # Early Dark Energy
    fEDE:  float = 0.0       # set to None to vary via sampler
    zcEDE: float = 1.0e8     # set to None to vary via sampler
    wnEDE: float = 1.0       # usually kept fixed

    # ------------------------------------------------------------------ #
    # Cobaya interface                                                     #
    # ------------------------------------------------------------------ #

    def initialize(self):
        """Resolve paths and sanity-check the PRIMAT installation."""

        base = os.path.dirname(os.path.abspath(__file__))
        if not self.PRIMAT_PATH:
            self.PRIMAT_PATH = os.path.join(base, "PRIMAT2024")
        elif not os.path.isabs(self.PRIMAT_PATH):
            self.PRIMAT_PATH = os.path.join(base, self.PRIMAT_PATH)

        self.primat_script = os.path.join(
            self.PRIMAT_PATH, "PythonInterface", "PyPRIMAT_FinalAbundances.m"
        )
        if not os.path.exists(self.primat_script):
            raise FileNotFoundError(
                f"PRIMAT script not found at {self.primat_script}"
            )

        if not self.MathKernelCommand:
            candidates = [
                "/Applications/Wolfram.app/Contents/MacOS/MathKernel",
                "/Applications/Mathematica.app/Contents/MacOS/MathKernel",
                "/usr/local/Wolfram/Mathematica/12.0/Executables/MathKernel",
                "MathKernel",
                "math13",
            ]
            for path in candidates:
                if path == "MathKernel" or os.path.exists(path):
                    self.MathKernelCommand = path
                    break
            if not self.MathKernelCommand:
                raise EnvironmentError(
                    "MathKernel not found. Set MathKernelCommand in PrimatTheory.yaml."
                )

        self.log.info("PrimatTheory initialised.")
        self.log.info(f"  PRIMAT script : {self.primat_script}")
        self.log.info(f"  MathKernel    : {self.MathKernelCommand}")

    def get_requirements(self):
        """
        Declare which sampler parameters this theory reads.
        omegabh2 is always required.  EDE / Nrelat are only requested
        when they are set to None in the YAML (i.e. varied by the sampler).
        """
        reqs = {"omegabh2": None}
        if self.fEDE is None:
            reqs["fEDE"] = None
        if self.zcEDE is None:
            reqs["zcEDE"] = None
        if self.Nrelat is None:
            reqs["Nrelat"] = None
        return reqs

    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Run PRIMAT for the current parameter point and store abundances
        in state["derived"] so Cobaya routes them to the likelihood and
        writes them to the chain.
        """
        omegabh2 = self.provider.get_param("omegabh2")

        fEDE   = self.fEDE   if self.fEDE   is not None else self.provider.get_param("fEDE")
        zcEDE  = self.zcEDE  if self.zcEDE  is not None else self.provider.get_param("zcEDE")
        Nrelat = self.Nrelat if self.Nrelat is not None else self.provider.get_param("Nrelat")
        wnEDE  = self.wnEDE if self.wnEDE is not None else self.provider.get_param("wnEDE")

        results = self._run_primat(omegabh2, Nrelat=Nrelat,
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
                f"Could not extract YHe or DH from PRIMAT output. "
                f"Available keys: {list(results.keys())}"
            )
            state["derived"] = {"YHe": np.nan, "DH": np.nan}
            return False

        state["derived"] = {"YHe": YHe, "DH": DH}

        if self.Verbose:
            self.log.info(
                f"PRIMAT result: omegabh2={omegabh2:.8f} → YHe={YHe:.8f}  D/H={DH:.6e}"
            )

        return True

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_mathematica(value):
        """Format a Python value as a Mathematica literal."""
        if isinstance(value, bool):
            return "True" if value else "False"
        return str(value)

    def _run_primat(self, omegabh2, Nrelat=0.0,
                    fEDE=0.0, zcEDE=1e8, wnEDE=1.0):
        """
        Invoke PRIMAT via MathKernel and return a dict of abundances,
        or None on failure.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            output_file = tmp.name

        try:
            use_EDE = fEDE > 0.0

            args = {
                "$ReducedNetwork":       self.ReducedNetwork,
                r"h2\[CapitalOmega]b0":  omegabh2,
                "Nrelat":                Nrelat,
                "$Verbose":              self.Verbose,
                "$EDEBool":              use_EDE,
            }
            if use_EDE:
                args["fEDE"]  = fEDE
                args["zcEDE"] = zcEDE
                args["wnEDE"] = wnEDE

            extra = "; ".join(
                f"{k}={self._to_mathematica(v)}" for k, v in args.items()
            ) + ";"
            option_output = f'$Outputfile="{output_file}"'

            if self.Verbose:
                self.log.info(f"PRIMAT args : {extra}")
                self.log.info(f"Output file : {output_file}")

            original_dir = os.getcwd()
            os.chdir(os.path.dirname(self.primat_script))
            try:
                proc = subprocess.run(
                    [
                        self.MathKernelCommand,
                        "-initfile", os.path.basename(self.primat_script),
                        extra,
                        option_output,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
            finally:
                os.chdir(original_dir)

            if self.Verbose:
                self.log.info(f"MathKernel exit code: {proc.returncode}")
                for line in (proc.stdout or "").splitlines()[:50]:
                    if line.strip():
                        self.log.info(f"  stdout: {line}")
                if proc.stderr:
                    self.log.info(f"  stderr: {proc.stderr}")

            if proc.returncode != 0:
                self.log.error(
                    f"PRIMAT failed (exit {proc.returncode}): {proc.stderr}"
                )
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

    @staticmethod
    def _extract_YHe(results):
        """Return Yp (He-4 mass fraction) from PRIMAT output dict."""
        for key in ["YP", "Yp", "YHe4", "Y_p", "YHe", "yp"]:
            if key in results:
                return results[key]
        return None

    @staticmethod
    def _extract_DH(results):
        """Return D/H ratio from PRIMAT output dict."""
        for key in ["D/H", "DH", "YD/YH", "D_H"]:
            if key in results:
                return results[key]
        YD = next((results[k] for k in ["YD", "Y_D", "D", "yD"] if k in results), None)
        YH = next((results[k] for k in ["YH", "Y_H", "H", "yH", "YH1", "Y_H1"] if k in results), None)
        if YD is not None and YH and YH > 0:
            return YD / YH
        return None
