# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A [Cobaya](https://cobaya.readthedocs.io) theory+likelihood package that wraps two BBN (Big Bang Nucleosynthesis) solvers — the Mathematica-based PRIMAT code and the pure-Python PyPRIMAT fallback — and evaluates a Gaussian likelihood against observed He-4 and D/H measurements.

## Installation

```bash
pip install -e .
# or on macOS with Homebrew-managed Python:
pip install -e . --break-system-packages
```

## Testing after changes

After any modification to the Cobaya wrapper (`primat_theory.py`, `primat_likelihood.py`, or the YAML defaults), run both integration tests:

```bash
# Test the PRIMAT (Mathematica MathKernel) path
cobaya-run yaml/testmma.yaml

# Test the PyPRIMAT (pure-Python) path
cobaya-run yaml/testpy.yaml
```

Each test passes once Cobaya has started sampling and begun writing chain files (a few seconds). You can interrupt with Ctrl-C at that point — full convergence is not required.

## Running

# Vary DeltaNeff (Neff - 3.044)
cobaya-run yaml/run_bbn_Nrelat.yaml

# BBN + EDE parameters
cobaya-run yaml/run_bbn_EDE.yaml

# Single-point test (no MCMC)
python pythontest/test_primat_point.py
python pythontest/test_primat_ede.py
```

PyPRIMAT has its own test suite (run from the `PyPRIMAT/` directory):

```bash
cd PyPRIMAT && pytest Tests/
# Single test file:
cd PyPRIMAT && pytest Tests/test_regression.py
```

## Architecture

```
primat_theory.py      PrimatTheory   Cobaya Theory: runs PRIMAT or PyPRIMAT, writes YHe/DH to state["derived"]
primat_likelihood.py  PrimatLikelihood  Cobaya Likelihood: Gaussian logp over YHe and DH
PrimatTheory.yaml     LaTeX labels for derived params (not numerical defaults)
PrimatLikelihood.yaml LaTeX labels for derived params (not numerical defaults)
yaml/                 Ready-to-use Cobaya run YAML files
PRIMAT/               Mathematica PRIMAT2024 code (not in repo, user-supplied)
PyPRIMAT/             Pure-Python BBN solver (fallback)
pythontest/           Quick single-point smoke tests using cobaya.model.get_model
```

### Data flow

1. Cobaya calls `PrimatTheory.calculate()` with the current `omegabh2` (and optionally `DeltaNeff`, `fEDE`, `zcEDE`, `wnEDE`).
2. The theory runs either PRIMAT via MathKernel subprocess (writing a temp CSV) or calls `PyPRIMAT.PyPRIMAT_FinalAbundances.compute_abundances()` directly.
3. Results are stored in `state["derived"] = {"YHe": ..., "DH": ...}`.
4. Cobaya injects `YHe` and `DH` as keyword arguments into `PrimatLikelihood.logp()`.
5. The likelihood returns the sum of two Gaussian log-PDFs (obs uncertainty ⊕ theoretical uncertainty in quadrature).

### BBN solver selection

`BBN_solver` in the YAML controls which solver is used. If `"PRIMAT"` is requested but no valid MathKernel is found (checked at `initialize()` via `subprocess.run([cmd, "-version"])`), it automatically falls back to `"PyPRIMAT"`.

### Parameter conventions

- Parameters set to a float in YAML are held fixed; set to `null` to have the sampler vary them.
- `DeltaNeff` = extra relativistic species beyond SM neutrinos (Neff − 3.044).
- `fEDE` = Early Dark Energy fraction at peak; EDE changes only the Hubble rate entering the BBN equations, not the nuclear rates.

### PyPRIMAT internals

See `PyPRIMAT/CLAUDE.md` for a full description. The entry point used by `PrimatTheory` is `PyPRIMAT/PyPRIMAT_FinalAbundances.py::compute_abundances()`, which returns a dict containing `YPBBN` (He-4 mass fraction) and `DoHx1e5` (D/H × 10⁵).
