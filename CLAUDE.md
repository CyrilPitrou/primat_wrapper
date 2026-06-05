# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A [Cobaya](https://cobaya.readthedocs.io) theory+likelihood package that wraps two BBN (Big Bang Nucleosynthesis) solvers — the Mathematica-based PRIMAT code and the pure-Python PyPRIMAT fallback — and evaluates a Gaussian likelihood against observed He-4 and D/H measurements.

## Installation

The three repos are kept **separate**. Typical layout (all siblings):

```
somewhere/
  PRIMAT/        ← Mathematica BBN code
  PyPRIMAT/      ← Python BBN code
  primat_tools/  ← this repo
```

```bash
# 1. Install PyPRIMAT as a Python package (editable)
pip install -e ../PyPRIMAT

# 2. Install this wrapper (editable)
pip install -e .
# or on macOS with Homebrew-managed Python:
pip install -e . --break-system-packages
```

Editable mode (`-e`) is required for this wrapper so that the YAML files
inside `primat_wrapper/` are found via `importlib.resources`.

### PRIMAT path discovery

The Mathematica PRIMAT code is found in this order:
1. Explicit `PRIMAT_PATH` in the Cobaya YAML
2. `$PRIMAT_DIR` environment variable (set once in your shell profile)
3. Sibling directory `../PRIMAT` relative to this repo root

### PyPRIMAT path discovery

`from pypr import PyPRclass` is tried first (works when installed via pip).
If not installed, `PyPRIMAT_PATH` in the YAML is used as a `sys.path` fallback.

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

```bash
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
primat_wrapper/                         The installable Python package
  primat_theory.py      PrimatTheory      Cobaya Theory: runs PRIMAT or PyPRIMAT, writes YHe/DH to state["derived"]
  primat_likelihood.py  PrimatLikelihood  Cobaya Likelihood: Gaussian logp over YHe and DH
  PrimatTheory.yaml     LaTeX labels for derived params (not numerical defaults)
  PrimatLikelihood.yaml LaTeX labels for derived params (not numerical defaults)
yaml/                 Ready-to-use Cobaya run YAML files
pythontest/           Quick single-point smoke tests using cobaya.model.get_model
```

`PRIMAT/` and `PyPRIMAT/` are **not inside this repo**; see installation notes above.

In Cobaya YAMLs the classes are referenced by their dotted package path,
e.g. `primat_wrapper.primat_theory.PrimatTheory`.

### Data flow

1. Cobaya calls `PrimatTheory.calculate()` with the current `omegabh2` (and optionally `DeltaNeff`, `fEDE`, `zcEDE`, `wnEDE`).
2. The theory runs either PRIMAT via MathKernel subprocess (writing a temp CSV) or calls `PyPR.PyPRclass(...).solve()` directly.
3. Results are stored in `state["derived"] = {"YHe": ..., "DH": ...}`.
4. Cobaya injects `YHe` and `DH` as keyword arguments into `PrimatLikelihood.logp()`.
5. The likelihood returns the sum of two Gaussian log-PDFs (obs uncertainty ⊕ theoretical uncertainty in quadrature).

### BBN solver selection

`BBN_solver` in the YAML controls which solver is used. The default is `"PyPRIMAT"`. If `"PRIMAT"` is requested but no valid MathKernel is found (checked at `initialize()` via `subprocess.run([cmd, "-version"])`), it automatically falls back to `"PyPRIMAT"`.

### Parameter conventions

- Parameters set to a float in YAML are held fixed; set to `null` to have the sampler vary them.
- `DeltaNeff` = extra relativistic species beyond SM neutrinos (Neff − 3.044).
- `fEDE` = Early Dark Energy fraction at peak; EDE changes only the Hubble rate entering the BBN equations, not the nuclear rates.

### PyPRIMAT internals

See `../PyPRIMAT/CLAUDE.md` for a full description. The entry point used by `PrimatTheory` is `pypr.PyPRclass(params).solve()`, which returns a dict containing `YPBBN` (He-4 mass fraction) and `DoH` (the raw D/H ratio, not scaled by 10⁵).
