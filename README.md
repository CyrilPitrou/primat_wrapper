# primat_wrapper

A [Cobaya](https://cobaya.readthedocs.io) theory+likelihood package that computes
primordial nucleosynthesis abundances by calling either the Mathematica
[PRIMAT](https://www2.iap.fr/users/pitrou/primat.htm) code or the pure-Python
PyPRIMAT code, and evaluates a Gaussian likelihood against observed He-4 and D/H
measurements.

## Components

| File | Cobaya class | Role |
|---|---|---|
| `primat_theory.py` | `PrimatTheory` | Runs PRIMAT or PyPRIMAT, provides `YHe` and `DH` as derived parameters |
| `primat_likelihood.py` | `PrimatLikelihood` | Gaussian log-likelihood from `YHe` and `DH` |
| `PrimatTheory.yaml` | — | Cobaya `params` defaults for `PrimatTheory` |
| `PrimatLikelihood.yaml` | — | Cobaya `params` defaults for `PrimatLikelihood` |
| `yaml/run_bbn.yaml` | — | Example run: BBN, baryons only |
| `yaml/run_bbn_DeltaNeff.yaml` | — | Example run: BBN, baryons and DeltaNeff = Neff − 3.044 |

## Requirements

- Python ≥ 3.10
- [Cobaya](https://cobaya.readthedocs.io) ≥ 3.5
- Either:
  - Mathematica / Wolfram Engine with a working kernel command
    (`MathKernel`, `math13`, etc.) and the
    [PRIMAT2024](https://www2.iap.fr/users/pitrou/primat.htm) code; or
  - The pure-Python PyPRIMAT code (used automatically as a fallback if
    no MathKernel is found)

## Installation

### 1. Clone or download the package

```bash
git clone https://your-repo/primat_wrapper.git
# or just place the primat_wrapper/ folder wherever you like
```

The directory structure should look like this:

```
primat_wrapper/
├── __init__.py
├── setup.py
├── primat_theory.py
├── primat_likelihood.py
├── PrimatTheory.yaml
├── PrimatLikelihood.yaml
├── yaml/
│   ├── run_bbn.yaml              ← BBN only (baryons)
│   ├── run_bbn_DeltaNeff.yaml       ← BBN + varying DeltaNeff
│   └── ...                       ← other example run files
├── PRIMAT/                        ← PRIMAT code goes here
│   └── PythonInterface/
│       └── PyPRIMAT_FinalAbundances.m
├── PyPRIMAT/                      ← PyPRIMAT code goes here (fallback)
│   └── PyPRIMAT_FinalAbundances.py
└── README.md
```

### 2. Install with pip (editable mode)

From **inside** the `primat_wrapper/` directory:

```bash
cd primat_wrapper/
pip install -e .
```

The `-e` flag installs in *editable* mode, meaning changes to the `.py` files
take effect immediately without reinstalling.

If your Python installation is externally managed (e.g. on macOS with Homebrew),
add `--break-system-packages`:

```bash
pip install -e . --break-system-packages
```

### 3. Verify the installation

```bash
python3 -c "
import inspect, pprint
from primat_wrapper.primat_theory import PrimatTheory
print('package :', inspect.getmodule(PrimatTheory).__package__)
pprint.pprint(PrimatTheory.get_defaults())
"
```

You should see `package: primat_wrapper` and a non-empty defaults dict.

## Running

Reference the classes using their fully-qualified module names:

```yaml
theory:
  primat_wrapper.primat_theory.PrimatTheory:
    BBN_solver: "PRIMAT"      # or "PyPRIMAT" to force the Python fallback
    PRIMAT_PATH: "PRIMAT"     # relative to package dir, or absolute
    MathKernelCommand: ""     # leave empty for auto-detection
    Verbose: False

likelihood:
  primat_wrapper.primat_likelihood.PrimatLikelihood:
```

Then run with Cobaya using one of the example YAML files in the `yaml/` folder:

```bash
cobaya-run yaml/run_bbn.yaml
# or, to also vary the effective number of relativistic degrees of freedom:
cobaya-run yaml/run_bbn_DeltaNeff.yaml
```

The `yaml/` folder contains several ready-to-use Cobaya run files. Each file
specifies the sampled parameters, priors, and likelihoods for a particular
analysis. Copy and modify one to suit your needs.

### BBN solver selection

`PrimatTheory` supports two BBN solvers, controlled by the `BBN_solver` option:

- **`PRIMAT`** (default): calls the Mathematica PRIMAT code via MathKernel.
  Requires a Mathematica / Wolfram Engine installation and the PRIMAT2024 code.
- **`PyPRIMAT`**: calls a pure-Python implementation. No Mathematica needed.
  Used automatically as a fallback if no valid MathKernel is found.

### MathKernel auto-detection

If `MathKernelCommand` is left empty (the default), `PrimatTheory` tries the
following commands in order:

1. `/Applications/Wolfram.app/Contents/MacOS/MathKernel` (macOS Wolfram Engine)
2. `/Applications/Mathematica.app/Contents/MacOS/MathKernel` (macOS Mathematica)
3. `/usr/local/Wolfram/Mathematica/12.0/Executables/MathKernel` (Linux)
4. `math13` (some HPC clusters)
5. `MathKernel` (any system where it is on `PATH`)

If none is found, `PrimatTheory` automatically falls back to PyPRIMAT.

To override, set `MathKernelCommand` explicitly in your run YAML:

```yaml
theory:
  primat_wrapper.primat_theory.PrimatTheory:
    MathKernelCommand: "/path/to/your/MathKernel"
```

## Configuration reference

All options are set as class attributes in the Python files and can be overridden
in your run YAML. The `PrimatTheory.yaml` and `PrimatLikelihood.yaml` files are
used only to provide Cobaya with LaTeX labels for the derived parameters `YHe`
and `DH`; all numerical defaults live in the Python source.

### PrimatTheory options

| Parameter | Default | Description |
|---|---|---|
| `BBN_solver` | `"PRIMAT"` | BBN solver to use: `"PRIMAT"` or `"PyPRIMAT"` |
| `PRIMAT_PATH` | `"PRIMAT"` | Path to PRIMAT directory (relative or absolute) |
| `PyPRIMAT_PATH` | `"PyPRIMAT"` | Path to PyPRIMAT directory (relative or absolute) |
| `MathKernelCommand` | `""` | Mathematica kernel command (auto-detected if empty) |
| `ReducedNetwork` | `True` | Use reduced nuclear network (faster, recommended for MCMC) |
| `DeltaNeff` | `0.0` | Extra relativistic species beyond SM neutrinos (`null` to vary) |
| `Verbose` | `False` | Log BBN inputs and outputs each step |
| `fEDE` | `0.0` | Early Dark Energy fraction (`null` to vary) |
| `zcEDE` | `1e8` | EDE critical redshift (`null` to vary) |
| `wnEDE` | `1.0` | EDE equation-of-state parameter (`null` to vary) |

### PrimatLikelihood options

| Parameter | Default | Description |
|---|---|---|
| `YHe_mean` | `0.2458` | Observed He-4 mass fraction |
| `YHe_sigma` | `0.0013` | Observational uncertainty on He-4 |
| `YHe_PRIMAT_sigma` | `0.0001091146` | PRIMAT theoretical uncertainty on He-4 |
| `DH_mean` | `2.527e-5` | Observed D/H ratio |
| `DH_sigma` | `0.030e-5` | Observational uncertainty on D/H |
| `DH_PRIMAT_sigma` | `2.754096e-7` | PRIMAT theoretical uncertainty on D/H |

Observational and theoretical uncertainties are added in quadrature.
Nuclear reaction errors are assumed not to depend on the baryon density
(approximately correct if the baryon density is not far from the standard value).
