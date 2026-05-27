# primat_wrapper

A [Cobaya](https://cobaya.readthedocs.io) theory+likelihood package that computes
primordial nucleosynthesis abundances by calling the Mathematica
[PRIMAT](https://www2.iap.fr/users/pitrou/primat.htm) code, and evaluates a
Gaussian likelihood against observed He-4 and D/H measurements.

## Components

| File | Cobaya class | Role |
|---|---|---|
| `primat_theory.py` | `PrimatTheory` | Runs PRIMAT, provides `YHe` and `DH` as derived parameters |
| `primat_likelihood.py` | `PrimatLikelihood` | Gaussian log-likelihood from `YHe` and `DH` |
| `PrimatTheory.yaml` | — | Default `params` block for `PrimatTheory` |
| `PrimatLikelihood.yaml` | — | Default `params` block for `PrimatLikelihood` |
| `run_bbn.yaml` | — | Example run: BBN only |
| `run_bbn_class.yaml` | — | Example run: BBN + CLASS (CMB) |

## Requirements

- Python ≥ 3.10
- [Cobaya](https://cobaya.readthedocs.io) ≥ 3.5
- Mathematica / Wolfram Engine with a working kernel command
  (`MathKernel`, `math13`, etc.)
- The [PRIMAT2024](https://www2.iap.fr/users/pitrou/primat.htm) code,
  placed in a folder called `PRIMAT2024/` inside the package directory
  (or pointed to by `PRIMAT_PATH` in your run YAML)

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
├── run_bbn.yaml
├── run_bbn_class.yaml
├── PRIMAT2024/          ← PRIMAT code goes here
│   └── PythonInterface/
│       └── PyPRIMAT_FinalAbundances.m
└── README.md
```

### 2. Install with pip (editable mode)

From **inside** the `primat_wrapper/` directory:

```bash
cd primat_wrapper/
pip install -e .
```

The `-e` flag installs in *editable* mode, meaning changes to the `.py` and `.yaml`
files take effect immediately without reinstalling.

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
print('package    :', inspect.getmodule(PrimatTheory).__package__)
print('yaml file  :', PrimatTheory.get_yaml_file())
print('defaults   :')
pprint.pprint(PrimatTheory.get_defaults())
"
```

You should see `package: primat_wrapper` and the `params` block from
`PrimatTheory.yaml` in the defaults.

## Running

Reference the classes using their fully-qualified module names:

```yaml
theory:
  primat_wrapper.primat_theory.PrimatTheory:
    PRIMAT_PATH: "PRIMAT2024"          # relative to package dir, or absolute
    MathKernelCommand: ""              # leave empty for auto-detection
    Verbose: False

likelihood:
  primat_wrapper.primat_likelihood.PrimatLikelihood:
```

Then run with Cobaya:

```bash
cobaya-run run_bbn.yaml
# or
cobaya-run run_bbn_class.yaml
```

### MathKernel auto-detection

If `MathKernelCommand` is left empty (the default), `PrimatTheory` tries the
following commands in order:

1. `/Applications/Wolfram.app/Contents/MacOS/MathKernel` (macOS Wolfram Engine)
2. `/Applications/Mathematica.app/Contents/MacOS/MathKernel` (macOS Mathematica)
3. `/usr/local/Wolfram/Mathematica/12.0/Executables/MathKernel` (Linux)
4. `math13` (some HPC clusters)
5. `MathKernel` (any system where it is on `PATH`)

To override, set `MathKernelCommand` explicitly in your run YAML:

```yaml
theory:
  primat_wrapper.primat_theory.PrimatTheory:
    MathKernelCommand: "/usr/local/bin/math13"
```

## Configuration reference

### PrimatTheory

| Parameter | Default | Description |
|---|---|---|
| `PRIMAT_PATH` | `"PRIMAT2024"` | Path to PRIMAT2024 directory |
| `MathKernelCommand` | `""` | Mathematica kernel command (auto-detected if empty) |
| `ReducedNetwork` | `True` | Use reduced nuclear network (faster, recommended for MCMC) |
| `Nrelat` | `0.0` | Extra relativistic species beyond SM (`null` to vary) |
| `Verbose` | `False` | Print PRIMAT inputs/outputs each step |
| `fEDE` | `0.0` | Early Dark Energy fraction (`null` to vary) |
| `zcEDE` | `1e8` | EDE critical redshift (`null` to vary) |
| `wnEDE` | `1.0` | EDE equation-of-state parameter |

### PrimatLikelihood

| Parameter | Default | Description |
|---|---|---|
| `YHe_mean` | `0.2458` | Observed He-4 mass fraction |
| `YHe_sigma` | `0.0013` | Observational uncertainty on He-4 |
| `YHe_PRIMAT_sigma` | `0.0001091146` | PRIMAT theoretical uncertainty on He-4 |
| `DH_mean` | `2.527e-5` | Observed D/H ratio |
| `DH_sigma` | `0.030e-5` | Observational uncertainty on D/H |
| `DH_PRIMAT_sigma` | `2.754096e-7` | PRIMAT theoretical uncertainty on D/H |

Observational and theoretical uncertainties are added in quadrature.
