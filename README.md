# primat_wrapper

A [Cobaya](https://cobaya.readthedocs.io) theory+likelihood package that computes
primordial nucleosynthesis abundances by calling either the Mathematica
[PRIMAT](https://www2.iap.fr/users/pitrou/primat.htm) code or the pure-Python
PyPRIMAT code, and evaluates a Gaussian likelihood against observed He-4 and D/H
measurements.

## Components

| File | Cobaya class | Role |
|---|---|---|
| `primat_wrapper/primat_theory.py` | `PrimatTheory` | Runs PRIMAT or PyPRIMAT, provides `YHe` and `DH` as derived parameters |
| `primat_wrapper/primat_likelihood.py` | `PrimatLikelihood` | Gaussian log-likelihood from `YHe` and `DH` |
| `primat_wrapper/PrimatTheory.yaml` | — | Cobaya `params` defaults for `PrimatTheory` |
| `primat_wrapper/PrimatLikelihood.yaml` | — | Cobaya `params` defaults for `PrimatLikelihood` |
| `yaml/run_bbn.yaml` | — | Example run: BBN, baryons only |
| `yaml/run_bbn_Nrelat.yaml` | — | Example run: BBN, baryons and DeltaNeff = Neff − 3.044 |

## Requirements

- Python ≥ 3.10
- [Cobaya](https://cobaya.readthedocs.io) ≥ 3.5
- [PyPRIMAT](https://github.com/CyrilPitrou/pyprimat_private) (pure-Python BBN solver, default)
- Optionally: Mathematica / Wolfram Engine with a working kernel command
  (`MathKernel`, `math13`, etc.) and the
  [PRIMAT](https://www2.iap.fr/users/pitrou/primat.htm) Mathematica code

## Installation

### 1. Arrange the repositories as siblings

Clone all three repositories into the same parent directory:

```bash
cd somewhere/
git clone git@github.com:CyrilPitrou/primat.git          PRIMAT
git clone git@github.com:CyrilPitrou/pyprimat_private.git PyPRIMAT
git clone git@github.com:CyrilPitrou/primat_tools.git   primat_tools
```

The layout should look like this:

```
somewhere/
├── PRIMAT/         ← Mathematica BBN code
├── PyPRIMAT/       ← Python BBN code
└── primat_tools/   ← this repository
    ├── pyproject.toml
    ├── README.md
    ├── primat_wrapper/
    └── yaml/
```

The three repos are **independent**; none lives inside another.

### 2. Install PyPRIMAT as a Python package

```bash
pip install -e ../PyPRIMAT
```

This makes `pypr` importable system-wide — no path configuration needed for
the Python solver.

### 3. Install this wrapper

```bash
pip install -e .
```

Editable mode (`-e`) is required so that the YAML files inside
`primat_wrapper/` are found correctly by Cobaya via `importlib.resources`.

On macOS with a Homebrew-managed Python, add `--break-system-packages` to both
`pip install` commands above.

### 4. Tell the wrapper where to find the Mathematica PRIMAT code

This step is only needed if you intend to use `BBN_solver: "PRIMAT"`. Add the
following line to your shell profile (`~/.zshrc`, `~/.bashrc`, or equivalent):

```bash
export PRIMAT_DIR=/absolute/path/to/PRIMAT
```

Then reload your shell (`source ~/.zshrc`) or open a new terminal.

The wrapper resolves the PRIMAT directory in this order:
1. Explicit `PRIMAT_PATH` key in your Cobaya YAML (highest priority)
2. `$PRIMAT_DIR` environment variable
3. `../PRIMAT` relative to the `primat_tools` repo root (last resort)

### 5. Verify the installation

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
    BBN_solver: "PyPRIMAT"    # default; use "PRIMAT" for the Mathematica solver
    MathKernelCommand: ""     # leave empty for auto-detection (PRIMAT only)
    Verbose: False

likelihood:
  primat_wrapper.primat_likelihood.PrimatLikelihood:
```

Then run with Cobaya using one of the example YAML files in the `yaml/` folder:

```bash
cobaya-run yaml/run_bbn.yaml
# or, to also vary the effective number of relativistic degrees of freedom:
cobaya-run yaml/run_bbn_Nrelat.yaml
```

The `yaml/` folder contains several ready-to-use Cobaya run files. Each file
specifies the sampled parameters, priors, and likelihoods for a particular
analysis. Copy and modify one to suit your needs.

### BBN solver selection

`PrimatTheory` supports two BBN solvers, controlled by the `BBN_solver` option:

- **`PyPRIMAT`** (default): calls the pure-Python PyPRIMAT implementation.
  No Mathematica needed; requires PyPRIMAT to be installed (see above).
- **`PRIMAT`**: calls the Mathematica PRIMAT code via MathKernel.
  Requires a Mathematica / Wolfram Engine installation and the PRIMAT code.
  Falls back to `PyPRIMAT` automatically if no valid MathKernel is found.

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
| `omegabh2` | — | **Required input**: physical baryon density Ωb h² (must be provided by the sampler or set as a fixed parameter) |
| `BBN_solver` | `"PyPRIMAT"` | BBN solver to use: `"PyPRIMAT"` or `"PRIMAT"` |
| `PRIMAT_PATH` | auto | Path to PRIMAT directory; overrides `$PRIMAT_DIR` and `../PRIMAT` |
| `PyPRIMAT_PATH` | auto | Fallback path to PyPRIMAT directory if not installed via pip |
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
