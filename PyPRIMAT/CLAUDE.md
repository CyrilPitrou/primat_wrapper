# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PyPRIMAT is a Python implementation of the PRIMAT package for precise Big Bang Nucleosynthesis (BBN) computations. It integrates coupled ODEs for the cosmological background (photon/neutrino temperatures, scale factor) and a nuclear reaction network to predict primordial abundances of H, D, He3, He4, Li7, and heavier nuclides.

## Running

Scripts live in `RunFiles/`. Run from the repo root so that `Rates/` data files resolve correctly:

```bash
python RunFiles/PyPRIMAT_run.py           # Standard SM run
python RunFiles/PyPRIMAT_EDE_run.py       # Early Dark Energy example
python RunFiles/PyPRIMAT_compare.py       # Comparison plots
```

The repo root must be on `sys.path` (the run scripts handle this automatically).

## Dependencies

- **numpy**, **scipy** — mandatory
- **numba** — recommended (JIT compilation for integrand kernels)
- **numdifftools** — recommended (numerical entropy derivatives; only needed if `analytic_entropy_derivative=False`)
- **vegas** — recommended (Monte Carlo integration for thermal weak rate corrections)
- **diffeqpy** — optional

## Architecture

```
PyPR/                  # Core package
  PyPR_config.py       # PyPRConfig: all physical constants + run-time flags
  PyPR_main.py         # PyPRclass: top-level driver
  PyPR_thermo.py       # Module-level thermodynamics (plasma + neutrinos)
  PyPR_nuclear_data.py # NuclearData: loads rate tables from Rates/
  PyPR_nuclear_net12.py# 12-reaction network (smallnet_flag=True)
  PyPR_nuclear_net63.py# 63-reaction network (smallnet_flag=False)
  PyPR_weak_rates.py   # n <-> p weak rate computation

Rates/
  thermo/              # QED pressure tables, neutrino interaction rates, T_γ/T_ν
  nuclear/
    key_primat_rates/  # 12 key reactions (default rates_dir)
    other_nucl_rates/  # 51 additional reactions for full network
  weak/                # Pre-tabulated n↔p forward/backward rates (HT/MT/LT)
  NEVO/                # NEVO non-instantaneous decoupling table

RunFiles/              # Example run scripts and comparison notebooks
```

### Execution flow

1. `PyPRclass.__init__` builds a `PyPRConfig`, calls `PyPRthermo.initialise(cfg)` (loads QED/neutrino tables), creates `NuclearData(cfg)`, and computes the cosmological background + weak rates.
2. `PyPRclass.solve()` integrates the nuclear network across three temperature eras:
   - **HT** (T > T_weak ≈ 1 MeV): n/p only
   - **MT** (T_weak → T_nucl ≈ 0.1 MeV): full network, BDF solver
   - **LT** (T_nucl → T_end ≈ 0.001 MeV): full network, BDF solver
3. Returns a dict: `Neff`, `YPBBN`, `YPCMB`, `DoH`, `He3oH`, `Li7oH`, neutrino Omegas.

### Background thermodynamics

Reads the pre-computed NEVO non-instantaneous neutrino decoupling table from `Rates/NEVO/NEVOPRIMAT_col_1_7.csv`, then solves the a(T) ODE using three distinct neutrino flavour temperatures (T_νe, T_νμ, T_ντ).

### Nuclear rate variation

Each reaction rate has a corresponding `p_<reaction>` parameter (e.g. `p_npdg`, `p_Li7paa`). Setting it to a non-zero float samples the rate at `median * exp(p * expsigma)` — useful for MCMC uncertainty propagation.

### Key configuration flags

| Flag | Default | Effect |
|------|---------|--------|
| `smallnet_flag` | True | 12-reaction network; False → 63-reaction full network |
| `tau_n_flag` | True | Normalise weak rates using τ_n (neutron lifetime) |
| `output_time_evolution` | False | Write `output_time_evolution.tsv` with full time series |
| `numerical_precision` | 1e-6 | `rtol` for all `solve_ivp` calls |

### `PyPR_thermo.py` design note

The module uses module-level state initialised by `initialise(cfg)`. Multiple independent `PyPRclass` instances in the same process will share (and overwrite) this state — do not run them concurrently.

## Validation before committing

After any modification, run the standard script and check the output:

```bash
python RunFiles/PyPRIMAT_run.py
```

The following values must hold (`Omegabh2=0.022425`). References were produced by
`RunFiles/PyPRIMAT_reference_run.py` (high-precision run, `numerical_precision=1e-10`,
`n_sampling=10000`, `sampling_nTOp=500`, `T_start_cosmo=100 MeV`).

**Small network** (`smallnet_flag=True`):

| Observable | Expected | Tolerance |
|------------|----------|-----------|
| YP (BBN) | 0.2469155 | ±1e-5 |
| D/H | 2.43647e-5 | ±3e-9 |

**Large network** (`smallnet_flag=False`):

| Observable | Expected | Tolerance |
|------------|----------|-----------|
| YP (BBN) | 0.2469188 | ±1e-5 |
| D/H | 2.43718e-5 | ±3e-9 |

A result outside these bounds indicates a regression.

## Citation

Pitrou, Coc, Uzan, Vangioni, *Physics Reports*, 04 (2018) 005.
