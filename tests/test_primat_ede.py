#!/usr/bin/env python
"""
Single-point smoke test of the wrapper with Early Dark Energy (EDE).

Builds one Cobaya model in which fEDE and zcEDE are free inputs, then evaluates
it twice -- once with an EDE component and once without (fEDE=0) -- to confirm
that EDE feeds through PrimatTheory and shifts the predicted abundances.

Usage:
    python pythontest/test_primat_ede.py
"""

import numpy as np
from cobaya.model import get_model

BBN_SOLVER = "PyPRIMAT"   # runs without Mathematica
OMEGABH2 = 0.022425

config = {
    "debug": False,
    "theory": {
        "primat_cobaya.primat_theory.PrimatTheory": {
            "BBN_solver": BBN_SOLVER,
            "ReducedNetwork": False,
            "DeltaNeff": 0.0,
            "wnEDE": 1.0,
            # null => read from the sampler/provider (declared in params below)
            "fEDE": None,
            "zcEDE": None,
            "Verbose": False,
        }
    },
    "likelihood": {
        "primat_cobaya.primat_likelihood.PrimatLikelihood": {
            "YHe_mean": 0.2458,
            "YHe_sigma": 0.0013,
            "DH_mean": 2.527e-5,
            "DH_sigma": 0.030e-5,
        }
    },
    "params": {
        "omegabh2": OMEGABH2,
        # Free inputs evaluated explicitly below (uniform priors => flat,
        # so the prior term cancels when comparing the two points).
        "fEDE": {"prior": {"min": 0.0, "max": 0.3}, "latex": "f_{\\rm EDE}"},
        "zcEDE": {"prior": {"min": 1.0e6, "max": 1.0e9}, "latex": "z_c^{\\rm EDE}"},
        "YHe": {"latex": "Y_p"},
        "DH": {"latex": "({\\rm D/H})"},
    },
}


def evaluate(model, label, point):
    result = model.logposterior(point)
    derived = dict(zip(model.parameterization.derived_params(), result.derived))
    loglike = sum(result.loglikes)
    print(f"{label}")
    print(f"  fEDE={point['fEDE']:g}  zcEDE={point['zcEDE']:g}")
    print(f"  YHe = {derived.get('YHe', float('nan')):.8f}"
          f"   D/H = {derived.get('DH', float('nan')):.6e}"
          f"   loglike = {loglike:.4f}")
    return derived, loglike


def test_ede():
    print("=" * 70)
    print(f"EDE smoke test  (solver={BBN_SOLVER}, omegabh2={OMEGABH2})")
    print("=" * 70)

    model = get_model(config)

    # zcEDE chosen near the BBN epoch (z ~ 1e9) so EDE actually boosts the
    # expansion rate while the abundances are being set.
    d_ede, ll_ede = evaluate(model, "With EDE:", {"fEDE": 0.08, "zcEDE": 1e9})
    d_std, ll_std = evaluate(model, "No EDE:  ", {"fEDE": 0.0, "zcEDE": 1e8})

    dYHe = d_ede["YHe"] - d_std["YHe"]
    print(f"\n  ΔYHe (EDE - no EDE) = {dYHe:+.6e}")
    print(f"  Δloglike            = {ll_ede - ll_std:+.4f}")

    # EDE raises the expansion rate during BBN, which must change Yp.
    assert np.isfinite(ll_ede) and np.isfinite(ll_std), "non-finite log-likelihood"
    assert abs(dYHe) > 1e-6, f"EDE produced no abundance shift: ΔYHe={dYHe:+.2e}"


if __name__ == "__main__":
    raise SystemExit(0 if test_ede() else 1)
