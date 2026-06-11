#!/usr/bin/env python
"""
Single-point smoke test of the PrimatTheory + PrimatLikelihood pair.

Builds a Cobaya model from PrimatTheory (BBN solver) and PrimatLikelihood
(Gaussian on YpBBN and D/H), evaluates it at one omegabh2 value, and prints
the resulting abundances and log-likelihood. This is the quickest way to check
that the wrapper is installed and wired correctly.

Usage:
    python tests/test_primat_point.py
"""

import numpy as np
from cobaya.model import get_model

OMEGABH2 = 0.022425  # Planck-like test value; the only parameter BBN depends on

config = {
    "debug": False,
    "theory": {
        "primat_cobaya.primat_theory.PrimatTheory": {
            "ReducedNetwork": False,
            "DeltaNeff": 0.0,
            "Verbose": False,
        }
    },
    "likelihood": {
        "primat_cobaya.primat_likelihood.PrimatLikelihood": {
            "Yp_mean": 0.2458,
            "Yp_sigma": 0.0013,
            "DH_mean": 2.527e-5,
            "DH_sigma": 0.030e-5,
        }
    },
    "params": {
        "omegabh2": OMEGABH2,
        # Derived abundances produced by PrimatTheory.
        # YpBBN = BBN-convention He-4 (consumed by likelihood).
        # YHe   = CMB-convention He-4 (YPCMB; consumed by CLASS in CMB runs).
        "YpBBN": {"latex": "Y_p^{\\rm BBN}"},
        "YHe":   {"latex": "Y_p^{\\rm CMB}"},
        "DH":    {"latex": "({\\rm D/H})"},
    },
}


def test_single_point():
    print("=" * 70)
    print(f"Single-point test  (omegabh2={OMEGABH2})")
    print("=" * 70)

    model = get_model(config)
    result = model.logposterior({"omegabh2": OMEGABH2})

    derived = dict(zip(model.parameterization.derived_params(), result.derived))

    print(f"  YpBBN (BBN conv) = {derived.get('YpBBN', float('nan')):.8f}")
    print(f"  YHe   (CMB conv) = {derived.get('YHe',   float('nan')):.8f}")
    print(f"  D/H              = {derived.get('DH',    float('nan')):.6e}")
    print(f"  log-likelihood = {sum(result.loglikes):.4f}")
    print(f"  log-posterior  = {result.logpost:.4f}")

    assert np.isfinite(result.logpost), "non-finite posterior"


if __name__ == "__main__":
    raise SystemExit(0 if test_single_point() else 1)
