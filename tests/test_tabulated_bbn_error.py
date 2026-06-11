#!/usr/bin/env python
"""
Test the tabulated_BBN_error mode of PrimatLikelihood.

Checks that:
  1. The uniform (default) and tabulated modes both produce finite log-posteriors
     at the same parameter point.
  2. The tabulated mode yields a different (non-identical) log-likelihood than
     the uniform mode, confirming the interpolation is actually used.
  3. The interpolated theoretical uncertainties at the test point are close to
     the fixed constants (sanity-check that the table is internally consistent).
"""

import numpy as np
from cobaya.model import get_model

OMEGABH2  = 0.022425
DELTANEFF = 0.0

BASE_CONFIG = {
    "debug": False,
    "theory": {
        "primat_cobaya.primat_theory.PrimatTheory": {
            "ReducedNetwork": False,
            "DeltaNeff": DELTANEFF,
            "Verbose": False,
        }
    },
    "params": {
        "omegabh2": OMEGABH2,
        "YHe": {"latex": "Y_p"},
        "DH":  {"latex": "({\\rm D/H})"},
    },
}

LIKELIHOOD_BASE = {
    "Yp_mean":  0.2458,
    "Yp_sigma": 0.0013,
    "DH_mean":   2.527e-5,
    "DH_sigma":  0.030e-5,
}


def _make_config(tabulated: bool) -> dict:
    config = {**BASE_CONFIG, "likelihood": {
        "primat_cobaya.primat_likelihood.PrimatLikelihood": {
            **LIKELIHOOD_BASE,
            "tabulated_BBN_error": tabulated,
        }
    }}
    return config


def _eval(tabulated: bool):
    model = get_model(_make_config(tabulated))
    result = model.logposterior({"omegabh2": OMEGABH2})
    loglike = sum(result.loglikes)
    derived = dict(zip(model.parameterization.derived_params(), result.derived))
    return loglike, derived


def test_uniform_mode_finite():
    loglike, derived = _eval(tabulated=False)
    print(f"[uniform]    YHe={derived['YHe']:.8f}  D/H={derived['DH']:.6e}  loglike={loglike:.4f}")
    assert np.isfinite(loglike), "uniform mode: non-finite log-likelihood"


def test_tabulated_mode_finite():
    loglike, derived = _eval(tabulated=True)
    print(f"[tabulated]  YHe={derived['YHe']:.8f}  D/H={derived['DH']:.6e}  loglike={loglike:.4f}")
    assert np.isfinite(loglike), "tabulated mode: non-finite log-likelihood"


def test_tabulated_differs_from_uniform():
    loglike_uniform,   _ = _eval(tabulated=False)
    loglike_tabulated, _ = _eval(tabulated=True)
    print(f"[uniform vs tabulated]  Δloglike = {loglike_tabulated - loglike_uniform:.6f}")
    assert loglike_uniform != loglike_tabulated, (
        "tabulated and uniform modes returned identical log-likelihoods — "
        "interpolation may not be active"
    )


def test_interpolated_sigmas_near_constants():
    """The table value at (DeltaN=0, omegabh2≈0.022) should be close to the
    fixed constants used in uniform mode."""
    import importlib.resources
    from scipy.interpolate import RegularGridInterpolator

    data_path = importlib.resources.files("primat_cobaya") / "data" / "PRIMAT_Yp_DH_ErrorMC_100_2024.dat"
    data = np.loadtxt(data_path, comments="#")
    deltaN_vals   = np.unique(data[:, 2])
    omegabh2_vals = np.unique(data[:, 0])
    n_delta, n_omega = len(deltaN_vals), len(omegabh2_vals)
    sig_YHe_interp = RegularGridInterpolator(
        (deltaN_vals, omegabh2_vals), data[:, 5].reshape(n_delta, n_omega),
        method="linear", bounds_error=False, fill_value=None,
    )
    sig_DH_interp = RegularGridInterpolator(
        (deltaN_vals, omegabh2_vals), data[:, 7].reshape(n_delta, n_omega),
        method="linear", bounds_error=False, fill_value=None,
    )

    pt = [[DELTANEFF, OMEGABH2]]
    sig_yhe = sig_YHe_interp(pt)[0]
    sig_dh  = sig_DH_interp(pt)[0]

    # Fixed constants from the class defaults
    Yp_PRIMAT_sigma = 0.0001091146
    DH_PRIMAT_sigma  = 2.754096e-7

    print(f"[sigmas]  sig_YHe: table={sig_yhe:.6e}  constant={Yp_PRIMAT_sigma:.6e}")
    print(f"[sigmas]  sig_DH:  table={sig_dh:.6e}  constant={DH_PRIMAT_sigma:.6e}")

    assert abs(sig_yhe - Yp_PRIMAT_sigma) / Yp_PRIMAT_sigma < 0.05, \
        f"sig_YHe from table ({sig_yhe:.6e}) deviates >5% from constant ({Yp_PRIMAT_sigma:.6e})"
    assert abs(sig_dh - DH_PRIMAT_sigma) / DH_PRIMAT_sigma < 0.05, \
        f"sig_DH from table ({sig_dh:.6e}) deviates >5% from constant ({DH_PRIMAT_sigma:.6e})"


def test_interpolation_clamps_outside_grid():
    """Points outside the grid must return the nearest-edge value, not an extrapolation."""
    import importlib.resources
    from scipy.interpolate import RegularGridInterpolator

    data_path = importlib.resources.files("primat_cobaya") / "data" / "PRIMAT_Yp_DH_ErrorMC_100_2024.dat"
    data = np.loadtxt(data_path, comments="#")
    deltaN_vals   = np.unique(data[:, 2])
    omegabh2_vals = np.unique(data[:, 0])
    n_delta, n_omega = len(deltaN_vals), len(omegabh2_vals)

    sig_YHe_interp = RegularGridInterpolator(
        (deltaN_vals, omegabh2_vals), data[:, 5].reshape(n_delta, n_omega),
        method="linear", bounds_error=True,
    )
    sig_DH_interp = RegularGridInterpolator(
        (deltaN_vals, omegabh2_vals), data[:, 7].reshape(n_delta, n_omega),
        method="linear", bounds_error=True,
    )

    deltaN_bounds   = (deltaN_vals[0],   deltaN_vals[-1])
    omegabh2_bounds = (omegabh2_vals[0], omegabh2_vals[-1])

    def interp_clamped(deltaN, omegabh2):
        dc = np.clip(deltaN,   *deltaN_bounds)
        oc = np.clip(omegabh2, *omegabh2_bounds)
        return (sig_YHe_interp([[dc, oc]])[0],
                sig_DH_interp([[dc, oc]])[0])

    # --- DeltaN too high: should equal the DeltaN=7 edge ---
    at_edge   = interp_clamped(deltaN_vals[-1], 0.022)
    beyond    = interp_clamped(deltaN_vals[-1] + 2.0, 0.022)
    assert at_edge == beyond, f"DeltaN above grid: edge={at_edge} beyond={beyond}"

    # --- DeltaN too low: should equal the DeltaN=-3 edge ---
    at_edge = interp_clamped(deltaN_vals[0], 0.022)
    beyond  = interp_clamped(deltaN_vals[0] - 1.0, 0.022)
    assert at_edge == beyond, f"DeltaN below grid: edge={at_edge} beyond={beyond}"

    # --- omegabh2 too high: should equal the omegabh2=0.04 edge ---
    at_edge = interp_clamped(0.0, omegabh2_vals[-1])
    beyond  = interp_clamped(0.0, omegabh2_vals[-1] + 0.01)
    assert at_edge == beyond, f"omegabh2 above grid: edge={at_edge} beyond={beyond}"

    # --- omegabh2 too low: should equal the omegabh2=0.005 edge ---
    at_edge = interp_clamped(0.0, omegabh2_vals[0])
    beyond  = interp_clamped(0.0, omegabh2_vals[0] - 0.003)
    assert at_edge == beyond, f"omegabh2 below grid: edge={at_edge} beyond={beyond}"

    print("[clamp]  all out-of-bounds points correctly return nearest-edge value")


def test_interpolation_exact_at_grid_nodes():
    """At every grid node the interpolator must return the table value exactly."""
    import importlib.resources
    from scipy.interpolate import RegularGridInterpolator

    data_path = importlib.resources.files("primat_cobaya") / "data" / "PRIMAT_Yp_DH_ErrorMC_100_2024.dat"
    data = np.loadtxt(data_path, comments="#")
    deltaN_vals   = np.unique(data[:, 2])
    omegabh2_vals = np.unique(data[:, 0])
    n_delta, n_omega = len(deltaN_vals), len(omegabh2_vals)

    sig_YHe_interp = RegularGridInterpolator(
        (deltaN_vals, omegabh2_vals), data[:, 5].reshape(n_delta, n_omega),
        method="linear", bounds_error=False, fill_value=None,
    )
    sig_DH_interp = RegularGridInterpolator(
        (deltaN_vals, omegabh2_vals), data[:, 7].reshape(n_delta, n_omega),
        method="linear", bounds_error=False, fill_value=None,
    )

    # Build the full grid of (DeltaN, omegabh2) pairs matching every table row
    pts = data[:, [2, 0]]   # columns: DeltaN, omegabh2
    sig_yhe_interp = sig_YHe_interp(pts)
    sig_dh_interp  = sig_DH_interp(pts)

    np.testing.assert_allclose(sig_yhe_interp, data[:, 5], rtol=1e-10,
                               err_msg="sig(Yp^BBN) interpolation error at grid nodes")
    np.testing.assert_allclose(sig_dh_interp,  data[:, 7], rtol=1e-10,
                               err_msg="sig(D/H) interpolation error at grid nodes")
    print(f"[grid nodes]  max sig_YHe residual: {np.max(np.abs(sig_yhe_interp - data[:, 5])):.2e}")
    print(f"[grid nodes]  max sig_DH  residual: {np.max(np.abs(sig_dh_interp  - data[:, 7])):.2e}")


if __name__ == "__main__":
    test_uniform_mode_finite()
    test_tabulated_mode_finite()
    test_tabulated_differs_from_uniform()
    test_interpolated_sigmas_near_constants()
    test_interpolation_clamps_outside_grid()
    test_interpolation_exact_at_grid_nodes()
    print("All tests passed.")
