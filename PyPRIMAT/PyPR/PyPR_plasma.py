# -*- coding: utf-8 -*-
"""
PyPR_plasma.py
==============
Thermodynamic functions for the SM plasma and neutrino bath used in PyPRIMAT.

Design
------
* A module-level ``_cfg`` is set once by ``initialise(cfg)`` called from
  ``PyPR_main.PyPRclass.__init__``.  This avoids passing cfg through every
  inner-loop call while keeping the design singleton-free across independent
  runs (each ``initialise`` replaces ``_cfg``).
* QED / neutrino-interaction tables are loaded from disk inside
  ``initialise``.
* Numba JIT is applied to the integrand kernels if available.
"""

import os
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

# Module-level state
_cfg = None
_rho_e_int_impl  = None
_drho_e_dT_impl  = None
_p_e_int_impl    = None

# QED pressure interpolants (set by _load_tables)
PofT   = None
dPdT   = None
d2PdT2 = None


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def initialise(cfg):
    """Bind to cfg, load tables, set up JIT kernels."""
    global _cfg
    _cfg = cfg
    _load_tables(cfg)
    _setup_integrand_impls(cfg)
    if cfg.verbose_flag:
        print("[init]  Tables loaded.")


def _load_tables(cfg):
    global PofT, dPdT, d2PdT2
    td = os.path.join(cfg.working_dir, "Rates", "plasma", "")

    t = np.loadtxt(td + "QED_P_int.txt")
    PofT = interp1d(t[:, 0], t[:, 1] + t[:, 2], bounds_error=False,
                    fill_value="extrapolate", assume_sorted=False, kind='linear')
    t = np.loadtxt(td + "QED_dP_intdT.txt")
    dPdT = interp1d(t[:, 0], t[:, 1] + t[:, 2], bounds_error=False,
                    fill_value="extrapolate", assume_sorted=False, kind='linear')
    t = np.loadtxt(td + "QED_d2P_intdT2.txt")
    d2PdT2 = interp1d(t[:, 0], t[:, 1] + t[:, 2], bounds_error=False,
                      fill_value="extrapolate", assume_sorted=False, kind='linear')


def _setup_integrand_impls(cfg):
    global _rho_e_int_impl, _drho_e_dT_impl, _p_e_int_impl, _dp_e_dT_impl
    me_val = cfg.me

    if cfg.numba_flag:
        try:
            from numba import njit

            @njit
            def _ri(E, Tg):
                return E**2 * (E**2 - (me_val / Tg)**2)**0.5 / (np.exp(E) + 1.)

            @njit
            def _di(E, Tg):
                return E**3 * (E**2 - (me_val / Tg)**2)**0.5 / np.cosh(E / 2.0)**2

            @njit
            def _pi(E, Tg):
                return (E**2 - (me_val / Tg)**2)**1.5 / (np.exp(E) + 1.)
            
            @njit
            def _qi(E, Tg):
                return E*(E**2 - (me_val / Tg)**2)**1.5 / np.cosh(E / 2.0)**2

            _rho_e_int_impl = _ri
            _drho_e_dT_impl = _di
            _p_e_int_impl   = _pi
            _dp_e_dT_impl   = _qi   
            return
        except ImportError:
            pass

    def _ri(E, Tg):
        return E**2 * (E**2 - (me_val / Tg)**2)**0.5 / (np.exp(E) + 1.)

    def _di(E, Tg):
        return E**3 * (E**2 - (me_val / Tg)**2)**0.5 / np.cosh(E / 2.0)**2

    def _pi(E, Tg):
        return (E**2 - (me_val / Tg)**2)**1.5 / (np.exp(E) + 1.)

    def _qi(E, Tg):
        return E*(E**2 - (me_val / Tg)**2)**1.5 / np.cosh(E / 2.0)**2

    _rho_e_int_impl = _ri
    _drho_e_dT_impl = _di
    _p_e_int_impl   = _pi
    _dp_e_dT_impl   = _qi   


# ---------------------------------------------------------------------------
# Photons
# ---------------------------------------------------------------------------

def rho_g(Tg):
    return 2. * (np.pi**2 / 30.) * Tg**4

def drho_g_dT(Tg):
    return 4. * rho_g(Tg) / Tg


# ---------------------------------------------------------------------------
# e±
# ---------------------------------------------------------------------------

def rho_e(Tg):
    me = _cfg.me
    if Tg < me / 30.:
        return 0.0
    r = quad(_rho_e_int_impl, me / Tg, 100., args=(Tg,),
             epsabs=1e-12, epsrel=1e-12)[0]
    return 4. / (2 * np.pi**2) * Tg**4 * r

def drho_e_dT(Tg):
    me = _cfg.me
    if Tg < me / 30.:
        return 0.0
    r = quad(_drho_e_dT_impl, me / Tg, 100., args=(Tg,),
             epsabs=1e-12, epsrel=1e-12)[0]
    return 1. / (2 * np.pi**2) * Tg**3 * r

def p_e(Tg):
    me = _cfg.me
    if Tg < me / 30.:
        return 0.0
    r = quad(_p_e_int_impl, me / Tg, 100., args=(Tg,),
             epsabs=1e-12, epsrel=1e-12)[0]
    return 4. / (6 * np.pi**2) * Tg**4 * r

def dp_e_dT(Tg):
    me = _cfg.me
    if Tg < me / 30.:
        return 0.0
    r = quad(_dp_e_dT_impl, me / Tg, 100., args=(Tg,),
             epsabs=1e-12, epsrel=1e-12)[0]
    return 1. / (6 * np.pi**2) * Tg**3 * r

# ---------------------------------------------------------------------------
# Neutrinos
# ---------------------------------------------------------------------------

def rho_nu(Tnu):
    """Energy density of one SM neutrino species (particle + antiparticle)."""
    return 2. * (7. / 8.) * (np.pi**2 / 30.) * Tnu**4

def drho_nu_dT(Tnu):
    return 4. * rho_nu(Tnu) / Tnu

# spl(T→∞)/T³: photons (g=2) + e+e- (g=4, fermion factor 7/8)
_sigma_inf = 11. * np.pi**2 / 45.

def T_nu_decoupling(Tg):
    """Neutrino temperature in the instantaneous-decoupling limit.

    Derived from entropy conservation: a³ spl(T_γ) = const and a T_ν = const,
    normalised so that T_ν → T_γ at T_γ >> m_e.
    """
    return Tg * (spl(Tg) / (_sigma_inf * Tg**3))**(1. / 3.)

def rho_nu_extra(Tg):
    """Energy density of DeltaNeff extra decoupled relativistic species.

    Uses the instantaneous-decoupling temperature rather than the NEVO
    SM neutrino temperature, since the extra species are fully decoupled
    and their temperature scales as 1/a exactly.
    """
    if _cfg.DeltaNeff == 0.:
        return 0.
    Tnu_dec = T_nu_decoupling(Tg)
    return _cfg.DeltaNeff * 2. * (7. / 8.) * (np.pi**2 / 30.) * Tnu_dec**4


# ---------------------------------------------------------------------------
# SM totals
# ---------------------------------------------------------------------------

def rho_SM(Tg, Tnue, Tnumu):
    return (rho_g(Tg) + rho_e(Tg) + Tg * dPdT(Tg) - PofT(Tg)
            + rho_nu(Tnue) + 2. * rho_nu(Tnumu)
            + rho_nu_extra(Tg))

def p_SM(Tg, Tnue, Tnumu):
    return (rho_g(Tg) / 3. + p_e(Tg) + PofT(Tg)
            + (rho_nu(Tnue) + 2. * rho_nu(Tnumu)) / 3.
            + rho_nu_extra(Tg) / 3.)


# ---------------------------------------------------------------------------
# Plasma entropy
# ---------------------------------------------------------------------------

def spl(Tg):
    rho_pl = rho_g(Tg) + rho_e(Tg)
    p_pl   = rho_g(Tg) / 3. + p_e(Tg)
    dQED   = Tg * dPdT(Tg) - PofT(Tg)
    dQEDp  = PofT(Tg)
    return (rho_pl + p_pl + dQED + dQEDp) / Tg

#Rewrite so as to return both because some functions are computed twice in the code, and this way we can save some time by not computing them twice.
def dspl_dT(Tg):
    rho_pl = rho_g(Tg) + rho_e(Tg)
    p_pl   = rho_g(Tg) / 3. + p_e(Tg)
    dQED   = Tg * dPdT(Tg) - PofT(Tg)
    dQEDp  = PofT(Tg)
    drho_pl_dT = drho_g_dT(Tg) + drho_e_dT(Tg)
    dp_pl_dT   = drho_g_dT(Tg) / 3. + dp_e_dT(Tg)
    ddQED_dT   = Tg * d2PdT2(Tg)
    ddQEDp_dT  = dPdT(Tg)
    ds_dT      = (drho_pl_dT + dp_pl_dT + ddQED_dT + ddQEDp_dT) / Tg - (rho_pl + p_pl + dQED + dQEDp) / Tg**2
    return ds_dT

def spl_and_dspl_dT(Tg):
    """Compute spl and dspl_dT together, sharing intermediate quantities."""
    rho_g_val  = rho_g(Tg)
    rho_e_val  = rho_e(Tg)
    p_e_val    = p_e(Tg)
    PofT_val   = PofT(Tg)
    dPdT_val   = dPdT(Tg)
    d2PdT2_val = d2PdT2(Tg)
    rho_pl = rho_g_val + rho_e_val
    p_pl   = rho_g_val / 3. + p_e_val
    dQED   = Tg * dPdT_val - PofT_val
    dQEDp  = PofT_val
    s      = (rho_pl + p_pl + dQED + dQEDp) / Tg
    drho_pl_dT = drho_g_dT(Tg) + drho_e_dT(Tg)
    dp_pl_dT   = drho_g_dT(Tg) / 3. + dp_e_dT(Tg)
    ddQED_dT   = Tg * d2PdT2_val
    ddQEDp_dT  = dPdT_val
    ds_dT      = (drho_pl_dT + dp_pl_dT + ddQED_dT + ddQEDp_dT) / Tg - s / Tg
    return s, ds_dT