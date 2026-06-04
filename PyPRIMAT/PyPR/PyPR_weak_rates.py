# -*- coding: utf-8 -*-
"""
PyPR_weak_rates.py
==================
n <--> p weak rates: computation from first principles and load/dispatch interface.

Merged from the former PyPR_nTOp.py (dispatcher) and PyPR_eval_nTOp.py (physics).
The n→p and p→n rate pairs are unified into single functions parameterised by
sgnq = +1 (n→p) or -1 (p→n).
"""

import os
import numpy as np
from scipy.special import gamma as scipy_gamma, spence
from scipy.integrate import quad
from scipy.interpolate import interp1d

exp_cutoff = 3e+2
epsrel_low = 1.e-1


# ---------------------------------------------------------------------------
# Fermi-Dirac helper functions — JIT-compiled when numba is available.
# These capture nothing from any enclosing scope (only the module-level
# exp_cutoff constant), so they can live at module level and be wrapped
# with @njit.  Call _setup_fd_impls(cfg.numba_flag) before first use.
# ---------------------------------------------------------------------------

def FD_nu3(E, phi, x):
    return 1. / (np.exp(x * E - phi) + 1.) if (x * E - phi) < exp_cutoff else 0.

def FD2(E, x):
    return 1. / (np.exp(x * E) + 1.) if (x * E) < exp_cutoff else 0.

def FD_nu_e2p0(E, phi, x):
    return E**2 / (np.exp(x * E - phi) + 1.) if (x * E - phi) < exp_cutoff else 0.

def FD_nu_e3p0(E, phi, x):
    return E**3 / (np.exp(x * E - phi) + 1.) if (x * E - phi) < exp_cutoff else 0.

def FD_nu_e4p2(E, phi, x):
    if (2. * phi < exp_cutoff) and (E * x + phi < exp_cutoff) and (2. * E * x < exp_cutoff):
        return (E**2 * np.exp(phi) * ((24. - E * x * (E * x + 8.)) * np.exp(E * x + phi)
                + np.exp(2 * E * x) * (E * x - 6.) * (E * x - 2.) + 12 * np.exp(2 * phi))
                / (np.exp(E * x) + np.exp(phi))**3)
    return 0.

def FD_nu_e2p2(E, phi, x):
    if (3. * phi < exp_cutoff) and (2 * E * x + phi < exp_cutoff) and (E * x < exp_cutoff):
        return (((E * x * (E * x - 4.) + 2.) * np.exp(2 * E * x + phi)
                 + (4. - E * x * (E * x + 4.)) * np.exp(E * x + 2 * phi)
                 + 2 * np.exp(3 * phi))
                / (np.exp(E * x) + np.exp(phi))**3)
    return 0.

def FD_nu_e4p1(E, phi, x):
    if (phi < exp_cutoff) and (E * x < exp_cutoff):
        return (np.exp(phi) * E**3 * (4 * np.exp(phi) + np.exp(E * x) * (4. - E * x))
                / (np.exp(E * x) + np.exp(phi))**2)
    return 0.

def FD_nu_e2p1(E, phi, x):
    if (phi < exp_cutoff) and (E * x < exp_cutoff):
        return (np.exp(phi) * E * (2 * np.exp(phi) + np.exp(E * x) * (2. - E * x))
                / (np.exp(E * x) + np.exp(phi))**2)
    return 0.

def FD_nu_e3p1(E, phi, x):
    if (phi < exp_cutoff) and (E * x < exp_cutoff):
        return (np.exp(phi) * E**2 * (3 * np.exp(phi) + np.exp(E * x) * (3. - E * x))
                / (np.exp(E * x) + np.exp(phi))**2)
    return 0.

def FD_nu_e3p2(E, phi, x):
    if (2. * phi < exp_cutoff) and (E * x + phi < exp_cutoff) and (2. * E * x < exp_cutoff):
        return (E * np.exp(phi)
                * ((12. - E * x * (E * x + 6.)) * np.exp(E * x + phi)
                   + np.exp(2. * E * x) * (E * x * (E * x - 6.) + 6.)
                   + 6 * np.exp(2. * phi))
                / (np.exp(E * x) + np.exp(phi))**3)
    return 0.


_fd_impls_initialized = False


def _setup_fd_impls(numba_flag):
    global FD_nu3, FD2, FD_nu_e2p0, FD_nu_e3p0, FD_nu_e4p2, FD_nu_e2p2, \
           FD_nu_e4p1, FD_nu_e2p1, FD_nu_e3p1, FD_nu_e3p2, _fd_impls_initialized
    if _fd_impls_initialized:
        return
    _fd_impls_initialized = True
    if not numba_flag:
        return
    try:
        from numba import njit
        FD_nu3      = njit(FD_nu3)
        FD2         = njit(FD2)
        FD_nu_e2p0  = njit(FD_nu_e2p0)
        FD_nu_e3p0  = njit(FD_nu_e3p0)
        FD_nu_e4p2  = njit(FD_nu_e4p2)
        FD_nu_e2p2  = njit(FD_nu_e2p2)
        FD_nu_e4p1  = njit(FD_nu_e4p1)
        FD_nu_e2p1  = njit(FD_nu_e2p1)
        FD_nu_e3p1  = njit(FD_nu_e3p1)
        FD_nu_e3p2  = njit(FD_nu_e3p2)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Coulomb + radiative correction factors
# ---------------------------------------------------------------------------

def FermiCoulomb(b, cfg):
    me      = cfg.me * cfg.MeV
    Gamma   = np.sqrt(1. - cfg.alphaem**2.) - 1.
    gamma1  = 1. + Gamma
    gamma2  = 3. + 2. * Gamma
    Fn_Comp = cfg.hbar * cfg.clight / me
    return ((1. + Gamma / 2.)
            * 4. * ((2. * cfg.radproton * b) / Fn_Comp) ** (2. * Gamma)
            / (scipy_gamma(gamma2)**2)
            * np.exp((np.pi * cfg.alphaem) / b)
            / ((1. - b**2) ** Gamma)
            * np.abs(scipy_gamma(gamma1 + (cfg.alphaem / b) * 1j))**2)


def RadCorrResum(b, y, en, cfg):
    mA        = 1.2e+3 * cfg.MeV
    Agndecay  = -0.34
    Cndecay   =  0.891
    deltand   = -0.00043
    Lndecay   =  1.02094
    Sndecay   =  1.02248
    NLLndecay = -0.0001

    me = cfg.me * cfg.MeV
    mn = cfg.mn * cfg.MeV
    mp = cfg.mp * cfg.MeV
    Q  = mn - mp

    Rd = 1. if b == 0 else np.arctanh(b) / b
    Sirlin = (3. * np.log(mp / me) - 3. / 4.
              + 4. * (Rd - 1.) * (y / (3. * en) - 3. / 2. + np.log(2. * y))
              + Rd * (2. * (1. + b**2) + y**2 / (6. * en**2) - 4. * b * Rd)
              - (4. / b) * spence(1. - (2 * b) / (1. + b)))
    return ((1. + cfg.alphaem / (2. * np.pi) * (Sirlin - 3. * np.log(mp / (2 * Q))))
            * (Lndecay + (cfg.alphaem / np.pi) * Cndecay
               + cfg.alphaem / (2 * np.pi) * deltand * 2 * np.pi / cfg.alphaem)
            * (Sndecay + 1. / (134. * 2. * np.pi) * (np.log(mp / mA) + Agndecay)
               + NLLndecay))


# ---------------------------------------------------------------------------
# Neutron-decay phase-space factor
# ---------------------------------------------------------------------------

def ComputeFn(cfg):
    """Compute the neutron-decay phase-space factor Fn."""
    me = cfg.me * cfg.MeV
    mn = cfg.mn * cfg.MeV
    mp = cfg.mp * cfg.MeV
    Q  = mn - mp

    def Fn_Born_int(E):
        if (-1. >= E) or (E >= 1):
            return E * (E - (Q / me))**2 * np.sqrt(E**2 - 1.)
        return 0.

    Fn_Born = quad(Fn_Born_int, 1., Q / me)[0]
    if cfg.nTOpBorn_flag:
        return Fn_Born

    def Fn_rad_int(e):
        b = np.sqrt(e**2 - 1.) / e
        q = Q / me
        return (e * (e - q)**2 * e * b
                * FermiCoulomb(b, cfg)
                * RadCorrResum(np.sqrt(e**2 - 1.) / e, q - e, e, cfg))

    Fn_rad = quad(Fn_rad_int, 1., Q / me)[0]

    gA         = cfg.gA
    deltakappa = cfg.deltakappa

    def ChiFMnDec(en, pe):
        f1n = ((1. + gA)**2. + 2. * deltakappa * gA) / (1. + 3. * gA**2)
        f2n = ((1. - gA)**2. - 2. * deltakappa * gA) / (1. + 3. * gA**2)
        f3n = (gA**2 - 1.) / (1. + 3. * gA**2)
        mnOme = mn / me
        return (f1n * (en - Q / me)**2 * (pe**2 / (mnOme * en))
                - f2n / mnOme * (en - Q / me)**3
                + (f1n + f2n + f3n) / (2. * mnOme) * (4. * (en - Q / me)**3 + 2 * (en - Q / me) * pe**2)
                + f3n / mnOme * (en - Q / me)**2 * pe**2 / en)

    def Fn_FM_int(pe):
        en = np.sqrt(pe**2 + 1.)
        b  = pe / en
        return (pe**2
                * ChiFMnDec(en, pe)
                * RadCorrResum(b, np.abs(en - Q / me), en, cfg)
                * FermiCoulomb(b, cfg))

    Fn_FM = quad(Fn_FM_int, 0., np.sqrt((Q / me)**2 - 1.))[0]
    return Fn_rad + Fn_FM


# ---------------------------------------------------------------------------
# Main rate computation
# ---------------------------------------------------------------------------

def ComputeWeakRates(Tvec, cfg):
    """
    Compute n<->p weak rates over the three temperature eras.

    Parameters
    ----------
    Tvec : [Tg_vec, Tnu_vec]  (arrays in MeV)
    cfg  : PyPRConfig

    Returns
    -------
    list of 9 arrays: T_HT, frwrd_HT, bkwrd_HT, T_MT, frwrd_MT, bkwrd_MT,
                      T_LT, frwrd_LT, bkwrd_LT
    """
    me = cfg.me * cfg.MeV
    mn = cfg.mn * cfg.MeV
    mp = cfg.mp * cfg.MeV
    Q  = mn - mp

    xi_nu  = cfg.munuOverTnu
    my_dir = cfg.working_dir

    Tg_vec, Tnu_vec = Tvec
    T_nuOverT = interp1d(Tg_vec * cfg.MeV_to_Kelvin, Tnu_vec / Tg_vec,
                         bounds_error=False, fill_value="extrapolate", kind='linear')

    _setup_fd_impls(cfg.numba_flag)

    # ------------------------------------------------------------------
    # Born rate integrands
    # ------------------------------------------------------------------
    def ChiFunc(E, p, x, znu, sgnq):
        return FD_nu3(E - sgnq * (Q / me), sgnq * xi_nu, znu) * FD2(-E, x) * (E - sgnq * (Q / me))**2

    def FermiStat(sgnq, sgnE, b):
        return FermiCoulomb(b, cfg) if (sgnq * sgnE) > 0 else 1.

    def IPENdp(p, x, znu, sgnq):
        E = np.sqrt(p**2 + 1.)
        return p**2 * (ChiFunc(E, p, x, znu, sgnq) + ChiFunc(-E, p, x, znu, sgnq))

    def _L_BORN_int(p, T, sgnq):
        x   = me / (cfg.kB * T)
        xnu = me / (cfg.kB * T * T_nuOverT(T))
        return IPENdp(p, x, xnu, sgnq)

    def _L_BORN(T, sgnq):
        x = me / (cfg.kB * T)
        return quad(_L_BORN_int, 0., max(7., 30. / x), args=(T, sgnq), epsrel=epsrel_low)[0]

    # ------------------------------------------------------------------
    # Finite-mass corrections
    # ------------------------------------------------------------------
    gA         = cfg.gA
    deltakappa = cfg.deltakappa

    def ChiFunc_FM(en, pe, x, znu, sgnq):
        M_sgnq = (mp + mn - sgnq * Q) / (2 * me)
        f_1 = ((1. + sgnq * gA)**2. + 2. * deltakappa * sgnq * gA) / (1. + 3. * gA**2)
        f_2 = ((1. - sgnq * gA)**2. - 2. * deltakappa * sgnq * gA) / (1. + 3. * gA**2)
        f_3 = (gA**2 - 1.) / (1. + 3. * gA**2)
        enu    = en - sgnq * Q / me
        FD2_en = FD2(-en, x)
        return (f_1 * FD_nu_e2p0(enu, 0, znu) * FD2_en * (pe**2 / (M_sgnq * en))
                + f_2 * FD_nu_e3p0(enu, 0, znu) * FD2_en * (-1. / M_sgnq)
                + (f_1 + f_2 + f_3) / (2. * x * M_sgnq)
                  * (FD_nu_e4p2(enu, 0, znu) * FD2_en + FD_nu_e2p2(enu, 0, znu) * FD2_en * pe**2)
                + (f_1 + f_2 + f_3) / (2. * M_sgnq)
                  * (FD_nu_e4p1(enu, 0, znu) * FD2_en + FD_nu_e2p1(enu, 0, znu) * FD2_en * pe**2)
                - (f_1 + f_2) / (x * M_sgnq)
                  * (FD_nu_e3p1(enu, 0, znu) * FD2_en + FD_nu_e2p1(enu, 0, znu) * FD2_en * pe**2 / (-en))
                - f_3 * 3. / (x * M_sgnq) * FD_nu_e2p0(enu, 0, znu) * FD2_en
                + f_3 / (3 * M_sgnq) * FD_nu_e3p1(enu, 0, znu) * FD2_en * pe**2 / en
                + f_3 * 2. / (2. * x * 3. * M_sgnq) * FD_nu_e3p2(enu, 0, znu) * FD2_en * pe**2 / en
                - (f_1 + f_2 + f_3) * 3. / (2. * x) * (1. - (mn / mp)**sgnq)
                  * (FD_nu_e2p1(enu, 0, znu) * FD2_en))

    def IPENdpFMCCR(p, x, znu, sgnq):
        eOFpe = np.sqrt(p**2 + 1.)
        b     = p / eOFpe
        return p**2 * (ChiFunc_FM(eOFpe,  p, x, znu, sgnq)
                       * RadCorrResum(b, np.abs(sgnq * Q / me - eOFpe), eOFpe, cfg)
                       * FermiStat(sgnq,  1, b)
                       + ChiFunc_FM(-eOFpe, p, x, znu, sgnq)
                       * RadCorrResum(b, np.abs(sgnq * Q / me + eOFpe), eOFpe, cfg)
                       * FermiStat(sgnq, -1, b))

    def _L_FMCCR_int(p, T, sgnq):
        x   = me / (cfg.kB * T)
        xnu = me / (cfg.kB * T * T_nuOverT(T))
        return IPENdpFMCCR(p, x, xnu, sgnq)

    def _L_FMCCR(T, sgnq):
        x = me / (cfg.kB * T)
        return quad(_L_FMCCR_int, 0., max(7., 30. / x), args=(T, sgnq), epsrel=epsrel_low)[0]

    # ------------------------------------------------------------------
    # T=0 radiative corrections
    # ------------------------------------------------------------------
    def IPENdpCCR(p, x, znu, sgnq):
        E = np.sqrt(p**2 + 1.)
        b = p / E
        return p**2 * (ChiFunc(E,  p, x, znu, sgnq)
                       * RadCorrResum(b, np.abs(sgnq * Q / me - E), E, cfg)
                       * FermiStat(sgnq,  1, b)
                       + ChiFunc(-E, p, x, znu, sgnq)
                       * RadCorrResum(b, np.abs(sgnq * Q / me + E), E, cfg)
                       * FermiStat(sgnq, -1, b))

    def _L_CCR_int(p, T, sgnq):
        x   = me / (cfg.kB * T)
        xnu = me / (cfg.kB * T * T_nuOverT(T))
        return IPENdpCCR(p, x, xnu, sgnq)

    def _L_CCR(T, sgnq):
        x = me / (cfg.kB * T)
        return quad(_L_CCR_int, 0., max(7., 30. / x), args=(T, sgnq), epsrel=epsrel_low)[0]

    # ------------------------------------------------------------------
    # Finite-temperature radiative corrections (optional, uses vegas)
    # ------------------------------------------------------------------
    if cfg.compute_nTOp_thermal_flag:
        try:
            import vegas
            _have_vegas = True
            n_eval = getattr(cfg, 'vegas_n_eval', 20000)
            n_itn  = getattr(cfg, 'vegas_n_itn',  20)
        except ImportError:
            _have_vegas = False
            from scipy.integrate import dblquad
            _epsrel_th = getattr(cfg, 'epsrel_thermal', 1.e-2)
            import warnings
            warnings.warn(
                "vegas not found: falling back to scipy.integrate.dblquad for thermal "
                "radiative corrections (epsrel={:.0e}).  Install vegas for better "
                "performance.".format(_epsrel_th),
                ImportWarning, stacklevel=2)

        def A(E, k):
            pE = np.sqrt(E**2 - 1.)
            return (2. * E**2 + k**2) * np.log((E + pE) / (E - pE)) - 4. * pE * E

        def B(E):
            pE = np.sqrt(E**2 - 1.)
            return 2. * E * np.log((E + pE) / (E - pE)) - 4. * pE

        def IPENCCRT(E, k, x, znu, sgnq):
            pE = np.sqrt(E**2 - 1.)

            def BE(EkBT):
                resvec = np.zeros(len(EkBT))
                my_index = np.where(np.abs(EkBT) < exp_cutoff)[0]
                resvec[my_index] = 1. / (np.exp(EkBT[my_index]) - 1.)
                return resvec

            def FD2_vec(en, xval):
                resvec = np.zeros(len(en))
                argvec = en * xval
                idx = np.where(np.abs(argvec) <= exp_cutoff)[0]
                resvec[idx] = 1. / (np.exp(argvec[idx]) + 1.)
                idx_ov = np.where(np.abs(argvec) > exp_cutoff)[0]
                resvec[idx_ov] = 1. / (np.exp(np.sign(argvec[idx_ov]) * exp_cutoff) + 1.)
                return resvec

            def Chitilde_vec(en, znuval, sgnq):
                q = Q / me
                resvec = np.zeros(len(en))
                argvec = znuval * (en - sgnq * q) - sgnq * xi_nu
                my_index = np.where(np.abs(argvec) < exp_cutoff)[0]
                resvec[my_index] = 1. / (np.exp(argvec[my_index]) + 1.)
                return resvec * (en - sgnq * q)**2

            return (cfg.alphaem / (2 * np.pi) * (BE(x * k) / k)
                    * (A(E, k) * (FD2_vec(-E, x) * FermiStat(sgnq,  1, pE / E)
                                  * (Chitilde_vec(E - k, znu, sgnq) + Chitilde_vec(E + k, znu, sgnq)
                                     - 2 * Chitilde_vec(E, znu, sgnq))
                                  + FD2_vec(E, x) * FermiStat(sgnq, -1, pE / E)
                                  * (Chitilde_vec(-E + k, znu, sgnq) + Chitilde_vec(-E - k, znu, sgnq)
                                     - 2 * Chitilde_vec(-E, znu, sgnq)))
                       - k * B(E) * (FD2_vec(-E, x) * FermiStat(sgnq,  1, pE / E)
                                     * (Chitilde_vec(E - k, znu, sgnq) - Chitilde_vec(E + k, znu, sgnq))
                                     + FD2_vec(E, x) * FermiStat(sgnq, -1, pE / E)
                                     * (Chitilde_vec(-E + k, znu, sgnq) - Chitilde_vec(-E - k, znu, sgnq)))))

        def IPENCCRDiffBremsstrahlung(E, k, x, znu, sgnq):
            q  = Q / me
            pE = np.sqrt(E**2 - 1.)
            Fp = (2. * E**2 + k**2) * np.log((E + pE) / (E - pE)) - 4. * pE * E + k * (2. * E * np.log((E + pE) / (E - pE)) - 4. * pE)
            Fm = (2. * E**2 + k**2) * np.log((E + pE) / (E - pE)) - 4. * pE * E - k * (2. * E * np.log((E + pE) / (E - pE)) - 4. * pE)

            def FD2_vec(en, xval):
                resvec = np.zeros(len(en))
                argvec = en * xval
                idx = np.where(np.abs(argvec) <= exp_cutoff)[0]
                resvec[idx] = 1. / (np.exp(argvec[idx]) + 1.)
                idx_ov = np.where(np.abs(argvec) > exp_cutoff)[0]
                resvec[idx_ov] = 1. / (np.exp(np.sign(argvec[idx_ov]) * exp_cutoff) + 1.)
                return resvec

            def Chitilde_vec(en, znuval, sgnq):
                q = Q / me
                resvec = np.zeros(len(en))
                argvec = znuval * (en - sgnq * q) - sgnq * xi_nu
                my_index = np.where(np.abs(argvec) < exp_cutoff)[0]
                resvec[my_index] = 1. / (np.exp(argvec[my_index]) + 1.)
                return resvec * (en - sgnq * q)**2

            res_fac  = cfg.alphaem / (2. * np.pi * k)
            res1_fac = FD2_vec(-E, x) * FermiStat(sgnq,  1, pE / E)
            res1vec  = Fp * Chitilde_vec(E + k, znu, sgnq)
            argvec   = k
            my_index = np.where(np.abs(argvec) < np.abs(E - sgnq * q))[0]
            res1vec[my_index] -= Fp[my_index] * FD2_vec(E[my_index] - sgnq * q, znu) * (np.abs(E[my_index] - sgnq * q) - k[my_index])**2
            res1vec *= res1_fac
            res2_fac = FD2_vec(E, x) * FermiStat(sgnq, -1, pE / E)
            res2vec  = Fm * Chitilde_vec(-E + k, znu, sgnq)
            my_index = np.where(np.abs(argvec) < np.abs(E + sgnq * q))[0]
            res2vec[my_index] -= Fp[my_index] * FD2_vec(-E[my_index] - sgnq * q, znu) * (np.abs(E[my_index] + sgnq * q) - k[my_index])**2
            res2vec *= res2_fac
            return res_fac * (res1vec + res2vec)

        def C1dE(E, x, znu, sgnq):
            pE = np.sqrt(E**2 - 1.)
            return (-(cfg.alphaem * E) / (2. * np.pi * pE) * (2. * np.pi**2) / (3. * x**2)
                    * (ChiFunc(E, pE, x, znu, sgnq) + ChiFunc(-E, pE, x, znu, sgnq)))

        def C2dE1dE2(e1v, e2v, x, znu, sgnq):
            resvec       = np.zeros(len(e1v))
            e1pe2        = e1v + e2v
            e1me2        = e1v - e2v
            min_e1pe2    = 2. + np.abs(e1me2)
            max_e1pe2    = 2. + max(10., 15. / x) + np.abs(e1me2)
            index_limits = np.where(((e1pe2 - min_e1pe2) > 0) * ((max_e1pe2 - e1pe2) > 0))[0]

            def FD2_vec(en, xval):
                resvec = np.zeros(len(en))
                argvec = en * xval
                idx = np.where(np.abs(argvec) <= exp_cutoff)[0]
                resvec[idx] = 1. / (np.exp(argvec[idx]) + 1.)
                idx_ov = np.where(np.abs(argvec) > exp_cutoff)[0]
                resvec[idx_ov] = 1. / (np.exp(np.sign(argvec[idx_ov]) * exp_cutoff) + 1.)
                return resvec

            def D_FD2_vec(en, xval):
                resvec = np.zeros(len(en))
                argvec = en * xval
                idx = np.where(np.abs(argvec) < exp_cutoff)[0]
                resvec[idx] = -xval * np.exp(argvec[idx]) / (np.exp(argvec[idx]) + 1.)**2
                return resvec

            def FD_nu3_vec(en, phi, xval):
                resvec = np.zeros(len(en))
                argvec = en * xval - phi
                idx = np.where(np.abs(argvec) < exp_cutoff)[0]
                resvec[idx] = 1. / (np.exp(argvec[idx]) + 1.)
                return resvec

            def ChiFunc_vec(E, p, x, znu, sgnq):
                return (FD_nu3_vec(E - sgnq * (Q / me), sgnq * xi_nu, znu)
                        * FD2_vec(-E, x) * (E - sgnq * (Q / me))**2)

            e1 = e1v[index_limits]
            e2 = e2v[index_limits]
            p1 = np.sqrt(e1**2 - 1.)
            p2 = np.sqrt(e2**2 - 1.)
            L_fac = np.log((e1 * e2 + p1 * p2 + 1.) / (e1 * e2 - p1 * p2 + 1.))
            resvec_limits = (cfg.alphaem / (2. * np.pi)
                             * (ChiFunc_vec(e1, p1, x, znu, sgnq) + ChiFunc_vec(-e1, p1, x, znu, sgnq))
                             * (-(1. / 4.) * np.log(((p1 + p2) / (p1 - p2))**2)**2
                                * (D_FD2_vec(e2, x) * p2 / p1 * e1**2 / e2 * (e1 + e2)
                                   + FD2_vec(e2, x) * e1**2 / (p1 * p2) * (e2 + e1 / e2**2))
                                + np.log(((p1 + p2) / (p1 - p2))**2)
                                * (D_FD2_vec(e2, x) * (p2**2 * e1 / e2 * (1. / p1**2 + 2.) - e1**2 * p2 / p1 * L_fac)
                                   + FD2_vec(e2, x) * (e1 / (p1**2 * e2**2) * (e2**2 + 2 * p1**2 + 1.)
                                                       - (e1**2 + e2**2) / (e1 + e2)
                                                       - (e1**2 * e2) / (p1 * p2) * L_fac))
                                - FD2_vec(e2, x) * (4. * e1 * p2 / p1 + 2. * e2 * L_fac)))
            resvec[index_limits] = resvec_limits
            return resvec

        def _L_ThermalTruePhoton_int(E, k, T, sgnq):
            x   = me / (cfg.kB * T)
            xnu = me / (cfg.kB * T * T_nuOverT(T))
            return IPENCCRT(E, k, x, xnu, sgnq)

        def _L_ThermalTruePhoton(T, sgnq):
            x     = me / (cfg.kB * T)
            E_max = max(10., 20. / x)
            k_max = max(10., 20. / x)
            if _have_vegas:
                integ = vegas.Integrator([[1.001, E_max], [0.001, k_max]])
                @vegas.batchintegrand
                def f_batch(xv):
                    E_val, k_val = np.transpose(xv)
                    return {'myres': _L_ThermalTruePhoton_int(E_val, k_val, T, sgnq)}
                integ(f_batch, nitn=n_itn, neval=n_eval)
                result = integ(f_batch, nitn=n_itn, neval=n_eval, adapt=True)
                return result['myres'].mean
            else:
                return dblquad(
                    lambda k, E: float(_L_ThermalTruePhoton_int(
                        np.atleast_1d(E), np.atleast_1d(k), T, sgnq)[0]),
                    1.001, E_max, 0.001, k_max, epsrel=_epsrel_th)[0]

        def _L_ThermalDiffBremsstrahlung_int(E, k, T, sgnq):
            x   = me / (cfg.kB * T)
            xnu = me / (cfg.kB * T * T_nuOverT(T))
            return IPENCCRDiffBremsstrahlung(E, k, x, xnu, sgnq)

        def _L_ThermalDiffBremsstrahlung(T, sgnq):
            x     = me / (cfg.kB * T)
            E_max = max(10., 20. / x)
            k_max = max(10., 20. / x)
            if _have_vegas:
                integ = vegas.Integrator([[1.001, E_max], [0.001, k_max]])
                @vegas.batchintegrand
                def f_batch(xv):
                    E_val, k_val = np.transpose(xv)
                    return {'myres': _L_ThermalDiffBremsstrahlung_int(E_val, k_val, T, sgnq)}
                integ(f_batch, nitn=n_itn, neval=n_eval)
                result = integ(f_batch, nitn=n_itn, neval=n_eval, adapt=True)
                return result['myres'].mean
            else:
                return dblquad(
                    lambda k, E: float(_L_ThermalDiffBremsstrahlung_int(
                        np.atleast_1d(E), np.atleast_1d(k), T, sgnq)[0]),
                    1.001, E_max, 0.001, k_max, epsrel=_epsrel_th)[0]

        def _L_Thermal_1_int(E, T, sgnq):
            return C1dE(E, me / (cfg.kB * T), me / (cfg.kB * T * T_nuOverT(T)), sgnq)

        def _L_Thermal_1(T, sgnq):
            return quad(_L_Thermal_1_int, 1., max(25., 150. * (cfg.kB * T) / me),
                        args=(T, sgnq), epsrel=1.e-2)[0]

        def _L_Thermal_2_3_int(e1pe2, e1me2, T, sgnq):
            x   = me / (cfg.kB * T)
            xnu = me / (cfg.kB * T * T_nuOverT(T))
            return 0.5 * C2dE1dE2((e1pe2 + e1me2) / 2., (e1pe2 - e1me2) / 2., x, xnu, sgnq)

        def _L_Thermal_2_3(T, sgnq):
            x    = me / (cfg.kB * T)
            half = max(10., 15. / x)
            res_2 = res_3 = 0.
            for min_e1me2, max_e1me2 in [(-half, -0.001), (0.001, half)]:
                lims = [2.001 + min(np.abs(min_e1me2), np.abs(max_e1me2)),
                        2.   + max(np.abs(min_e1me2), np.abs(max_e1me2))]
                if _have_vegas:
                    integ = vegas.Integrator([lims, [min_e1me2, max_e1me2]])
                    @vegas.batchintegrand
                    def f_batch(xv):
                        e1pe2, e1me2 = np.transpose(xv)
                        return {'myres': _L_Thermal_2_3_int(e1pe2, e1me2, T, sgnq)}
                    integ(f_batch, nitn=n_itn, neval=n_eval)
                    result = integ(f_batch, nitn=n_itn, neval=n_eval, adapt=True)
                    val = result['myres'].mean
                else:
                    val = dblquad(
                        lambda e1me2, e1pe2: float(_L_Thermal_2_3_int(
                            np.atleast_1d(e1pe2), np.atleast_1d(e1me2), T, sgnq)[0]),
                        lims[0], lims[1], min_e1me2, max_e1me2, epsrel=_epsrel_th)[0]
                if min_e1me2 < 0:
                    res_2 = val
                else:
                    res_3 = val
            return res_2 + res_3

        def _L_CCRTh_compute(T, sgnq):
            if sgnq == -1 and T < 10**(8.2):
                return 0.
            return (_L_ThermalTruePhoton(T, sgnq)
                    + _L_ThermalDiffBremsstrahlung(T, sgnq)
                    + _L_Thermal_1(T, sgnq)
                    + _L_Thermal_2_3(T, sgnq))

        if cfg.verbose_flag:
            print("[weak] Re-evaluating n <--> p thermal corrections. This may take a while ...")

        _T_th      = np.logspace(np.log10(cfg.T_end), np.log10(cfg.T_start), cfg.sampling_nTOp_thermal)
        L_nTh_data = np.vectorize(lambda T: _L_CCRTh_compute(T, +1))(_T_th)
        L_pTh_data = np.vectorize(lambda T: _L_CCRTh_compute(T, -1))(_T_th)

        if cfg.save_nTOp_thermal_flag:
            _td = my_dir + "/Rates/weak/"
            os.makedirs(_td, exist_ok=True)
            np.savetxt(_td + "nTOp_thermal_corrections.txt", np.c_[_T_th, L_nTh_data])
            np.savetxt(_td + "pTOn_thermal_corrections.txt", np.c_[_T_th, L_pTh_data])

        if cfg.verbose_flag:
            print("n <--> p thermal corrections computed")

        T_th, L_nTh, L_pTh = _T_th, L_nTh_data, L_pTh_data

    else:
        _td   = my_dir + "/Rates/weak/"
        T_th, L_nTh = np.loadtxt(_td + "nTOp_thermal_corrections.txt", unpack=True)
        T_th, L_pTh = np.loadtxt(_td + "pTOn_thermal_corrections.txt",  unpack=True)

    L_nTOpCCRTh = interp1d(T_th, L_nTh, bounds_error=False, fill_value="extrapolate", kind='quadratic')
    L_pTOnCCRTh = interp1d(T_th, L_pTh, bounds_error=False, fill_value="extrapolate", kind='quadratic')

    # ------------------------------------------------------------------
    # Assembled rates  (sgnq = +1: n→p,  sgnq = -1: p→n)
    # ------------------------------------------------------------------
    def nTOp_rate_(T, sgnq):
        if cfg.nTOpBorn_flag:
            return _L_BORN(T, sgnq)
        L_CCRTh = L_nTOpCCRTh(T) if sgnq == 1 else L_pTOnCCRTh(T)
        return _L_CCR(T, sgnq) + _L_FMCCR(T, sgnq) + L_CCRTh

    nTOp_frwrd_vec = np.vectorize(lambda T: nTOp_rate_(T, +1))
    nTOp_bkwrd_vec = np.vectorize(lambda T: nTOp_rate_(T, -1))

    T_HT = np.logspace(np.log10(cfg.T_start), np.log10(cfg.T_weak), cfg.sampling_nTOp)
    T_MT = np.logspace(np.log10(cfg.T_weak),  np.log10(cfg.T_nucl), cfg.sampling_nTOp)
    T_LT = np.logspace(np.log10(cfg.T_nucl),  np.log10(cfg.T_end),  cfg.sampling_nTOp)

    frwrd_HT = nTOp_frwrd_vec(T_HT)
    bkwrd_HT = nTOp_bkwrd_vec(T_HT)
    frwrd_MT = nTOp_frwrd_vec(T_MT)
    bkwrd_MT = nTOp_bkwrd_vec(T_MT)
    frwrd_LT = nTOp_frwrd_vec(T_LT)
    bkwrd_LT = nTOp_bkwrd_vec(T_LT)

    if cfg.save_nTOp_flag:
        _td = my_dir + "/Rates/weak/"
        os.makedirs(_td, exist_ok=True)
        np.savetxt(_td + "nTOp_frwrd_HT.txt", np.c_[T_HT, frwrd_HT])
        np.savetxt(_td + "nTOp_bkwrd_HT.txt", np.c_[T_HT, bkwrd_HT])
        np.savetxt(_td + "nTOp_frwrd_MT.txt", np.c_[T_MT, frwrd_MT])
        np.savetxt(_td + "nTOp_bkwrd_MT.txt", np.c_[T_MT, bkwrd_MT])
        np.savetxt(_td + "nTOp_frwrd_LT.txt", np.c_[T_LT, frwrd_LT])
        np.savetxt(_td + "nTOp_bkwrd_LT.txt", np.c_[T_LT, bkwrd_LT])

    return [T_HT, frwrd_HT, bkwrd_HT, T_MT, frwrd_MT, bkwrd_MT, T_LT, frwrd_LT, bkwrd_LT]


# ---------------------------------------------------------------------------
# Load / dispatch interface
# ---------------------------------------------------------------------------

def InterpolateWeakRates(cfg):
    """Load pre-tabulated weak rates from disk and return interpolants."""
    nd = os.path.join(cfg.working_dir, "Rates", "weak", "")

    def _load(fname):
        tab = np.loadtxt(nd + fname)
        return interp1d(tab[:, 0], tab[:, 1], bounds_error=False,
                        fill_value="extrapolate", kind='quadratic')

    return [_load("nTOp_frwrd_HT.txt"), _load("nTOp_bkwrd_HT.txt"),
            _load("nTOp_frwrd_MT.txt"), _load("nTOp_bkwrd_MT.txt"),
            _load("nTOp_frwrd_LT.txt"), _load("nTOp_bkwrd_LT.txt")]


def RecomputeWeakRates(Tvec, cfg):
    """
    Recompute weak rates from scratch or load pre-tabulated values,
    depending on ``cfg.compute_nTOp_flag``.

    Parameters
    ----------
    Tvec : [Tg_vec, Tnu_vec]  (arrays in MeV)
    cfg  : PyPRConfig

    Returns
    -------
    list of 6 interp1d objects: frwrd/bkwrd for HT, MT, LT eras.
    """
    if cfg.compute_nTOp_flag:
        (T_HT, frwrd_HT, bkwrd_HT,
         T_MT, frwrd_MT, bkwrd_MT,
         T_LT, frwrd_LT, bkwrd_LT) = ComputeWeakRates(Tvec, cfg)

        def _interp(T, v):
            return interp1d(T, v, bounds_error=False,
                            fill_value="extrapolate", kind='quadratic')

        return [_interp(T_HT, frwrd_HT), _interp(T_HT, bkwrd_HT),
                _interp(T_MT, frwrd_MT), _interp(T_MT, bkwrd_MT),
                _interp(T_LT, frwrd_LT), _interp(T_LT, bkwrd_LT)]
    else:
        return InterpolateWeakRates(cfg)
