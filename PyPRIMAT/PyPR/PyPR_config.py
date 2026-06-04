# -*- coding: utf-8 -*-
"""
PyPR_config.py
==============
Central configuration for PyPRIMAT.

Physical constants and derived unit conversions are *fixed* and computed once
here.  All run-time flags and cosmological/nuclear parameters are carried in a
``PyPRConfig`` instance and can be overridden by passing a parameter dictionary
to ``PyPRConfig(params)``.

No file I/O happens here.  Nuclear rate data are loaded separately in
``PyPR_nuclear_data.py``.
"""

import os
import numpy as np
from scipy.special import zeta

# ---------------------------------------------------------------------------
# Default parameter values exposed as a plain dict so callers can inspect them
# ---------------------------------------------------------------------------
DEFAULT_PARAMS: dict = {
    # ---- general behaviour ------------------------------------------------
    "verbose_flag":               False,
    "debug_flag":                 False,
    "numerical_precision":        1.e-6,   # for finite differences (solve_ivp) 
    "numba_flag":                 True,  # will be re-checked at runtime. 
    "analytic_entropy_derivative": True, # Use analytic derivative of entropy for plasma thermodynamics
    "numdiff_flag":               True,  # will be re-checked at runtime and used only if analytic_entropy_derivative is False. 


    # ---- fundamental constants (overridable for sensitivity studies) --------
    "GN":                         6.70883e-45,   # Newton's constant [MeV^-2]

    # ---- background thermodynamics ----------------------------------------
    "T_start_cosmo_MeV":          40.0,
    "n_sampling":                 2000,

    # ---- n <--> p weak rates ----------------------------------------------
    "compute_nTOp_flag":          True,
    "sampling_nTOp":              60,
    "nTOpBorn_flag":              False,
    "compute_nTOp_thermal_flag":  False,
    "sampling_nTOp_thermal":      100,
    "vegas_n_eval":               20000,   # evaluations per vegas iteration
    "vegas_n_itn":                20,      # number of vegas iterations
    "epsrel_thermal":             1.e-2,   # fallback tolerance if vegas unavailable
    "tau_n_flag":                 True, # Use neutron lifetime to normalize weak rates (instead of absolute normalization from GF, Vud, gA, etc.)
    "tau_n":                      878.4,  # neutron lifetime [s]; overrides the class-level constant when tau_n_flag=True
    "std_tau_n":                  0.5,    # 1σ uncertainty on tau_n [s], used for MC sampling
    "save_nTOp_flag":             False,
    "save_nTOp_thermal_flag":     False,
    "output_time_evolution":      False,
    "output_rates_time_evolution": False,
    "output_n_points":            500,
    "output_file":                "results/output_tables.tsv",

    # ---- nuclear network --------------------------------------------------
    "rates_dir":                  "key_primat_rates/",
    "smallnet_flag":              True,
    "NP_nuclear_flag":            False,

    # ---- cosmological inputs ----------------------------------------------
    "Omegabh2":                   0.022425,
    "DeltaNeff":                  0.,
    "munuOverTnu":                0.,

    # ---- Early Dark Energy ------------------------------------------------
    "fEDE":                       0.,     # EDE fraction at peak; 0 = disabled
    "zcEDE":                      1.e8,   # redshift of EDE peak
    "wnEDE":                      1.,     # EDE equation-of-state parameter

    # ---- nuclear rate MCMC weights (all zero = median) --------------------
    # 12 key reactions
    "p_npdg": 0., "p_dpHe3g": 0., "p_ddHe3n": 0., "p_ddtp": 0.,
    "p_tpag": 0., "p_tdan": 0., "p_taLi7g": 0., "p_He3ntp": 0.,
    "p_He3dap": 0., "p_He3aBe7g": 0., "p_Be7nLi7p": 0., "p_Li7paa": 0.,
    # 51 additional reactions (full net)
    "p_Li7paag": 0., "p_Be7naa": 0., "p_Be7daap": 0., "p_daLi6g": 0.,
    "p_Li6pBe7g": 0., "p_Li6pHe3a": 0., "p_B8naap": 0., "p_Li6He3aap": 0.,
    "p_Li6taan": 0., "p_Li6tLi8p": 0., "p_Li7He3Li6a": 0., "p_Li8He3Li7a": 0.,
    "p_Be7tLi6a": 0., "p_B8tBe7a": 0., "p_B8nLi6He3": 0., "p_B8nBe7d": 0.,
    "p_Li6tLi7d": 0., "p_Li6He3Be7d": 0., "p_Li7He3aad": 0., "p_Li8He3aat": 0.,
    "p_Be7taad": 0., "p_Be7tLi7He3": 0., "p_B8dBe7He3": 0., "p_B8taaHe3": 0.,
    "p_Be7He3ppaa": 0., "p_ddag": 0., "p_He3He3app": 0., "p_Be7pB8g": 0.,
    "p_Li7daan": 0., "p_dntg": 0., "p_ttann": 0., "p_He3nag": 0.,
    "p_He3tad": 0., "p_He3tanp": 0., "p_Li7taan": 0., "p_Li7He3aanp": 0.,
    "p_Li8dLi7t": 0., "p_Be7taanp": 0., "p_Be7He3aapp": 0., "p_Li6nta": 0.,
    "p_He3tLi6g": 0., "p_anpLi6g": 0., "p_Li6nLi7g": 0., "p_Li6dLi7p": 0.,
    "p_Li6dBe7n": 0., "p_Li7nLi8g": 0., "p_Li7dLi8p": 0., "p_Li8paan": 0.,
    "p_annHe6g": 0., "p_ppndp": 0., "p_Li7taann": 0.,
    # NP nuclear rate shifts
    "NP_delta_npdg": 0., "NP_delta_dpHe3g": 0., "NP_delta_ddHe3n": 0.,
    "NP_delta_ddtp": 0., "NP_delta_tpag": 0., "NP_delta_tdan": 0.,
    "NP_delta_taLi7g": 0., "NP_delta_He3ntp": 0., "NP_delta_He3dap": 0.,
    "NP_delta_He3aBe7g": 0., "NP_delta_Be7nLi7p": 0., "NP_delta_Li7paa": 0.,
    "NP_delta_Li7paag": 0., "NP_delta_Be7naa": 0., "NP_delta_Be7daap": 0.,
    "NP_delta_daLi6g": 0., "NP_delta_Li6pBe7g": 0., "NP_delta_Li6pHe3a": 0.,
    "NP_delta_B8naap": 0., "NP_delta_Li6He3aap": 0., "NP_delta_Li6taan": 0.,
    "NP_delta_Li6tLi8p": 0., "NP_delta_Li7He3Li6a": 0., "NP_delta_Li8He3Li7a": 0.,
    "NP_delta_Be7tLi6a": 0., "NP_delta_B8tBe7a": 0., "NP_delta_B8nLi6He3": 0.,
    "NP_delta_B8nBe7d": 0., "NP_delta_Li6tLi7d": 0., "NP_delta_Li6He3Be7d": 0.,
    "NP_delta_Li7He3aad": 0., "NP_delta_Li8He3aat": 0., "NP_delta_Be7taad": 0.,
    "NP_delta_Be7tLi7He3": 0., "NP_delta_B8dBe7He3": 0., "NP_delta_B8taaHe3": 0.,
    "NP_delta_Be7He3ppaa": 0., "NP_delta_ddag": 0., "NP_delta_He3He3app": 0.,
    "NP_delta_Be7pB8g": 0., "NP_delta_Li7daan": 0., "NP_delta_dntg": 0.,
    "NP_delta_ttann": 0., "NP_delta_He3nag": 0., "NP_delta_He3tad": 0.,
    "NP_delta_He3tanp": 0., "NP_delta_Li7taan": 0., "NP_delta_Li7He3aanp": 0.,
    "NP_delta_Li8dLi7t": 0., "NP_delta_Be7taanp": 0., "NP_delta_Be7He3aapp": 0.,
    "NP_delta_Li6nta": 0., "NP_delta_He3tLi6g": 0., "NP_delta_anpLi6g": 0.,
    "NP_delta_Li6nLi7g": 0., "NP_delta_Li6dLi7p": 0., "NP_delta_Li6dBe7n": 0.,
    "NP_delta_Li7nLi8g": 0., "NP_delta_Li7dLi8p": 0., "NP_delta_Li8paan": 0.,
    "NP_delta_annHe6g": 0., "NP_delta_ppndp": 0., "NP_delta_Li7taann": 0.,
}


class PyPRConfig:
    """
    Immutable physical constants + mutable run-time parameters.

    Usage::

        cfg = PyPRConfig()                    # all defaults
        cfg = PyPRConfig({"Omegabh2": 0.022, "smallnet_flag": False})

    After construction every key in ``DEFAULT_PARAMS`` is an attribute, plus
    all physical constants listed below.
    """

    # ------------------------------------------------------------------
    # Class-level physical constants (identical for every instance)
    # ------------------------------------------------------------------

    # CGS base units (dimensionless by convention)
    Kelvin: float = 1.
    second: float = 1.
    cm:     float = 1.
    gram:   float = 1.

    @property
    def erg(self) -> float:
        return self.gram * self.cm**2 / self.second

    # Fundamental constants (PDG)
    kB:     float = 1.380649e-16          # Boltzmann [erg/K]
    clight: float = 2.99792458e+10        # speed of light [cm/s]
    hbar:   float = 6.62607015 / (2 * np.pi) * 1e-27  # Planck [erg·s]
    Mpc:    float = 3.08567758149e+24     # [cm]
    MeV:    float = 1.602176634e-6        # [erg]
    keV:    float = 1.602176634e-9        # [erg]

    # Conversion factors (MeV-based natural units <-> CGS)
    @property
    def MeV_to_Kelvin(self) -> float:
        return self.MeV / self.kB

    @property
    def MeV_to_secm1(self) -> float:
        return self.MeV / self.hbar

    @property
    def MeV_to_g(self) -> float:
        return self.MeV / self.clight**2

    @property
    def MeV_to_cmm1(self) -> float:
        return self.MeV / (self.hbar * self.clight)

    @property
    def MeV4_to_gcmm3(self) -> float:
        return self.MeV_to_g * self.MeV_to_cmm1**3

    # Temperature eras [K]
    @property
    def T_start_cosmo(self) -> float:
        return self.T_start_cosmo_MeV * self.MeV_to_Kelvin

    @property
    def T_start(self) -> float:
        return 10.0 * self.MeV_to_Kelvin

    @property
    def T_weak(self) -> float:
        return 1.0 * self.MeV_to_Kelvin

    @property
    def T_nucl(self) -> float:
        return 0.1 * self.MeV_to_Kelvin

    @property
    def T_end(self) -> float:
        return 1.e-3 * self.MeV_to_Kelvin

    # Electroweak sector (PDG)
    alphaem: float = 1. / 137.035999084
    GF:      float = 1.1663787e-5 * 1.e-6   # [MeV^-2]
    mZ:      float = 91.1876e3               # [MeV]

    @property
    def sW2(self) -> float:
        return 0.5 * (1. - np.sqrt(1. - 2.*np.sqrt(2.)*np.pi*self.alphaem / (self.GF * self.mZ**2)))

    @property
    def geL(self) -> float:
        return 0.5 + self.sW2

    @property
    def geR(self) -> float:
        return self.sW2

    @property
    def gmuL(self) -> float:
        return -0.5 + self.sW2

    @property
    def gmuR(self) -> float:
        return self.sW2

    # Fermion masses [MeV]
    me: float = 0.51099895
    mn: float = 939.56542052
    mp: float = 938.27208816

    # Gravity [MeV^-2]
    GN: float = 6.70883e-39 * 1.e-6

    @property
    def Mpl(self) -> float:
        return 1. / np.sqrt(self.GN)

    # Weak rate nuclear structure constants (PDG)
    gA:          float = 1.2756
    kappa_p:     float = 2.79284734463 - 1.
    kappa_n:     float = -1.91304273
    tau_n:       float = 878.4             # neutron lifetime [s]
    Vud:         float = 0.9738
    radproton:   float = 0.8409e-13        # proton charge radius [cm]

    @property
    def deltakappa(self) -> float:
        return self.kappa_p - self.kappa_n

    # CMB / cosmological fixed quantities
    T0CMB: float = 2.7255                  # photon temperature today [K]

    @property
    def s0bar(self) -> float:
        return 4. * np.pi**2 / 45.

    @property
    def s0CMB(self) -> float:
        return self.s0bar * (self.T0CMB / self.MeV_to_Kelvin)**3  # [MeV^3]

    @property
    def n0CMB(self) -> float:
        return (2. * zeta(3)) / np.pi**2 * (self.T0CMB / self.MeV_to_Kelvin)**3  # [MeV^3]

    # Atomic masses
    ma:         float = 931.494061         # 1 u.m.a. [MeV]
    He4Overma:  float = 4.0026032541
    HOverma:    float = 1.00782503223

    @property
    def mB(self) -> float:
        percentHe = 24.7 / 100.
        return ((1. - percentHe) * self.HOverma + percentHe * self.He4Overma / 4.) * self.ma

    @property
    def maOvermB(self) -> float:
        return self.ma / self.mB

    @property
    def HubbleOverh(self) -> float:
        return 100. * (1.e+5 * self.cm * self.MeV_to_cmm1) / (self.second * self.MeV_to_secm1) / (self.Mpc * self.MeV_to_cmm1)

    @property
    def rhocOverh2(self) -> float:
        return 3. / (8. * np.pi * self.GN) * self.HubbleOverh**2  # [MeV^4/h^2]

    # Nuclear data (list of nuclides, binding energies, spins)
    Nuclides: dict = {
        "n":   [1, 0], "p":   [0, 1], "H2":  [1, 1], "H3":  [2, 1],
        "He3": [1, 2], "He4": [2, 2], "He6": [4, 2], "Li6": [3, 3],
        "Li7": [4, 3], "Be7": [3, 4], "Li8": [5, 3], "B8":  [3, 5],
    }
    NuclExcessMass: dict = {
        "n":   8071.3171,  "p":   7288.9706,  "H2":  13135.722,
        "H3":  14949.81,   "He3": 14931.218,  "He4": 2424.9156,
        "He6": 17592.10,   "Li6": 14086.8789, "Li7": 14907.105,
        "Be7": 15769.,     "Li8": 20945.80,   "Be8": 4941.67,    "B8": 22921.6,
    }
    NuclSpin: dict = {
        "n":   0.5, "p":   0.5, "H2":  1.,  "H3":  0.5,
        "He3": 0.5, "He4": 0.,  "He6": 0.,  "Li6": 1.,
        "Li7": 1.5, "Be7": 1.5, "Li8": 2.,  "Be8": 0., "B8": 2.,
    }

    # ------------------------------------------------------------------
    # Constructor: merge user params over defaults
    # ------------------------------------------------------------------
    def __init__(self, params: dict | None = None):
        # Initialise every default as an instance attribute
        for key, value in DEFAULT_PARAMS.items():
            setattr(self, key, value)

        user_keys = set(params.keys()) if params else set()

        # Apply user overrides
        if params:
            unknown = set(params.keys()) - set(DEFAULT_PARAMS.keys())
            if unknown:
                import warnings
                warnings.warn(
                    f"PyPRConfig: unknown parameter keys ignored: {unknown}",
                    stacklevel=2,
                )
            for key, value in params.items():
                if key in DEFAULT_PARAMS:
                    setattr(self, key, value)

        # Detect optional libraries for flags not explicitly set by the caller.
        # Messages are stored for deferred printing (after the banner).
        self._init_messages = []

        if self.numba_flag:
            try:
                import numba  # noqa: F401
                self._init_messages.append('[init]  numba detected: using it for JIT compilation.')
            except ImportError:
                self.numba_flag = False
                self._init_messages.append('[init]  numba not detected: running without JIT compilation.')

        if self.numdiff_flag and (not self.analytic_entropy_derivative):
            try:
                import numdifftools  # noqa: F401
                self._init_messages.append('[init]  numdifftools detected: using it for numerical derivative of entropy.')
            except ImportError:
                self.numdiff_flag = False
                self._init_messages.append('[init]  numdifftools not detected: using finite differences for numerical derivatives of entropy.')

        # Derived cosmological quantity (depends on Omegabh2)
        self._update_derived()

    def _update_derived(self):
        """Recompute quantities that depend on mutable parameters."""
        self.Omegabh2_to_eta0b = (self.rhocOverh2 / self.n0CMB) / (self.ma / self.maOvermB)
        self.eta0b = self.Omegabh2_to_eta0b * self.Omegabh2

    # Convenience: allow dict-style access for backwards compat if needed
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)
        self._update_derived()

    # Path helper: working dir is two levels above this file (package root)
    @property
    def working_dir(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
