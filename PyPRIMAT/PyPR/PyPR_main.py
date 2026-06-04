# -*- coding: utf-8 -*-
"""
PyPR_main.py
============
Main class for PyPRIMAT.

Design
------
* ``PyPRclass.__init__(params)`` accepts an optional dict of parameters,
  builds a ``PyPRConfig``, loads all data files (thermodynamics tables and
  nuclear rate tables), and pre-computes the thermal background.
* ``PyPRclass.solve()`` runs the full nuclear network ODE integration and
  returns the BBN predictions.
* ``PyPRclass.PyPRresults()`` calls ``solve()`` and returns the result dict
  (for backwards compatibility).

"""

import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.special import zeta

from PyPR.PyPR_config       import PyPRConfig
from PyPR.PyPR_nuclear_data import NuclearData
import PyPR.PyPR_plasma      as PyPRthermo
import PyPR.PyPR_weak_rates   as PyPRnTOp


__version__ = "0.1.0"

# Column order for abundance interpolators; names match PyPRConfig.Nuclides keys
_NUC_NAMES_SMALL = ["n", "p", "H2", "H3", "He3", "He4", "Li7", "Be7"]
_NUC_NAMES_FULL  = ["n", "p", "H2", "H3", "He3", "He4", "Li7", "Be7",
                    "He6", "Li8", "Li6", "B8"]

_BANNER = """
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                 ┃
┃   ░█▀█░█░█░█▀█░█▀▄░▀█▀░█▄█░█▀█░▀█▀              ┃
┃   ░█▀▀░░█░░█▀▀░█▀▄░░█░░█░█░█▀█░░█░              ┃
┃   ░▀░░░░▀░░▀░░░▀░▀░▀▀▀░▀░▀░▀░▀░░▀░              ┃
┃                                                 ┃
┃    Welcome to PyPRIMAT v{version} — Cyril Pitrou    ┃
┃                                                 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
""".format(version=__version__)


class PyPRclass:
    """
    Main PyPRIMAT class.

    Parameters
    ----------
    params : dict, optional
        Run-time parameters overriding defaults (see ``PyPR_config.DEFAULT_PARAMS``).
    """

    def __init__(self, params=None):

        # ------------------------------------------------------------------
        # 1. Build configuration
        # ------------------------------------------------------------------
        self.cfg = PyPRConfig(params or {})
        cfg = self.cfg
        self.N = {name: NZ[0]           for name, NZ in cfg.Nuclides.items()}
        self.Z = {name: NZ[1]           for name, NZ in cfg.Nuclides.items()}
        self.A = {name: NZ[0] + NZ[1]   for name, NZ in cfg.Nuclides.items()}

        if cfg.verbose_flag:
            print(_BANNER)
            for msg in cfg._init_messages:
                print(msg)
            self._t0 = time.time()

        # ------------------------------------------------------------------
        # 2. Initialise thermodynamics module (loads QED/neutrino tables)
        # ------------------------------------------------------------------
        PyPRthermo.initialise(cfg)

        # ------------------------------------------------------------------
        # 3. Build EDE energy-density function if fEDE > 0
        # ------------------------------------------------------------------
        self._setup_EDE()

        # ------------------------------------------------------------------
        # 4. Load nuclear rate data from disk (once, here)
        # ------------------------------------------------------------------
        self.nuclear_data = NuclearData(cfg)

        # ------------------------------------------------------------------
        # 5. Compute or load thermal background + cosmological functions
        # ------------------------------------------------------------------
        self._setup_background_and_cosmo()
        self._setup_derived_cosmo()

        # ------------------------------------------------------------------
        # 6. Compute or load n <--> p weak rates
        # ------------------------------------------------------------------
        self._setup_weak_rates()
        self._results = None

        if cfg.verbose_flag:
            print(f"[init] Initialisation complete in {time.time()-self._t0:.1f} s")

    # ======================================================================
    # Private: Early Dark Energy setup
    # ======================================================================

    def _setup_EDE(self):
        """Build the EDE energy-density function from cfg.fEDE/zcEDE/wnEDE.

        Sets self._rho_EDE(Tg) to a callable if fEDE > 0, else None.
        Must be called after PyPRthermo.initialise() since it evaluates rho_g.
        """
        cfg = self.cfg
        if cfg.fEDE == 0.:
            self._rho_EDE = None
            return

        thermo  = PyPRthermo
        acEDE   = 1. / (1. + cfg.zcEDE)
        amaxEDE = acEDE * (4. / (3. * cfg.wnEDE - 1.))**(1. / (3. * cfg.wnEDE + 3.))
        TmaxEDE = cfg.T0CMB / amaxEDE / cfg.MeV_to_Kelvin   # [MeV]
        TcEDE   = cfg.T0CMB / acEDE   / cfg.MeV_to_Kelvin   # [MeV]

        #The final Neff value in the standard case (3.044) is hard coded here.
        rhocEDEac = (cfg.fEDE / (1. - cfg.fEDE)
                     * thermo.rho_g(TmaxEDE)
                     * (1. + 3.044 * 7./8. * (4./11.)**(4./3.))
                     / 2.
                     * (1. + 4. / (3. * cfg.wnEDE - 1.)))

        def rho_EDE(T):
            return 2. * rhocEDEac / (1. + (TcEDE / T)**(3. * cfg.wnEDE + 3.))

        self._rho_EDE = rho_EDE

    # ======================================================================
    # Private: background thermodynamics + cosmological setup
    # ======================================================================

    # Friedmann expansion rate
    def _Hubble(self, Tg, Tnue, Tnumu, Tnutau):
        cfg     = self.cfg
        thermo  = PyPRthermo
        rho_pl  = thermo.rho_g(Tg) + thermo.rho_e(Tg) - thermo.PofT(Tg) + Tg * thermo.dPdT(Tg)
        rho_3nu = thermo.rho_nu(Tnue) + thermo.rho_nu(Tnumu) + thermo.rho_nu(Tnutau)
        rho_tot = rho_pl + rho_3nu + thermo.rho_nu_extra(Tg)
        if self._rho_EDE is not None: rho_tot += self._rho_EDE(Tg)
        return cfg.MeV_to_secm1 * (rho_tot * 8. * np.pi / (3. * cfg.Mpl**2))**0.5

    def _setup_background_and_cosmo(self):
        """NEVO cosmological background.

        Reads the pre-computed NEVO neutrino-decoupling table and builds
        all interpolants used by _setup_derived_cosmo, _setup_weak_rates,
        and solve().
        """
        cfg    = self.cfg
        thermo = PyPRthermo

        Tstartcosmo  = cfg.T_start_cosmo / cfg.MeV_to_Kelvin
        Tstart = cfg.T_start / cfg.MeV_to_Kelvin   # [MeV]
        Tend   = cfg.T_end   / cfg.MeV_to_Kelvin   # [MeV]

        # ------------------------------------------------------------------
        # Step 1 – Load NEVO table and build neutrino-temperature interpolants
        # ------------------------------------------------------------------
        nevo_path = cfg.working_dir + "/Rates/NEVO/NEVOPRIMAT_col_1_7.csv"
        table = np.loadtxt(nevo_path, delimiter=',', usecols=range(6))
        # Column layout (0-indexed):
        #   0: x = me / (kB T_com)   [dimensionless]
        #   1: z = a T_γ normalised  [dimensionless]
        #   2: T_νe  / T_com         [dimensionless]
        #   3: T_νμ  / T_com         [dimensionless]
        #   4: T_ντ  / T_com         [dimensionless]
        #   5: N_NEVO (heating fn)   [dimensionless, same units as s̄ = s(T)/T³]

        x      = table[:, 0]
        z      = table[:, 1]

        Tg_tab = cfg.me * z / x          # T_γ [MeV]
        Tnue_tab   = table[:, 2] * cfg.me / x   # T_νe  [MeV]
        Tnumu_tab  = table[:, 3] * cfg.me / x   # T_νμ  [MeV]
        Tnutau_tab = table[:, 4] * cfg.me / x   # T_ντ  [MeV]
        N_NEVO_tab = table[:, 5]                 # dimensionless

        # Ensure the table is ordered by decreasing T_γ (high→low)
        if Tg_tab[0] < Tg_tab[-1]:
            Tg_tab, Tnue_tab, Tnumu_tab, Tnutau_tab, N_NEVO_tab = (
                arr[::-1] for arr in (Tg_tab, Tnue_tab, Tnumu_tab, Tnutau_tab, N_NEVO_tab))

        # Interpolants for neutrino temperatures as functions of T_γ.
        # Outside the table we keep T_να / T_γ = constant (boundary value), so
        # that neutrino temperatures scale exactly like photon temperature — the
        # correct behaviour in the instantaneous-decoupling limit both at high T
        # (before any heating) and at low T (well after freeze-out).
        _ratio_ue   = Tnue_tab   / Tg_tab
        _ratio_umu  = Tnumu_tab  / Tg_tab
        _ratio_utau = Tnutau_tab / Tg_tab

        # ratio(T_γ) interpolants with constant extrapolation at boundaries
        _ratiofn_kw = dict(bounds_error=False, kind='linear')
        _ratiofn_ue   = interp1d(Tg_tab, _ratio_ue,   fill_value=(_ratio_ue[-1],   _ratio_ue[0]),   **_ratiofn_kw)
        _ratiofn_umu  = interp1d(Tg_tab, _ratio_umu,  fill_value=(_ratio_umu[-1],  _ratio_umu[0]),  **_ratiofn_kw)
        _ratiofn_utau = interp1d(Tg_tab, _ratio_utau, fill_value=(_ratio_utau[-1], _ratio_utau[0]), **_ratiofn_kw)

        def Tnue_of_Tg(Tg):
            return _ratiofn_ue(Tg) * Tg

        def Tnumu_of_Tg(Tg):
            return _ratiofn_umu(Tg) * Tg

        def Tnutau_of_Tg(Tg):
            return _ratiofn_utau(Tg) * Tg

        N_NEVO_of_Tg = interp1d(Tg_tab, N_NEVO_tab, bounds_error=False,
                                 fill_value=(0., 0.), kind='linear')

        # ------------------------------------------------------------------
        # Step 2 – Solve a(T) ODE / invert to T(a)
        # ------------------------------------------------------------------
        def _sbar(T):
            return thermo.spl(T) / T**3   # dimensionless

        if not cfg.analytic_entropy_derivative:
            if cfg.numdiff_flag:
                from numdifftools import Derivative
                _dsbardT = Derivative(_sbar, n=1)
            else:
                def _dsbardT(T):
                    dToT = 1.e-3
                    return (_sbar((1. + dToT) * T) - _sbar((1. - dToT) * T)) / (2. * dToT * T)

        #print('At 10 MeV sbar(T) = ',_sbar(10),' and dsbar/dT = ',_dsbardT(10))
        #print('At 1 MeV sbar(T) = ',_sbar(1.),' and dsbar/dT = ',_dsbardT(1.))
        #print('At .1 MeV sbar(T) = ',_sbar(.1),' and dsbar/dT = ',_dsbardT(.1))
        #print('At .01 MeV sbar(T) = ',_sbar(.01),' and dsbar/dT = ',_dsbardT(.01))

        if cfg.analytic_entropy_derivative:
            def _dlnadlnT_NEVO(lnT, y):
                T = np.exp(lnT)
                s, ds_dT = thermo.spl_and_dspl_dT(T)
                sb     = s / T**3
                dsbdT  = ds_dT / T**3 - 3. * s / T**4
                N = float(N_NEVO_of_Tg(T))
                return [-(3. * sb + T * dsbdT) / (N + 3. * sb)]
        else:
            def _dlnadlnT_NEVO(lnT, y):
                T   = np.exp(lnT)
                sb  = _sbar(T)
                N   = float(N_NEVO_of_Tg(T))
                return [-(3. * sb + T * _dsbardT(T)) / (N + 3. * sb)]       

        z0   = cfg.T0CMB / cfg.MeV_to_Kelvin   # [MeV]
        zend = z0 / (_sbar(Tend) / cfg.s0bar) ** (1. / 3.)
        lna_end = np.log(zend / Tend)

        T_sol = np.logspace(np.log(Tend), np.log(Tstartcosmo), cfg.n_sampling, base=np.e)
        _t_nevo_a0 = time.time()
        sol_lna = solve_ivp(_dlnadlnT_NEVO,
                            [np.log(Tend), np.log(Tstartcosmo)],
                            [lna_end],
                            t_eval=np.log(T_sol),
                            method='LSODA', rtol=0.1*cfg.numerical_precision, atol=1e-10)
        if cfg.debug_flag:
            print((f"[bckg] Finished a(T) solve in {time.time()- _t_nevo_a0:.2f} s "
                   f"(status={sol_lna.status}, nfev={sol_lna.nfev})"), flush=True)
        _lnalnT = interp1d(sol_lna.t, sol_lna.y[0].flatten(),
                           bounds_error=False, fill_value="extrapolate")
        
        def a_of_T(T):
            return np.exp(_lnalnT(np.log(T)))
        
        #Alternative method where we integrate log(a*T) as a function of T, which is what Mathematica solves. 
        # This is more direct but slightly less stable (requires higher precision and more careful handling of the derivative of sbar)
        #def _dlnaTdlnT_NEVO(lnT, y):
        #    T   = np.exp(lnT)
        #    sb  = _sbar(T)
        #    N   = float(N_NEVO_of_Tg(T))
        #    # laTCN = ln(a·T) is what Mathematica solves, so d(ln a)/d(ln T) = RHS - 1
        #    # = (𝒩 - T·ds̄/dT)/(𝒩 + 3s̄) - 1 = -(3s̄ + T·ds̄/dT)/(𝒩 + 3s̄)
        #    return [(N - T * _dsbardT(T)) / (N + 3. * sb)]
        #
        #PRIMAT method based on integration of aT as function of T (with logs)
        #lnaT_end = np.log(zend)
        #_t_nevo_at0 = time.time()
        #sol_lnaT = solve_ivp(_dlnaTdlnT_NEVO,
        #                    [np.log(Tend), np.log(Tstartcosmo)],
        #                    [lnaT_end],
        #                    t_eval=np.log(T_sol),
        #                    method='LSODA', rtol=0.1*cfg.numerical_precision, atol=1e-11)
        #if cfg.debug_flag:
        #    print((f"[bckg] Finished aT(T) solve in {time.time()-_t_nevo_at0:.2f} s "
        #           f"(status={sol_lnaT.status}, nfev={sol_lnaT.nfev})"), flush=True)
        #_lnaTlnT = interp1d(sol_lnaT.t, sol_lnaT.y[0].flatten(),
        #                   bounds_error=False, fill_value="extrapolate")
        #def a_of_T(T):
        #    return np.exp(_lnaTlnT(np.log(T)))/T

        
        # ------------------------------------------------------------------
        # Step 3 – Invert a(T) → T(a), then integrate dt/d(ln a) = 1/H(a)
        # ------------------------------------------------------------------
        T_grid = T_sol                          # already sampled low→high
        a_grid = np.array([a_of_T(T) for T in T_grid])   # low a → high a

        T_of_a = interp1d(a_grid, T_grid, bounds_error=False, fill_value="extrapolate")

        a_ini = a_of_T(Tstartcosmo)
        a_fin = a_of_T(Tend)


        # ------------------------------------------------------------------
        # Step 4 – Integrate dt/d(ln a) = 1/H(a)
        # ------------------------------------------------------------------
        def Hubble_NEVO(Tg):
            return self._Hubble(Tg, Tnue_of_Tg(Tg), Tnumu_of_Tg(Tg), Tnutau_of_Tg(Tg))

        t_ini = 1. / (2. * Hubble_NEVO(Tstartcosmo))

        a_samp = np.logspace(np.log(a_ini), np.log(a_fin), cfg.n_sampling, base=np.e)
 
        def _dtdlna(lna, t):
            return [1. / Hubble_NEVO(T_of_a(np.exp(lna)))]

        _t_nevo_t0 = time.time()
        sol_t = solve_ivp(_dtdlna,
                          [np.log(a_ini), np.log(a_fin)],
                          [t_ini],
                          t_eval=np.log(a_samp),
                          method='LSODA', rtol=cfg.numerical_precision, atol=1e-12)
        if cfg.debug_flag:
            print((f"[bckg] Finished t(a) solve in {time.time()-_t_nevo_t0:.2f} s "
                   f"(status={sol_t.status}, nfev={sol_t.nfev})"), flush=True)

        t_of_lna = interp1d(sol_t.t, sol_t.y[0].flatten(),
                            bounds_error=False, fill_value="extrapolate")

        # ------------------------------------------------------------------
        # Step 5 – Sample on the common time grid; set instance attributes
        # ------------------------------------------------------------------
        a_arr  = np.exp(sol_t.t)       # a values at ODE evaluation points
        t_vec  = sol_t.y[0].flatten()  # corresponding t [s]
        Tg_vec = T_of_a(a_arr)         # T_γ [MeV]

        Tnue_vec   = Tnue_of_Tg(Tg_vec)
        Tnumu_vec  = Tnumu_of_Tg(Tg_vec)
        Tnutau_vec = Tnutau_of_Tg(Tg_vec)
        # Energy-weighted average neutrino temperature (for weak rates / Omega_ν)
        Tnu_avg_vec = ((Tnue_vec**4 + Tnumu_vec**4 + Tnutau_vec**4) / 3.)**0.25

        self._t_vec      = t_vec
        self._Tg_vec     = Tg_vec
        self._Tnu_vec    = Tnu_avg_vec   # average, used by _setup_derived_cosmo and _setup_weak_rates
        self._Tnue_vec   = Tnue_vec
        self._Tnumu_vec  = Tnumu_vec
        self._Tnutau_vec = Tnutau_vec

        self._t_of_T = interp1d(Tg_vec, t_vec, bounds_error=False,
                                 fill_value="extrapolate", kind='linear')
        self._T_of_t = interp1d(t_vec, Tg_vec, bounds_error=False,
                                 fill_value="extrapolate", kind='linear')
        self._TnuofT = interp1d(Tg_vec, Tnu_avg_vec, bounds_error=False,
                                 fill_value="extrapolate", kind='linear')
        self._a_of_T = np.vectorize(a_of_T)
        self._a_of_t = interp1d(t_vec, a_arr, bounds_error=False,
                                 fill_value=(a_arr[0], a_arr[-1]))
        self._N_NEVO_of_Tg = N_NEVO_of_Tg

    def _setup_derived_cosmo(self):
        """Build N_eff and relic-neutrino Omega functions from the stored background.

        Called after _setup_background_and_cosmo.
        Requires self._Tg_vec, self._Tnu_vec to be set.
        """
        cfg    = self.cfg
        thermo = PyPRthermo

        # N_eff
        def N_eff(Tg, Tnue, Tnumu, Tnutau):
            rho_g   = thermo.rho_g(Tg)
            rho_rad = thermo.rho_nu(Tnue) + thermo.rho_nu(Tnumu) + thermo.rho_nu(Tnutau) + rho_g + thermo.rho_nu_extra(Tg)
            return (rho_rad - rho_g) / rho_g / ((7. / 8.) * (4. / 11.) ** (4. / 3.))

        self._N_eff = N_eff

        # Relic neutrino abundances
        def Omeganuh2_relnu():
            Tnu0 = self._Tnu_vec[-1] / self._Tg_vec[-1] * cfg.T0CMB / cfg.MeV_to_Kelvin
            return (7. * np.pi**2 / 120. * Tnu0**4) / cfg.rhocOverh2

        def Omeganuh2_nrnu():
            Tnu0 = self._Tnu_vec[-1] / self._Tg_vec[-1] * cfg.T0CMB / cfg.MeV_to_Kelvin
            return (3. / 2. * zeta(3) / np.pi**2 * Tnu0**3) / cfg.rhocOverh2

        self._Omeganuh2_relnu = Omeganuh2_relnu
        self._Omeganuh2_nrnu  = Omeganuh2_nrnu

    # ======================================================================
    # Private: weak rates
    # ======================================================================

    def _setup_weak_rates(self):
        cfg = self.cfg
        _t_weak0 = time.time()
        (self._nTOp_frwrd_HT, self._nTOp_bkwrd_HT,
         self._nTOp_frwrd_MT, self._nTOp_bkwrd_MT,
         self._nTOp_frwrd_LT, self._nTOp_bkwrd_LT) = \
            PyPRnTOp.RecomputeWeakRates([self._Tg_vec, self._Tnue_vec], cfg)
        if cfg.debug_flag:
            print((f"[weak] Finished recomputation of n <--> p weak rates in "
                   f"{time.time()-_t_weak0:.2f} s"), flush=True)

        # Normalisation factor
        _t_norm0 = time.time()
        if cfg.tau_n_flag:
            Fn = PyPRnTOp.ComputeFn(cfg)
            self._NormWeakRates = 1. / (Fn * cfg.tau_n)   # [s^-1]
        else:
            GFtilde2 = (cfg.GF * cfg.Vud)**2 * (1. + 3. * cfg.gA**2) / (2. * np.pi**3)
            self._NormWeakRates = cfg.MeV_to_secm1 * (GFtilde2 * cfg.me**5)   
        if cfg.debug_flag:
            print((f"[weak] Finished weak-rate normalisation setup in "
                   f"{time.time()-_t_norm0:.2f} s"), flush=True)

    # ======================================================================
    # solve(): integrate nuclear network ODEs
    # ======================================================================

    def solve(self):
        """
        Integrate the nuclear network over the three temperature eras and
        return a dict of BBN observables.
        """
        cfg       = self.cfg
        t_vec     = self._t_vec
        Tg_vec    = self._Tg_vec
        Tnu_vec   = self._Tnu_vec
        a_of_t    = self._a_of_t
        T_of_t    = self._T_of_t
        t_of_T    = self._t_of_T
        NormWR    = self._NormWeakRates
        nd        = self.nuclear_data

        if cfg.verbose_flag:
            _t0 = time.time()

        # ------------------------------------------------------------------
        # Temperature era boundaries [s]
        # ------------------------------------------------------------------
        t_start = t_of_T(cfg.T_start / cfg.MeV_to_Kelvin)
        t_weak  = t_of_T(cfg.T_weak  / cfg.MeV_to_Kelvin)
        t_nucl  = t_of_T(cfg.T_nucl  / cfg.MeV_to_Kelvin)
        t_end   = t_of_T(cfg.T_end   / cfg.MeV_to_Kelvin)

        # ------------------------------------------------------------------
        # Baryon density for the nuclear network
        # ------------------------------------------------------------------
        def nB(a):
            return cfg.n0CMB * cfg.eta0b / a**3   # [MeV^3]

        def etab_of_T(T_K):
            T_MeV = T_K / cfg.MeV_to_Kelvin
            ngCMB = (2. * zeta(3)) / np.pi**2 * T_MeV**3
            return nB(self._a_of_T(T_MeV)) / ngCMB

        def rhoB_BBN(a):
            n0B = cfg.n0CMB * cfg.eta0b
            return cfg.ma * n0B * cfg.MeV4_to_gcmm3 / a**3  # [g cm^-3]

        # ------------------------------------------------------------------
        # Nuclear network (import here to avoid circular init-time imports)
        # ------------------------------------------------------------------
        from PyPR.PyPR_nuclear_net import UpdateNuclearRates
        PyPRnucl = UpdateNuclearRates(nd, cfg)

        # ------------------------------------------------------------------
        # Local thermal equilibrium abundance
        # ------------------------------------------------------------------
        def YA(name, Yn, Yp, T):
            x     = cfg.Nuclides[name]
            A     = x[0] + x[1]
            Z     = x[1]
            N     = A - Z
            Mass  = (A * cfg.ma * cfg.MeV
                     + cfg.keV * cfg.NuclExcessMass[name]
                     - Z * cfg.me * cfg.MeV)
            BindE = (N * cfg.NuclExcessMass["n"]
                     + Z * cfg.NuclExcessMass["p"]
                     - cfg.NuclExcessMass[name])
            NormYA = (Mass / ((cfg.mn * cfg.MeV)**(A - Z)
                              * (cfg.mp * cfg.MeV)**Z))**(3. / 2.)
            return ((2 * cfg.NuclSpin[name] + 1)
                    * zeta(3)**(A - 1) * np.pi**((1 - A) / 2.)
                    * 2**((3 * A - 5) / 2.)
                    * NormYA
                    * (cfg.kB * T)**(3. / 2. * (A - 1))
                    * etab_of_T(T)**(A - 1)
                    * Yp**Z * Yn**(A - Z)
                    * np.exp(BindE * cfg.keV / (cfg.kB * T)))

        # ------------------------------------------------------------------
        # High-temperature (HT) era: only n and p
        # ------------------------------------------------------------------
        if cfg.verbose_flag:
            print("[nucl] Solving neutron decoupling at high temperature era")

        nTOp_frwrd_HT = self._nTOp_frwrd_HT
        nTOp_bkwrd_HT = self._nTOp_bkwrd_HT

        def nTOp_frwrd_HT_norm(T): return NormWR * nTOp_frwrd_HT(T)
        def nTOp_bkwrd_HT_norm(T): return NormWR * nTOp_bkwrd_HT(T)

        def Yn_i_func(T):
            b = nTOp_bkwrd_HT_norm(T)
            return b / (b + nTOp_frwrd_HT_norm(T))

        def Y_prime_HT(t, Y):
            T_K = T_of_t(t) * cfg.MeV_to_Kelvin
            f   = nTOp_frwrd_HT_norm(T_K)
            b   = nTOp_bkwrd_HT_norm(T_K)
            return b * Y[1] - f * Y[0], f * Y[0] - b * Y[1]

        Yn_i = Yn_i_func(cfg.T_start)
        Yp_i = 1. - Yn_i
        _t_ht0 = time.time()
        sol_HT = solve_ivp(Y_prime_HT, [t_start, t_weak], [Yn_i, Yp_i],
                           method='LSODA', rtol=cfg.numerical_precision, atol=1e-10)
        if cfg.debug_flag:
            print((f"[nucl] [HT] Finished solve_ivp in {time.time()-_t_ht0:.2f} s "
                   f"(status={sol_HT.status}, nfev={sol_HT.nfev})"), flush=True)
        Yn_HT_f, Yp_HT_f = sol_HT.y[0][-1], sol_HT.y[1][-1]

        if cfg.verbose_flag:
            print(f"[nucl] HT done in {time.time()-_t0:.2f} s")

        # ------------------------------------------------------------------
        # ODE systems for the full network
        # ------------------------------------------------------------------
        nTOp_frwrd_MT     = self._nTOp_frwrd_MT
        nTOp_bkwrd_MT     = self._nTOp_bkwrd_MT
        nTOp_frwrd_LT     = self._nTOp_frwrd_LT
        nTOp_bkwrd_LT     = self._nTOp_bkwrd_LT

        def make_nTOp_pair(frwrd_raw, bkwrd_raw):
            def f(T): return NormWR * frwrd_raw(T)
            def b(T): return NormWR * bkwrd_raw(T)
            return f, b

        if cfg.smallnet_flag:
            def _Y_prime(t, Y, nTOp_f, nTOp_b):
                rho  = rhoB_BBN(a_of_t(t))
                T_K  = T_of_t(t) * cfg.MeV_to_Kelvin
                return PyPRnucl.rhs(Y, T_K, rho, nTOp_f, nTOp_b)

            def _Jac(t, Y, nTOp_f, nTOp_b):
                rho = rhoB_BBN(a_of_t(t))
                T_K = T_of_t(t) * cfg.MeV_to_Kelvin
                return PyPRnucl.Jacobian(Y, T_K, rho, nTOp_f, nTOp_b)

            nTOp_f_MT, nTOp_b_MT = make_nTOp_pair(nTOp_frwrd_MT, nTOp_bkwrd_MT)
            nTOp_f_LT, nTOp_b_LT = make_nTOp_pair(nTOp_frwrd_LT, nTOp_bkwrd_LT)

            def Y_prime_MT(t, Y): return _Y_prime(t, Y, nTOp_f_MT, nTOp_b_MT)
            def Jacobian_MT(t, Y): return _Jac(t, Y, nTOp_f_MT, nTOp_b_MT)
            def Y_prime_LT(t, Y): return _Y_prime(t, Y, nTOp_f_LT, nTOp_b_LT)
            def Jacobian_LT(t, Y): return _Jac(t, Y, nTOp_f_LT, nTOp_b_LT)

        else:
            def _Y_prime_MT(t, Y, nTOp_f, nTOp_b):
                rho = rhoB_BBN(a_of_t(t))
                T_K = T_of_t(t) * cfg.MeV_to_Kelvin
                return PyPRnucl.rhsMT(Y, T_K, rho, nTOp_f, nTOp_b)

            def _Y_prime_LT(t, Y, nTOp_f, nTOp_b):
                rho = rhoB_BBN(a_of_t(t))
                T_K = T_of_t(t) * cfg.MeV_to_Kelvin
                return PyPRnucl.rhsLT(Y, T_K, rho, nTOp_f, nTOp_b)

            nTOp_f_MT, nTOp_b_MT = make_nTOp_pair(nTOp_frwrd_MT, nTOp_bkwrd_MT)
            nTOp_f_LT, nTOp_b_LT = make_nTOp_pair(nTOp_frwrd_LT, nTOp_bkwrd_LT)

            def Y_prime_MT(t, Y): return _Y_prime_MT(t, Y, nTOp_f_MT, nTOp_b_MT)
            def Jacobian_MT(t, Y):
                rho = rhoB_BBN(a_of_t(t)); T_K = T_of_t(t)*cfg.MeV_to_Kelvin
                return PyPRnucl.JacobianMT(Y, T_K, rho, nTOp_f_MT, nTOp_b_MT)
            def Y_prime_LT(t, Y): return _Y_prime_LT(t, Y, nTOp_f_LT, nTOp_b_LT)
            def Jacobian_LT(t, Y):
                rho = rhoB_BBN(a_of_t(t)); T_K = T_of_t(t)*cfg.MeV_to_Kelvin
                return PyPRnucl.JacobianLT(Y, T_K, rho, nTOp_f_LT, nTOp_b_LT)

        # ------------------------------------------------------------------
        # Mid-temperature (MT) era
        # ------------------------------------------------------------------
        if cfg.verbose_flag:
            print("[nucl] Solving nuclear network at mid temperature era")

        Yd_i   = YA("H2",  Yn_HT_f, Yp_HT_f, cfg.T_weak)
        Yt_i   = YA("H3",  Yn_HT_f, Yp_HT_f, cfg.T_weak)
        YHe3_i = YA("He3", Yn_HT_f, Yp_HT_f, cfg.T_weak)
        Ya_i   = YA("He4", Yn_HT_f, Yp_HT_f, cfg.T_weak)
        YLi7_i = YA("Li7", Yn_HT_f, Yp_HT_f, cfg.T_weak)
        YBe7_i = YA("Be7", Yn_HT_f, Yp_HT_f, cfg.T_weak)

        #if cfg.debug_flag:
        #    print(" IC at MT:", Yn_HT_f, Yp_HT_f, Yd_i, Yt_i, YHe3_i, Ya_i, YLi7_i, YBe7_i)

        if cfg.smallnet_flag:
            Yi_MT = [Yn_HT_f, Yp_HT_f, Yd_i, Yt_i, YHe3_i, Ya_i, YLi7_i, YBe7_i]
            _t_mt0 = time.time()
            sol_MT = solve_ivp(Y_prime_MT, [t_weak, t_nucl], Yi_MT,
                               method='BDF', jac=Jacobian_MT, rtol=cfg.numerical_precision, atol=1e-13)
            if cfg.debug_flag:
                print((f"[nucl] [MT] Finished solve_ivp (small network) in {time.time()-_t_mt0:.2f} s "
                       f"(status={sol_MT.status}, nfev={sol_MT.nfev})"), flush=True)
            (Yn_MT, Yp_MT, Yd_MT, Yt_MT,
             YHe3_MT, Ya_MT, YLi7_MT, YBe7_MT) = [sol_MT.y[i][-1] for i in range(8)]
        else:
            YHe6_i = YA("He6", Yn_HT_f, Yp_HT_f, cfg.T_weak)
            YLi8_i = YA("Li8", Yn_HT_f, Yp_HT_f, cfg.T_weak)
            YLi6_i = YA("Li6", Yn_HT_f, Yp_HT_f, cfg.T_weak)
            YB8_i  = YA("B8",  Yn_HT_f, Yp_HT_f, cfg.T_weak)
            Yi_MT  = [Yn_HT_f, Yp_HT_f, Yd_i, Yt_i, YHe3_i, Ya_i, YLi7_i, YBe7_i,
                      YHe6_i, YLi8_i, YLi6_i, YB8_i]
            _t_mt0 = time.time()
            sol_MT = solve_ivp(Y_prime_MT, [t_weak, t_nucl], Yi_MT,
                               method='BDF', jac=Jacobian_MT, rtol=cfg.numerical_precision, atol=1e-13)
            if cfg.debug_flag:
                print((f"[nucl] [MT] Finished solve_ivp (full network) in {time.time()-_t_mt0:.2f} s "
                       f"(status={sol_MT.status}, nfev={sol_MT.nfev})"), flush=True)
            (Yn_MT, Yp_MT, Yd_MT, Yt_MT, YHe3_MT, Ya_MT, YLi7_MT, YBe7_MT,
             YHe6_MT, YLi8_MT, YLi6_MT, YB8_MT) = [sol_MT.y[i][-1] for i in range(12)]

        #if cfg.debug_flag:
        #    print(" MT end:", Yn_MT, Yp_MT, Yd_MT, Yt_MT, YHe3_MT, Ya_MT, YLi7_MT, YBe7_MT)
        if cfg.verbose_flag:
            print(f"[nucl] MT done in {time.time()-_t0:.2f} s")

        # ------------------------------------------------------------------
        # Low-temperature (LT) era
        # ------------------------------------------------------------------
        if cfg.verbose_flag:
            print("[nucl] Solving nuclear network at low temperature era")

        if cfg.smallnet_flag:
            Yi_LT  = [Yn_MT, Yp_MT, Yd_MT, Yt_MT, YHe3_MT, Ya_MT, YLi7_MT, YBe7_MT]
            _t_lt0 = time.time()
            sol_LT = solve_ivp(Y_prime_LT, [t_nucl, t_end], Yi_LT,
                               method='BDF', jac=Jacobian_LT, rtol=10.*cfg.numerical_precision, atol=1e-15)
            if cfg.debug_flag:
                print((f"[nucl] [LT] Finished solve_ivp (small network) in {time.time()-_t_lt0:.2f} s "
                       f"(status={sol_LT.status}, nfev={sol_LT.nfev})"), flush=True)
            (Yn_f, Yp_f, Yd_f, Yt_f,
             YHe3_f, Ya_f, YLi7_f, YBe7_f) = [sol_LT.y[i][-1] for i in range(8)]
        else:
            Yi_LT = [Yn_MT, Yp_MT, Yd_MT, Yt_MT, YHe3_MT, Ya_MT, YLi7_MT, YBe7_MT,
                     YHe6_MT, YLi8_MT, YLi6_MT, YB8_MT]
            _t_lt0 = time.time()
            sol_LT = solve_ivp(Y_prime_LT, [t_nucl, t_end], Yi_LT,
                               method='BDF', jac=Jacobian_LT, rtol=10.*cfg.numerical_precision, atol=1e-15)
            if cfg.debug_flag:
                print((f"[nucl] [LT] Finished solve_ivp (full network) in {time.time()-_t_lt0:.2f} s "
                       f"(status={sol_LT.status}, nfev={sol_LT.nfev})"), flush=True)
            (Yn_f, Yp_f, Yd_f, Yt_f, YHe3_f, Ya_f, YLi7_f, YBe7_f,
             YHe6_f, YLi8_f, YLi6_f, YB8_f) = [sol_LT.y[i][-1] for i in range(12)]

        #if cfg.debug_flag:
        #    print(" LT end:", Yn_f, Yp_f, Yd_f, Yt_f, YHe3_f, Ya_f, YLi7_f, YBe7_f)
        if cfg.verbose_flag:
            print(f"[nucl] LT done in {time.time()-_t0:.1f} s")
            print("-" * 50)
            print("Predicted primordial abundances at the end of BBN")
            print("-" * 50)
            print(f"Yp    = {Yp_f}")
            print(f"Yd    = {Yd_f}")
            print(f"Yt    = {Yt_f}")
            print(f"YHe3  = {YHe3_f}")
            print(f"Ya    = {Ya_f}")
            print(f"YLi7  = {YLi7_f}")
            print(f"YBe7  = {YBe7_f}")

        # ------------------------------------------------------------------
        # Store final Y values for direct access (used by get_quantity)
        # ------------------------------------------------------------------
        if cfg.smallnet_flag:
            self._Y_final = dict(zip(_NUC_NAMES_SMALL,
                [Yn_f, Yp_f, Yd_f, Yt_f, YHe3_f, Ya_f, YLi7_f, YBe7_f]))
        else:
            self._Y_final = dict(zip(_NUC_NAMES_FULL,
                [Yn_f, Yp_f, Yd_f, Yt_f, YHe3_f, Ya_f, YLi7_f, YBe7_f,
                 YHe6_f, YLi8_f, YLi6_f, YB8_f]))

        # ------------------------------------------------------------------
        # Build abundance interpolator (always, so __getitem__ works)
        # ------------------------------------------------------------------
        self._abundance_names = (_NUC_NAMES_SMALL if cfg.smallnet_flag
                                 else _NUC_NAMES_FULL)
        n_nuc = len(self._abundance_names)
        _Y_HT = np.zeros((sol_HT.t.size, n_nuc))
        _Y_HT[:, 0] = sol_HT.y[0]
        _Y_HT[:, 1] = sol_HT.y[1]
        _t_nuc = np.concatenate((sol_HT.t, sol_MT.t[1:], sol_LT.t[1:]))
        _Y_nuc = np.vstack((_Y_HT, sol_MT.y.T[1:, :], sol_LT.y.T[1:, :]))
        self._Y_of_t = interp1d(_t_nuc, _Y_nuc, axis=0, bounds_error=False,
                                fill_value=(0, _Y_nuc[-1]))

        # ------------------------------------------------------------------
        # Optional output: full time evolution of background + abundances
        # ------------------------------------------------------------------
        if cfg.output_time_evolution:
            self._write_time_evolution(
                sol_HT, sol_MT, sol_LT, t_weak, t_nucl,
                nTOp_frwrd_HT_norm, nTOp_bkwrd_HT_norm,
                nTOp_f_MT, nTOp_b_MT, nTOp_f_LT, nTOp_b_LT,
                PyPRnucl,
            )

        # ------------------------------------------------------------------
        # Final observables
        # ------------------------------------------------------------------
        Tg_last  = self._Tg_vec[-1]
        Tnu_last = self._Tnu_vec[-1]

        Neff = self._N_eff(Tg_last, Tnu_last, Tnu_last, Tnu_last)

        YPBBN  = 4. * Ya_f
        YPCMB  = ((cfg.He4Overma / 4.) * YPBBN
                  / ((cfg.He4Overma / 4.) * YPBBN + cfg.HOverma * (1. - YPBBN)))

        self._results = {
            "Neff":            Neff,
            "Omeganurel":      self._Omeganuh2_relnu() * 1e+6,
            "OneOverOmeganunr": 1. / (self._Omeganuh2_nrnu() * 1e-6),
            "YPCMB":           YPCMB,
            "YPBBN":           YPBBN,
            "DoH":             Yd_f / Yp_f,
            "He3oH":           (Yt_f + YHe3_f) / Yp_f,
            "He3oHe4":         (Yt_f + YHe3_f) / Ya_f,
            "Li7oH":           (YLi7_f + YBe7_f) / Yp_f,
        }
        return self._results

    def _write_time_evolution(self, sol_HT, sol_MT, sol_LT, t_weak, t_nucl,
                              nTOp_frwrd_HT_norm, nTOp_bkwrd_HT_norm,
                              nTOp_f_MT, nTOp_b_MT, nTOp_f_LT, nTOp_b_LT,
                              PyPRnucl):
        cfg = self.cfg
        if cfg.smallnet_flag:
            nuc_cols = ["Yn", "Yp", "Yd", "Yt", "YHe3", "Ya", "YLi7", "YBe7"]
        else:
            nuc_cols = ["Yn", "Yp", "Yd", "Yt", "YHe3", "Ya", "YLi7", "YBe7",
                        "YHe6", "YLi8", "YLi6", "YB8"]
        n_nuc = len(nuc_cols)

        Y_of_t = self._Y_of_t

        # Uniform log-spaced output grid from T_start_cosmo to end of LT era
        t_cosmo = self._t_of_T(cfg.T_start_cosmo / cfg.MeV_to_Kelvin)
        t_end   = sol_LT.t[-1]
        t_out   = np.logspace(np.log10(t_cosmo), np.log10(t_end), cfg.output_n_points)

        a_out = self._a_of_t(t_out)
        T_out = self._T_of_t(t_out)

        Tnue_of_t   = interp1d(self._t_vec, self._Tnue_vec,   bounds_error=False,
                               fill_value="extrapolate", kind='linear')
        Tnumu_of_t  = interp1d(self._t_vec, self._Tnumu_vec,  bounds_error=False,
                               fill_value="extrapolate", kind='linear')
        Tnutau_of_t = interp1d(self._t_vec, self._Tnutau_vec, bounds_error=False,
                               fill_value="extrapolate", kind='linear')
        Tnue_out   = Tnue_of_t(t_out)
        Tnumu_out  = Tnumu_of_t(t_out)
        Tnutau_out = Tnutau_of_t(t_out)

        H_out = np.array([
            self._Hubble(T_out[i], Tnue_out[i], Tnumu_out[i], Tnutau_out[i])
            for i in range(t_out.size)
        ])

        # Weak rates: zero before nuclear network starts
        t_start = sol_HT.t[0]
        T_K_out = T_out * cfg.MeV_to_Kelvin
        weak_n_to_p_out = np.zeros_like(t_out)
        weak_p_to_n_out = np.zeros_like(t_out)

        mask_ht = (t_out >= t_start) & (t_out <= t_weak)
        mask_mt = (t_out >  t_weak)  & (t_out <= t_nucl)
        mask_lt =  t_out >  t_nucl

        weak_n_to_p_out[mask_ht] = nTOp_frwrd_HT_norm(T_K_out[mask_ht])
        weak_p_to_n_out[mask_ht] = nTOp_bkwrd_HT_norm(T_K_out[mask_ht])
        weak_n_to_p_out[mask_mt] = nTOp_f_MT(T_K_out[mask_mt])
        weak_p_to_n_out[mask_mt] = nTOp_b_MT(T_K_out[mask_mt])
        weak_n_to_p_out[mask_lt] = nTOp_f_LT(T_K_out[mask_lt])
        weak_p_to_n_out[mask_lt] = nTOp_b_LT(T_K_out[mask_lt])

        # Abundances: zero before nuclear network starts
        Y_out = np.zeros((len(t_out), n_nuc))
        mask_nuc = t_out >= t_start
        Y_out[mask_nuc] = Y_of_t(t_out[mask_nuc])

        if cfg.output_rates_time_evolution:
            rxn_rate_cols = sorted(
                name for name in dir(PyPRnucl)
                if name.endswith("_frwrd") and callable(getattr(PyPRnucl, name))
            )
            rxn_rate_out = np.zeros((len(t_out), len(rxn_rate_cols)))
            rxn_rate_out[mask_nuc] = np.column_stack([
                getattr(PyPRnucl, name)(T_K_out[mask_nuc]) for name in rxn_rate_cols
            ])
        else:
            rxn_rate_cols = []
            rxn_rate_out = np.empty((len(t_out), 0))

        Nheating_out = self._N_NEVO_of_Tg(T_out)

        import os
        out_path = (cfg.output_file if os.path.isabs(cfg.output_file)
                    else os.path.join(cfg.working_dir, cfg.output_file))
        out_data = np.column_stack((a_out, T_out, t_out, H_out,
                                    Tnue_out, Tnumu_out, Tnutau_out, Nheating_out,
                                    Y_out,
                                    weak_n_to_p_out, weak_p_to_n_out, rxn_rate_out))
        out_header = "\t".join(["a", "T", "t", "H",
                                 "Tnue", "Tnumu", "Tnutau", "Nheating"]
                               + nuc_cols
                               + ["n_to_p_weak_rate", "p_to_n_weak_rate"] + rxn_rate_cols)
        np.savetxt(out_path, out_data, delimiter='\t', header=out_header, comments='')

        if cfg.verbose_flag or cfg.debug_flag:
            print(f"[output] Time-evolution data written to {out_path}")

    # ======================================================================
    # Public API
    # ======================================================================

    @property
    def T_of_t(self):
        """T_γ(t) interpolator [MeV], available after initialisation."""
        return self._T_of_t

    @property
    def t_of_T(self):
        """t(T_γ) interpolator [s], available after initialisation."""
        return self._t_of_T

    def __getitem__(self, species):
        """Return Y(t) for a species name (e.g. 'H2', 'He4', 'Li7').

        Calls solve() automatically if needed.
        """
        self._ensure_solved()
        if species not in self._abundance_names:
            raise KeyError(
                f"Unknown species '{species}'. Available: {self._abundance_names}"
            )
        idx = self._abundance_names.index(species)
        def fn(t):
            t_arr = np.atleast_1d(np.asarray(t, dtype=float))
            vals  = self._Y_of_t(t_arr)[:, idx]
            return float(vals[0]) if np.ndim(t) == 0 else vals
        return fn

    def _ensure_solved(self):
        if self._results is None:
            self.solve()

    def PyPRresults(self):
        """Return the BBN result dict, running ``solve()`` first if needed."""
        self._ensure_solved()
        return self._results

    # Convenience accessors
    def Neff(self):          self._ensure_solved(); return self._results["Neff"]
    def Omeganurel(self):    self._ensure_solved(); return self._results["Omeganurel"]
    def Omeganunonrel(self): self._ensure_solved(); return 1. / self._results["OneOverOmeganunr"]
    def YPCMB(self):         self._ensure_solved(); return self._results["YPCMB"]
    def YPBBN(self):         self._ensure_solved(); return self._results["YPBBN"]
    def DoH(self):           self._ensure_solved(); return self._results["DoH"]
    def He3oH(self):         self._ensure_solved(); return self._results["He3oH"]
    def Li7oH(self):         self._ensure_solved(); return self._results["Li7oH"]

    def get_quantity(self, quantity):
        """Return a scalar BBN quantity by name.

        Accepts any key from the result dict ('YPBBN', 'DoH', 'He3oH',
        'Li7oH', 'Neff', 'YPCMB', ...) or a nuclide name from
        cfg.Nuclides ('H2', 'He4', 'Li7', ...) for the final mass fraction Y.
        """
        self._ensure_solved()
        if quantity in self._results:
            return self._results[quantity]
        if quantity in self._Y_final:
            return self._Y_final[quantity]
        raise ValueError(
            f"Unknown quantity '{quantity}'. "
            f"Valid result keys: {list(self._results.keys())}. "
            f"Valid nuclide names: {list(self._Y_final.keys())}."
        )


# ---------------------------------------------------------------------------
# MC result classes
# ---------------------------------------------------------------------------

class MCQuantityResult:
    """MC statistics for a single BBN quantity.

    Attributes
    ----------
    central : float
        Value at nominal rates (all p_* = 0).
    mean : float
        Mean of MC samples.
    std : float
        1σ standard deviation of MC samples.
    values : np.ndarray, shape (num_mc,)
        All individual MC sample values.
    """
    __slots__ = ('central', 'mean', 'std', 'values')

    def __init__(self, central, samples):
        self.central = float(central)
        self.values  = np.asarray(samples)
        self.mean    = float(np.mean(self.values))
        self.std     = float(np.std(self.values))

    def __repr__(self):
        return (f"MCQuantityResult(central={self.central:.6g}, "
                f"mean={self.mean:.6g}, std={self.std:.6g}, "
                f"n={len(self.values)})")


class MCResult:
    """MC results for one or more BBN quantities, indexed by name.

    Usage::

        mc = mc_uncertainty(100, ['YPBBN', 'DoH'], params=...)
        mc['YPBBN'].mean
        mc['YPBBN'].std
        mc['YPBBN'].values
        mc['DoH'].central
    """
    def __init__(self, data):
        self._data = data   # dict: str -> MCQuantityResult

    def __getitem__(self, quantity):
        return self._data[quantity]

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        lines = [f"MCResult({len(self._data)} quantities):"]
        for k, v in self._data.items():
            lines.append(f"  {k}: central={v.central:.6g}, "
                         f"mean={v.mean:.6g}, std={v.std:.6g}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level MC worker (must be at module level for joblib pickling)
# ---------------------------------------------------------------------------

def _mc_run_one(base_params, rate_keys, quantities, seed):
    """Single MC worker: randomise nuclear rates and return all requested quantities."""
    rng    = np.random.default_rng(seed)
    p_vals = rng.standard_normal(len(rate_keys))
    params = {**base_params,
              **{k: float(v) for k, v in zip(rate_keys, p_vals)}}
    inst   = PyPRclass(params=params)
    inst.solve()
    return [inst.get_quantity(q) for q in quantities]


def mc_uncertainty(num_mc, quantity, params=None, n_jobs=-1, seed=0):
    """Estimate nuclear rate uncertainties on BBN observables via Monte Carlo.

    Each MC sample draws all active nuclear rate offsets p_* independently from
    N(0,1) and runs a full PyPRIMAT solve.  The 12 key rates are varied for the
    small network; all 63 for the full network.

    Parameters
    ----------
    num_mc : int
        Number of MC samples.
    quantity : str or list of str
        A key from the result dict ('YPBBN', 'DoH', 'He3oH',
        'Li7oH', 'Neff', 'YPCMB', ...) or a nuclide name ('H2', 'He4',
        'Li7', ...) for the final mass fraction Y.  Pass a list to evaluate
        multiple quantities in one MC pass (more efficient than separate calls).
    params : dict, optional
        Base parameters for PyPRclass (e.g. Omegabh2, smallnet_flag).
    n_jobs : int
        Number of parallel workers passed to joblib.Parallel (-1 = all CPUs).
    seed : int
        Base random seed; sample i uses seed + i for reproducibility.
        When evaluating on a parameter grid (e.g. scanning Ω_b h²), use the
        **same seed at every grid point** so that sample i draws the same rate
        vector p_* everywhere.  This correlates the MC noise across the grid,
        making any finite-sample bias cancel when comparing predictions at
        different parameter values.

    Returns
    -------
    MCResult
        Dict-like object indexed by quantity name.  Each value is an
        ``MCQuantityResult`` with attributes ``central``, ``mean``, ``std``,
        and ``values``.

    Example
    -------
    >>> mc = mc_uncertainty(100, ['YPBBN', 'DoH'], params={'Omegabh2': 0.022})
    >>> mc['YPBBN'].central
    >>> mc['YPBBN'].std
    >>> mc['DoH'].values   # full sample array
    """
    from joblib import Parallel, delayed
    from PyPR.PyPR_config import DEFAULT_PARAMS

    quantities = [quantity] if isinstance(quantity, str) else list(quantity)

    base_params = dict(params or {})
    base_params.setdefault('verbose_flag', False)
    base_params.setdefault('debug_flag',   False)

    all_p_keys = [k for k in DEFAULT_PARAMS if k.startswith('p_')]
    smallnet   = base_params.get('smallnet_flag', DEFAULT_PARAMS['smallnet_flag'])
    rate_keys  = all_p_keys[:12] if smallnet else all_p_keys

    central_inst = PyPRclass(params=base_params)
    central_inst.solve()
    centrals = [central_inst.get_quantity(q) for q in quantities]

    raw = Parallel(n_jobs=n_jobs)(
        delayed(_mc_run_one)(base_params, rate_keys, quantities, seed + i)
        for i in range(num_mc)
    )
    samples = np.array(raw)   # shape (num_mc, len(quantities))

    return MCResult({
        q: MCQuantityResult(centrals[j], samples[:, j])
        for j, q in enumerate(quantities)
    })
