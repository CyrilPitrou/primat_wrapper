# -*- coding: utf-8 -*-
import PyPR.PyPR_init as PyPRini
import PyPR.PyPR_main as PyPRmain
import PyPR.PyPR_thermo as PyPRthermo



def compute_abundances(omegabh2=0.022425, Nrelat=0.0, fEDE=0., zcEDE=1e8, wnEDE=1.0):
    """
    Compute BBN abundances using PyPRIMAT with EDE parameters.
    Returns (Yp, D/H) tuple.
    We could return directly a dictionary if we want but we keep things simple for the moment.
    """
    acEDE = 1/(1+zcEDE)
    amaxEDE = acEDE * (4/(3*wnEDE-1))**(1/(3*wnEDE+3))
    TmaxEDE = PyPRini.T0CMB/amaxEDE/PyPRini.MeV_to_Kelvin # MeV
    TcEDE = PyPRini.T0CMB/acEDE/PyPRini.MeV_to_Kelvin # MeV
    rhocEDEac = fEDE/(1-fEDE)*PyPRthermo.rho_g(TmaxEDE)*(1+ 3.044*7/8*(4/11)**(4/3.)) /2.*(1 + 4/(3*wnEDE-1))

    def rho_NP(T_NP):
        return 2*rhocEDEac/(1+(TcEDE/T_NP)**(3*wnEDE+3))

    def p_NP(T_NP):
        return -rho_NP(T_NP) #If we do this the only change is in Hubble factor, as we want.

    def drho_NP_dT(T_NP):
        return 0 #If we do this the only change is in Hubble factor, as we want.

    # Set PyPR flags and parameters
    PyPRini.verbose = True
    PyPRini.aTid_flag = True
    PyPRini.compute_bckg_flag = True #True is slower but more accurate since expansion is modified by EDE
    PyPRini.compute_nTOp_flag = False #True is slower and we do not modify the rates with EDE. But we should be careful in general.
    PyPRini.compute_nTOp_thermal_flag = False
    PyPRini.save_bckg_flag = False #Only the first time this is needed
    PyPRini.save_nTOp_flag = False #Only the first time this is needed
    PyPRini.NP_nu_flag = False
    PyPRini.NP_e_flag = True #That is where there is space in PyPR for extra energy density.
    PyPRini.Omegabh2 = omegabh2
    PyPRini.eta0b = PyPRini.Omegabh2_to_eta0b * PyPRini.Omegabh2
    PyPRini.DeltaNeff  = Nrelat #Not exactly the same definition in PRIMAT but who cares ?
    PyPRini.nacreii_flag = False
    PyPRini.smallnet_flag = True

    # Run PyPR with EDE
    PyPREDE = PyPRmain.PyPRclass(rho_NP, p_NP, drho_NP_dT)
    results = PyPREDE.PyPRresults()
    # [Neff, Ω_ν h^2 × 10^6 (rel), ∑ m_ν / (Ω_ν h^2) [eV], YP (CMB), YP (BBN), D/H × 10^5, 3He/H × 10^5, 7Li/H × 10^10]
    YP = results[4]
    DH = 1e-5 * results[5]
    return YP, DH
