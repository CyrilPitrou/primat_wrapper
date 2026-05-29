# -*- coding: utf-8 -*-
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.special import zeta

class PyPRclass(object):
    def __init__(self,my_rho_NP=0.,my_p_NP=0.,my_drho_NP_dT=0.,my_delta_rho_NP=0.):
        #############################
        # PyPRIMAT initialization #
        #############################
        banner = """
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃                                                 ┃
        ┃   ░█▀█░█░█░█▀█░█▀▄░▀█▀░█▄█░█▀█░▀█▀              ┃
        ┃   ░█▀▀░░█░░█▀▀░█▀▄░░█░░█░█░█▀█░░█░              ┃
        ┃   ░▀░░░░▀░░▀░░░▀░▀░▀▀▀░▀░▀░▀░▀░░▀░              ┃
        ┃                                                 ┃
        ┃            Welcome to PyPRIMAT                  ┃
        ┃                                                 ┃
        ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
        """
        import PyPR.PyPR_init as PyPRini
        #print('PyPRini.verbose_flag --> ',PyPRini.verbose_flag)
        if(PyPRini.verbose_flag):
            print(banner)
            start_time = time.time()

        ###############################
        # Checking if numba is installed and setting flag accordingly
        ###############################
        try:
            import numba
            PyPRini.numba_flag = True
            if(PyPRini.verbose_flag):
                print('Using Numba to speed up some integrations in PyPR_thermo.py')
        except ImportError:
            PyPRini.numba_flag = False
        try:
            import numdifftools
            PyPRini.numdiff_flag = True
            if(PyPRini.verbose_flag):
                print('Using numdifftools.Derivative to compute numerical derivatives in PyPR_main.py')
        except ImportError:
            PyPRini.numdiff_flag = False

        import PyPR.PyPR_thermo as PyPRthermo
        # Loading New Physics species (constructor default: none)
        PyPRthermo.rho_NP,PyPRthermo.p_NP,PyPRthermo.drho_NP_dT,PyPRthermo.delta_rho_NP=my_rho_NP,my_p_NP,my_drho_NP_dT,my_delta_rho_NP


        ################################
        # PyPRIMAT working directory #
        ################################
        my_dir = PyPRini.working_dir
        
        ##############################
        # Definition of temperatures #
        ##############################
        Tstart_MeV = PyPRini.T_start/PyPRini.MeV_to_Kelvin
        Tend_MeV = PyPRini.T_end/PyPRini.MeV_to_Kelvin

        ##################
        # Thermodynamics #
        ##################
        # Units adopted for background:
        # - Time in [s]
        # - Energy, temperature in [MeV]

        # Expansion rate from Friedmann equation
        def Hubble(Tg,Tnue,Tnumu,T_NP=0.):
            rho_pl = PyPRthermo.rho_g(Tg)+PyPRthermo.rho_e(Tg)-PyPRthermo.PofT(Tg)+Tg*PyPRthermo.dPdT(Tg)
            rho_3nu = PyPRthermo.rho_nu(Tnue)+2.*PyPRthermo.rho_nu(Tnumu)
            rho_tot = rho_pl+rho_3nu
            if(PyPRini.NP_thermo_flag):
                rho_tot += PyPRthermo.rho_NP(T_NP)
            if(PyPRini.NP_nu_flag):
                rho_tot += PyPRthermo.rho_NP(Tnue)
            if(PyPRini.NP_e_flag):
                rho_tot += PyPRthermo.rho_NP(Tg)
            return PyPRini.MeV_to_secm1*(rho_tot*8.*np.pi/(3.*PyPRini.Mpl**2))**0.5
        # Computing the background (if not pre-stored)
        if(PyPRini.compute_bckg_flag):
            # Integrated Boltzmann equations for temperature of species
            # Neutrino temperature evolution
            def dTnudt(Tg,Tnue,Tnumu,T_NP=0.):
                Hubble_T = Hubble(Tg,Tnue,Tnumu,T_NP)
                num = -12.*Hubble_T*PyPRthermo.rho_nu(Tnue)
                delta_rho_nu = (PyPRthermo.delta_rho_nue(Tg,Tnue,Tnumu)+2.*PyPRthermo.delta_rho_numu(Tg,Tnue,Tnumu))
                if(PyPRini.NP_thermo_flag):
                    delta_rho_nu += PyPRthermo.delta_rho_NP(Tg,Tnue,Tnumu,T_NP)
                num += delta_rho_nu
                den = 3.*PyPRthermo.drho_nu_dT(Tnue)
                if(PyPRini.NP_nu_flag):
                    num -= 3.*Hubble_T*(PyPRthermo.rho_NP(Tnue)+PyPRthermo.p_NP(Tnue))
                    den += PyPRthermo.drho_NP_dT(Tnue)
                return num/den
            # Plasma temperature evolution
            def dTgdt(Tg,Tnue,Tnumu,T_NP=0.):
                Hubble_T = Hubble(Tg,Tnue,Tnumu,T_NP)
                num = -(Hubble_T*(4.*PyPRthermo.rho_g(Tg)+3.*(PyPRthermo.rho_e(Tg)+PyPRthermo.p_e(Tg))+3.*Tg*PyPRthermo.dPdT(Tg)))
                # Sum of collision terms must vanish
                delta_rho_g = -(PyPRthermo.delta_rho_nue(Tg,Tnue,Tnumu)+2.*PyPRthermo.delta_rho_numu(Tg,Tnue,Tnumu))
                if(PyPRini.NP_thermo_flag):
                    delta_rho_g -= PyPRthermo.delta_rho_NP(Tg,Tnue,Tnumu,T_NP) # traceless collision operator
                num += delta_rho_g
                den = PyPRthermo.drho_g_dT(Tg)+PyPRthermo.drho_e_dT(Tg)+Tg*PyPRthermo.d2PdT2(Tg)
                if(PyPRini.NP_e_flag):
                    num -= 3.*Hubble_T*(PyPRthermo.rho_NP(Tg)+PyPRthermo.p_NP(Tg))
                    den += PyPRthermo.drho_NP_dT(Tg)
                return num/den
            # NP temperature evolution
            def dTNPdt(Tg,Tnue,Tnumu,T_NP):
                Hubble_T = Hubble(Tg,Tnue,Tnumu,T_NP)
                rho_NP = PyPRthermo.rho_NP(T_NP)
                p_NP = PyPRthermo.p_NP(T_NP)
                num = -3.*Hubble_T*(rho_NP+p_NP)
                delta_rho_NP = PyPRthermo.delta_rho_NP(Tg,Tnue,Tnumu,T_NP)
                num += delta_rho_NP
                den = PyPRthermo.drho_NP_dT(T_NP)
                return num/den
            def dTtotdt(t,T_vec):
                if(PyPRini.NP_thermo_flag):
                    Tg,Tnu,T_NP = T_vec
                    y_vec = dTgdt(Tg,Tnu,Tnu,T_NP),dTnudt(Tg,Tnu,Tnu,T_NP),dTNPdt(Tg,Tnue,Tnumu,T_NP)
                    return y_vec
                else:
                    Tg,Tnu = T_vec
                    y_vec = dTgdt(Tg,Tnu,Tnu),dTnudt(Tg,Tnu,Tnu)
                    return y_vec
            # Solution of Boltzmann equations for background thermodynamics
            tfin = PyPRini.t_end # [s]
            if(PyPRini.NP_thermo_flag):
                tini = 1./(2.*Hubble(Tstart_MeV,Tstart_MeV,Tstart_MeV,PyPRini.Tstart_NP)) # [s]
                sol_thermo_sampling = np.logspace(np.log10(tini),np.log10(tfin),PyPRini.n_sampling)
                sol_thermo_sampling[0],sol_thermo_sampling[-1] = tini,tfin
                Tini_vec = [Tstart_MeV,Tstart_MeV,PyPRini.Tstart_NP]
                sol_thermo = solve_ivp(dTtotdt,[tini,tfin],Tini_vec,t_eval=sol_thermo_sampling,method='LSODA',rtol=1.e-6,atol=1.e-9)
                t_vec = sol_thermo.t
                Tg_vec = sol_thermo.y[0][:]
                Tnu_vec = sol_thermo.y[1][:]
                TNP_vec = sol_thermo.y[2][:]
            else:
                tini = 1./(2.*Hubble(Tstart_MeV,Tstart_MeV,Tstart_MeV)) # s
                sol_thermo_sampling = np.logspace(np.log10(tini),np.log10(tfin),PyPRini.n_sampling)
                sol_thermo_sampling[0],sol_thermo_sampling[-1] = tini,tfin
                Tini_vec = [Tstart_MeV,Tstart_MeV]
                sol_thermo = solve_ivp(dTtotdt,[tini,tfin],Tini_vec,t_eval=sol_thermo_sampling,method='LSODA',rtol=1.e-6,atol=1.e-9)
                t_vec = sol_thermo.t
                Tg_vec = sol_thermo.y[0][:]
                Tnu_vec = sol_thermo.y[1][:]
            # Save results for background thermodynamics
            if(PyPRini.save_bckg_flag):
                if(PyPRini.NP_thermo_flag):
                    np.savetxt(my_dir+"/PyPRrates/"+"thermo/Tgamma_Tnu_TNP.txt",np.c_[t_vec,Tg_vec,Tnu_vec,TNP_vec])
                else:
                    np.savetxt(my_dir+"/PyPRrates/"+"thermo/Tgamma_Tnu.txt",np.c_[t_vec,Tg_vec,Tnu_vec])
        else:
            if(PyPRini.NP_thermo_flag):
                t_vec,Tg_vec,Tnu_vec,TNP_vec = np.loadtxt(my_dir+"/PyPRrates/"+"thermo/Tgamma_Tnu_TNP.txt",unpack=True)
            else:
                t_vec,Tg_vec,Tnu_vec = np.loadtxt(my_dir+"/PyPRrates/"+"thermo/Tgamma_Tnu.txt",unpack=True)
                
        # Interpolation of Tnu(T) (and NP) for non-instantaneous decoupling effects in a(T)
        if(PyPRini.aTid_flag):
            TnuofT = interp1d(Tg_vec[:],Tnu_vec[:],bounds_error=False,fill_value="extrapolate",kind='linear')
            if(PyPRini.NP_thermo_flag):
                TNPofT = interp1d(Tg_vec[:],TNP_vec[:],bounds_error=False,fill_value="extrapolate",kind='linear')
        
        ################
        # N effective  #
        ################
        # Definition as extra radiation density relative to photons in units of 8/7 x (11/4)^(4/3)
        def N_eff(Tg,Tnue,Tnumu,T_NP=0.):
            rho_gamma = PyPRthermo.rho_g(Tg)
            rho_rad_tot = PyPRthermo.rho_nu(Tnue)+2.*PyPRthermo.rho_nu(Tnumu)+rho_gamma
            if(PyPRini.NP_thermo_flag):
                rho_rad_tot += PyPRthermo.rho_NP(T_NP)
            elif(PyPRini.NP_nu_flag):
                rho_rad_tot += PyPRthermo.rho_NP(Tnue)
            elif(PyPRini.NP_e_flag):
                rho_rad_tot += PyPRthermo.rho_NP(Tg)
            # normalization of extra radiation as neutrinos
            normDeltaNeff = (7./8.)*(4./11.)**(4./3.)
            return (rho_rad_tot-rho_gamma)/rho_gamma/normDeltaNeff
            
        ################################
        # Relic abundance of neutrinos #
        ################################
        # Cosmic abundance of single species of relativistic nu
        def Omeganuh2_relnu():
            Tnu0 = Tnu_vec[-1]/Tg_vec[-1]*PyPRini.T0CMB/PyPRini.MeV_to_Kelvin
            return (7.*np.pi**2/120.*Tnu0**4)/PyPRini.rhocOverh2 # dimensionless
        # Cosmic abundance of non-relativistic nu over sum of nu masses
        def Omeganuh2_nrnu():
            Tnu0 = Tnu_vec[-1]/Tg_vec[-1]*PyPRini.T0CMB/PyPRini.MeV_to_Kelvin
            return (3./2.*zeta(3)/np.pi**2*Tnu0**3)/PyPRini.rhocOverh2 # MeV
        
        ######################################################
        # FRW cosmological backround in radiation domination #
        ######################################################
        # Relation between time and temperature of the thermal bath
        t_of_T = interp1d(Tg_vec[:],t_vec[:],bounds_error=False,fill_value="extrapolate",kind='linear')
        t_of_T_vec = np.vectorize(t_of_T)
        T_of_t = interp1d(t_vec[:],Tg_vec[:],bounds_error=False,fill_value="extrapolate",kind='linear')
        T_of_t_vec = np.vectorize(T_of_t)
        
        ######################################################
        # Relation of scale factor with temperature and time #
        ######################################################
        # Non-instantaneous decoupling effects on the entropy of the plasma
        if(PyPRini.aTid_flag):
            # Heat rate due to neutrino (and NP) interactions with the plasma
            def N_nu_rate(T):
                Tnu = TnuofT(T)
                qdot_pl = -(PyPRthermo.delta_rho_nue(T,Tnu,Tnu)+2.*PyPRthermo.delta_rho_numu(T,Tnu,Tnu))
                Hubble_T = Hubble(T,Tnu,Tnu)
                if(PyPRini.NP_thermo_flag):
                    TNP = TNPofT(T)
                    qdot_pl -= PyPRthermo.delta_rho_NP(T,Tnu,Tnu,TNP)
                    Hubble_T = Hubble(T,Tnu,Tnu,TNP)
                res = -qdot_pl/Hubble_T/T**4
                return res
            # Plasma entropy density normalized to T^3 (constant after e+- annihilation)
            def sbar(T):
                return PyPRthermo.spl(T)/T**3
            # Numerical derivative of the above wrt to temperature
            if(PyPRini.numdiff_flag):
                from numdifftools import Derivative
                dsbardT = Derivative(sbar,n=1)
            else:
                def dsbardT(T):
                    dToT = 1.e-3
                    return (sbar((1.+dToT)*T)-sbar((1.-dToT)*T))/(2.*dToT*T)
            # dlog(a*T)/dlog(T)
            def dlnadlnT(lnT):
                T = np.exp(lnT)
                sbar_T = sbar(T)
                N_nu_T = N_nu_rate(T)
                return -(3.*sbar_T+T*dsbardT(T))/(3.*sbar_T+N_nu_T)
            # Log of scale factor as a function of log of temperature of thermal bath
            Tini_vec = [np.log(Tend_MeV),np.log(Tstart_MeV)]
            # Initial conditions using z = a*T and entropy conservation
            z0 = PyPRini.T0CMB/PyPRini.MeV_to_Kelvin # a0 = 1 --> z0 = T0
            # Assuming no change in plasma entropy per comoving volume after end of BBN
            zend = (z0/(sbar(Tend_MeV)/PyPRini.s0bar)**(1/3)) # iff d(spl*a^3) = 0
            # aend conveniently allows to sample from end of BBN instead of today
            T_sol_vec = np.logspace(np.log10(Tend_MeV),np.log10(Tstart_MeV),PyPRini.n_sampling)
            def dlna(lnT,y):
                return dlnadlnT(lnT)
            sol_lnalnT = solve_ivp(dlna,Tini_vec,[np.log(zend/Tend_MeV)],t_eval=np.log(T_sol_vec),method='LSODA',rtol=1.e-6,atol=1.e-9)
            sol_lnT = np.array(sol_lnalnT.t[:]).flatten()
            sol_lna = np.array(sol_lnalnT.y[:]).flatten()
            # log(a) as a function of log(T)
            lnalnT = interp1d(sol_lnT,sol_lna,bounds_error=False,fill_value="extrapolate")
        
        # Scale factor as a function of temperature of thermal bath
        def a_of_T(T):
            # Including non-instantaneous decoupling effects
            if(PyPRini.aTid_flag):
                return np.exp(lnalnT(np.log(T)))
            # Using instantaneous approximation
            else:
                spl_T = PyPRthermo.spl(T)
                return (PyPRini.s0CMB/spl_T)**(1./3.)
        a_of_T_vec = np.vectorize(a_of_T)
        # Scale factor as a function of time
        a_in = a_of_T(Tg_vec[0])
        a_fin = a_of_T(Tg_vec[-1])
        a_of_t = interp1d(t_vec[:],a_of_T_vec(Tg_vec),bounds_error=False,fill_value=(a_in,a_fin))
        
        ##########################################
        # Baryon density for the nuclear network #
        ##########################################
        # Baryon number density obtained as n0B = rho0B/mB
        # mB = averaged baryon mass in MeV: assumes helium fraction of 24.7%
        # rho0B = atomic density (it includes electron mass + binding energies)
        def nB(a):
            n0B = PyPRini.n0CMB*PyPRini.eta0b # baryon number density of today MeV^3
            return n0B/a**3 # MeV^3
        # Baryon-to-photon ratio as a function of temperature given in [K]
        def etab_of_T(T_K):
            T_MeV = T_K/PyPRini.MeV_to_Kelvin
            ngCMB = (2.*zeta(3))/(np.pi**2)*T_MeV**3
            return nB(a_of_T(T_MeV))/ngCMB
        # Baryon energy density adopted in the nuclear network
        # rhoB = nucleonic density (i.e. rho0B measured by CMB x ma/mB)
        def rhoB_BBN(a):
            n0B = PyPRini.n0CMB*PyPRini.eta0b
            rho0BmaOvermB = PyPRini.ma*n0B
            return rho0BmaOvermB*PyPRini.MeV4_to_gcmm3/a**3 # CGS, a0 = 1
            
        ######################################################
        # Definition of temperature eras for nuclear network #
        ######################################################
        t_start = t_of_T(PyPRini.T_start/PyPRini.MeV_to_Kelvin)
        t_weak = t_of_T(PyPRini.T_weak/PyPRini.MeV_to_Kelvin)
        t_nucl = t_of_T(PyPRini.T_nucl/PyPRini.MeV_to_Kelvin)
        t_end = t_of_T(PyPRini.T_end/PyPRini.MeV_to_Kelvin)

        ##############################
        # Import n <--> p weak rates #
        ###############################
        import PyPR.PyPR_eval_nTOp as PyPRevalnTOp
        import PyPR.PyPR_nTOp as PyPRnTOp
        nTOp_frwrd_HT,nTOp_bkwrd_HT,nTOp_frwrd_MT,nTOp_bkwrd_MT,nTOp_frwrd_LT,nTOp_bkwrd_LT = PyPRnTOp.RecomputeWeakRates([Tg_vec,Tnu_vec])
        
        ############################
        # Weak rates normalization #
        ############################
        if(PyPRini.tau_n_flag):
            Fn = PyPRevalnTOp.ComputeFn()
            NormWeakRates = 1./(Fn*PyPRini.tau_n) # normalization in [s-1]
        else:
            GFtilde2 = (PyPRini.GF*PyPRini.Vud)**2*(1+3.*PyPRini.gA**2)/(2.*np.pi**3)
            NormWeakRates = PyPRini.MeV_to_secm1*(GFtilde2*PyPRini.me**5) # normalization in [s-1]
        
        ##################################
        # High temperature era: only p,n #
        ##################################
        # Initial conditions from detailed balance
        def Yni(T):
            return nTOp_bkwrd_HT(T)/(nTOp_bkwrd_HT(T) + nTOp_frwrd_HT(T))
        def Ypi(T):
            return (1.-Yni(T))

        # Weak rates at HT
        def nTOp_frwrd(T):
            return NormWeakRates*nTOp_frwrd_HT(T)
        def nTOp_bkwrd(T):
            return NormWeakRates*nTOp_bkwrd_HT(T)
            
        def Yn_prime_HT(t,Y):
            T_t = T_of_t(t)*PyPRini.MeV_to_Kelvin # temperature in [K]
            return nTOp_bkwrd(T_t)*Y[1]-nTOp_frwrd(T_t)*Y[0]
            
        def Yp_prime_HT(t,Y):
            T_t = T_of_t(t)*PyPRini.MeV_to_Kelvin # temperature in [K]
            return nTOp_frwrd(T_t)*Y[0]-nTOp_bkwrd(T_t)*Y[1]
            
        def Y_prime_HT(t,Y):
            dY = Yn_prime_HT(t,Y),Yp_prime_HT(t,Y)
            return dY

        #############################
        # High temperature solution #
        #############################
        if(PyPRini.verbose_flag):
            print(" ")
            print("Solving neutron decoupling at high temperature era")
            
        # HT era definition
        t_init = t_start
        t_fin = t_weak
        #print("t_init and t_fin = ",t_init,t_fin)

        # HT initial conditions
        Yn_i = Yni(PyPRini.T_start)
        Yp_i = Ypi(PyPRini.T_start)
        #print("Final Yn_i,Yp_i = ",Yn_i,Yp_i)
        
        # Solving HT network
        Yi_vec = [Yn_i,Yp_i]
        sol_at_HT = solve_ivp(Y_prime_HT,[t_init,t_fin],Yi_vec,method='LSODA',rtol=1.e-6,atol=1.e-9)
        Yn_HT_f,Yp_HT_f = sol_at_HT.y[0][-1],sol_at_HT.y[1][-1]
        
        if(PyPRini.verbose_flag):
            print("--- running time: %s seconds ---" % (time.time() - start_time))
            print(" ")
            #print("Final Yn_HT_f,Yp_HT_f = ",Yn_HT_f,Yp_HT_f)
        
        ########################
        # Import nuclear rates #
        ########################
        if(PyPRini.smallnet_flag):
            import PyPR.PyPR_nuclear_net12 as PyPRnuclear
            PyPRnucl = PyPRnuclear.UpdateNuclearRates(PyPRini.p_npdg,PyPRini.p_dpHe3g,PyPRini.p_ddHe3n,PyPRini.p_ddtp,PyPRini.p_tpag,PyPRini.p_tdan,PyPRini.p_taLi7g,PyPRini.p_He3ntp,PyPRini.p_He3dap,PyPRini.p_He3aBe7g,PyPRini.p_Be7nLi7p,PyPRini.p_Li7paa)
        else:
            import PyPR.PyPR_nuclear_net63 as PyPRnuclear
            PyPRnucl = PyPRnuclear.UpdateNuclearRates(PyPRini.p_npdg,PyPRini.p_dpHe3g,PyPRini.p_ddHe3n,PyPRini.p_ddtp,PyPRini.p_tpag,PyPRini.p_tdan,PyPRini.p_taLi7g,PyPRini.p_He3ntp,PyPRini.p_He3dap,PyPRini.p_He3aBe7g,PyPRini.p_Be7nLi7p,PyPRini.p_Li7paa,PyPRini.p_Li7paag,PyPRini.p_Be7naa,PyPRini.p_Be7daap,PyPRini.p_daLi6g,PyPRini.p_Li6pBe7g,PyPRini.p_Li6pHe3a,PyPRini.p_B8naap,PyPRini.p_Li6He3aap,PyPRini.p_Li6taan,PyPRini.p_Li6tLi8p,PyPRini.p_Li7He3Li6a,PyPRini.p_Li8He3Li7a,PyPRini.p_Be7tLi6a,PyPRini.p_B8tBe7a,PyPRini.p_B8nLi6He3,PyPRini.p_B8nBe7d,PyPRini.p_Li6tLi7d,PyPRini.p_Li6He3Be7d,PyPRini.p_Li7He3aad,PyPRini.p_Li8He3aat,PyPRini.p_Be7taad,PyPRini.p_Be7tLi7He3,PyPRini.p_B8dBe7He3,PyPRini.p_B8taaHe3,PyPRini.p_Be7He3ppaa,PyPRini.p_ddag,PyPRini.p_He3He3app,PyPRini.p_Be7pB8g,PyPRini.p_Li7daan,PyPRini.p_dntg,PyPRini.p_ttann,PyPRini.p_He3nag,PyPRini.p_He3tad,PyPRini.p_He3tanp,PyPRini.p_Li7taan,PyPRini.p_Li7He3aanp,PyPRini.p_Li8dLi7t,PyPRini.p_Be7taanp,PyPRini.p_Be7He3aapp,PyPRini.p_Li6nta,PyPRini.p_He3tLi6g,PyPRini.p_anpLi6g,PyPRini.p_Li6nLi7g,PyPRini.p_Li6dLi7p,PyPRini.p_Li6dBe7n,PyPRini.p_Li7nLi8g,PyPRini.p_Li7dLi8p,PyPRini.p_Li8paan,PyPRini.p_annHe6g,PyPRini.p_ppndp,PyPRini.p_Li7taann)
        
        #################################################
        # Local thermal equilibrium for nuclear species #
        #################################################
        def YA(name,Yn,Yp,T):
            x = PyPRini.Nuclides[name]
            A = x[0]+x[1]
            Z = x[1]
            N = A-Z
            Mass = A*PyPRini.ma*PyPRini.MeV+PyPRini.keV*PyPRini.NuclExcessMass[name]-Z*PyPRini.me*PyPRini.MeV
            BindingE = N*PyPRini.NuclExcessMass["n"] + Z*PyPRini.NuclExcessMass["p"]-PyPRini.NuclExcessMass[name]
            NormYA = (Mass/((PyPRini.mn*PyPRini.MeV)**(A-Z)*(PyPRini.mp*PyPRini.MeV)**Z))**(3/2)
            #print(" T and etab_of_T(T) = ",T,etab_of_T(T))
            #print("Mass Binding and NormYA = ",Mass,BindingE,NormYA)
            return (2*PyPRini.NuclSpin[name]+1)*zeta(3)**(A-1)*np.pi**((1-A)/2)*2**((3*A-5)/2)*NormYA*(PyPRini.kB*T)**(3/2*(A-1))*etab_of_T(T)**(A-1)*Yp**Z*Yn**(A-Z) *np.exp(BindingE*PyPRini.keV/(PyPRini.kB*T))
        
        #########################################################
        # Nuclear network: Final yields for p,d,t,He3,a,Li7,Be7 #
        #########################################################
        if(PyPRini.smallnet_flag):
            def Y_prime(t,Y):
                rhoBBN = rhoB_BBN(a_of_t(t))
                T_t = T_of_t(t)*PyPRini.MeV_to_Kelvin # temperature in [K]
                dY = PyPRnucl.dYndt(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYpdt(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYddt(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYtdt(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYHe3dt(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYadt(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYLi7dt(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYBe7dt(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd)
                return dY
                
            def Jacobian(t,Y):
                rhoBBN = rhoB_BBN(a_of_t(t))
                T_t = T_of_t(t)*PyPRini.MeV_to_Kelvin # temperature in [K]
                return PyPRnucl.Jacobian(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd)
        else:
            def Y_prime_MT(t,Y):
                rhoBBN = rhoB_BBN(a_of_t(t))
                T_t = T_of_t(t)*PyPRini.MeV_to_Kelvin # temperature in [K]
                dY = PyPRnucl.dYndtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYpdtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYddtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYtdtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYHe3dtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYadtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYLi7dtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYBe7dtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYHe6dtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYLi8dtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYLi6dtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYB8dtMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd)
                return dY
                
            def Jacobian_MT(t,Y):
                rhoBBN = rhoB_BBN(a_of_t(t))
                T_t = T_of_t(t)*PyPRini.MeV_to_Kelvin # temperature in [K]
                return PyPRnucl.JacobianMT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd)
        
            def Y_prime_LT(t,Y):
                rhoBBN = rhoB_BBN(a_of_t(t))
                T_t = T_of_t(t)*PyPRini.MeV_to_Kelvin # temperature in [K]
                dY = PyPRnucl.dYndtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYpdtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYddtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYtdtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYHe3dtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYadtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYLi7dtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYBe7dtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYHe6dtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYLi8dtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYLi6dtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd),PyPRnucl.dYB8dtLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd)
                return dY

            def Jacobian_LT(t,Y):
                rhoBBN = rhoB_BBN(a_of_t(t))
                T_t = T_of_t(t)*PyPRini.MeV_to_Kelvin # temperature in [K]
                return PyPRnucl.JacobianLT(Y,T_t,rhoBBN,nTOp_frwrd,nTOp_bkwrd)
        
        ############################
        # Mid temperature solution #
        ############################
        if(PyPRini.verbose_flag):
            print("Solving nuclear network at mid temperature era")
            
        # MT era definition
        t_init = t_weak
        t_fin = t_nucl
        #print("Middle range t_init and t_fin = ",t_init,t_fin)
        
        # Weak rates at MT
        def nTOp_frwrd(T):
            return NormWeakRates*nTOp_frwrd_MT(T)
        def nTOp_bkwrd(T):
            return NormWeakRates*nTOp_bkwrd_MT(T)
        
        # Initial conditions at MT
        Yn_i = Yn_HT_f
        Yp_i = Yp_HT_f
        Yd_i = YA("d",Yn_i,Yp_i,PyPRini.T_weak)
        Yt_i = YA("t",Yn_i,Yp_i,PyPRini.T_weak)
        YHe3_i = YA("He3",Yn_i,Yp_i,PyPRini.T_weak)
        Ya_i = YA("a",Yn_i,Yp_i,PyPRini.T_weak)
        YLi7_i = YA("Li7",Yn_i,Yp_i,PyPRini.T_weak)
        YBe7_i = YA("Be7",Yn_i,Yp_i,PyPRini.T_weak)
        #print(" Initial conditions at MT are ",Yn_i,Yp_i,Yd_i,Yt_i,YHe3_i,Ya_i,YLi7_i,YBe7_i)
        if(PyPRini.smallnet_flag == False):
            YHe6_i = YA("He6",Yn_i,Yp_i,PyPRini.T_weak)
            YLi8_i = YA("Li8",Yn_i,Yp_i,PyPRini.T_weak)
            YLi6_i = YA("Li6",Yn_i,Yp_i,PyPRini.T_weak)
            YB8_i = YA("B8",Yn_i,Yp_i,PyPRini.T_weak)
        
        # Solving MT network
        if(PyPRini.smallnet_flag):
            Yi_vec = [Yn_i,Yp_i,Yd_i,Yt_i,YHe3_i,Ya_i,YLi7_i,YBe7_i]
            sol_at_MT = solve_ivp(Y_prime,[t_init,t_fin],Yi_vec,method='BDF',jac=Jacobian,rtol=1.e-6,atol=1.e-9)
            Yn_MT_f,Yp_MT_f,Yd_MT_f,Yt_MT_f,YHe3_MT_f,Ya_MT_f,YLi7_MT_f,YBe7_MT_f = sol_at_MT.y[0][-1],sol_at_MT.y[1][-1],sol_at_MT.y[2][-1],sol_at_MT.y[3][-1],sol_at_MT.y[4][-1],sol_at_MT.y[5][-1],sol_at_MT.y[6][-1],sol_at_MT.y[7][-1]
        else:
            Yi_vec = [Yn_i,Yp_i,Yd_i,Yt_i,YHe3_i,Ya_i,YLi7_i,YBe7_i,YHe6_i,YLi8_i,YLi6_i,YB8_i]
            sol_at_MT = solve_ivp(Y_prime_MT,[t_init,t_fin],Yi_vec,method='BDF',jac=Jacobian_MT,rtol=1.e-6,atol=1.e-9)
            Yn_MT_f,Yp_MT_f,Yd_MT_f,Yt_MT_f,YHe3_MT_f,Ya_MT_f,YLi7_MT_f,YBe7_MT_f,YHe6_MT_f,YLi8_MT_f,YLi6_MT_f,YB8_MT_f = sol_at_MT.y[0][-1],sol_at_MT.y[1][-1],sol_at_MT.y[2][-1],sol_at_MT.y[3][-1],sol_at_MT.y[4][-1],sol_at_MT.y[5][-1],sol_at_MT.y[6][-1],sol_at_MT.y[7][-1],sol_at_MT.y[8][-1],sol_at_MT.y[9][-1],sol_at_MT.y[10][-1],sol_at_MT.y[11][-1]
        
        if(PyPRini.verbose_flag):
            print("--- running time: %s seconds ---" % (time.time() - start_time))
            print(" ")
            #print(" Middle range end Yn Yp Yd Yt YHe3 Ya YLi7 YBe7",Yn_MT_f,Yp_MT_f,Yd_MT_f,Yt_MT_f,YHe3_MT_f,Ya_MT_f,YLi7_MT_f,YBe7_MT_f)
        
        ############################
        # Low temperature solution #
        ############################
        if(PyPRini.verbose_flag):
            print("Solving nuclear network at low temperature era")
        
        # LT era definition
        t_init = t_nucl
        t_fin = t_end

        # Weak rates at LT
        def nTOp_frwrd(T):
            return NormWeakRates*nTOp_frwrd_LT(T)
        def nTOp_bkwrd(T):
            return NormWeakRates*nTOp_bkwrd_LT(T)
            
        # Initial conditions at LT
        Yn_i = Yn_MT_f
        Yp_i = Yp_MT_f
        Yd_i = Yd_MT_f
        Yt_i = Yt_MT_f
        YHe3_i = YHe3_MT_f
        Ya_i = Ya_MT_f
        YLi7_i = YLi7_MT_f
        YBe7_i = YBe7_MT_f
        if(PyPRini.smallnet_flag == False):
            YHe6_i = YHe6_MT_f
            YLi8_i = YLi8_MT_f
            YLi6_i = YLi6_MT_f
            YB8_i = YB8_MT_f
            
        if(PyPRini.smallnet_flag):
            Yi_vec = [Yn_i,Yp_i,Yd_i,Yt_i,YHe3_i,Ya_i,YLi7_i,YBe7_i]
            sol_at_LT = solve_ivp(Y_prime,[t_init,t_fin],Yi_vec,method='BDF',jac=Jacobian,atol=1.e-11)
            Yn_f,Yp_f,Yd_f,Yt_f,YHe3_f,Ya_f,YLi7_f,YBe7_f = sol_at_LT.y[0][-1],sol_at_LT.y[1][-1],sol_at_LT.y[2][-1],sol_at_LT.y[3][-1],sol_at_LT.y[4][-1],sol_at_LT.y[5][-1],sol_at_LT.y[6][-1],sol_at_LT.y[7][-1]
        else:
            Yi_vec = [Yn_i,Yp_i,Yd_i,Yt_i,YHe3_i,Ya_i,YLi7_i,YBe7_i,YHe6_i,YLi8_i,YLi6_i,YB8_i]
            sol_at_LT = solve_ivp(Y_prime_LT,[t_init,t_fin],Yi_vec,method='BDF',jac=Jacobian_LT,atol=1.e-15)
            Yn_f,Yp_f,Yd_f,Yt_f,YHe3_f,Ya_f,YLi7_f,YBe7_f,YHe6_f,YLi8_f,YLi6_f,YB8_f = sol_at_LT.y[0][-1],sol_at_LT.y[1][-1],sol_at_LT.y[2][-1],sol_at_LT.y[3][-1],sol_at_LT.y[4][-1],sol_at_LT.y[5][-1],sol_at_LT.y[6][-1],sol_at_LT.y[7][-1],sol_at_LT.y[8][-1],sol_at_LT.y[9][-1],sol_at_LT.y[10][-1],sol_at_LT.y[11][-1]

        if(PyPRini.verbose_flag):
            print("--- running time: %s seconds ---" % (time.time() - start_time))
            print(" ")

        if(PyPRini.verbose_flag):
            print("-------------------------------------------------")
            print("Predicted primordial abundances at the end of BBN")
            print("-------------------------------------------------")
            print("Yp = ",Yp_f)
            print("Yd = ",Yd_f)
            print("Yt = ",Yt_f)
            print("YHe3 = ",YHe3_f)
            print("Ya = ",Ya_f)
            print("YLi7 = ",YLi7_f)
            print("YBe7 = ",YBe7_f)
            print(" ")
            print("--- PyPRIMAT runned in: %s seconds ---" % (time.time() - start_time))
            
        #####################
        # Final predictions #
        #####################
        # N effective at the end of BBN era
        if(PyPRini.NP_thermo_flag):
            self.Neff_f = N_eff(Tg_vec[-1],Tnu_vec[-1],Tnu_vec[-1],TNP_vec[-1])
        else:
            self.Neff_f = N_eff(Tg_vec[-1],Tnu_vec[-1],Tnu_vec[-1])
        # Abundance of a single species of relativistic neutrino x 10^6
        self.Omeganurel_f = Omeganuh2_relnu()*1.e+6
        # Inverse of abundance of non-relativistic nu in units of sum of nu masses in [eV]
        self.OneOverOmeganunr_f = 1./(Omeganuh2_nrnu()*1.e-6)
        # Primordial helium-4 abundance as (nucleon) mass fraction (BBN definition)
        self.YPBBN_f = 4.*Ya_f
        # Primordial helium-4 abundance as (baryon) mass fraction (CMB definition)
        self.YPCMB_f = (PyPRini.He4Overma/4.)*self.YPBBN_f/((PyPRini.He4Overma/4.)*self.YPBBN_f+PyPRini.HOverma*(1.-self.YPBBN_f))
        # Primordial deuterium abundance as relative number density to hydrogen x 10^5
        self.DoHx1e5_f = Yd_f/Yp_f*1e+5
        # Primordial helium-3 abundance as relative number density to hydrogen x 10^5
        self.He3oHx1e5_f = (Yt_f+YHe3_f)/Yp_f*1e+5 # includes decay of tritium
        # Primordial lithium-7 abundance as relative number density to hydrogen x 10^10
        self.Li7oHx1e10_f = (YLi7_f+YBe7_f)/Yp_f*1e+10 # includes decay of beryllium-7
        # PRymordial output
        self.res = np.array([self.Neff_f,self.Omeganurel_f,self.OneOverOmeganunr_f,self.YPCMB_f,self.YPBBN_f,self.DoHx1e5_f,self.He3oHx1e5_f,self.Li7oHx1e10_f])
        
    def PyPRresults(self):
        return self.res

    def Neff(self):
        return self.Neff_f
        
    def Omeganurel(self):
        return self.Omeganurel_f
        
    def Omeganunonrel(self):
        return 1./self.OneOverOmeganunr_f
        
    def YPCMB(self):
        return self.YPCMB_f
        
    def YPBBN(self):
        return self.YPBBN_f
        
    def DoH(self):
        return self.DoHx1e5_f*1.e-5
        
    def He3oH(self):
        return self.He3oHx1e5_f*1.e-5
        
    def Li7oH(self):
        return self.Li7oHx1e10_f*1.e-10
