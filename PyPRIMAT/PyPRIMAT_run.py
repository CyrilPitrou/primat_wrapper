# -*- coding: utf-8 -*-
#import numpy as np
import PyPR.PyPR_init as PyPRini # We only import PyPR_init to set initial conditions.
import time

# Cosmological parameters
Nrelat = 0.
omegabh2 = 0.022425

PyPRini.smallnet_flag = True
start_time = time.time()

PyPRini.verbose_flag = True
PyPRini.debug_flag = True
PyPRini.aTid_flag = True
PyPRini.sampling_nTOp = 50 
PyPRini.sampling_nTOp_thermal = 50
PyPRini.compute_bckg_flag = True #True is slower but more accurate since expansion is modified by EDE
PyPRini.compute_nTOp_flag = True #True is slower and we do not modify the rates with EDE. But we should be careful in general.
PyPRini.compute_nTOp_thermal_flag = False
PyPRini.save_bckg_flag = False #Only the first time True is needed
PyPRini.save_nTOp_flag = False #Only the first time True is needed
PyPRini.NP_nu_flag = False
PyPRini.NP_e_flag = False 
PyPRini.Omegabh2 = omegabh2
PyPRini.eta0b = PyPRini.Omegabh2_to_eta0b * PyPRini.Omegabh2
PyPRini.DeltaNeff  = Nrelat #Not exactly the same definition in PRIMAT but who cares ?
PyPRini.smallnet_flag = True

#Now we import the main PyPR_main
import PyPR.PyPR_main as PyPRmain
#print(" ")
#print(" #################################################")
#print("    PyPRIMAT: Run with small network ")
#print(" #################################################")
PyPRrun = PyPRmain.PyPRclass()
res = PyPRrun.PyPRresults()

print(" ")
print(" Neff --> ",res['Neff'])
print(" Ωνh2 x 10^6 (rel) --> ",res['Omeganurel'])
print(" Σmν/Ωνh2 [eV] --> ",res['OneOverOmeganunr'])
print(" YP (CMB) --> ",res['YPCMB'])
print(" YP (BBN) --> ",res['YPBBN'])
print(" D/H x 10^5 --> ",res['DoHx1e5'])
print(" He3/H x 10^5 --> ",res['He3oHx1e5'])
print(" Li7/H x 10^10 --> ",res['Li7oHx1e10'])
print(" ")
print("--- running time: %s seconds ---" % (time.time() - start_time))
