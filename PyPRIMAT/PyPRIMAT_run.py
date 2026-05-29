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
PyPRini.aTid_flag = True
PyPRini.compute_bckg_flag = True #True is slower but more accurate since expansion is modified by EDE
PyPRini.compute_nTOp_flag = False #True is slower and we do not modify the rates with EDE. But we should be careful in general.
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
print(" Neff --> ",res[0])
print(" Ωνh2 x 10^6 (rel) --> ",res[1])
print(" Σmν/Ωνh2 [eV] --> ",res[2])
print(" YP (CMB) --> ",res[3])
print(" YP (BBN) --> ",res[4])
print(" D/H x 10^5 --> ",res[5])
print(" He3/H x 10^5 --> ",res[6])
print(" Li7/H x 10^10 --> ",res[7])
print(" ")
print("--- running time: %s seconds ---" % (time.time() - start_time))
