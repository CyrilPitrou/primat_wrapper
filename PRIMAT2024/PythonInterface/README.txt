%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 READ me file for the PythonInterface folder
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

*'PythonWrapper' folder

Examples of how PRIMAT can be called within python and its results retrieved (via storing in the disk in csv files) are provided. This can be used to template a python code which must call PRIMAT.

1) First, PRIMAT_python_basic.ipynb calls the Mathematica Kernel with the PyPRIMAT_FinalAbundances.m file as argument (which internally calls PRIMAT) and also extra arguments. These extra arguments can take the form of a dictionary in python, which is then translated in a string and passed as argument to the Mathematica Kernel (and then translated into Mathematica expressions, so as to set e.g. options). 
After PyPRIMAT_FinalAbundances.m has been successfully called, it outputs its results in a .csv file which is then read inside python. This is the easiest way to interface python and PRIMAT. 

As a pedagogical example we run its once with the default PRIMAT parameters when using the small network (12 reactions), and once when replacing some rates with the Parthenope rates.

2) A second more advanced example shows how to build Schramm diagrams, that is diagrams of abundances, with uncertainty due to nuclear rates, as a function of the baryon-to-photon ratio. This example calls PyPRIMAT-MonteCarlo.m, and takes much longer to evaluate because for each baryon abundance, it evaluates a Monte-Carlo on the uncertainty of nuclear rates. The user can easily modify the number of baryon abundance points, and the number of runs in each Monte-Carlo. It is left to the user to gather the plots and polish the matplotlib plots.