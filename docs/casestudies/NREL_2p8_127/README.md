# Calibrating the NREL 2.8-127

This case study will show how to use `amrwind-frontend` to set up a
small, uniform flow domain with an NREL 2.8-127 ADM turbine.  The
results of the ADM runs are then compared to the nominal performance
curve.


Take a look at the following notebooks:

-
  [AMRWind_GE2-8-127_SetupWS9.ipynb](AMRWind_GE2-8-127_SetupWS9.ipynb):
  How to set up the case for a single turbine using the python
  interface.

- [PlotCurves1.ipynb](PlotCurves1.ipynb): Taking the OpenFAST output
  files, and plotting the performance characteristics against the
  nominal curve.

-
  [AMRWind_GE2-8-127_FarmSetup.ipynb](AMRWind_GE2-8-127_FarmSetup.ipynb):
  How to use `amrwind-frontend` to set up multiple turbines in a wind
  farm or multiple condition cases.