# openfast

Postprocessing of openfast variables
## Inputs: 
```
  name                : An arbitrary name (Required)
  filename            : Openfast output file (Required)
  vars                : Variables to extract from the openfast file (Required)
  extension           : The extension to use for the csv files (Optional, Default: '.csv')
  output_dir          : Directory to save results (Optional, Default: './')
  useregex            : Use regex expansion in vars list (Optional, Default: False)
```

## Actions: 
```
  csv                 : ACTION: Writes the openfast variables to a csv file (Optional)
    individual_files  : Write each variable to a separate csv file (Optional, Default: True)
  operate             : ACTION: Operates on the openfast data and saves to a csv file (Optional)
    operations        : List of operations to perform (mean,std,DEL,pwelch) (Required)
    trange            : Times to apply operation over (Optional, Default: [])
    awc_period        : Average over equal periods for AWC forcing (Optional, Default: False)
    awc               : AWC case name [baseline,n0,n1p,n1m,n1p1m_cl00,n1p1m_cl90] (Optional, Default: 'baseline')
    St                : Forcing Strouhal number (Optional, Default: 0.3)
    diam              : Turbine diameter (Optional, Default: 0)
    U_st              : Wind speed to define Strouhal number (Optional, Default: 0)
    nperseg           : Number of samples per segment used in pwelch (Optional, Default: 4096)
  spanwiseloading     : ACTION: Reformats time history csv data to spanwise loading profiles (Optional)
    bladefile         : AeroDyn blade file (Required)
    bladevars         : List of blade variables to extract, such as Alpha, Cl, Cd, etc. (Required)
    meancsvfile       : mean csv file (output from above) (Required)
    savecsvfile       : output csv file (Required)
    radialstations    : list of radial blade stations (Required)
    prefix            : Prefix in front of each openfast var (Optional, Default: 'AB1N')
```

## Notes


Currently the `openfast` executor is only capable of acting on text
output from OpenFAST, corresponding to OutFileFmt=1 in the fst input
file.
    
The `useregex` option allows multiple variables in the `vars` list to
be specified through a regex [regular
expression](https://en.wikipedia.org/wiki/Regular_expression).  For
instance, `^Rot` will match any variable that starts with `Rot`, such
as `RotSpeed` or `RotTorq`.
    

## Example

```yaml
openfast:
- name: NREL5MW_SECLOADS
  filename: RUNDIR/T0_NREL5MW_v402_ROSCO/openfast-cpp/5MW_Land_DLL_WTurb_cpp/5MW_Land_DLL_WTurb_cpp.out
  vars: 
  - Time
  - '^Rot'
  - 'AB1N...Alpha'
  - 'AB1N...Phi'
  - 'AB1N...Cl'
  - 'AB1N...Cd'
  - 'AB1N...Fx'
  - 'AB1N...Fy'
  - RotSpeed
  output_dir: RESULTSDIR
  useregex: True
  csv:  # Store information to CSV files
    individual_files: False
  operate:
    operations: 
    - mean
    trange: [300, 900]
  spanwiseloading:
    bladefile: RUNDIR/T0_NREL5MW_v402_ROSCO/openfast/5MW_Baseline/NRELOffshrBsline5MW_AeroDyn_blade.dat
    bladevars: [Alpha, Phi, Cl, Cd, Fx, Fy]
    meancsvfile: RESULTSDIR/NREL5MW_SECLOADS_mean.csv
    savecsvfile: RESULTSDIR/NREL5MW_SECLOADS_mean_rpts.csv
    radialstations: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    prefix: AB1N
```

