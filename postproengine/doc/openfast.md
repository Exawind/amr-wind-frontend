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
