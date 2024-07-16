# openfast

Postprocessing of openfast variables
## Inputs: 
```
  name                : An arbitrary name (Required)
  filename            : Openfast output file (Required)
  vars                : Variables to extract from the openfast file (Required)
  extension           : The extension to use for the csv files (Optional, Default: '.csv')
  output_dir          : Directory to save results (Optional, Default: './')
```

## Actions: 
```
  csv                 : ACTION: Writes the openfast variables to a csv file (Optional)
    individual_files  : Write each variable to a separate csv file (Optional, Default: True)
  operate             : ACTION: Operates on the openfast data and saves to a csv file (Optional)
    operations        : List of operations to perform (mean,std,DEL) (Required)
    trange            : Times to apply operation over (Optional, Default: [])
    awc_period        : Average over equal periods for AWC forcing (Optional, Default: False)
    awc               : AWC case name [baseline,n0,n1p,n1m,n1p1m_cl00,n1p1m_cl90] (Optional, Default: 'baseline')
    St                : Forcing Strouhal number (Optional, Default: 0.3)
    diam              : Turbine diameter (Optional, Default: 0)
    U_st              : Wind speed to define Strouhal number (Optional, Default: 0)
```
