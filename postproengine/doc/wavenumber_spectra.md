# wavenumber_spectra

Calculates 2D wavenumber spectra in x and y
## Inputs: 
```
  name                : An arbitrary name (Required)
  ncfile              : NetCDF file of horizontal planes (Required)
  group               : Group name in netcdf file (Required)
  trange              : Which times to average over (Required)
  num_bins            : How many wavenumber bins to use for spectra (Optional, Default: 50)
  type                : What type of spectra to compute: 'energy', 'horiz', 'vertical', 'kol'. Default is all. (Optional, Default: ['energy', 'horiz', 'vertical', 'kol'])
  varnames            : Variables to extract from the netcdf file (Optional, Default: ['velocityx', 'velocityy', 'velocityz'])
  iplanes             : i-index of planes to postprocess (Optional, Default: None)
  csvfile             : Filename to save spectra to (Optional, Default: '')
  xaxis               : Which axis to use on the abscissa (Optional, Default: 'x')
  yaxis               : Which axis to use on the ordinate (Optional, Default: 'y')
  C_kol               : Kolmogorov constant to use for theoretical spectra (Optional, Default: 1.5)
  diss_rate           : Dissipation rate for theoretical spectra (Optional, Default: 1.0)
  remove_endpoint_x   : Remove one endpoint in x before FFT if periodic signal is sampled twice at endpoints. (Optional, Default: False)
  remove_endpoint_y   : Remove one endpoint in y before FFT if periodic signal is sampled twice at endpoints. (Optional, Default: False)
```

## Actions: 

## Example
```yaml
    wavenumber_spectra:
        name: Spectra_027
        ncfile: XYdomain_027_30000.nc
        group: Farm_XYdomain027
        csvfile: E_spectra_Z027.csv
        trange:  [15000, 20000]
        iplanes: 0 
    
```
