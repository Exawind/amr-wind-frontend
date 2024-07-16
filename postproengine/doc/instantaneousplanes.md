# instantaneousplanes

Make instantaneous plots from netcdf sample planes
## Inputs: 
```
  name                : An arbitrary name (Required)
  ncfile              : NetCDF sampling file (Required)
  iters               : Which iterations to pull from netcdf file (Required)
  iplane              : Which plane to pull from netcdf file (Required)
  xaxis               : Which axis to use on the abscissa (Required)
  yaxis               : Which axis to use on the ordinate (Required)
  times               : Which times to pull from netcdf file (overrides netCDF) (Optional, Default: [])
  group               : Which group to pull from netcdf file (Optional, Default: None)
  title               : Title of the plot (Optional, Default: '')
  varnames            : Variables to extract from the netcdf file (Optional, Default: ['velocityx', 'velocityy', 'velocityz'])
  plotfunc            : Function to plot (lambda expression) (Optional, Default: 'lambda u, v, w: np.sqrt(u**2 + v**2)')
  clevels             : Color levels (eval expression) (Optional, Default: 'np.linspace(0, 12, 121)')
  xlabel              : Label on the X-axis (Optional, Default: 'X [m]')
  ylabel              : Label on the Y-axis (Optional, Default: 'Y [m]')
  dpi                 : Figure resolution (Optional, Default: 125)
  figsize             : Figure size (inches) (Optional, Default: [12, 3])
  savefile            : Filename to save the picture (Optional, Default: '')
```

## Actions: 
