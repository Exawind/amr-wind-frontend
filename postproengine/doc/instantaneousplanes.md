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
```
  interpolate         : ACTION: Interpolate data from an arbitrary set of points (Optional)
    pointlocationfunction: Function to call to generate point locations. Function should have no arguments and return a list of points (Required)
    pointcoordsystem  : Coordinate system for point interpolation.  Options: XYZ, A1A2 (Required)
    varnames          : List of variable names to extract. (Required)
    savefile          : Filename to save the interpolated data (Optional, Default: '')
    method            : Interpolation method [Choices: linear, nearest, slinear, cubic, quintic, pchip] (Optional, Default: 'linear')
    iplane            : Which plane to interpolate on (Optional, Default: 0)
    iters             : Which time iterations to interpolate from (Optional, Default: None)
```
