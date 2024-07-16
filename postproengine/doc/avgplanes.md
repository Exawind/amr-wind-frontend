# avgplanes

Average netcdf sample planes
## Inputs: 
```
  name                : An arbitrary name (Required)
  ncfile              : NetCDF sampling file (Required)
  tavg                : Which times to average over (Optional, Default: [])
  xaxis               : Which axis to use on the abscissa (Optional, Default: 'x')
  yaxis               : Which axis to use on the ordinate (Optional, Default: 'y')
  savepklfile         : Name of pickle file to save results (Optional, Default: '')
  group               : Which group to pull from netcdf file (Optional, Default: None)
  varnames            : Variables to extract from the netcdf file (Optional, Default: ['velocityx', 'velocityy', 'velocityz'])
```

## Actions: 
  rotorAvgVel         : ACTION: Computes the rotor averaged velocity (Optional)
    iplane            : List of iplane values (Required)
    Diam              : Turbine Diameter (Required)
    zc                : Center of rotor disk in z (Optional, Default: None)
    yc                : Center of rotor disk in y (Optional, Default: None)
    savefile          : csv filename to save results (Optional, Default: None)
  plot                : ACTION: Plot rotor averaged planes (Optional)
    dpi               : Figure resolution (Optional, Default: 125)
    figsize           : Figure size (inches) (Optional, Default: [12, 8])
    savefile          : Filename to save the picture (Optional, Default: '')
    clevels           : Color levels (eval expression) (Optional, Default: 'np.linspace(0, 12, 121)')
    xlabel            : Label on the X-axis (Optional, Default: 'X [m]')
    ylabel            : Label on the Y-axis (Optional, Default: 'Y [m]')
    title             : Title of the plot (Optional, Default: '')
    plotfunc          : Function to plot (lambda expression) (Optional, Default: 'lambda u, v, w: u')
