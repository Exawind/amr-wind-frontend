# avgplanes

Average netcdf sample planes
## Inputs: 
```
  name                : An arbitrary name (Required)
  ncfile              : NetCDF sampling file (Required)
  tavg                : Which times to average over (Optional, Default: [])
  loadpklfile         : Load previously computed results from this pickle file (Optional, Default: '')
  savepklfile         : Name of pickle file to save results (Optional, Default: '')
  group               : Which group to pull from netcdf file (Optional, Default: None)
  varnames            : Variables to extract from the netcdf file (Optional, Default: ['velocityx', 'velocityy', 'velocityz'])
```

## Actions: 
```
  rotorAvgVel         : ACTION: Computes the rotor averaged velocity (Optional)
    iplane            : List of iplane values (Required)
    Diam              : Turbine Diameter (Required)
    zc                : Center of rotor disk in z (Optional, Default: None)
    yc                : Center of rotor disk in y (Optional, Default: None)
    savefile          : csv filename to save results (Optional, Default: None)
  wakeThickness       : ACTION: Computes the wake displacement and momentum thickness (Optional)
    iplane            : List of iplane values (Required)
    noturbine_pkl_file: pickle file containing rotor planes for the case with no turbine (Optional, Default: None)
    U_inf             : constant value for U_inf for cases with uniform inflow (Optional, Default: None)
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
  interpolate         : ACTION: Plot rotor averaged planes (Optional)
    pointlocationfunction: Function to call to generate point locations. Function should have no arguments and return a list of points (Required)
    pointcoordsystem  : Coordinate system for point interpolation.  Options: XYZ, A1A2 (Required)
    varnames          : List of variable names to extract. (Required)
    savefile          : Filename to save the interpolated data (Optional, Default: '')
    method            : Interpolation method [Choices: linear, nearest, slinear, cubic, quintic, pchip] (Optional, Default: 'linear')
    iplane            : Which plane to interpolate on (Optional, Default: 0)
    iters             : Which time iterations to interpolate from (Optional, Default: None)
```

## Example
```yaml
avgplanes:
  - name: avg_smallXYplane
    ncfile:
    - /lustre/orion/cfd162/world-shared/lcheung/AdvancedControlsWakes/Runs/LowWS_LowTI.Frontier/oneturb_7x2/rundir_baseline/post_processing/XY_35000.nc
    - /lustre/orion/cfd162/world-shared/lcheung/AdvancedControlsWakes/Runs/LowWS_LowTI.Frontier/oneturb_7x2/rundir_baseline/post_processing/XY_50000.nc
    - /lustre/orion/cfd162/world-shared/lcheung/AdvancedControlsWakes/Runs/LowWS_LowTI.Frontier/oneturb_7x2/rundir_baseline/post_processing/XY_65000.nc    
    - /lustre/orion/cfd162/world-shared/lcheung/AdvancedControlsWakes/Runs/LowWS_LowTI.Frontier/oneturb_7x2/rundir_baseline/post_processing/XY_77500.nc
    tavg: [17800, 18500]
    plot:
      plotfunc: 'lambda u, v, w: np.sqrt(u**2 + v**2)'
      title: 'AVG horizontal velocity'
      xaxis: x           # Which axis to use on the abscissa 
      yaxis: y           # Which axis to use on the ordinate 
      iplane: 1    

```
