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
  axis_rotation       : Degrees to rotate axis for velocitya1,a2,a3 transformation (Optional, Default: 0)
```

## Actions: 
```
  rotorAvgVel         : ACTION: Computes the rotor averaged velocity (Optional)
    iplane            : List of iplane values (Optional, Default: None)
    Diam              : Turbine Diameter (Required)
    xc                : Center of rotor disk on the xaxis (Optional, Default: None)
    yc                : Center of rotor disk in the yaxis (Optional, Default: None)
    xaxis             : Which axis to use on the abscissa (Optional, Default: 'y')
    yaxis             : Which axis to use on the ordinate (Optional, Default: 'z')
    savefile          : csv filename to save results (Optional, Default: None)
    avgfunc           : Function to average (lambda expression) (Optional, Default: 'lambda db: db["velocityx_avg"]')
    wake_meandering_stats_file: The lateral and vertical wake center will be read from yc_mean and zc_mean columns of this file, overriding yc and zc. (Optional, Default: None)
  wakeThickness       : ACTION: Computes the wake displacement and momentum thickness (Optional)
    iplane            : List of iplane values (Required)
    noturbine_pkl_file: pickle file containing rotor planes for the case with no turbine (Optional, Default: None)
    U_inf             : constant value for U_inf for cases with uniform inflow (Optional, Default: None)
    savefile          : csv filename to save results (Optional, Default: None)
  contourplot         : ACTION: Plot rotor averaged planes (Optional)
    dpi               : Figure resolution (Optional, Default: 125)
    figsize           : Figure size (inches) (Optional, Default: [12, 8])
    savefile          : Filename to save the picture (Optional, Default: '')
    clevels           : Color levels (eval expression) (Optional, Default: '41')
    cmap              : Color map name (Optional, Default: 'coolwarm')
    iplane            : Which plane to pull from netcdf file (Optional, Default: 0)
    xaxis             : Which axis to use on the abscissa (Optional, Default: 'x')
    yaxis             : Which axis to use on the ordinate (Optional, Default: 'y')
    xlabel            : Label on the X-axis (Optional, Default: 'X [m]')
    ylabel            : Label on the Y-axis (Optional, Default: 'Y [m]')
    title             : Title of the plot (Optional, Default: '')
    plotfunc          : Function to plot (lambda expression) (Optional, Default: 'lambda db: 0.5*(db["uu_avg"]+db["vv_avg"]+db["ww_avg"])')
    axis_rotation     : Degrees to rotate a1,a2,a3 axis for plotting. (Optional, Default: 0)
    xscalefunc        : Function to scale the x-axis (lambda expression) (Optional, Default: 'lambda x: x')
    yscalefunc        : Function to scale the y-axis (lambda expression) (Optional, Default: 'lambda y: y')
    postplotfunc      : Function to call after plot is created. Function should have arguments func(fig, ax) (Optional, Default: '')
    fontsize          : Fontsize for figure (Optional, Default: 14)
    figname           : Name/number of figure to create plot in (Optional, Default: None)
    axesnumfunc       : Function to determine which subplot axes to create plot in (lambda expression with iplane as arg) (Optional, Default: None)
    cbar              : Boolean to include colorbar (Optional, Default: True)
    cbar_label        : Label for colorbar (Optional, Default: None)
    cbar_nticks       : Number of ticks to include on colorbar (Optional, Default: None)
  interpolate         : ACTION: Interpolate data from an arbitrary set of points (Optional)
    pointlocationfunction: Function to call to generate point locations. Function should have no arguments and return a list of points (Required)
    pointcoordsystem  : Coordinate system for point interpolation.  Options: XYZ, A1A2 (Required)
    varnames          : List of variable names to extract. (Required)
    savefile          : Filename to save the interpolated data (Optional, Default: '')
    method            : Interpolation method [Choices: linear, nearest, slinear, cubic, quintic, pchip] (Optional, Default: 'linear')
    iplane            : Which plane to interpolate on (Optional, Default: 0)
    iters             : Which time iterations to interpolate from (Optional, Default: None)
  circavg             : ACTION: Circumferential average data into radial profiles (Optional)
    centerpoint       : Center point to use for radial averaging (Required)
    r1                : Inner radius (Required)
    r2                : Outer radius (Required)
    Nr                : Number of points in radial direction (Required)
    pointcoordsystem  : Coordinate system for point interpolation.  Options: XYZ, A1A2 (Required)
    varnames          : List of variable names to average. (Required)
    savefile          : Filename to save the radial profiles (Required)
    iplane            : Which plane(s) to interpolate on (Optional, Default: 0)
    theta1            : Theta start (Optional, Default: 0.0)
    theta2            : Theta end (Optional, Default: 6.283185307179586)
    Ntheta            : Number of points in theta (Optional, Default: 180)
    wake_meandering_stats_file: For streamwise planes, wake center will be read from columns of these file, overiding centerpoint. (Optional, Default: None)
```

## Example
```yaml
avgplanes:
  name: Wake YZ plane
  ncfile:
  - /lustre/orion/cfd162/world-shared/lcheung/ALCC_Frontier_WindFarm/farmruns/LowWS_LowTI/ABL_ALM_10x10/rundir_baseline/post_processing/rotor_*.nc
  tavg: [25400,26000]
  group: T08_rotor
  varnames: ['velocitya1','velocitya2','velocitya3']
  verbose: True

  contourplot:
    iplane: 6
    xaxis: 'a1'
    yaxis: 'a2'
    xlabel: 'Lateral axis [m]'
    ylabel: 'Vertical axis [m]'
    clevels: "121"
    plotfunc: "lambda db: db['velocitya3_avg']"
    savefile: 'avg_plane.png'

  rotorAvgVel:
    iplane: [0,1,2,3,4,5,6,7,8,9,10]
    Diam: 240
    yc: 150
    xaxis: 'a1'
    yaxis: 'a2'
    avgfunc: "lambda db: db['velocitya3_avg']"
    savefile: test.csv

```
