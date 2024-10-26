# phaseavgplanes

Phase average netcdf sample planes
## Inputs: 
```
  name                : An arbitrary name (Required)
  ncfile              : NetCDF sampling file (Required)
  tstart              : Time to start phase averaging (Required)
  tend                : Time to end phase averaging (Required)
  tstart              : Time period of phase averaging (Required)
  calcavg             : Also calculate average variables (Optional, Default: False)
  calcrestress        : Also calculate Reynolds stresses (Optional, Default: False)
  saveavgpklfile      : Name of pickle file to save average results (Optional, Default: '')
  loadavgpklfile      : Name of pickle file to load average results (Optional, Default: '')
  loadpklfile         : Load previously computed results from this pickle file (Optional, Default: '')
  savepklfile         : Name of pickle file to save results (Optional, Default: '')
  group               : Which group to pull from netcdf file (Optional, Default: None)
  varnames            : Variables to extract from the netcdf file (Optional, Default: ['velocityx', 'velocityy', 'velocityz'])
  axis_rotation       : Degrees to rotate axis for velocitya1,a2,a3 transformation (Optional, Default: 0)
```

## Actions: 
```
  reynoldsstress1     : ACTION: Calculate Reynolds stress (version 1) (Optional)
    savepklfile       : Name of pickle file to save phase averaged results (Optional, Default: '')
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
    subtractpklfile   : Name of pickle file to subtract from dataframe (Optional, Default: '')
```

## Example
```yaml
```
  phaseavgplanes:
  - name: LowWSLowTI Baseline
    ncfile:
    - /lustre/orion/cfd162/world-shared/lcheung/AdvancedControlsWakes/Runs/LowWS_LowTI.Frontier/oneturb_7x2/rundir_baseline/post_processing/XZ_*.nc
    tstart: 17650
    tend: 18508.895705521474
    tperiod: 122.6993865030675
    varnames: ['velocityx', 'velocityy', 'velocityz', 'tke']
    calcavg: True
    contourplot:
      title: Baseline
      plotfunc: 'lambda db: db["velocityx_phavg"] - db["velocityx_avg"]'   #'lambda db: np.sqrt(db["velocityx_avg"]**2 + db["velocityy_avg"]**2)'
      xaxis: x         # Which axis to use on the abscissa
      yaxis: z         # Which axis to use on the ordinate
      iplane: [0]
      clevels: 'np.linspace(-1, 1, 101)'
```

```
