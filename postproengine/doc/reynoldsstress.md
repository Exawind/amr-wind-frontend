# reynoldsstress

Reynolds-Stress average netcdf sample planes
## Inputs: 
```
  name                : An arbitrary name (Required)
  ncfile              : NetCDF sampling file (Required)
  tavg                : Which times to average over (Optional, Default: [])
  meanpklfile         : Name of pickle file which contains mean results (Optional, Default: '')
  savepklfile         : Name of pickle file to save results (Optional, Default: '')
  group               : Which group to pull from netcdf file (Optional, Default: None)
  varnames            : Variables to extract from the netcdf file (Optional, Default: ['velocityx', 'velocityy', 'velocityz'])
```

## Actions: 
  contourplot         : ACTION: Plot rotor averaged planes (Optional)
    dpi               : Figure resolution (Optional, Default: 125)
    figsize           : Figure size (inches) (Optional, Default: [12, 8])
    savefile          : Filename to save the picture (Optional, Default: '')
    clevels           : Color levels (eval expression) (Optional, Default: '41')
    iplane            : Which plane to pull from netcdf file (Optional, Default: 0)
    xaxis             : Which axis to use on the abscissa (Optional, Default: 'x')
    yaxis             : Which axis to use on the ordinate (Optional, Default: 'y')
    xlabel            : Label on the X-axis (Optional, Default: 'X [m]')
    ylabel            : Label on the Y-axis (Optional, Default: 'Y [m]')
    title             : Title of the plot (Optional, Default: '')
    plotfunc          : Function to plot (lambda expression) (Optional, Default: 'lambda db: 0.5*(db["uu_avg"]+db["vv_avg"]+db["ww_avg"])')

## Example
```yaml
reynoldsstress:
  - name: test
    ncfile: /lustre/orion/cfd162/proj-shared/lcheung/AWAKEN/Neutral/5kmX5km_turbine1/post_processing/sampling_41000.nc
    tavg: [20886.5, 21486.5]
    contourplot:
      plotfunc: 'lambda db: 0.5*(db["uu_avg"]+db["vv_avg"]+db["ww_avg"])'
```
