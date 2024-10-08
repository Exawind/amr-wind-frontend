# instantaneousplanes

Make instantaneous plots from netcdf sample planes
## Inputs: 
```
  name                : An arbitrary name (Required)
  ncfile              : NetCDF sampling file (Required)
  iters               : Which iterations to pull from netcdf file (Required)
  xaxis               : Which axis to use on the abscissa (Required)
  yaxis               : Which axis to use on the ordinate (Required)
  times               : Which times to pull from netcdf file (overrides iters) (Optional, Default: [])
  trange              : Pull a range of times from netcdf file (overrides iters) (Optional, Default: None)
  group               : Which group to pull from netcdf file (Optional, Default: None)
  varnames            : Variables to extract from the netcdf file (Optional, Default: ['velocityx', 'velocityy', 'velocityz'])
  iplane              : Which plane to pull from netcdf file (Required)
```

## Actions: 
```
  plot                : ACTION: Plot instantaneous fields for all iterations (Optional)
    title             : Title of the plot (Optional, Default: '')
    plotfunc          : Function to plot (lambda expression) (Optional, Default: "lambda db,i: np.sqrt(db['velocityx'][i]**2 + db['velocityy'][i]**2)")
    clevels           : Color levels (eval expression) (Optional, Default: 'np.linspace(0, 12, 121)')
    cmap              : Color map name (Optional, Default: 'coolwarm')
    cbar              : Boolean to include colorbar (Optional, Default: True)
    cbar_label        : Label for colorbar (Optional, Default: None)
    cbar_nticks       : Number of ticks to include on colorbar (Optional, Default: None)
    xlabel            : Label on the X-axis (Optional, Default: 'X [m]')
    ylabel            : Label on the Y-axis (Optional, Default: 'Y [m]')
    dpi               : Figure resolution (Optional, Default: 125)
    figsize           : Figure size (inches) (Optional, Default: [12, 3])
    fontsize          : Fontsize for labels and axis (Optional, Default: 14)
    savefile          : Filename to save the picture (Optional, Default: '')
    postplotfunc      : Function to call after plot is created. Function should have arguments func(fig, ax) (Optional, Default: '')
    xscalefunc        : Function to scale the x-axis (lambda expression) (Optional, Default: 'lambda x: x')
    yscalefunc        : Function to scale the y-axis (lambda expression) (Optional, Default: 'lambda y: y')
    figname           : Name/number of figure to create plot in (Optional, Default: None)
    axesnumfunc       : Function to determine which subplot axes to create plot in (lambda expression with iplane as arg) (Optional, Default: None)
    axesnumfunc       : Function to determine which subplot axes to create plot in (lambda expression with iplane as arg) (Optional, Default: None)
    axisscale         : Aspect ratio of figure axes (options:equal,scaled,tight,auto,image,square) (Optional, Default: 'scaled')
  interpolate         : ACTION: Interpolate data from an arbitrary set of points (Optional)
    pointlocationfunction: Function to call to generate point locations. Function should have no arguments and return a list of points (Required)
    pointcoordsystem  : Coordinate system for point interpolation.  Options: XYZ, A1A2 (Required)
    varnames          : List of variable names to extract. (Required)
    savefile          : Filename to save the interpolated data (Optional, Default: '')
    method            : Interpolation method [Choices: linear, nearest, slinear, cubic, quintic, pchip] (Optional, Default: 'linear')
    iplane            : Which plane to interpolate on (Optional, Default: 0)
    iters             : Which time iterations to interpolate from (Optional, Default: None)
  animate             : ACTION: Generate animation from static images of planes (Optional)
    name              : Name of video (Required)
    fps               : Frame per second (Optional, Default: 1)
    imagefilename     : savefile name of images (Required)
    times             : Override parent times for animation (Optional, Default: None)
  plot_radial         : ACTION: Plot instantaneous field in polar coordinate (Optional)
    title             : Title of the plot (Optional, Default: '')
    plotfunc          : Function to plot (lambda expression) (Optional, Default: "lambda db,i: db['velocityx'][i]")
    cmap              : Color map name (Optional, Default: 'coolwarm')
    cbar              : Boolean to include colorbar (Optional, Default: True)
    dpi               : Figure resolution (Optional, Default: 125)
    figsize           : Figure size (inches) (Optional, Default: [8, 5])
    savefile          : Filename to save the picture (Optional, Default: '')
    vmin              : Minimum color range (Optional, Default: None)
    vmax              : Maximum color range (Optional, Default: None)
    LR                : Extent of radial grid (Required)
    NR                : Number of points in radial direction (Required)
    NTheta            : Number of points in azimuthal direction (Required)
```

## Example
```yaml
instantaneousplanes:
  name: Wake YZ plane
  ncfile: ./data_converter/PA_1p25_new2/YZslice_01.00D_456.00s_1556.00s_n1m.nc
  iters: [0,]
  #trange: [456,457]
  xaxis: 'y'
  yaxis: 'z'
  varnames: ['velocityx','velocityy','velocityz']
  iplane: 0

  plot:
    plotfunc: "lambda db, i: db['velocityx'][i]"
    savefile: 'inst_figs_n1m/inst_test_{time}.png'
    figsize: [8,5]
    dpi: 125
    xlabel: 'Y [m]'
    ylabel: 'Z [m]'
    clevels: 'np.linspace(2,7,121)'
    cbar: False
    cmap: 'viridis'

  animate:
    name: 'output.mp4'
    fps: 20
    imagefilename: './inst_figs_n1m/inst_test_{time}.png'
    #times: 'np.arange(456,1556.5,0.5)'

  plot_radial:
    plotfunc: "lambda db, i: db['velocityx'][i]"
    savefile: 'radial_test_{time}.png'
    figsize: [8,5]
    dpi: 125
    clevels: 'np.linspace(2,7,121)'
    cmap: 'viridis'
    LR: 89.0
    NR: 256
    NTheta: 256
    vmin: 2
    vmax: 7
    xc: 375
    yc: 90
    cbar: True

```
