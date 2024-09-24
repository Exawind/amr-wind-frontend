# wake_meander

Compute wake meandering statistics
## Inputs: 
```
  iplane              : i-index of planes to postprocess (Optional, Default: None)
  name                : An arbitrary name (Required)
  ncfile              : NetCDF sampling files of cross-flow planes (Required)
  group               : Which group to pull from netcdf file (Optional, Default: None)
  savepklfile         : Name of pickle file to save wake object (Optional, Default: '')
  varnames            : Variables to extract from the netcdf file (Optional, Default: ['velocityx', 'velocityy', 'velocityz'])
  trange              : Which times to postprocess (Required)
  yhub                : Lateral hub-height center (Optional, Default: None)
  zhub                : Vertical hub-height (Optional, Default: None)
  method              : Method for computing wake center. Options include: ConstantArea, ConstantFlux, Gaussian (Required)
  diam                : Rotor diameter (Optional, Default: 0)
  Uinf                : U velocity for approximating rotor thrust for ConstantFlux method (Optional, Default: None)
  Ct                  : Thrust coefficient for approximating rotor thrust for ConstantFlux method (Optional, Default: 0)
  rotthrust           : Target rotor thrust for ConstantFlux method. (Optional, Default: None)
  savefile            : File to save timeseries of wake centers, per iplane (Optional, Default: '')
  output_dir          : Directory to save results (Optional, Default: './')
  weighted_center     : If True, calculate the velocity-deficit-weighted "center of mass"; if False, calculate the geometric center of the wake. (Optional, Default: True)
  axis_rotation       : Degrees to rotate axis for velocitya1,a2,a3 transformation (Optional, Default: 0)
  xaxis               : Which axis to use on the abscissa (Optional, Default: 'y')
  yaxis               : Which axis to use on the ordinate (Optional, Default: 'z')
```

## Actions: 
```
  plot                : ACTION: Plot contour with wake boundary and center  (Optional)
    dpi               : Figure resolution (Optional, Default: 125)
    figsize           : Figure size (inches) (Optional, Default: [12, 8])
    savefile          : Filename to save the picture (Optional, Default: '')
    xlabel            : Label on the X-axis (Optional, Default: 'X [m]')
    ylabel            : Label on the Y-axis (Optional, Default: 'Y [m]')
    title             : Title of the plot (Optional, Default: '')
    cmin              : Minimum contour level (Optional, Default: None)
    cmax              : Maximum contour level (Optional, Default: None)
    iter              : Iteration in time to plot (Optional, Default: 0)
  statistics          : ACTION: Compute wake meandering statistics (Optional)
    savefile          : Filename to save statistics (Optional, Default: '')
    mean              : Boolean to compute mean wake center (Optional, Default: True)
    std               : Boolean to compute std wake center (Optional, Default: True)
    anisotropy        : Boolean to compute wake anisotropy metric (Optional, Default: False)
    compute_uv        : Boolean to compute eigenvectors of PCA (Optional, Default: False)
    pklfile           : File to save eigenvectors of PCA (Optional, Default: '')
```

## Example
```yaml
    wake_meander:
        iplane:
            - 5
            - 6
        name: Wake YZ plane
        ncfile: YZcoarse_103125.nc
        trange: [27950,28450]
        group: T0_YZdomain
        yhub: 1000
        zhub: 150
        method: ConstantArea
        #method: ConstantFlux
        #method: Gaussian
        diam: 240
        savefile: wake_center_{iplane}.csv
        output_dir: ./wake_meandering/
        Uinf: 9.0
        Ct: 1.00

        plot:
            xlabel: 'Y [m]'
            ylabel: 'Z [m]'
            iter: 0
            savefile: wake_center_{iplane}.png

        statistics:
            savefile: wake_stats_{iplane}.csv
            mean: True
            std: True
            anisotropy: True
            compute_uv: True
            pklfile: pcs.pkl
    
```
