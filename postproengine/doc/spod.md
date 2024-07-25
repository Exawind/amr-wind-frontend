# spod

Compute SPOD eigenvectors and eigenvalues
## Inputs: 
```
  name                : An arbitrary name (Required)
  ncfile              : NetCDF sampling file (Required)
  trange              : Which times to average over (Optional, Default: [])
  group               : Which group to pull from netcdf file (Optional, Default: None)
  nperseg             : Number of snapshots per segment to specify number of blocks. (Required)
  yc                  : Lateral wake center (Required)
  zc                  : Vertical wake center (Required)
  wake_meandering_stats_file: The lateral and vertical wake center will be read from yc_mean and zc_mean columns of this file, overriding yc and zc. (Optional, Default: None)
  LR_factor           : Factor of blade-radius to define the radial domain extent. (Optional, Default: 1.4)
  NR                  : Number of points in the radial direction. (Optional, Default: 256)
  NTheta              : Number of points in the azimuthal direction. (Optional, Default: 256)
  remove_temporal_mean: Boolean to remove temporal mean from SPOD. (Optional, Default: True)
  remove_azimuthal_mean: Boolean to remove azimuthal mean from SPOD. (Optional, Default: False)
  iplane              : i-index of plane to postprocess (Optional, Default: 0)
  correlations        : List of correlations to include in SPOD. Separate U,V,W components with dash. Examples: U-V-W, U,V,W,V-W  (Optional, Default: ['U'])
  output_dir          : Directory to save results (Optional, Default: './')
  savepklfile         : Name of pickle file to save results (Optional, Default: '')
  loadpklfile         : Name of pickle file to load to perform actions (Optional, Default: '')
  compute_eigen_vectors: Boolean to compute eigenvectors or just eigenvalues (Optional, Default: True)
  sort                : Boolean to included sorted wavenumber and frequency indices by eigenvalue (Optional, Default: True)
  save_num_modes      : Number of eigenmodes to save, ordered by eigenvalue. Modes will be save in array of shape (save_num_mods,NR). (Optional, Default: None)
```

## Actions: 
```
  plot_eigvals        : ACTION: Plots the leading eigenvalues and corresponding wavenumber and frequencies (Optional)
    num               : Number of eigenvalues to plot (Optional, Default: 10)
    figsize           : Figure size (inches) (Optional, Default: [16, 4])
    savefile          : Filename to save the picture (Optional, Default: '')
    title             : Title of the plot (Optional, Default: '')
    dpi               : Figure resolution (Optional, Default: 125)
    correlations      : List of correlations to plot (Optional, Default: ['U'])
    Uinf              : Velocity for compute strouhal frequency (Required)
  plot_eigmodes       : ACTION: Plots leading eigenmodes (Optional)
    num               : Number of eigenvectors to include in reconstruction (Optional, Default: 1)
    figsize           : Figure size (inches) (Optional, Default: [16, 4])
    savefile          : Filename to save the picture (Optional, Default: '')
    title             : Title of the plot (Optional, Default: '')
    dpi               : Figure resolution (Optional, Default: 125)
    correlations      : List of correlations to plot (Optional, Default: ['U'])
    Uinf              : Velocity for compute strouhal frequency (Optional, Default: 0)
    St                : Plot leading eigenmodes at fixed Strouhal frequency (Optional, Default: None)
    itime             : Time iteration to plot (Optional, Default: 0)
```

## Example
```yaml
spod:
  name: Wake YZ plane
  ncfile: ./YZslice_00.50D_456.00s_1556.00s_cl00.nc
  group: xslice
  trange: [456.00,1556.50]
  nperseg: 256
  diam: 127
  yc: 375.0
  zc: 90
  wake_meandering_stats_file: ./data_converter/wake_meandering_baseline_area_1p4R/wake_meandering/wake_stats_00.50D.csv
  correlations:
    - U
    - V
    - W
    - U-V-W
    - V-W
  output_dir: ./test
  savepklfile: test.pkl
  #loadpklfile: ./test/test.pkl

  plot_eigvals:
    num: 11
    savefile: ./test/test_eigvals.png
    correlations:
      - U
    Uinf: 6.4

  plot_eigmodes:
    num: 2
    St: 0.3
    Uinf: 6.4
    savefile: ./test/test_eigmode.png
    correlations:
      - U
    ```
