# spod

Compute SPOD eigenvectors and eigenvalues
## Inputs: 
```
  name                : An arbitrary name (Required)
  ncfile              : NetCDF sampling file (Required)
  trange              : Which times to average over (Optional, Default: [])
  group               : Which group to pull from netcdf file (Optional, Default: None)
  nperseg             : Number of snapshots per segment to specify number of blocks. Default is 1 block. (Optional, Default: None)
  xc                  : Wake center on xaxis (Optional, Default: None)
  yc                  : Wake center on yaxis (Optional, Default: None)
  xaxis               : Which axis to use on the abscissa (Optional, Default: 'y')
  yaxis               : Which axis to use on the ordinate (Optional, Default: 'z')
  wake_meandering_stats_file: The lateral and vertical wake center will be read from yc_mean and zc_mean columns of this file, overriding yc and zc. (Optional, Default: None)
  LR_factor           : Factor of blade-radius to define the radial domain extent. (Optional, Default: 1.4)
  NR                  : Number of points in the radial direction. (Optional, Default: 256)
  NTheta              : Number of points in the azimuthal direction. (Optional, Default: 256)
  remove_temporal_mean: Boolean to remove temporal mean from SPOD. (Optional, Default: True)
  remove_azimuthal_mean: Boolean to remove azimuthal mean from SPOD. (Optional, Default: False)
  iplane              : List of i-index of plane to postprocess (Optional, Default: None)
  correlations        : List of correlations to include in SPOD. Separate U,V,W components with dash. Examples: U-V-W, U,V,W,V-W  (Optional, Default: ['U'])
  output_dir          : Directory to save results (Optional, Default: './')
  savepklfile         : Name of pickle file to save results (Optional, Default: '')
  loadpklfile         : Name of pickle file to load to perform actions (Optional, Default: None)
  compute_eigen_vectors: Boolean to compute eigenvectors or just eigenvalues (Optional, Default: True)
  sort                : Boolean to included sorted wavenumber and frequency indices by eigenvalue (Optional, Default: True)
  save_num_modes      : Number of eigenmodes to save, ordered by eigenvalue. Modes will be save in array of shape (save_num_mods,NR). (Optional, Default: None)
  cylindrical_velocities: Boolean to use cylindrical velocity components instead of cartesian. If True U->U_x, V->U_r, W->U_\Theta (Optional, Default: False)
  save_all_proj_coeff : Boolean to precompute and store the projection coefficients of the original signal onto all POD eigenvectors. Will significantly slow down SPOD computation. Use save_num_modes to only store a subset (Optional, Default: False)
  varnames            : Variables to extract from the netcdf file (Optional, Default: ['velocityx', 'velocityy', 'velocityz'])
  verbose             : Print extra information. (Optional, Default: True)
  nowindow            : Do not window time fourier transform with single block (e.g.,, for periodic signals in time). (Optional, Default: False)
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
  radial_shear_stress_flux: ACTION: Compute radial shear stress flux contribution from individual streamwise SPOD modes (Optional)
    number_of_modes   : Number of individual eigenvectors to include in reconstruction, sorted by eigenvalues (Optional, Default: 1)
    savefile          : Filename to save results (Optional, Default: '')
    correlations      : List of correlations (Optional, Default: ['U'])
    decompose_radial_velocity: Boolean to apply SPOD decomposition to radial velocity in addition to streamwise velocity. Radial velocity must be included in correlations. (Optional, Default: False)
    r                 : Radius value to compute radial shear stress flux. (Required)
    Uinf              : Inflow velocity for defining Strouhal number. (Optional, Default: None)
    ktheta_list       : List of kthetas to including in reconstruction. Override number of modes.  (Optional, Default: None)
    St_list           : List of Strouhal numbers to including in reconstruction. Override number of modes.  (Optional, Default: None)
  unit_tests          : ACTION: Run SPOD unit test suite (Optional)
    correlations      : List of correlations (Optional, Default: ['U'])
```

## Example

```yaml
spod:
  name: Wake YZ plane
  ncfile: /lustre/orion/cfd162/world-shared/lcheung/ALCC_Frontier_WindFarm/farmruns/LowWS_LowTI/ABL_ALM_10x10/rundir_AWC0/post_processing/rotor_*.nc
  iplane:
    - 7
    - 8
  group: T00_rotor
  trange: [25400,26000]
  nperseg: 256
  diam: 240
  #xc: 480
  yc: 150
  NR: 128
  NTheta: 128
  LR_factor: 1.2
  xaxis: 'a1'
  yaxis: 'a2'
  varnames: ['velocitya1','velocitya2','velocitya3']
  wake_meandering_stats_file:
    - ./T00_wake_meandering/wake_stats_7.csv
    - ./T00_wake_meandering/wake_stats_8.csv
  cylindrical_velocities: False
  correlations:
    - U
  output_dir: ./T00_spod_results/
  savepklfile: spod_{iplane}.pkl
  save_num_modes: 100
  verbose: True
  #loadpklfile: ./test/test.pkl

  plot_eigvals:
    num: 11
    savefile: ./eigvals_{iplane}.png
    correlations:
      - U
    Uinf: 6.5

  plot_eigmodes:
    num: 1
    Uinf: 6.5
    savefile: ./eigmode_{iplane}.png
    correlations:
      - U
```
    
