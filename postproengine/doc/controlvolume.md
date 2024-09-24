# controlvolume

Control volume analysis 
## Inputs: 
```
  name                : An arbitrary name (Required)
  Uinf                : U inflow (Required)
  diam                : Turbine diameter (Required)
  axis                : Order of axis (Required)
  center              : Center of control volume (Required)
  rho                 : Density (Required)
  box_dims            : Dimensions of control volume in turbine diameters (Required)
  bot_avg_file        : Bot avg pkl file (Required)
  bot_rs_file         : Bot rs pkl file (Required)
  top_avg_file        : top avg pkl file (Required)
  top_rs_file         : top rs pkl file (Required)
  lft_avg_file        : lft avg pkl file (Required)
  lft_rs_file         : lft rs pkl file (Required)
  rht_avg_file        : rht avg pkl file (Required)
  rht_rs_file         : rht rs pkl file (Required)
  x_avg_files         : x avg pkl files (Required)
  x_rs_files          : x rs pkl files (Required)
```

## Actions: 
```
  table               : ACTION: Print table of results from control volume analysis (Optional)
  plot_totals         : ACTION: Plot totals from control volume (Optional)
    savefile          : filename to save plot (Optional, Default: None)
  plot_contributions  : ACTION: Plot contributions from control volume (Optional)
    savefile          : filename to save plot (Optional, Default: None)
```

## Example
```yaml
controlvolume:
  name: Streamwise CV
  Uinf: 9.03
  diam: 240
  center: [2280.0,1000.0,150.0]
  axis: ['x','y','z']
  rho: 1.2456
  box_dims: [6,1,1]
  body_force: [0.00014295185866400572, 0.0008354682029301641, 0.0]

  bot_avg_file: '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_avg_XY.pkl'
  bot_rs_file:  '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_rs_XY.pkl'

  top_avg_file: '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_avg_XY.pkl'
  top_rs_file:  '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_rs_XY.pkl'

  lft_avg_file: '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_avg_XZl.pkl'
  lft_rs_file:  '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_rs_XZl.pkl'

  rht_avg_file: '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_avg_XZr.pkl'
  rht_rs_file:  '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_rs_XZr.pkl'

  x_avg_files:
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_avg_YZwake1.pkl'
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_avg_YZwake2.pkl'
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_avg_YZwake3.pkl'
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_avg_YZwake4.pkl'
  x_rs_files:
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_rs_YZwake1.pkl'
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_rs_YZwake2.pkl'
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_rs_YZwake3.pkl'
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_rs_YZwake4.pkl'

  table:

  plot_totals:
    savefile: 'test_cv_total.png'

  plot_contributions:
    savefile: 'test_cv_contributions.png'
    
```
