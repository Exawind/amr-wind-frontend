# controlvolume

Control volume analysis 
## Inputs: 
```
  name                : An arbitrary name (Required)
  inflow_velocity_XYZ : Inflow velocity from incflo.velocity setting in AMR-Wind (XYZ coordinates) (Required)
  diam                : Turbine diameter (Required)
  box_center_XYZ      : Center of control volume (XYZ coordinates) (Optional, Default: None)
  box_fr_center_XYZ   : Center of control volume on front face (XYZ coordinates) (Optional, Default: None)
  box_fr_streamwise_offset: Streamwise offset when specifying front face of control volume, in turbine diameters (Optional, Default: 0)
  rho                 : Density (Required)
  latitude            : Latitude (Required)
  body_force_XYZ      : Body force from AMR-Wind input file (XYZ coordinates) (Required)
  streamwise_box_size : Streamwise dimension of control volume in turbine diameters  (Required)
  lateral_box_size    : Lateral dimension of control volume in turbine diameters  (Required)
  vertical_box_size   : Vertical dimension of control volume in turbine diameters  (Required)
  compute_pressure_gradient: To approximate the streamwise pressure gradient based on finite different between streamwise planes (Optional, Default: True)
  bot_avg_file        : Bot avg pkl file (Required)
  bot_rs_file         : Bot rs pkl file (Required)
  varnames            : Variable names  (Optional, Default: ['velocityx', 'velocityy', 'velocityz'])
  top_avg_file        : top avg pkl file (Required)
  top_rs_file         : top rs pkl file (Required)
  lft_avg_file        : lft avg pkl file (Required)
  lft_rs_file         : lft rs pkl file (Required)
  rht_avg_file        : rht avg pkl file (Required)
  rht_rs_file         : rht rs pkl file (Required)
  streamwise_avg_files: streamwise avg pkl files (Required)
  streamwise_rs_files : streamwise rs pkl files (Required)
  savepklfile         : Name of pickle file to save results (Optional, Default: '')
```

## Actions: 
```
  print_table         : ACTION: Print table of results from control volume analysis (Optional)
  plot_totals         : ACTION: Plot totals from control volume (Optional)
    savefile          : filename to save plot (Optional, Default: None)
  plot_contributions  : ACTION: Plot contributions from control volume (Optional)
    savefile          : filename to save plot (Optional, Default: None)
```

## Example

```yaml
controlvolume:
  name: Streamwise CV
  inflow_velocity_XYZ: [9.03,0,0]
  diam: 240
  latitude: 39.55
  box_center_XYZ: [3000.0,1000.0,150.0]
  #box_fr_center_XYZ: [2280.0,1000.0,150.0]
  #box_fr_streamwise_offset: 0
  streamwise_box_size: 6
  lateral_box_size: 1
  vertical_box_size: 1
  rho: 1.2456
  body_force_XYZ: [0.00014295185866400572, 0.0008354682029301641, 0.0]
  varnames: ['velocitya1','velocitya2','velocitya3']
  compute_pressure_gradient: True
  savepklfile: 'Control_Volume_a123.pkl'
  #varnames: ['velocityx','velocityy','velocityz']

  bot_avg_file: '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_avg_XY.pkl'
  bot_rs_file:  '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_rs_XY.pkl'

  top_avg_file: '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_avg_XY.pkl'
  top_rs_file:  '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_rs_XY.pkl'

  lft_avg_file: '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_avg_XZl.pkl'
  lft_rs_file:  '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_rs_XZl.pkl'

  rht_avg_file: '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_avg_XZr.pkl'
  rht_rs_file:  '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_rs_XZr.pkl'

  streamwise_avg_files:
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_avg_YZwake1.pkl'
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_avg_YZwake2.pkl'
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_avg_YZwake3.pkl'
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_avg_YZwake4.pkl'
  streamwise_rs_files:
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_rs_YZwake1.pkl'
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_rs_YZwake2.pkl'
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_rs_YZwake3.pkl'
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_rs_YZwake4.pkl'

  print_table:

  plot_totals:
    savefile: 'test_cv_total_a1a2a3.png'

  plot_contributions:
    savefile: 'test_cv_contributions_a1a2a3.png'
```

