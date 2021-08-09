# Turbine repository

## Overview

Overview

## YAML file structure

Each yaml file in the `turbines/` directory should have a specific
structure.  There should be a `turbines:` section at the top, with all
turbines underneath.

For instance, the definition of the `nre5mwAM` 
```yaml
turbines:
  nre5mwALM:
    turbinetype_name:             "NREL5MW ALM"
    turbinetype_comment:          "Any comment you want to add"
    Actuator_type:                TurbineFastLine
    Actuator_openfast_input_file: OpenFAST_NREL5MW/nrel5mw.fst
    Actuator_rotor_diameter:      126
    Actuator_hub_height:          90
    Actuator_num_points_blade:    64
    Actuator_num_points_tower:    12
    Actuator_epsilon:             [10.0, 10.0, 10.0]
    Actuator_epsilon_tower:       [5.0, 5.0, 5.0]
    Actuator_openfast_start_time: 0.0
    Actuator_openfast_stop_time:  1000.0
    Actuator_nacelle_drag_coeff:  0.0
    Actuator_nacelle_area:        0.0
    Actuator_output_frequency:    10
    turbinetype_filedir:          OpenFAST_NREL5MW
```

Each field 
