# Customizing `amrwind_frontend`


## Local preferences directory

`local`

```bash
./amrwind_frontend.py --localconfigdir CONFIGDIR
```

## Overriding and customizing inputs

```yaml
popupwindow:
  # === Set up a local run ===
  localrun:
    inputwidgets:
      - name:       localrun_exe
        defaultval: '/projects/wind_uq/lcheung/AMRWindBuilds/tcf.20210610/amr-wind.lcheung/build/amr_wind'
      - name:       localrun_modules
        defaultval: "module load cde/prod/gcc/7.2.0/openmpi/3.1.6 cde/prod/gcc/7.2.0/hdf5 cde/prod/gcc/7.2.0/netcdf-c/4.7.3"
```

```yaml
inputwidgets:
  - name:       submitscript_template
    # See https://learnxinyminutes.com/docs/yaml/ for YAML tricks
    # This is a default LSF scipt from summit
    defaultval: |
      #!/bin/bash
      # Begin LSF Directives
      #BSUB -P ABC123
      #BSUB -W 3:00
      #BSUB -nnodes 2048
      #BSUB -alloc_flags gpumps
      #BSUB -J RunSim123
      #BSUB -o RunSim123.%J
      #BSUB -e RunSim123.%J
      cd $MEMBERWORK/abc123
      cp $PROJWORK/abc123/RunData/Input.123 ./Input.123
      date
      jsrun -n 4092 -r 2 -a 12 -g 3 ./a.out
      cp my_output_file /ccs/proj/abc123/Output.123
```
