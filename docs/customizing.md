# Customizing `amrwind_frontend`

The `amrwind_frontend` code can be customized so that it loads custom
defaults parameters, UI widgets, and additional program elements.
This is done by loading yaml files with inputs which can override or
extend what's already in `amrwind_frontend`.

## Local preferences directory

By default, `amrwind_frontend` will look in the `local` subdirectory
for any yaml files to load parameters.  It automatically load any file
with a `.yaml` extension, so the directory structure can look
something like

```
local
|-- README.md
|-- defaultvals.yaml
\-- localpreferences.yaml
```

and it load the `defaultvals.yaml` and `localpreferences.yaml` files.

However, if there are yaml files located in another directory which
should be loaded instead of `local/`, this can be specified at the
command line via

```bash
./amrwind_frontend.py --localconfigdir CONFIGDIR
```
and it will load all the yaml files from `CONFIGDIR` instead.

## Overriding and customizing inputs 

The structure of the yaml files in `local` should match what's already
being used in the [config.yaml](../config.yaml) file.  We'll provide a
couple of examples of below.  All of these snippets can be included in
a single yaml file, or spread out into several yaml files in the
`local` subdirectory.

### Changing the turbine repository directory

The location of the turbine repository directory is set by the
`preferences_turbinedir` input widget, with a default value of
`turbines/`.  This can be changed adding the following section of yaml
code:

```yaml
inputwidgets:
  # Change the turbine repo dir to myturbinedir
  - name:       preferences_turbinedir
    defaultval: myturbinedir
```

### Setting the `AMR-Wind` executable paths

To start a local AMR-Wind run,`amrwind_frontend` depends on the
knowing location of the AMR-Wind executable and any modules which must
be loaded beforehand.  To set these values, override the default
values of the `localrun_exe` and `localrun_modules` widgets in the
`popupwindow` section of the yaml file.

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

### Setting the submission script template

Similarly, the default submission script is a somewhat basic LSF
script. This can be customized for specific platforms via the
`submitscript_template` variable:

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
