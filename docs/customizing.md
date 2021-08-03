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

## Customizing the submission script

The submission script process can also be customized using the
approprate yaml entries in `local` preferences folder.  By default,
the `amrwind-frontend` contains a very basic LSF script which looks
like this:

```bash
#!/bin/bash
# Begin LSF Directives
#BSUB -W submitscript_runtime
#BSUB -n submitscript_numnodes
#BSUB -J submitscript_jobname
#BSUB -o %J.out
#BSUB -e %J.err

submitscript_modules 

mpirun -np submitscript_numnodes submitscript_exe submitscript_inputfile
```

This is connected to the submission script window that appears when
you select **"Run -> Job submission"** from the menu bar:  

![](tutorial1/images/submitscript_basic_filledout.png)

The entries in the fields above get inserted into the submission
script template and saved into a new submission.  

### How it works

Note that there are certain keywords like `submitscript_runtime`,
`submitscript_numnodes`, `submitscript_jobname`, etc., in the template
script above.  Those are placeholders for values which will come from
the inputfields.  All of the input fields are defined in the
`submitscript` section of the `popupwindow` definitions in
[`config.yaml`](../config.yaml).

For instance, the `submitscript_runtime` input is defined by this widget:
```yaml
popupwindow:
  submitscript:
    inputwidgets:
      - name:       submitscript_runtime   # Internal name to amrwind-frontend
        label:      "Run time (HH:MM:SS)"  # Description next to input field
        inputtype:  str
        defaultval: "1:00:00"              # Default value
        outputdef:
          replacevar: submitscript_runtime # Keyword that's replaced in template script
```

The structure of this yaml input is governed by the specifics of the
[TK yaml](https://github.com/lawrenceccheung/tkyamlgui) library, but
the important things to note are:

- The `defaultval` field is the default value that shows up in the
  input form.  You can change this to anything you'd like
  
- The `replacevar` parameter under `outputdef` controls what keyword
  is replaced in the template script.  You can change this to match
  anything in the script.

For instance, if you wanted to change the default run time for all
scripts to `24:00:00`, create a yaml file in the `local` directory
with the following:

```yaml
popupwindow:
  submitscript:
    inputwidgets:
      - name:       submitscript_runtime   # Internal name to amrwind-frontend
        defaultval: "24:00:00"             # Default value now 24hrs
```

and `defaultval` will get updated (all other parameters will remain
unchanged.)

### Setting the submission script template

Obviously the default submission script listed above is insufficient
for many platforms.  We can customize it for specific uses via the
`submitscript_template` variable:

```yaml
inputwidgets:
  - name:       submitscript_template
    # See https://learnxinyminutes.com/docs/yaml/ for YAML tricks
    # This is a default LSF scipt from summit
    defaultval: |
      #!/bin/bash
      # Begin LSF Directives
      #BSUB -P submitscript_project
      #BSUB -W submitscript_runtime
      #BSUB -nnodes submitscript_numnodes
      #BSUB -alloc_flags gpumps
      #BSUB -J submitscript_jobname
      #BSUB -o submitscript_jobname.%J
      #BSUB -e submitscript_jobname.%J

      submitscript_modules 

      jsrun -n submitscript_ncpu submitscript_exe submitscript_inputfile
```

Note that there's now a `submitscript_project` and a
`submitscript_ncpu` keyword that have been added.  These can be added
to the form by having this yaml file included in the `local`
subdirectory.

```yaml
popupwindow:
  submitscript:
    inputwidgets:
      - name:       submitscript_project
        label:      Project name
        inputtype:  str
        defaultval: MYPROJECT
        outputdef:
          replacevar: submitscript_project
      - name:       submitscript_ncpu
        label:      Total number of cpu's
        inputtype:  int
        defaultval: 16
        outputdef:
          replacevar: submitscript_ncpu
```

Now when the job submission form is pulled up, the two new fields can
be seen:  

![](images/jobsubmission_customized.png)

