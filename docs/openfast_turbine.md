# Installing/using an OpenFAST turbine model

**Contents**
- [Introduction](#introduction)
- [Preliminaries](#preliminaries)
- [Compiling ROSCO](#compiling-roscox)
- [Editing the OpenFAST settings](#editing-the-openfast-settings)
- [Configure the frontend entry](#configure-the-frontend-entry)

## Introduction

Let's say that somebody gave you an OpenFAST turbine model, and you'd
like to use it within AMR-Wind, and of course, make it work with the
AMR-Wind frontend tool.  This document will go through some
step-by-step instructions to make sure that the turbine model is set
up and installed correctly.

To make things more concrete, we'll use the NREL 2.8-127 model
developed by Eliot Quon at NREL (see
https://github.com/NREL/openfast-turbine-models), although these
instructions should be generalizable to any OpenFAST model.

## Preliminaries

### Downloading the model

The first thing to do is to download the OpenFAST model to where
things will be run:  

```bash
$ git clone https://github.com/NREL/openfast-turbine-models.git
```

Then the particular model files will need to be copied over to where
the amrwind-frontend tool can find them.  We'll be editing the
OpenFAST files as well, so let's make a copy of them:  

```bash
$ cp -av openfast-turbine-models/IEA-scaled/NREL-2.8-127/OpenFAST ~/amrwind-frontend/turbines/OpenFAST_NREL2p8-127
```

Here we're just interested in the NREL-2.8-127 model.  This command
assumes that you downloaded amrwind-frontend to your home directory
(`~/amrwind-frontend`), and we'll put them in the `turbines` directory
called `OpenFAST_NREL2p8-127`.


### Checking the OpenFAST version

The OpenFAST model that you downloaded needs to be compatible with the
version of OpenFAST used in AMR-Wind.  If you're unsure of what
version was compiled in AMR-Wind, you can take a look at the output
logs of a sample run, and at the beginning of the run there is an
OpenFAST informational section:

```
Creating ExtSolver instance: OpenFAST

 **************************************************************************************************
 OpenFAST

 Copyright (C) 2021 National Renewable Energy Laboratory
 Copyright (C) 2021 Envision Energy USA LTD

 This program is licensed under Apache License Version 2.0 and comes with ABSOLUTELY NO WARRANTY.
 See the "LICENSE" file distributed with this software for details.
 **************************************************************************************************

 OpenFAST-v2.6.0
 Compile Info:
  - Compiler: GCC version 7.2.0
  - Architecture: 64 bit
  - Precision: double
  - Date: Aug  3 2021
  - Time: 19:33:05
```

This run shows **`OpenFAST-v2.6.0`**, for instance, and also says it
was compiled with `gcc` version 7.2.0.  This works well because the
current turbine models on
https://github.com/NREL/openfast-turbine-models.git are also set up
for OpenFAST version 2.6.

However, let's say that you'd like to use a newer version of OpenFAST,
for instance, version 3.0.0.  In that case, we'll need to make a few
edits to the turbine model.

In the OpenFAST v2.6 ServoDyn file (`NREL-2p8-127_ServoDyn.dat` in
this example), there is a section that looks like this:

```
---------------------- TUNED MASS DAMPER ---------------------------------------
False                  CompNTMD    - Compute nacelle tuned mass damper {true/false} (flag)
"none"                 NTMDfile    - Name of the file for nacelle tuned mass damper (quoted string) [unused when CompNTMD is false]
False                  CompTTMD    - Compute tower tuned mass damper {true/false} (flag)
"none"                 TTMDfile    - Name of the file for tower tuned mass damper (quoted string) [unused when CompTTMD is false]
```

For OpenFAST v3.0.0, replace those lines in the ServoDyn file with this section:  

```
---------------------- STRUCTURAL CONTROL --------------------------------------
0             NumBStC      - Number of blade structural controllers (integer)
"unused"      BStCfiles    - Name of the files for blade structural controllers (quoted strings) [unused when NumBStC==0]
0             NumNStC      - Number of nacelle structural controllers (integer)
"unused"      NStCfiles    - Name of the files for nacelle structural controllers (quoted strings) [unused when NumNStC==0]
0             NumTStC      - Number of tower structural controllers (integer)
"unused"      TStCfiles    - Name of the files for tower structural controllers (quoted strings) [unused when NumTStC==0]
0             NumSStC      - Number of substructure structural controllers (integer)
"unused"      SStCfiles    - Name of the files for substructure structural controllers (quoted strings) [unused when NumSStC==0]
```

Then you should be set to run with the newer OpenFAST version.

## Compiling ROSCO

The next thing we need to do is to get a compiled version of the
turbine DISCON controller.  This is necessary unless you want to run
with a simple default controller, or use fixed pitch/fixed rpm all of
the time.

The NREL 2.8-127 model uses the [ROSCO](https://github.com/NREL/ROSCO)
controller, also developed by NREL.  We'll need to download and
compile it.

First let's clone the ROSCO repository:

```bash
$ git clone --recursive https://github.com/NREL/ROSCO.git
Cloning into 'ROSCO'...
remote: Enumerating objects: 8698, done.
remote: Counting objects: 100% (1461/1461), done.
remote: Compressing objects: 100% (304/304), done.
remote: Total 8698 (delta 1240), reused 1291 (delta 1154), pack-reused 7237
Receiving objects: 100% (8698/8698), 18.24 MiB | 21.59 MiB/s, done.
Resolving deltas: 100% (5820/5820), done.
Updating files: 100% (617/617), done.
```

Then go in and make a build directory:

```bash
$ cd ROSCO/ROSCO
$ mkdir build
$ ls -l
total 16
drwxrwsr-x 2 lcheung wg-WindHFM 4096 Apr 22 10:40 build
-rwxrwxr-x 1 lcheung wg-WindHFM 2657 Apr 22 10:37 CMakeLists.txt
drwxrwsr-x 2 lcheung wg-WindHFM 4096 Apr 22 10:37 rosco_registry
drwxrwsr-x 3 lcheung wg-WindHFM 4096 Apr 22 10:37 src
$ cd build
```

Now make sure that the correct compiler modules are loaded.  For this
example, we'll use these `gcc` modules on the Sandia HPC machines:

```bash
$ module purge
$ module load cde/prod/compiler/gcc/7.2.0 cde/prod/gcc/7.2.0/openmpi/3.1.6 cde/prod/gcc/7.2.0/hdf5 cde/prod/gcc/7.2.0/netcdf-c/4.7.3 cde/prod/cmake/3.17.1
```

**Note:** The compiler version needs to match the compiler used to in
  compiling AMR-Wind (see above).  For instance, you can't use the
  `intel` compiler for ROSCO if you used `gcc` to compile AMR-Wind.

Then run cmake to configure the build files:  

```bash
$ cmake ..
-- The Fortran compiler identification is GNU 7.2.0
-- Check for working Fortran compiler: /projects/cde/v1/spack/opt/spack/linux-rhel7-x86_64/gcc-4.8.5/gcc-7.2.0-ibhp57j/bin/gfortran
-- Check for working Fortran compiler: /projects/cde/v1/spack/opt/spack/linux-rhel7-x86_64/gcc-4.8.5/gcc-7.2.0-ibhp57j/bin/gfortran - works
-- Detecting Fortran compiler ABI info
-- Detecting Fortran compiler ABI info - done
-- Checking whether /projects/cde/v1/spack/opt/spack/linux-rhel7-x86_64/gcc-4.8.5/gcc-7.2.0-ibhp57j/bin/gfortran supports Fortran 90
-- Checking whether /projects/cde/v1/spack/opt/spack/linux-rhel7-x86_64/gcc-4.8.5/gcc-7.2.0-ibhp57j/bin/gfortran supports Fortran 90 - yes
-- CMAKE_Fortran_COMPILER_ID = GNU
-- Setting system file as: src/SysFiles/SysGnuLinux.f90
-- Configuring done
-- Generating done
-- Build files have been written to: /projects/wind_uq/lcheung/LDRD_AWC/ROSCO/ROSCO/build
```

Once `cmake` successfully runs, you can compile the ROSCO controller
using the `make` command:

```
$ make
Scanning dependencies of target discon
[  9%] Building Fortran object CMakeFiles/discon.dir/src/Constants.f90.o
[ 18%] Building Fortran object CMakeFiles/discon.dir/src/ROSCO_Types.f90.o
[ 27%] Building Fortran object CMakeFiles/discon.dir/src/Filters.f90.o
[ 36%] Building Fortran object CMakeFiles/discon.dir/src/Functions.f90.o
[ 45%] Building Fortran object CMakeFiles/discon.dir/src/ControllerBlocks.f90.o
[ 54%] Building Fortran object CMakeFiles/discon.dir/src/Controllers.f90.o
[ 63%] Building Fortran object CMakeFiles/discon.dir/src/SysFiles/SysGnuLinux.f90.o
[ 72%] Building Fortran object CMakeFiles/discon.dir/src/ReadSetParameters.f90.o
[ 81%] Building Fortran object CMakeFiles/discon.dir/src/ROSCO_IO.f90.o
[ 90%] Building Fortran object CMakeFiles/discon.dir/src/DISCON.F90.o
[100%] Linking Fortran shared library libdiscon.so
[100%] Built target discon
```

Check the results of the compilation process.  There should be a
compiled library called libdiscon.so now.  

```bash
$ ls -l libdiscon.so
-rwxrwxr-x 1 lcheung wg-WindHFM 365000 Apr 22 10:48 libdiscon.so
```

We can move that to the same location where the other OpenFAST files
are copied.  This step is optional, but keeps the file organization
nice and neat.

```bash
$ cp -av libdiscon.so ~/amrwind-frontend/turbines/OpenFAST_NREL2p8-127/
```

## Editing the OpenFAST settings

Now before we actually run the turbine model, we need to change some
of the default parameters in the OpenFAST files so that they are
suitable for running in a combined AMR-Wind+OpenFAST context.

### `CompInflow` in the FST file

In the file `NREL-2p8-127.fst`, change the `CompInflow` flag to be 2 (external from OpenFOAM):

```
---------------------- FEATURE SWITCHES AND FLAGS ------------------------------
1                      CompElast   - Compute structural dynamics (switch) {1=ElastoDyn; 2=ElastoDyn + BeamDyn for blades}
2                      CompInflow  - Compute inflow wind velocities (switch) {0=still air; 1=InflowWind; 2=external from OpenFOAM}
```

### `WakeMod` in the Aerodyn file

In the file `NREL-2p8-127_AeroDyn15.dat`, change the `WakeMod` to be 0 (none):  

```
======  General Options  ============================================================================
False                  Echo        - Echo the input to "<rootname>.AD.ech"?  (flag)
default                DTAero      - Time interval for aerodynamic calculations {or "default"} (s)
0                      WakeMod     - Type of wake/induction model (switch) {0=none, 1=BEMT, 2=DBEMT, 3=OLAF} [WakeMod cannot be 2 or 3 when linearizing]
```

### File paths

And lastly, most file paths the OpenFAST model are _relative_ paths.
However, the discon library file path should be an _absolute_ path in
order for things to work reliably with amrwind-frontend's
expectations.  In the ServoDyn file `NREL-2p8-127_ServoDyn.dat`,
change the `DLL_FileName` to point to the absolute filename location
of `libdiscon.so`:

```
---------------------- BLADED INTERFACE ---------------------------------------- [used only with Bladed Interface]
"/home/USER/amrwind-frontend/turbines/OpenFAST_NREL2p8-127/libdiscon.so" DLL_FileName - Name/location of the dynamic library {.dll [Windows] or .so [Linux]} in the Bladed-DLL format (-) [used only with Bladed Interface]
```

## Configure the frontend entry

Finally, let's complete the process and set up turbine entries for
amrwind-frontend to use.  This will follow the same process as in the
[turbine repository doc](turbinerepo.md).  Create a file called
`nrel28-127.yaml` (the name can be arbitrary) in the `turbines/`
subdirectory of the amrwind-frontend repo.  Inside the yaml file, it
should have entries like:

```yaml
turbines:
  nrel28_127ALM:
    turbinetype_name:    "NREL 2.8-127 ALM"
    turbinetype_comment: 
    Actuator_type:       TurbineFastLine
    Actuator_openfast_input_file: OpenFAST_NREL2p8-127/NREL-2p8-127.fst
    Actuator_rotor_diameter:      127
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
    turbinetype_filedir: OpenFAST_NREL2p8-127

  nrel28_127ADM:
    turbinetype_name:    "NREL 2.8-127 ADM"
    turbinetype_comment: 
    Actuator_type:       TurbineFastDisk
    Actuator_openfast_input_file: OpenFAST_NREL2p8-127/NREL-2p8-127.fst
    Actuator_rotor_diameter:      127
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
    turbinetype_filedir: OpenFAST_NREL2p8-127
```

Note that the tags `nrel28_127ALM` and `nrel28_127ADM` are arbitrary,
they can be anything as long as they're unique.  The
`turbinetype_name` fields are also arbitrary, but also should be
unique.  Note that we're using the same set of OpenFAST files for both
the actuator line and actuator disk models, the only difference in the
specification is the `Actuator_type` field.  Also note that the
`Actuator_epsilon` values shown here might not be the optimal ones to
use, but deciding on the best values is another discussion.
