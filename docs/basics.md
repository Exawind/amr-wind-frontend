# Basics

**Contents**
- [Using the GUI](#using-the-gui)
- [Using the python interface](#using-the-python-interface)

## Using the GUI

## Using the python interface


### Loading the module
```python
>>> import sys
>>> sys.path.insert(1, 'amrwind-frontend')
>>> import amrwind_frontend as amrwind
```

### Start a case
```python
>>> case1=amrwind.MyApp.init_nogui()
```

Note that multiple cases can be initiated simultaneously:
```python
>>> case1=amrwind.MyApp.init_nogui()
>>> case2=amrwind.MyApp.init_nogui()
```
The inputs and parameters for `case1` and `case2` will be completely independent.

If you want to see what the current setup looks like, we can print the
input file using `writeAMRWindInput()`:

```python
>>> print(case1.writeAMRWindInput(''))
# --- Simulation time control parameters ---
time.stop_time                           = 100.0               # Max (simulated) time to evolve [s]
time.max_step                            = -1
time.fixed_dt                            = -1.0                # Fixed timestep size (in seconds). If negative, then time.cfl is used
incflo.verbose                           = 0
io.check_file                            = chk
incflo.use_godunov                       = true
incflo.godunov_type                      = ppm
incflo.gravity                           = 0.0 0.0 -9.81       # Gravitational acceleration vector (x,y,z) [m/s^2]
incflo.density                           = 1.0                 # Fluid density [kg/m^3]
transport.viscosity                      = 1.872e-05           # Fluid dynamic viscosity [kg/m-s]
transport.laminar_prandtl                = 0.7                 # Laminar prandtl number
transport.turbulent_prandtl              = 0.3333              # Turbulent prandtl number

# --- Geometry and Mesh ---
geometry.prob_lo                         = 0.0 0.0 0.0
geometry.prob_hi                         = 1000.0 1000.0 1000.0
amr.n_cell                               = 48 48 48            # Number of cells in x, y, and z directions
amr.max_level                            = 0
geometry.is_periodic                     = 1 1 0
zlo.type                                 = no_slip_wall
zhi.type                                 = no_slip_wall

# --- ABL parameters ---
ICNS.source_terms                        =
incflo.velocity                          = 10.0 0.0 0.0
ABLForcing.abl_forcing_height            = 0.0
time.plot_interval                       = 1000
io.plot_file                             = plt
io.KE_int                                = -1

#---- extra params ----
#== END AMR-WIND INPUT ==
```

#### Loading a case

```python
case1.loadAMRWindInput(filename)
```

### Setting parameters

#### Using `loadAMRWindInput`

#### Using `setAMRWindInput`

#### Retrieving parameters

### Internal versus AMR-Wind parameters

- AMR-Wind Input file parameters: 

- amrwind-frontend internal parameters: 

Every AMR-Wind parameter used in the GUI will have a corresponding
internal parameter in amrwind-frontend, but not all internal
parameters will correspond to an AMR-Wind parameters.  For instance,
the AMR-Wind parameter for gravity is `incflo.gravity`, and is linked
to the internal amrwind-frontend parameter `gravity`.  However, the
ABL averaging times variable `ablstats_avgt` is used only in the
post-processing section of amrwind-frontend, and has no corresponding
variable in the AMR-Wind input file.

### Getting help

If you need help with a specific parameter, or not sure how something
is spelled exactly, you can search for it using `getInputHelp()`.  For
instance, to find all parameters or names which contain the string
`velocity`, use

```python
>>> case1.getInputHelp('velocity')
INTERNAL NAME                            AMRWIND PARAMETER                        DEFAULT VAL / DESCRIPTION
ConstValue_velocity                      ConstValue.velocity.value
xlo_velocity                             xlo.velocity
xhi_velocity                             xhi.velocity
ylo_velocity                             ylo.velocity
yhi_velocity                             yhi.velocity
zlo_velocity                             zlo.velocity
zhi_velocity                             zhi.velocity
ABL_bndry_var_names                      ABL.bndry_var_names                      'velocity temperature'
ABL_velocity                             incflo.velocity                          [10.0, 0.0, 0.0]
perturb_velocity                         ABL.perturb_velocity                     0
sampling_fields                          sampling.fields                          'velocity'
```

