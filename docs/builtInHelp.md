# Accessing the built in help

If you're using `amrwind-frontend` with the python interface, there
are a couple ways to accessing the built-in help.  The first is a more
general keyword search (`getInputHelp()`), and the second provides
more detailed help (`tellMeAbout()`).

## Using `getInputHelp()`

For a more general search of all inputs which mention a specific term
or keyword, try the `getInputHelp(keyword)` method.  This searches all
inputs, both the official AMR-Wind inputs and the extended inputs from
amrwind-frontend, and returns the search results.

**Example:**

To search for all inputs which mention `velocity`, do the following:

```python
>>> import amrwind_frontend as amrwind
>>> case=amrwind.MyApp.init_nogui()
>>> case.getInputHelp("velocity")
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

```python
>>> case.getInputHelp("gravity")
INTERNAL NAME                            AMRWIND PARAMETER                        DEFAULT VAL / DESCRIPTION
gravity                                  incflo.gravity                           [0.0, 0.0, -9.81]
```

## Using `tellMeAbout()`

On the other hand, if need additional details on a specific input, you
can use `tellMeAbout(input)`.  The argument `input` can be either the
internal name or AMR-Wind name.

**Example:**

To get detailed information about `incflo.gravity`, try

```python
>>> case.tellMeAbout("incflo.gravity")
Internal name       : gravity
AMR-Wind name       : incflo.gravity
Help                : Gravitational acceleration vector (x,y,z) [m/s^2]
Variable type       : [<type 'float'>, <type 'float'>, <type 'float'>]
GUI Label           : Gravity
Default value       : [0.0, 0.0, -9.81]
Option list         : None
>>>
```

**Example**

To see what are the valid options to set `time_control` (an
amrwind-frontend internal parameter only), try

```python
>>> case.tellMeAbout("time_control")
Internal name       : time_control
AMR-Wind name       : None
Help                : None
Variable type       : moretypes.listbox
GUI Label           : March with
Default value       : const dt
Option list         : ['const dt', 'max cfl']
```
