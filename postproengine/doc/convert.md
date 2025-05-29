# convert

Converts netcdf sample planes to different file formats
## Inputs: 
```
  name                : An arbitrary name (Required)
  filelist            : NetCDF sampling file (Required)
  trange              : Which times to pull from netcdf file, e.g., [tstart,tend] (Required)
```

## Actions: 
```
  bts                 : ACTION: Converts data to bts files (Optional)
    iplane            : Index of x location to read (Optional, Default: None)
    xc                : Location in flow to center plane on in the abscissa. (Optional, Default: None)
    yc                : Location to center plane on in the ordinate. (Optional, Default: None)
    btsfile           : bts file name to save results (Required)
    ID                : bts file ID. 8="periodic", 7="non-periodic" (Optional, Default: 8)
    turbine_height    : Height of the turbine. (Required)
    group             : Which group to pull from netcdf file (Optional, Default: None)
    diam              : Diameter for computing rotor averaged velocity (Required)
    xaxis             : Which axis to use on the abscissa (Optional, Default: 'y')
    yaxis             : Which axis to use on the ordinate (Optional, Default: 'z')
    varnames          : Variables to extract from the netcdf file (Optional, Default: ['velocityx', 'velocityy', 'velocityz'])
  nalu_to_amr         : ACTION: Converts a list of planes from Nalu-Wind to AMR-Wind (Optional)
    savefile          : Name of AMR-Wind file (Required)
    coordfile         : Nalu-Wind coordinate file (Optional, Default: None)
    groupname         : netCDF group name (Optional, Default: 'plane')
```
