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
    iplane            : Index of x location to read (Required)
    yhh               : Location in flow to use as hub height location in y (Required)
    zhh               : Location in flow to use as hub height location in z (Required)
    btsfile           : bts file name to save results (Required)
    ID                : bts file ID. 8="periodic", 7="non-periodic" (Optional, Default: 8)
    turbine_height    : Height of the turbine (if different than zc) (Optional, Default: None)
    group             : Which group to pull from netcdf file (Optional, Default: None)
  nalu_to_amr         : ACTION: Converts a list of planes from Nalu-Wind to AMR-Wind (Optional)
    savefile          : Name of AMR-Wind file (Required)
    coordfile         : Nalu-Wind coordinate file (Optional, Default: None)
    groupname         : netCDF group name (Optional, Default: 'plane')
```
