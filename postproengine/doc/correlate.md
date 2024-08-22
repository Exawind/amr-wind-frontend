# correlate

Calculate the two-point correlation and integral lengthscale
## Inputs: 
```
  name                : An arbitrary name (Required)
  ncfile              : NetCDF file (Required)
  group               : Group name in netcdf file (Required)
  timerange           : Time range to evaluate the correlation over (Required)
  iplane              : Plane number to probe (Required)
  probelength         : Probe length (Required)
  probelocationfunction: Function to call to generate point locations. Function should have no arguments and return a list of (i,j,k) indices (Required)
  plotprobept         : Make a plot of the probe locations (Optional, Default: False)
  saveprefix          : Filename prefix for all saved files (Optional, Default: '')
```

## Actions: 
```
  integrallengthscale : ACTION: Calculate the integral lengthscale (Optional)
    savefile          : YAML file to save the results to (Required)
```

## Example
```yaml
correlate:
  - name: two-point correlation (AMR-Wind)
    ncfile: /lustre/orion/cfd162/scratch/lcheung/sampling_80000.nc
    group: p_hub
    timerange: [20000, 21000]
    iplane: 0
    probelength: 1000
    probelocationfunction: spectrapoints.probelocations
    plotprobept: True
    saveprefix: correlation
    integrallengthscale:
      savefile: lengthscale.yaml

Note that in spectrapoints.py, the probelocations function is defined as:
def probelocations(s=1):
    import numpy as np
    ds = 10
    startx = np.arange(100,200,ds)
    starty = np.arange(100,200,ds)[::s]
    startp = []
    yoffset=0
    [[startp.append([x,y+yoffset*iy,0]) for x in startx] for iy, y in enumerate(starty)]
    return startp

```
