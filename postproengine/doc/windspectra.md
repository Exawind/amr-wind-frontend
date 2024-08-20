# windspectra

Calculate the wind spectra in time
## Inputs: 
```
  name                : An arbitrary name (Required)
  ncfile              : NetCDF file (Required)
  group               : Group name in netcdf file (Required)
  pointlocationfunction: Function to call to generate point locations. Function should have no arguments and return a list of (i,j,k) indices (Required)
  csvfile             : Filename to save spectra to (Required)
  timeindices         : Which indices to use from netcdf file (Optional, Default: [])
  avgbins             : Averaging time windows (Optional, Default: [])
  thirdoctaveband     : Use 1/3 octave band averaging (Optional, Default: False)
  normalize           : Normalize the output spectra f and U^2 (Optional, Default: True)
```

## Actions: 
```
  kaimal              : ACTION: Get the Kaimal spectra (Optional)
    csvfile           : CSV file to save Kaimal spectra to (Required)
    ustarsource       : Source of ustar information (Options: specified/ablstatsfile) (Required)
    ustar             : Group name in netcdf file (Optional, Default: 0.0)
    z                 : z-height for Kaimal spectra (Required)
    ablstatsfile      : NetCDF abl statistics file (Optional, Default: '')
    avgt              : Average time over ustar (Optional, Default: [])
```

## Example
```yaml
windspectra:
- name: spectra1
  ncfile: /lustre/orion/cfd162/scratch/lcheung/sampling_80000.nc
  group: p_bot
  pointlocationfunction: spectrapoints.getptlist
  csvfile: spectra1.csv
  kaimal:
    ustarsource: ablstatsfile
    ablstatsfile:  /lustre/orion/cfd162/scratch/lcheung/abl_statistics80000.nc
    avgt: [20000, 25000]
    #ustar: 0.289809
    csvfile: kaimal1.csv
    z: 26.5

```
