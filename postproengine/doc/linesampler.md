# linesampler

Process line sample files
## Inputs: 
```
  name                : An arbitrary name (Required)
  ncfile              : A list of netcdf files (Required)
  group               : Which group to use in the netcdf file (Optional, Default: None)
  varnames            : Variables to extract from the netcdf file (Optional, Default: ['velocityx', 'velocityy', 'velocityz'])
```

## Actions: 
```
  average             : ACTION: Time average the line (Optional)
    savefile          : Filename to save the radial profiles (Required)
    tavg              : Times to average over (Optional, Default: [])
```

## Example

```yaml
linesampler:
- name: metmast_1k
  ncfile: 
  - /gpfs/lcheung/HFM/exawind-benchmarks/convective_abl/post_processing/metmast_30000.nc
  group: virtualmast
  varnames: ['velocityx', 'velocityy', 'velocityz', 'temperature']
  average:
    tavg: [15000, 16000]
    savefile: ../results/avgmast_1000.csv
```

