# probeplt

Probe PLT output at arbitrary points
## Inputs: 
```
  name                : An arbitrary name (Optional, Default: 'sample')
  pltdirs             : An arbitrary name (Required)
  maxlevel            : Maximum level for plt (Optional, Default: 0)
  varnames            : Variable names to extract (Optional, Default: ['velocityx', 'velocityy', 'velocityz'])
```

## Actions: 
```
  samplegrid          : ACTION: Sample a regular Cartesian grid (Optional)
    filesuffix        : The output filename suffix (Required)
    origin            : Origin point for sampling (Required)
    axis1             : First axis of sampling volume (Required)
    axis2             : Second axis of sampling volume (Required)
    axis3             : Third axis of sampling volume (Required)
    Npoints           : Number of points along each axis [N1, N2, N3] (Required)
  samplexyz           : ACTION: Sample a set of (x,y,z) given by a file (Optional)
    filesuffix        : The output filename suffix (Required)
    xyzfile           : The filename containing a list of (x,y,z) points  (Required)
```

## Example

Example input for `probeplt`
```yaml
globalattributes:
  verbose: True
  executeorder:
  - probeplt

probeplt:
    name: STUFF
    maxlevel: 0
    pltdirs: 
    - /tscratch/lcheung/MMSEI/GABLS/GABLS1-AMR-Wind.ERFRestart3/plt18000
    - /tscratch/lcheung/MMSEI/GABLS/GABLS1-AMR-Wind.ERFRestart3/plt19000
    samplegrid:
        filesuffix: morestuff.dat
        origin: [200, 0, 0]
        axis1:  [0, 400, 0]
        axis2:  [0, 0, 400]
        axis3:  [0, 0, 0]
        Npoints: [129, 129, 1]
    samplexyz:
        filesuffix: comparestuff
        xyzfile: xyz.dat
```

Note that `filesuffix` will be appended to the plt names to create the
output filenames.  In the example above, the files that get output are
```
plt18000_comparestuff.csv
plt18000_morestuff.dat.csv
plt19000_comparestuff.csv
plt19000_morestuff.dat.csv
``` 


