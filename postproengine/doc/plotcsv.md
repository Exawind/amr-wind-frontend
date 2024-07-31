# plotcsv

Make plots of csv files
## Inputs: 
```
  name                : An arbitrary name (Required)
  csvfiles            : A list of dictionaries containing csv files (Required)
  dpi                 : Figure resolution (Optional, Default: 125)
  figsize             : Figure size (inches) (Optional, Default: [12, 3])
  savefile            : Filename to save the picture (Optional, Default: '')
  xlabel              : Label on the X-axis (Optional, Default: 'Time [s]')
  ylabel              : Label on the Y-axis (Optional, Default: '')
  title               : Title of the plot (Optional, Default: '')
  legendopts          : Dictionary with legend options (Optional, Default: {})
  postplotfunc        : Function to call after plot is created. Function should have arguments func(fig, ax) (Optional, Default: '')
```

## Actions: 

## Example
```yaml
plotcsv:
  - name: plotfiles
    xlabel: 'Time'
    ylabel: 'Power'
    title: 'Turbine power'
    legendopts: {'loc':'upper left'}
    csvfiles:
    - {'file':'T0.csv', 'xcol':'Time', 'ycol':'GenPwr', 'lineopts':{'color':'r', 'lw':2, 'label':'T0'}}
    
```
