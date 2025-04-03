# plotcsv

Make plots of csv files
## Inputs: 
```
  name                : An arbitrary name (Required)
  csvfiles            : A list of dictionaries containing csv files (Required)
  dpi                 : Figure resolution (Optional, Default: 125)
  figsize             : Figure size (inches) (Optional, Default: [12, 3])
  savefile            : Filename to save the picture (Optional, Default: '')
  xlabel              : Label on the X-axis (Optional, Default: None)
  ylabel              : Label on the Y-axis (Optional, Default: None)
  xscale              : Scale on the X-axis (options: linear/log/symlog/logit) (Optional, Default: 'linear')
  yscale              : Scale on the Y-axis (options: linear/log/symlog/logit) (Optional, Default: 'linear')
  title               : Title of the plot (Optional, Default: '')
  legendopts          : Dictionary with legend options (Optional, Default: {})
  postplotfunc        : Function to call after plot is created. Function should have arguments func(fig, ax) (Optional, Default: '')
  figname             : Name/number of figure to create plot in (Optional, Default: None)
  axesnum             : Which subplot axes to create plot in (Optional, Default: None)
  fontsize            : Fontsize for figure (Optional, Default: 12)
  stylefile           : Load a custom style file (Optional, Default: None)
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

Note that the csvfiles dictionary list can also include xscalefunc and yscalefunc lambda functions 
to manipulate x and y inputs.  For instance,
```yaml
'xscalefunc':'lambda x:x-72.5'
```
shifts the x data by 72.5.  Similarly,
```yaml
'yscalefunc':'lambda y:y*2.0'
```
Multiples y by 2.0.  If ycol is the empty string '', then the lambda function input is the whole dataframe.  
This allows you to provide the function
```yaml
'yscalefunc':'lambda y:y["BldPitch1"]+["BldPitch1"]'
```
    
