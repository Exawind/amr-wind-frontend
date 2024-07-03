# Post-processing engine

A plugin architecture for postprocessing AMR-Wind data

## input file structure

The inputs will take this form
```yaml
task1:
  key1: value1
  key2: value2
  action1:
    param1: val1
  action2:
    param1: val1

task2:
  - name: postprocessing1
    key1: value1		 
    key2: value2
  - name: postprocessing2
    key1: valueA		 
    key2: valueB		     
```

## How to call the post-processing engine
```python
import postproengine as ppeng	
import ruamel.yaml	
yaml = ruamel.yaml.YAML(typ='unsafe', pure=True)
Loader=yaml.load

# Load the yaml file
with open(yamlfile, 'r') as fp:
    yamldict = Loader(fp)

# Run the driver
ppeng.driver(yamldict)
```

## Typical plugin file structure

```python
from postproengine import registerplugin, mergedicts, registeraction
import os.path

"""
See README.md for details on the structure of classes here
"""

@registerplugin
class postpro_task1():
    """
    Example task
    """
    name      = "task1"                # Name of task (this is same as the name in the yaml)
    blurb     = "Description of task"  # Description of task
    inputdefs = [
        {'key':'name',     'required':True,  'help':'An arbitrary name',  'default':''},
        {'key':'filename', 'required':True,  'help':'OpenFAST inputfile', 'default':''},
        {'key':'prop1',    'required':False, 'help':'random property',    'default':2},
    ]
    actionlist = {}                    # Dictionary for holding sub-actions

    # --- Stuff required for main task ---
    def __init__(self, inputs):
        self.yamldictlist = []
        inputlist = inputs if isinstance(inputs, list) else [inputs]
        for indict in inputlist:
            self.yamldictlist.append(mergedicts(indict, self.inputdefs))
        print('Initialized '+self.name)
        print(self.yamldictlist)
        print(self.actionlist)
        return
    
    def execute(self):
        # Do any task-related here
        print('Running task 1')
        for yamldict in self.yamldictlist:
            # Run any necessary stuff for this task
            # [....]
            
            # Do any sub-actions required for this task
            for a in self.actionlist:
                action = self.actionlist[a]
                # Check to make sure required actions are there
                if action.required and (action.actionname not in yamldict.keys()):
                    # This is a problem, stop things
                    raise ValueError('Required action %s not present'%action.actionname)
                if action.actionname in yamldict.keys():
                    actionitem = action(self, yamldict[action.actionname])
                    actionitem.execute()
        return

    # --- Inner classes for action list ---
    @registeraction(actionlist)
    class action1():
        actionname = 'action1'
        blurb      = 'A description of action'
        required   = True
        actiondefs = [
            {'key':'name',     'required':True,  'help':'An arbitrary name',  'default':''},
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            print('Initialized '+self.actionname+' inside '+parent.name)
            print(self.actiondict)
            return

        def execute(self):
            print('Executing action1')
            return

```

## Self-documenting the tasks and actions

This part we still have to work out, but it might work something like this:
```python
>>> import postproengine as ppeng

>>> ppeng.print_inputs()
---
task1: Description of task
  name                : An arbitrary name
  filename            : OpenFAST inputfile
  prop1               : random property
  action1
    name              : An arbitrary name

---
task2: Description of task
  name                : An arbitrary name
  filename            : NetCDF inputfile
```

## Using the driver script

You can run the postprocessing engine through the driver script
[ppengine.py](../utilities/ppengine.py) via
```
python ppengine.py vizplanes.yaml
```

## Using the template notebook

You can run the postprocessing engine and capture all of the output in
notebook from the command line.  Take a look at the
[ppengine.ipynb](../utilities/ppengine.ipynb) notebook.

Run that notebook from the command line with
```bash
export YAMLFILE=INPUT.yaml
jupyter nbconvert --to notebook --execute ppengine.ipynb --output ${PWD}/OUTPUT.ipynb
```

where 
- `INPUT.yaml` is the name of the yaml file with the configuration (default: will look for postpro.yaml)
- `OUTPUT.ipynb` is the name of the output notebook file

Optional environment variables
- `WORKDIR`: this the base path where all of the netcdf sample files are relative to (the notebook will change to `WORKDIR` before running)
- `VERBOSE`: will turn on verbosity if this variable is set
- `TITLE`: an optional title for notebook
