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
  key1: value1
  key2: value2
```

## How to call the post-processing engine
```python
import postproengine as ppeng	
import ruamel.yaml	
yaml = ruamel.yaml.YAML(typ='unsafe', pure=True)
Loader=yaml.load

# Load the yaml file
with open(yamlfile, 'r') as fp:
    yamldict = Loader(fp, **loaderkwargs)

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
    name      = "task1"
    blurb     = """Description of task"""
    inputdefs = [
        {'key':'name',     'required':True,  'help':'An arbitrary name',  'default':''},
        {'key':'filename', 'required':True,  'help':'OpenFAST inputfile', 'default':''},
        {'key':'prop1',    'required':False, 'help':'random property',    'default':2},
    ]
    actionlist = {}

    # --- Stuff required for main task ---
    def __init__(self, inputs):
        self.yamldict = mergedicts(inputs, self.inputdefs)
        print('Initialized '+self.name)
        print(self.yamldict)
        print(self.actionlist)
        return
    
    def execute(self):
        # Do any task-related here
        print('Running task 1')
        
        # Do any sub-actions required for this task
        for a in self.actionlist:
            action = self.actionlist[a]
            # Check to make sure required actions are there
            if action.required and (action.actionname not in self.yamldict.keys()):
                # This is a problem, stop things
                raise ValueError('Required action %s not present'%action.actionname)
            if action.actionname in self.yamldict.keys():
                actionitem = action(self, self.yamldict[action.actionname])
                actionitem.execute()
        return

    # --- Stuff required for action list ---
    @registeraction(actionlist)
    class action1():
        actionname = 'action1'
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