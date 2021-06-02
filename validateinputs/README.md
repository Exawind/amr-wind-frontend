# Validation plugin structure


```bash
$ tree validateinputs/
validateinputs/
├── README.md         # This file
├── __init__.py       # Has header and library info
├── checkactuator.py  # Validation plugin for actuator inputs
├── plugin2.py
└── ...
```

Inside `__init__.py` it defines 
```python
class CheckStatus(Enum):
    PASS = 1
    SKIP = 2
    FAIL = 3
    WARN = 4
```

Here's an example of a condition check 
```python
from validateinputs import registerplugin
from validateinputs import CheckStatus as status

@registerplugin
class Check_max_level(): 
    name = "max_level"

    def check(self, app):
        max_level = app.inputvars['max_level'].getval()

        checkstatus                = {}   # Dict containing return status
        checkstatus['subname']     = ''   # Additional name info
        if max_level >= 0:
            checkstatus['result']  = status.PASS  
            checkstatus['mesg']    = 'max_level = %i >= 0'%max_level
        else:
            checkstatus['result']  = status.FAIL
            checkstatus['mesg']    = 'max_level = %i < 0'%max_level            
        return [checkstatus]              # Must be a list of dicts

```
