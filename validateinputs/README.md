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

Here's an example of a condition check 
```python
from validateinputs import registerplugin
from validateinputs import CheckStatus as status

@registerplugin                # Class decorator to register check condition
class CheckCondition(): 
    name   = "Condition"       # Name of condition to check
	active = True              # Optional, assumed true if not present

    def check(self, app):      # Must be fuction called check
		checkstatus = {}       # Dict which holds
		checkstatus['subname'] = 'Test XYZ'
		checkstatus['result']  = CheckStatus.PASS
		checkstatus['mesg']    = 'All good'
		return [checkstatus]   # Must be a list of dicts
```
