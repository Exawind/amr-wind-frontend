from validateinputs import registerplugin
from validateinputs import CheckStatus as status

"""
See README.md for details on the structure of classes here
"""

@registerplugin
class PluginA(): 
    name   = "Plugin A"
    #active = False
    def check(self, app):
        return [{'subname':'', 'result':status.SKIP, 'mesg':''}]


@registerplugin
class Check_max_level(): 
    name = "max_level"

    def check(self, app):
        max_level = app.inputvars['max_level'].getval()

        checkstatus = {}       # Dict which holds
        checkstatus['subname'] = ''
        if max_level >= 0:
            checkstatus['result']  = status.PASS  
            checkstatus['mesg']    = 'max_level = %i >= 0'%max_level
        else:
            checkstatus['result']  = status.FAIL
            checkstatus['mesg']    = 'max_level = %i < 0'%max_level            
        return [checkstatus]   # Must be a list of dicts
