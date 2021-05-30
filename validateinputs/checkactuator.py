from validateinputs import registerplugin

"""

# --------Validation class template -----------
@registerplugin
class PluginB(): 
    name = "Plugin B"

    def check(self, app):
        print("Plugin B")
"""

@registerplugin
class PluginA(): 
    name = "Plugin A"

    def check(self, app):
        print(self.name)


@registerplugin
class PluginB(): 
    name = "Plugin B"

    def check(self, app):
        print(self.name)
        print("max_level: "+repr(app.inputvars['max_level'].getval()))
