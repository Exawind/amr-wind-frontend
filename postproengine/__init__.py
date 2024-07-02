import os
import traceback
import copy
import io

# Load imp or importlib depending on what's available
try:
    from importlib import util
    useimp = False
except:
    import imp
    useimp = True

# See https://gist.github.com/dorneanu/cce1cd6711969d581873a88e0257e312
# for more information

"""
Plugin file structure
validateinputs/
|-- __init__.py
|-- plugin1.py
|-- plugin2.py
|-- ...
"""

# The list of all plugins is kept and built here
pluginlist = {}
def registerplugin(f):
    """
    Register all the plugins to pluginlist
    """
    pluginlist[f.name]=f
    return f

def registeraction(alist):
    """Decorator to add action"""
    def inner(f):
        alist[f.actionname]=f
    return inner


def mergedicts(inputdict, inputdefs):
    """
    Merge the input dictionary with the task  
    """
    outputdict = copy.deepcopy(inputdict)
    for key in inputdefs:
        if key['required'] and (key['key'] not in outputdict):
            # This is a problem, stop things
            raise ValueError('Required key %s not present'%key['key'])
        if (not key['required']) and (key['key'] not in outputdict):
            outputdict[key['key']] = key['default']
    return outputdict

# Small utility to automatically load modules
def load_module(path):
    name = os.path.split(path)[-1]
    if useimp:
        module = imp.load_source(name.split('.')[0], path)
    else:
        spec = util.spec_from_file_location(name, path)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module

def print_inputs(subset=[], plist=pluginlist):
    """
    Prints out the inputs required for every plugin
    """
    if len(subset)>0:
        looptasks = subset
    else:
        looptasks = plist.keys()
    for task in looptasks:
        # Print out info on each task
        inputdefs = plist[task].inputdefs
        print('---')
        print(task+': '+plist[task].blurb)
        for input in inputdefs:
            print('  %-20s: %s'%(input['key'], input['help']))
        # Loop through the action items
        if len(plist[task].actionlist)>0:
            for action in plist[task].actionlist:
                print('  %s'%action)
                for input in plist[task].actionlist[action].actiondefs:
                    print('    %-18s: %s'%(input['key'], input['help']))
        print()
    return

def driver(yamldict, subset=[], plist=pluginlist, verbose=False):
    """
    Run through and execute all tasks
    """
    if len(subset)>0:
        looptasks = subset
    else:
        looptasks = plist.keys()
    for task in looptasks:
        if task in yamldict:
            taskitem = plist[task](yamldict[task], verbose=verbose)
            taskitem.execute(verbose=verbose)
    return

def test():
    # Only execute this if test.py is included
    yamldict = {
        'task1':[{'name':'MyTask1',
                 'filename':'myfile',
                 'action1':{'name':'myaction'}}],
        'task2':{'name':'MyTask2', 'filename':'myfile2', },
    }
    driver(yamldict)
    return

# ------------------------------------------------------------------
# Get current path
path    = os.path.abspath(__file__)
dirpath = os.path.dirname(path)

# Load all plugins in this directory
for fname in os.listdir(dirpath):
    # Load only "real modules"
    if not fname.startswith('.') and \
       not fname.startswith('__') and fname.endswith('.py'):
        try:
            load_module(os.path.join(dirpath, fname))
        except Exception:
            traceback.print_exc()

