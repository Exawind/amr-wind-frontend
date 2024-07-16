import sys
import os
import traceback
import copy
import io
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

scriptpath = os.path.dirname(os.path.realpath(__file__))

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
    if bool(inputdict) is False:
        outputdict = {}
    else:
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


def print_readme(f, plist=pluginlist, docformat='markdown'):
    """
    Print the docs/README.md for all postpro engine executors 
    """
    header="""
# Postprocessing engine workflows

The following workflows are available:

"""
    f.write(header)
    looptasks = plist.keys()
    for task in looptasks:
        executormd = task+'.md'
        blurb      = plist[task].blurb
        f.write(f'- [{task}]({executormd}): {blurb}\n')
    return

def print_executor(f, task, docformat='markdown'):
    """
    Write the documentation page for a task
    """
    taskname = task.name
    blurb    = task.blurb
    inputdefs = task.inputdefs
    f.write(f'# {taskname}\n')
    f.write('\n')
    f.write(f'{blurb}\n')
    f.write('## Inputs: \n')
    # Write the inputs
    f.write('```\n')
    for input in inputdefs:
        if input['required']:
            extrainfo = '(Required)'
        else:
            extrainfo = '(Optional, Default: %s)'%repr(input['default'])
        f.write('  %-20s: %s %s\n'%(input['key'], input['help'], extrainfo))
    f.write('```\n\n')
    f.write('## Actions: \n')
    # Loop through the action items
    if len(task.actionlist)>0:
        f.write('```\n')
        for action in task.actionlist:
            description = task.actionlist[action].blurb
            if task.actionlist[action].required:
                description += ' (Required)'
            else:
                description += ' (Optional)'
            f.write('  %-20s: ACTION: %s\n'%(action, description))
            for input in task.actionlist[action].actiondefs:
                if input['required']:
                    extrainfo = '(Required)'
                else:
                    extrainfo = '(Optional, Default: %s)'%repr(input['default'])
                f.write('    %-18s: %s %s\n'%(input['key'], input['help'], extrainfo))
        f.write('```\n')
    # Write any examples
    if hasattr(task, 'example'):
        f.write('\n')
        f.write('## Example\n')
        f.write('```yaml')
        f.write(task.example)
        f.write('```\n')
    return

def makedocs(rootpath=scriptpath, docdir='doc', docformat='markdown'):
    """
    Create the documentation directory
    """
    docpath = os.path.join(rootpath, docdir)
    if not os.path.exists(docpath):
        os.makedirs(docpath)
        
    with open(os.path.join(docpath, 'README.md'), 'w') as f:
        print_readme(f)

    looptasks = pluginlist.keys()
    for task in looptasks:
        mdfile = os.path.join(docpath, pluginlist[task].name+'.md')
        print(mdfile)
        with open(mdfile, 'w') as f:
            print_executor(f, pluginlist[task])
    return

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
            if input['required']:
                extrainfo = '(Required)'
            else:
                extrainfo = '(Optional, Default: %s)'%repr(input['default'])
            print('  %-20s: %s %s'%(input['key'], input['help'], extrainfo))
        # Loop through the action items
        if len(plist[task].actionlist)>0:
            for action in plist[task].actionlist:
                description = plist[task].actionlist[action].blurb
                if plist[task].actionlist[action].required:
                    description += ' (Required)'
                else:
                    description += ' (Optional)'
                print('  %-20s: ACTION: %s'%(action, description))
                for input in plist[task].actionlist[action].actiondefs:
                    if input['required']:
                        extrainfo = '(Required)'
                    else:
                        extrainfo = '(Optional, Default: %s)'%repr(input['default'])
                    print('    %-18s: %s %s'%(input['key'], input['help'], extrainfo))
        if hasattr(plist[task], 'example'):
            print()
            print('Example')
            print('-------')
            print(plist[task].example)
        print()
    return

class contourplottemplate():
    """
    A contour plot template that can be used by other executors
    """
    actionname = 'plot'
    blurb      = 'Plot rotor averaged planes'
    required   = False
    actiondefs = [
        {'key':'dpi',       'required':False,  'default':125,
         'help':'Figure resolution', },
        {'key':'figsize',   'required':False,  'default':[12,8],
         'help':'Figure size (inches)', },
        {'key':'savefile',  'required':False,  'default':'',
         'help':'Filename to save the picture', },
        {'key':'clevels',   'required':False,  'default':'41', 
         'help':'Color levels (eval expression)',},
        {'key':'iplane',   'required':False,  'default':0,
         'help':'Which plane to pull from netcdf file', },            
        {'key':'xaxis',    'required':False,  'default':'x',
         'help':'Which axis to use on the abscissa', },
        {'key':'yaxis',    'required':False,  'default':'y',
         'help':'Which axis to use on the ordinate', },            
        {'key':'xlabel',    'required':False,  'default':'X [m]',
         'help':'Label on the X-axis', },
        {'key':'ylabel',    'required':False,  'default':'Y [m]',
         'help':'Label on the Y-axis', },
        {'key':'title',     'required':False,  'default':'',
         'help':'Title of the plot',},
        {'key':'plotfunc',  'required':False,  'default':'lambda db: 0.5*(db["uu_avg"]+db["vv_avg"]+db["ww_avg"])',
         'help':'Function to plot (lambda expression)',},
    ]
    plotdb = None
    def __init__(self, parent, inputs):
        self.actiondict = mergedicts(inputs, self.actiondefs)
        self.parent = parent
        print('Initialized '+self.actionname+' inside '+parent.name)
        # Don't forget to initialize plotdb in inherited classes!
        return

    def execute(self):
        print('Executing '+self.actionname)
        figsize  = self.actiondict['figsize']
        xaxis    = self.actiondict['xaxis']
        yaxis    = self.actiondict['yaxis']
        dpi      = self.actiondict['dpi']
        iplanes  = self.actiondict['iplane']
        savefile = self.actiondict['savefile']
        xlabel   = self.actiondict['xlabel']
        ylabel   = self.actiondict['ylabel']
        clevels  = eval(self.actiondict['clevels'])
        title    = self.actiondict['title']
        plotfunc = eval(self.actiondict['plotfunc'])
        if not isinstance(iplanes, list): iplanes = [iplanes,]
        for iplane in iplanes:
            fig, ax = plt.subplots(1,1,figsize=(figsize[0],figsize[1]), dpi=dpi)
            plotq = plotfunc(self.plotdb)
            c=plt.contourf(self.plotdb[xaxis][iplane,:,:], 
                           self.plotdb[yaxis][iplane,:,:], plotq[iplane, :, :], 
                           levels=clevels,cmap='coolwarm', extend='both')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.05)
            cbar=fig.colorbar(c, ax=ax, cax=cax)
            cbar.ax.tick_params(labelsize=7)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.axis('scaled')
            ax.set_title(eval("f'{}'".format(title)))

            if len(savefile)>0:
                savefname = savefile.format(iplane=iplane)
                plt.savefig(savefname)
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

