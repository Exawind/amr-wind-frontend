import sys
import os
import traceback
import copy
import io
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import OrderedDict
import numpy as np

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
pluginlist = OrderedDict()
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
        f.write('\n```\n')
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

# =====================================================
# --- ADD REUSABLE CLASSES AND METHODS HERE ---
# =====================================================

def compute_axis1axis2_coords(db):
    """
    Computes the native axis1 and axis2 coordinate system for a given
    set of sample planes.

    Note: this assumes db has the origin, axis1/2, and offset definitions
    """

    # Check to make sure db has everything needed
    if ('origin' not in db) or \
       ('axis1' not in db) or \
       ('axis2' not in db) or \
       ('axis3' not in db) or \
       ('offsets') not in db:
        print('Need to ensure that the sample plane data includes origin, axis1, axis2, axis3, and offset information')
        return

    # Pull out the coordate definitions
    axis1  = np.array(db['axis1'])
    axis2  = np.array(db['axis2'])
    axis3  = np.array(db['axis3'])
    origin = np.array(db['origin'])
    offsets= db['offsets']
    if not isinstance(offsets,list):
        offsets = [offsets]
    
    # Compute the normals
    n1 = axis1/np.linalg.norm(axis1)
    n2 = axis2/np.linalg.norm(axis2)
    if np.linalg.norm(axis3) > 0.0:
        n3 = axis3/np.linalg.norm(axis3)
    else:
        n3 = axis3
        
    db['a1'] = np.full_like(db['x'], 0.0)
    db['a2'] = np.full_like(db['x'], 0.0)
    db['a3'] = np.full_like(db['x'], 0.0)    
    ijk_idx = db['x'].shape

    for k in range(ijk_idx[0]):
        # Loop through each plane
        for j in range(ijk_idx[1]):
            for i in range(ijk_idx[2]):
                x = db['x'][k, j, i]
                y = db['y'][k, j, i]
                z = db['z'][k, j, i]
                plane_origin = origin + axis3*offsets[k]
                pt = np.array([x, y, z])
                a1coord  = np.dot(pt-plane_origin, n1)
                a2coord  = np.dot(pt-plane_origin, n2)
                a3coord  = np.dot(pt-plane_origin, n3)
                db['a1'][k,j,i] = a1coord
                db['a2'][k,j,i] = a2coord
                db['a3'][k,j,i] = a3coord                
    return

# ------- reusable contour plot class -----------------
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

        # Convert to native axis1/axis2 coordinates if necessary
        if ('a1' in [xaxis, yaxis]) or \
           ('a2' in [xaxis, yaxis]) or \
           ('a3' in [xaxis, yaxis]):
            compute_axis1axis2_coords(self.plotdb)
        
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

# =====================================================

    
def runtaskdict(taskdict, plist, looptasks, verbose):
    for task in looptasks:
        if task in taskdict:
            taskitem = plist[task](taskdict[task], verbose=verbose)
            taskitem.execute(verbose=verbose)
    return
    
def driver(yamldict, plist=pluginlist, verbose=None):
    """
    Run through and execute all tasks
    """
    looptasks = plist.keys()

    # Get the global attributes
    globattr = yamldict['globalattributes'] if 'globalattributes' in yamldict else {}
    
    # Set the verbosity
    # Take verbosity from globalattributes
    verbose_attr = globattr['verbose'] if 'verbose' in globattr else False
    # Override with verbose if necessary
    verbosity = verbose if verbose is not None else verbose_attr

    # Load any user defined modules
    if 'udfmodules' in globattr:
        udfmodules = globattr['udfmodules']
        for module in udfmodules:
            name = os.path.splitext(os.path.basename(module))[0]
            mod = load_module(module)
            sys.modules[name] = mod

            
    # Check if executeorder is present
    if 'executeorder' in globattr:
        exeorder = globattr['executeorder']
        for item in exeorder:
            if isinstance(item, str):
                if item in plist.keys():
                    # item is the exact name of an executor, run it:
                    taskitem = plist[item](yamldict[item], verbose=verbosity)
                    taskitem.execute(verbose=verbosity)
                else:
                    # item is an entire workflow, run that
                    runtaskdict(yamldict[item], plist, looptasks, verbosity)
            else:
                # item is a workflow with a specific set of tasks
                wflowname  = next(iter(item))
                wflowtasks = item[wflowname]
                runtaskdict(yamldict[wflowname], plist, wflowtasks, verbosity)
    else:
        # Run everything in yamldict
        runtaskdict(yamldict, plist, looptasks, verbosity)  
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

