# Get the location where this script is being run
import sys, os
scriptpath = os.path.dirname(os.path.realpath(__file__))
basepath   = os.path.dirname(scriptpath)
# Add any possible locations of amr-wind-frontend here
amrwindfedirs = ['../',
                 basepath]
for x in amrwindfedirs: sys.path.insert(1, x)

import numpy as np
import copy
import pandas as pd

from postproengine import registerplugin, mergedicts, registeraction
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

try:
    import yt
    hasyt = True
except:
    hasyt = False

def loadplt(pltdir):
    ds = yt.load(pltdir, 
                 unit_system="mks",
                 units_override={"length_unit": (1.0, "m"), 
                                 "mass_unit": (1.0, "kg"), 
                                 "velocity_unit": (1.0, "m/s"), 
                                 "time_unit": (1.0, "s")})
    
    return ds

def get_coveringgrid_vars(ds, varlist, maxlevel=0):
    cg = ds.covering_grid(maxlevel,
                          ds.domain_left_edge,
                          ds.domain_dimensions)
    outdict = {}
    for v in varlist:
        outdict[v] = cg[v].to_ndarray()
    return outdict

def interpVar2Grid(olddat, x, y, z, v):
    """
    """
    interp = RegularGridInterpolator(
        (olddat['x'][:,0,0], olddat['y'][0,:,0], olddat['z'][0,0,:]),
        olddat[v],
        bounds_error=False,
        fill_value=None,
        )
    interpvar = interp((x, y, z))
    return interpvar

def makeNewGrid(origin, axis1, axis2, axis3, n):
    # Define empty arrays
    x     = np.zeros((n[0], n[1], n[2]))
    y     = np.zeros((n[0], n[1], n[2]))
    z     = np.zeros((n[0], n[1], n[2]))
    # Get the dx
    dx1   = axis1/(n[0]-1)
    dx2   = axis2/(n[1]-1)
    if n[2] > 1:
        dx3   = axis3/(n[2]-1)
    else:
        dx3   = np.zeros(3)
        
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                pt = origin + i*dx1 + j*dx2 + k*dx3
                x[i,j,k] = pt[0]
                y[i,j,k] = pt[1]
                z[i,j,k] = pt[2]
    return x,y,z

@registerplugin
class postpro_probeplt():
    """
    Postprocess averaged planes
    """
    # Name of task (this is same as the name in the yaml)
    name      = "probeplt"
    # Description of task
    blurb     = "Probe PLT output at arbitrary points"
    inputdefs = [
        # -- Execute parameters ----
        {'key':'name',         'required':False,  'default':'sample',       'help':'An arbitrary name',},
        {'key':'pltdirs',      'required':True,   'default':[],
         'help':'An arbitrary name',},
        {'key':'maxlevel',     'required':False,  'default':0,
         'help':'Maximum level for plt',},
        {'key':'varnames',  'required':False,     'default':['velocityx', 'velocityy', 'velocityz'],
         'help':'Variable names to extract',},
    ]
    actionlist = {}                    # Dictionary for holding sub-actions
    example = """
Example input for `probeplt`
```yaml
globalattributes:
  verbose: True
  executeorder:
  - probeplt

probeplt:
    name: STUFF
    maxlevel: 0
    pltdirs: 
    - /tscratch/lcheung/MMSEI/GABLS/GABLS1-AMR-Wind.ERFRestart3/plt18000
    - /tscratch/lcheung/MMSEI/GABLS/GABLS1-AMR-Wind.ERFRestart3/plt19000
    samplegrid:
        filesuffix: morestuff.dat
        origin: [200, 0, 0]
        axis1:  [0, 400, 0]
        axis2:  [0, 0, 400]
        axis3:  [0, 0, 0]
        Npoints: [129, 129, 1]
    samplexyz:
        filesuffix: comparestuff
        xyzfile: xyz.dat
```

Note that `filesuffix` will be appended to the plt names to create the
output filenames.  In the example above, the files that get output are
```
plt18000_comparestuff.csv
plt18000_morestuff.dat.csv
plt19000_comparestuff.csv
plt19000_morestuff.dat.csv
``` 

"""

    # --- Stuff required for main task ---
    def __init__(self, inputs, verbose=False):
        if not hasyt:
            raise ValueError('ERROR: yt not loaded and required for probeplt')
        
        self.yamldictlist = []
        inputlist = inputs if isinstance(inputs, list) else [inputs]
        for indict in inputlist:
            self.yamldictlist.append(mergedicts(indict, self.inputdefs))
        if verbose: print('Initialized '+self.name)
        return

    def execute(self, verbose=False):
        if verbose: print('Running '+self.name)

        outputdict = {}
        
        # Loop through each plt list
        for isect, sect in enumerate(self.yamldictlist):
            name          = sect['name']
            pltdirs       = sect['pltdirs']
            varlist       = sect['varnames']
            maxlevel      = sect['maxlevel']
            self.varlist  = varlist
            
            # Check to make sure it has all required actions
            for a in self.actionlist:
                action = self.actionlist[a]
                # Check to make sure required actions are there
                if action.required and (action.actionname not in self.yamldictlist[isect].keys()):
                    # This is a problem, stop things
                    raise ValueError('Required action %s not present'%action.actionname)
            pltoutputs = []

            # Load each plt directory and sample it
            for iplt, pltdir in enumerate(pltdirs):
                if verbose: print(pltdir)

                # Load the plt directory
                ds = loadplt(pltdir)
                self.ds = ds
                self.pltdir = pltdir

                fullvars = copy.deepcopy(varlist)
                if 'x' not in fullvars: fullvars.append('x')
                if 'y' not in fullvars: fullvars.append('y')
                if 'z' not in fullvars: fullvars.append('z')
                dim     = ds.domain_dimensions
                self.cgrid   = get_coveringgrid_vars(ds, fullvars, maxlevel=maxlevel)
                actionoutputs = {}

                # Go through actionlist and process
                runactions = [item for item in self.yamldictlist[isect].keys() if item in self.actionlist.keys()]
                for runaction in runactions:
                    action = self.actionlist[runaction]
                    actionitem = action(self, self.yamldictlist[isect][action.actionname])
                    actionoutputs[action.actionname] = actionitem.execute(verbose=verbose)
                pltoutputs.append(actionoutputs)

            outputdict[name] = pltoutputs

        return outputdict

    # --- Inner classes for action list ---
    @registeraction(actionlist)
    class samplegrid():
        actionname = 'samplegrid'
        blurb      = 'Sample a regular Cartesian grid'
        required   = False
        actiondefs = [
            {'key':'filesuffix',  'required':True,  'default':'',       'help':'The output filename suffix',},
            {'key':'origin',      'required':True,  'help':'Origin point for sampling',  'default':None},
            {'key':'axis1',       'required':True,  'help':'First axis of sampling volume',  'default':None},
            {'key':'axis2',       'required':True,  'help':'Second axis of sampling volume',  'default':None},
            {'key':'axis3',       'required':True,  'help':'Third axis of sampling volume',  'default':None},
            {'key':'Npoints',     'required':True,  'help':'Number of points along each axis [N1, N2, N3]',  'default':None},
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent
            print('Initialized '+self.actionname+' inside '+parent.name)
            return

        def execute(self, verbose=False):
            if verbose: print('Executing '+self.actionname)

            # Load the inputs
            filesuffix = self.actiondict['filesuffix']
            origin  = self.actiondict['origin']
            axis1   = np.array(self.actiondict['axis1'])
            axis2   = np.array(self.actiondict['axis2'])
            axis3   = np.array(self.actiondict['axis3'])
            Npoints = self.actiondict['Npoints']

            newdat = {}
            newdat['x'], newdat['y'], newdat['z'] = makeNewGrid(origin, axis1, axis2, axis3, Npoints)

            for v in self.parent.varlist:
                newdat[v] = interpVar2Grid(self.parent.cgrid,
                                           newdat['x'],
                                           newdat['y'],
                                           newdat['z'],
                                           v)

            # Flatten the array
            flatdat = {}
            for k, g in newdat.items():
                flatdat[k] = g.flatten()
                
            # Write this back out
            filename = Path(self.parent.pltdir).name
            filename = filename + '_' + filesuffix + '.csv'
            savedf = pd.DataFrame(flatdat)
            if verbose:
                print(f'Saving {filename}')
            savedf.to_csv(filename,index=False,sep=',')
            return newdat

    @registeraction(actionlist)
    class samplexyz():
        actionname = 'samplexyz'
        blurb      = 'Sample a set of (x,y,z) given by a file'
        required   = False
        actiondefs = [
            {'key':'filesuffix',  'required':True,  'default':'',       'help':'The output filename suffix',},
            {'key':'xyzfile',     'required':True,  'default':'',       'help':'The filename containing a list of (x,y,z) points ',},
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent
            print('Initialized '+self.actionname+' inside '+parent.name)
            return

        def execute(self, verbose=False):
            if verbose: print('Executing '+self.actionname)

            filesuffix = self.actiondict['filesuffix']
            xyzfile    = self.actiondict['xyzfile']

            xyzdat     = np.loadtxt(xyzfile)
            newdat = {}
            newdat['x'] = xyzdat[:,0]
            newdat['y'] = xyzdat[:,1]
            newdat['z'] = xyzdat[:,2]

            for v in self.parent.varlist:
                newdat[v] = interpVar2Grid(self.parent.cgrid,
                                           newdat['x'],
                                           newdat['y'],
                                           newdat['z'],
                                           v)

            # Write this back out
            filename = Path(self.parent.pltdir).name
            filename = filename + '_' + filesuffix + '.csv'
            savedf = pd.DataFrame(newdat)
            if verbose:
                print(f'Saving {filename}')
            savedf.to_csv(filename,index=False,sep=',')
            return newdat
