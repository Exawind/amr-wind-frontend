# Get the location where this script is being run
import sys, os
scriptpath = os.path.dirname(os.path.realpath(__file__))
basepath   = os.path.dirname(scriptpath)
# Add any possible locations of amr-wind-frontend here
amrwindfedirs = ['../',
                 basepath]
for x in amrwindfedirs: sys.path.insert(1, x)

from postproengine import registerplugin, mergedicts, registeraction, contourplottemplate
import postproamrwindsample_xarray as ppsamplexr
import postproamrwindsample as ppsample
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


"""
Plugin for post processing averaged planes

See README.md for details on the structure of classes here
"""

@registerplugin
class postpro_linesampler():
    """
    Make plots of instantaneous planes
    """
    # Name of task (this is same as the name in the yaml
    name      = "linesampler"
    # Description of task
    blurb     = "Process line sample files"
    inputdefs = [
        # --- Required parameters ----
        {'key':'name',     'required':True,  'default':'',
         'help':'An arbitrary name',},
        {'key':'ncfile', 'required':True,  'default':'',
         'help':'A list of netcdf files',},
        # --- optional parameters ----
        {'key':'group', 'required':False,  'default':None,
         'help':'Which group to use in the netcdf file',},
        {'key':'varnames',  'required':False,  'default':['velocityx', 'velocityy', 'velocityz'],
         'help':'Variables to extract from the netcdf file',},
    ]
    actionlist = {}                    # Dictionary for holding sub-actions
    example = """
```yaml
linesampler:
- name: metmast_1k
  ncfile: 
  - /gpfs/lcheung/HFM/exawind-benchmarks/convective_abl/post_processing/metmast_30000.nc
  group: virtualmast
  varnames: ['velocityx', 'velocityy', 'velocityz', 'temperature']
  average:
    tavg: [15000, 16000]
    savefile: ../results/avgmast_1000.csv
```
"""
    # --- Stuff required for main task ---
    def __init__(self, inputs, verbose=False):
        self.yamldictlist = []
        inputlist = inputs if isinstance(inputs, list) else [inputs]
        for indict in inputlist:
            self.yamldictlist.append(mergedicts(indict, self.inputdefs))
        if verbose: print('Initialized '+self.name)
        return
    
    def execute(self, verbose=False):
        self.verbose=verbose
        if verbose: print('Running '+self.name)

        for iline, line in enumerate(self.yamldictlist):
            # Run any necessary stuff for this task
            self.ncfile   = line['ncfile']
            self.varnames = line['varnames']
            self.group    = line['group']
            
            # Do any sub-actions required for this task
            for a in self.actionlist:
                action = self.actionlist[a]
                # Check to make sure required actions are there
                if action.required and (action.actionname not in self.yamldictlist[plotitemiter].keys()):
                    # This is a problem, stop things
                    raise ValueError('Required action %s not present'%action.actionname)
                if action.actionname in self.yamldictlist[iline].keys():
                    actionitem = action(self, self.yamldictlist[iline][action.actionname])
                    actionitem.execute()
        return

    @registeraction(actionlist)
    class average():
        actionname = 'average'
        blurb      = 'Time average the line'
        required   = False
        actiondefs = [
            {'key':'savefile',  'required':True,  'default':'',
             'help':'Filename to save the radial profiles', },
            {'key':'tavg',       'required':False,  'default':[],  'help':'Times to average over', },
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent
            print('Initialized '+self.actionname+' inside '+parent.name)
            return

        def execute(self):
            print('Executing '+self.actionname)
            tavg     = self.actiondict['tavg']
            savefile = self.actiondict['savefile']
            ncfile   = self.parent.ncfile
            varnames = self.parent.varnames
            group    = self.parent.group
            
            ds   = ppsamplexr.avgLineXR(ncfile, tavg, varnames,
                                        groupname=group,
                                        verbose=self.parent.verbose,
                                        includeattr=False, gettimes=False)
            
            # Save data to csv file
            ds.pop('group')     # Remove the group from being written
            dfcsv = pd.DataFrame()
            for k, g in ds.items():
                dfcsv[k] = g
            dfcsv.to_csv(savefile,index=False,sep=',')

            return
