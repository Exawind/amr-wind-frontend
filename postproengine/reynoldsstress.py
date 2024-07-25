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
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
Plugin for creating Reynolds stress averages on sample planes

See README.md for details on the structure of classes here
"""

@registerplugin
class postpro_reynoldsstress():
    """
    Postprocess averaged planes
    """
    # Name of task (this is same as the name in the yaml)
    name      = "reynoldsstress"
    # Description of task
    blurb     = "Reynolds-Stress average netcdf sample planes"
    inputdefs = [
        # -- Execute parameters ----
        {'key':'name',     'required':True,  'default':'',
         'help':'An arbitrary name',},
        {'key':'ncfile',   'required':True,  'default':'',
        'help':'NetCDF sampling file', },
        {'key':'tavg',    'required':False,  'default':[],
         'help':'Which times to average over', },
        {'key':'meanpklfile', 'required':False,  'default':'',
        'help':'Name of pickle file which contains mean results', }, 
        {'key':'savepklfile', 'required':False,  'default':'',
        'help':'Name of pickle file to save results', },
        {'key':'group',   'required':False,  'default':None,
         'help':'Which group to pull from netcdf file', },
        {'key':'varnames',  'required':False,  'default':['velocityx', 'velocityy', 'velocityz'],
         'help':'Variables to extract from the netcdf file',},                
    ]
    example = """
reynoldsstress:
  - name: test
    ncfile: /lustre/orion/cfd162/proj-shared/lcheung/AWAKEN/Neutral/5kmX5km_turbine1/post_processing/sampling_41000.nc
    tavg: [20886.5, 21486.5]
    contourplot:
      plotfunc: 'lambda db: 0.5*(db["uu_avg"]+db["vv_avg"]+db["ww_avg"])'
"""    
    actionlist = {}                    # Dictionary for holding sub-actions

    # --- Stuff required for main task ---
    def __init__(self, inputs, verbose=False):
        self.yamldictlist = []
        inputlist = inputs if isinstance(inputs, list) else [inputs]
        for indict in inputlist:
            self.yamldictlist.append(mergedicts(indict, self.inputdefs))
        if verbose: print('Initialized '+self.name)
        return

    def execute(self, verbose=False):
        if verbose: print('Running '+self.name)
        # Loop through and create plots
        for planeiter, plane in enumerate(self.yamldictlist):
            tavg     = plane['tavg']
            ncfile   = plane['ncfile']
            group    = plane['group']
            pklfile  = plane['savepklfile']
            meanpkl  = plane['meanpklfile']
            varnames = plane['varnames']

            # Convert ncfile to list of files if necessary
            if type(ncfile) is not list:
                filelist = [ncfile]
            else:
                filelist = ncfile
                
            # Get the averaging window
            if tavg==[]:
                for fileiter, file in enumerate(filelist):
                    timevec = ppsample.getVar(ppsample.loadDataset(file), 'time')
                    if fileiter == 0:
                        tavg = [timevec[0].data,timevec[-1].data]
                    else:
                        if timevec[0].data < tavg[0]: tavg[0] = timevec[0].data
                        if timevec[-1].data > tavg[1]: tavg[1] = timevec[-1].data
                print("No time interval specified. Averaging over entire file: ",tavg)
                
            # Load the plane
            if meanpkl == '':
                meandb = None
            else:
                # Load it from the pickle file
                pfile          = open(meanpkl, 'rb')
                meandb         = pickle.load(pfile)
                pfile.close()
            
            # Get the reynolds-stress averages
            self.dbReAvg  = ppsamplexr.ReynoldsStress_PlaneXR(filelist, tavg,
                                                              avgdb = meandb,
                                                              varnames=varnames,
                                                              savepklfile=pklfile,
                                                              groupname=group,
                                                              verbose=verbose, includeattr=True)
            
            # Do any sub-actions required for this task
            for a in self.actionlist:
                action = self.actionlist[a]
                # Check to make sure required actions are there
                if action.required and (action.actionname not in self.yamldictlist[planeiter].keys()):
                    # This is a problem, stop things
                    raise ValueError('Required action %s not present'%action.actionname)
                if action.actionname in self.yamldictlist[planeiter].keys():
                    actionitem = action(self, self.yamldictlist[planeiter][action.actionname])
                    actionitem.execute()
        return 

    @registeraction(actionlist)
    class contourplot(contourplottemplate):
        actionname = 'contourplot'
        def __init__(self, parent, inputs):
            super().__init__(parent, inputs)
            self.plotdb = self.parent.dbReAvg
            return
        
