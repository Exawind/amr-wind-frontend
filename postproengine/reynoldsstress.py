# Get the location where this script is being run
import sys, os
scriptpath = os.path.dirname(os.path.realpath(__file__))
basepath   = os.path.dirname(scriptpath)
# Add any possible locations of amr-wind-frontend here
amrwindfedirs = ['../',
                 basepath]
for x in amrwindfedirs: sys.path.insert(1, x)

from postproengine import registerplugin, mergedicts, registeraction, contourplottemplate
from postproengine import compute_axis1axis2_coords
import postproamrwindsample_xarray as ppsamplexr
import postproamrwindsample as ppsample
import numpy as np
import pandas as pd
import itertools
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from postproengine import interpolatetemplate, circavgtemplate

"""
Plugin for creating Reynolds stress averages on sample planes

See README.md for details on the structure of classes here
"""

@registerplugin
class postpro_reynoldsstress():
    """
    Postprocess reynolds stresses 
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
  name: Example YZ plane
  ncfile: YZcoarse_103125.nc
  tavg: [27950,28450]
  group: T0_YZdomain

  radial_stress:
    yc: 1000
    zc: 150
    iplane: 
      - 5
    wake_center_files: 
      - ./wake_meandering/wake_center_5.csv
      
  contourplot:
    #plotfunc: 'lambda db: 0.5*(db["uu_avg"]+db["vv_avg"]+db["ww_avg"])'
    plotfunc: 'lambda db: (-db["ux_avg_uxur_avg"])'
    savefile: test_rs.png
    xaxis: y
    yaxis: z
    xlabel: 'Y [m]'
    ylabel: 'Z [m]'
    iplane: 5
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
            self.pklfile  = plane['savepklfile']
            meanpkl  = plane['meanpklfile']
            self.varnames = plane['varnames']

            # Get the averaging window
            if tavg==[]:
                for fileiter, file in enumerate(ncfile):
                    timevec = ppsample.getVar(ppsample.loadDataset(file), 'time')
                    if fileiter == 0:
                        tavg = [timevec[0].data,timevec[-1].data]
                    else:
                        if timevec[0].data < tavg[0]: tavg[0] = timevec[0].data
                        if timevec[-1].data > tavg[1]: tavg[1] = timevec[-1].data
                print("No time interval specified. Averaging over entire file: ",tavg)
                
            self.tavg = tavg
            # Load the plane
            if meanpkl == '':
                meandb = None
            else:
                # Load it from the pickle file
                pfile          = open(meanpkl, 'rb')
                meandb         = pickle.load(pfile)
                pfile.close()
            
            # Get the reynolds-stress averages
            self.dbReAvg  = ppsamplexr.ReynoldsStress_PlaneXR(ncfile, tavg,
                                                              avgdb = meandb,
                                                              varnames=self.varnames,
                                                              savepklfile=self.pklfile,
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
    class compute_radial_shear_stress():
        actionname = 'radial_stress'
        blurb      = 'Computes the radial Reynolds shear stress from the Cartesian stresses'
        required   = False
        actiondefs = [
                {'key':'iplane', 'required':False, 'default':None, 'help':'List of iplane values. Default is all planes in ncfile.'},
                {'key':'yc', 'required':False, 'default':0, 'help':'Specified lateral center of wake, yc'},
                {'key':'zc', 'required':False, 'default':0, 'help':'Specified vertical center of wake, zc'},
                {'key':'wake_center_files', 'required':False, 'default':None, 'help':'csv files containing time series of wake centers for each iplane. yc and zc will be compute based on mean centers over the specified time interval'},
                {'key':'yaxis', 'required':False, 'default':'y', 'help':'Axis for lateral flow direction (y,a1,a2)'},
                {'key':'zaxis', 'required':False, 'default':'z', 'help':'Axis for vertical flow direction (z,a1,a2)'},
        ]

        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent
            print('Initialized '+self.actionname+' inside '+parent.name)
            return

        def execute(self):
            yc = self.actiondict['yc']
            zc = self.actiondict['zc']
            yaxis = self.actiondict['yaxis']
            zaxis = self.actiondict['zaxis']
            iplanes = self.actiondict['iplane']
            wake_center_files = self.actiondict['wake_center_files']

            if iplanes == None:
                iplanes = list(range(0,self.parent.dbReAvg['x'].shape[0]))

            if not isinstance(iplanes, list): iplanes = [iplanes,]
            if not wake_center_files == None and not isinstance(wake_center_files, list): wake_center_files = [wake_center_files,]

            if wake_center_files != None and len(wake_center_files) != len(iplanes):
                print("Error: len(wake_center_files) != len(iplanes). Exiting.")
                sys.exit()

            corr_mapping = {
                'velocityx': 'u',
                'velocityy': 'v',
                'velocityz': 'w',
                'velocitya1': 'ua1',
                'velocitya2': 'ua2',
                'velocitya3': 'ua3'
            }

            combinations = itertools.combinations_with_replacement(self.parent.varnames, 2)

            corrlist = [
                [f"{corr_mapping[var1]}{corr_mapping[var2]}_avg", var1, var2]
                for var1, var2 in combinations
        ]

            self.parent.dbReAvg['uxur_avg']    = np.zeros_like(self.parent.dbReAvg[corrlist[0][0]])

            for iplaneiter, iplane in enumerate(iplanes):
                if wake_center_files != None:
                    df = pd.read_csv(wake_center_files[iplaneiter])
                    filtered_df = df[(df['t'] >= self.parent.tavg[0]) & (df['t'] <= self.parent.tavg[-1])]
                    y_center = filtered_df['yc'].mean()
                    z_center = filtered_df['zc'].mean()
                else:
                    y_center = yc
                    z_center = zc

            # Convert to native axis1/axis2 coordinates if necessary
            if ('a1' in [yaxis, zaxis]) or ('a2' in [yaxis, zaxis]):
                compute_axis1axis2_coords(self.parent.dbReAvg)
                uv_avg  = self.parent.dbReAvg[corr_mapping['velocity'+yaxis]+'ua3_avg'][iplane,:,:]
                uw_avg  = self.parent.dbReAvg[corr_mapping['velocity'+zaxis]+'ua3_avg'][iplane,:,:]
            else:
                uv_avg  = self.parent.dbReAvg['u'+corr_mapping['velocity'+yaxis]+'_avg'][iplane,:,:]
                uw_avg  = self.parent.dbReAvg['u'+corr_mapping['velocity'+zaxis]+'_avg'][iplane,:,:]

            YY = self.parent.dbReAvg[yaxis][iplane,:,:]
            ZZ = self.parent.dbReAvg[zaxis][iplane,:,:]

            Theta = np.arctan2(ZZ-z_center,YY-y_center)

            self.parent.dbReAvg['uxur_avg'][iplane,:,:] = uv_avg * np.sin(Theta) + uw_avg * np.cos(Theta)

            # Overwrite picklefile
            if len(self.parent.pklfile)>0:
                dbfile = open(self.parent.pklfile, 'wb')
                pickle.dump(self.parent.dbReAvg, dbfile, protocol=2)
                dbfile.close()

    @registeraction(actionlist)
    class compute_turbulent_fluxes():
        actionname = 'turbulent_fluxes'
        blurb      = 'Computes the turbulent fluxes'
        required   = False
        actiondefs = [
                {'key':'iplane', 'required':False, 'default':None, 'help':'List of iplane values. Default is all planes in ncfile.'},
                {'key':'include_radial', 'required':False, 'default':None, 'help':'Boolean to compute radial reynolds shear stress flux.'},
        ]

        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent
            print('Initialized '+self.actionname+' inside '+parent.name)
            return

        def execute(self):
            include_radial = self.actiondict['include_radial']
            iplanes = self.actiondict['iplane']

            if iplanes == None:
                iplanes = list(range(0,self.parent.dbReAvg['x'].shape[0]))

            corr_mapping = {
                'velocityx': 'u',
                'velocityy': 'v',
                'velocityz': 'w',
                'velocitya1': 'ua1',
                'velocitya2': 'ua2',
                'velocitya3': 'ua3'
            }

            combinations = itertools.combinations_with_replacement(self.parent.varnames, 2)

            corrlist = [
                f"{corr_mapping[var1]}{corr_mapping[var2]}_avg"
                for var1, var2 in combinations 
            ]

            turbulent_transport_list = [
                [f"{corr_mapping[var1]}_avg_{var2}", var1+"_avg", var2]
                for var1 in self.parent.varnames for var2 in corrlist
            ]

            if include_radial:
                self.parent.dbReAvg['ux_avg_uxur_avg'] = np.zeros_like(self.parent.dbReAvg[corrlist[0]])

            for term in turbulent_transport_list:
                self.parent.dbReAvg[term[0]] = np.zeros_like(self.parent.dbReAvg[corrlist[0]])

            for iplaneiter, iplane in enumerate(iplanes):
                for term in turbulent_transport_list:
                    u_avg  = self.parent.dbReAvg[term[1]][iplane,:,:]
                    uu_avg = self.parent.dbReAvg[term[2]][iplane,:,:]

                    self.parent.dbReAvg[term[0]][iplane,:,:] = u_avg * uu_avg

                if include_radial:
                    if 'velocitya3' in self.parent.varnames:
                        u_avg  = self.parent.dbReAvg['velocitya3_avg'][iplane,:,:]
                    else:
                        u_avg  = self.parent.dbReAvg['velocityx_avg'][iplane,:,:]
                    self.parent.dbReAvg['ux_avg_uxur_avg'][iplane,:,:] = u_avg * self.parent.dbReAvg['uxur_avg'][iplane,:,:]

            # Overwrite picklefile
            if len(self.parent.pklfile)>0:
                dbfile = open(self.parent.pklfile, 'wb')
                pickle.dump(self.parent.dbReAvg, dbfile, protocol=2)
                dbfile.close()

    @registeraction(actionlist)
    class contourplot(contourplottemplate):
        actionname = 'contourplot'
        def __init__(self, parent, inputs):
            super().__init__(parent, inputs)
            self.plotdb = self.parent.dbReAvg
            return

    @registeraction(actionlist)
    class interpolate(interpolatetemplate):
        """
        Add the default interpolation template
        """
        actionname = 'interpolate'
        def __init__(self, parent, inputs):
            super().__init__(parent, inputs)
            self.interpdb = self.parent.dbReAvg
            return

    @registeraction(actionlist)
    class circavg(circavgtemplate):
        """
        Add the default circumferential average template
        """
        actioname = 'circavg'
        def __init__(self, parent, inputs):
            super().__init__(parent, inputs)
            self.interpdb = self.parent.dbReAvg
            return
