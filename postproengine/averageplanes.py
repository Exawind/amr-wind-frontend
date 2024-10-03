# Get the location where this script is being run
import sys, os
scriptpath = os.path.dirname(os.path.realpath(__file__))
basepath   = os.path.dirname(scriptpath)
# Add any possible locations of amr-wind-frontend here
amrwindfedirs = ['../',
                 basepath]
for x in amrwindfedirs: sys.path.insert(1, x)

from postproengine import registerplugin, mergedicts, registeraction, contourplottemplate
from postproengine import compute_axis1axis2_coords, get_mapping_xyz_to_axis1axis2
import postproamrwindsample_xarray as ppsamplexr
import postproamrwindsample as ppsample
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from postproengine import interpolatetemplate, circavgtemplate
from postproengine import compute_axis1axis2_coords
import re

"""
Plugin for post processing averaged planes

See README.md for details on the structure of classes here
"""

def loadpickle(picklefile):
    pfile          = open(picklefile, 'rb')
    ds             = pickle.load(pfile)
    pfile.close()
    return ds

@registerplugin
class postpro_averageplanes():
    """
    Postprocess averaged planes
    """
    # Name of task (this is same as the name in the yaml)
    name      = "avgplanes"
    # Description of task
    blurb     = "Average netcdf sample planes"
    inputdefs = [
        # -- Execute parameters ----
        {'key':'name',     'required':True,  'default':'',
         'help':'An arbitrary name',},
        {'key':'ncfile',   'required':True,  'default':'',
        'help':'NetCDF sampling file', },
        {'key':'tavg',    'required':False,  'default':[],
            'help':'Which times to average over', },
        {'key':'loadpklfile', 'required':False,  'default':'',
        'help':'Load previously computed results from this pickle file', },        
        {'key':'savepklfile', 'required':False,  'default':'',
        'help':'Name of pickle file to save results', },
        {'key':'group',   'required':False,  'default':None,
         'help':'Which group to pull from netcdf file', },
        {'key':'varnames',  'required':False,  'default':['velocityx', 'velocityy', 'velocityz'],
         'help':'Variables to extract from the netcdf file',},        
        {'key':'axis_rotation',  'required':False,  'default':0,
         'help':'Degrees to rotate axis for velocitya1,a2,a3 transformation',},                
        
    ]
    actionlist = {}                    # Dictionary for holding sub-actions
    example = """
avgplanes:
  name: Wake YZ plane
  ncfile:
  - /lustre/orion/cfd162/world-shared/lcheung/ALCC_Frontier_WindFarm/farmruns/LowWS_LowTI/ABL_ALM_10x10/rundir_baseline/post_processing/rotor_*.nc
  tavg: [25400,26000]
  group: T08_rotor
  varnames: ['velocitya1','velocitya2','velocitya3']
  verbose: True

  contourplot:
    iplane: 6
    xaxis: 'a1'
    yaxis: 'a2'
    xlabel: 'Lateral axis [m]'
    ylabel: 'Vertical axis [m]'
    clevels: "121"
    plotfunc: "lambda db: db['velocitya3_avg']"
    savefile: 'avg_plane.png'

  rotorAvgVel:
    iplane: [0,1,2,3,4,5,6,7,8,9,10]
    Diam: 240
    yc: 150
    xaxis: 'a1'
    yaxis: 'a2'
    avgfunc: "lambda db: db['velocitya3_avg']"
    savefile: test.csv
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
        if verbose: print('Running '+self.name)
        # Loop through and create plots
        for iplane, plane in enumerate(self.yamldictlist):
            tavg     = plane['tavg']
            ncfile   = plane['ncfile']
            group    = plane['group']
            loadpkl  = plane['loadpklfile']            
            pklfile  = plane['savepklfile']
            self.varnames = plane['varnames']
            self.axis_rotation = plane['axis_rotation']

            #Get all times if not specified
            if isinstance(ncfile, str):
                filelist = [ncfile]
            else:
                filelist = []
                for fileiter in range(0,len(ncfile)):
                    filelist.append(ncfile[fileiter])
                    
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
            if len(loadpkl)>0:
                # Load from existing file
                pfile          = open(loadpkl, 'rb')
                self.dbavg     = pickle.load(pfile)
                pfile.close()                
            else:
                # Compute the result
                self.dbavg  = ppsamplexr.avgPlaneXR(ncfile, tavg, varnames=self.varnames, groupname=group,includeattr=True, savepklfile=pklfile, verbose=verbose,axis_rotation=self.axis_rotation)

            
            # Do any sub-actions required for this task
            for a in self.actionlist:
                action = self.actionlist[a]
                # Check to make sure required actions are there
                if action.required and (action.actionname not in self.yamldictlist[0].keys()):
                    # This is a problem, stop things
                    raise ValueError('Required action %s not present'%action.actionname)
                if action.actionname in self.yamldictlist[iplane].keys():
                    actionitem = action(self, self.yamldictlist[iplane][action.actionname])
                    actionitem.execute()
        return 

    # --- Inner classes for action list ---
    @registeraction(actionlist)
    class compute_rotor_averaged_velocity():
        actionname = 'rotorAvgVel'
        blurb      = 'Computes the rotor averaged velocity'
        required   = False
        actiondefs = [
            {'key':'iplane',   'required':False,  'help':'List of iplane values',  'default':None},
            {'key':'Diam', 'required':True,  'help':'Turbine Diameter',  'default':0},
            {'key':'xc',       'required':False,  'help':'Center of rotor disk on the xaxis',  'default':None},
            {'key':'yc',       'required':False,  'help':'Center of rotor disk in the yaxis',  'default':None},
            {'key':'xaxis',    'required':False,  'default':'y',
            'help':'Which axis to use on the abscissa', },
            {'key':'yaxis',    'required':False,  'default':'z',
            'help':'Which axis to use on the ordinate', },
            {'key':'savefile',  'required':False,  'default':None,'help':'csv filename to save results'},
            {'key':'avgfunc',  'required':False,  'default':'lambda db: db["velocityx_avg"]',
            'help':'Function to average (lambda expression)',},
            {'key':'wake_meandering_stats_file','required':False,  'default':None,
         'help':'The lateral and vertical wake center will be read from yc_mean and zc_mean columns of this file, overriding yc and zc.', },
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent
            print('Initialized '+self.actionname+' inside '+parent.name)
            return

        def extract_1d_from_meshgrid(self,Z):
            # Check along axis 0
            unique_rows = np.unique(Z, axis=0)
            if unique_rows.shape[0] == 1:
                return unique_rows[0]

            # Check along axis 1
            unique_cols = np.unique(Z, axis=1)
            if unique_cols.shape[1] == 1:
                return unique_cols[:, 0]


        def execute(self):
            print('Executing '+self.actionname)
            xc = self.actiondict['xc']
            yc = self.actiondict['yc']
            wake_meandering_stats_file = self.actiondict['wake_meandering_stats_file']
            Radius =  self.actiondict['Diam']/2.0
            iplanes = self.actiondict['iplane']
            savefile = self.actiondict['savefile']
            xaxis    = self.actiondict['xaxis']
            yaxis    = self.actiondict['yaxis']
            avgfunc = eval(self.actiondict['avgfunc'])
            rotor_avg = {}
            planeloc = {}
            if iplanes == None:
                iplanes = list(range(len(self.parent.dbavg['offsets'])))
            if not isinstance(iplanes, list): iplanes = [iplanes,]

            if not wake_meandering_stats_file == None and not isinstance(wake_meandering_stats_file, list): wake_meandering_stats_file = [wake_meandering_stats_file,]
            if wake_meandering_stats_file != None and len(wake_meandering_stats_file) != len(iplanes):
                print("Error: len(wake_meandering_stats_file) != len(iplanes). Exiting.")
                sys.exit()


            # Convert to native axis1/axis2 coordinates if necessary
            natural_axes = False
            if ('a1' in [xaxis, yaxis]) or ('a2' in [xaxis, yaxis]) or ('a3' in [xaxis, yaxis]):
                natural_axes = True
                compute_axis1axis2_coords(self.parent.dbavg,rot=0)
                R = get_mapping_xyz_to_axis1axis2(self.parent.dbavg['axis1'],self.parent.dbavg['axis2'],self.parent.dbavg['axis3'],rot=0)
                origin = self.parent.dbavg['origin']
                origina1a2a3 = R@self.parent.dbavg['origin']
                offsets = self.parent.dbavg['offsets']
                offsets = [offsets] if (not isinstance(offsets, list)) and (not isinstance(offsets,np.ndarray)) else offsets

            for iplaneiter, iplane in enumerate(iplanes):
                iplane = int(iplane)
                if not natural_axes:
                    plane_loc = self.parent.dbavg['x'][iplane,0,0]
                else:
                    plane_loc = origina1a2a3[-1] + offsets[iplane]

                x = self.parent.dbavg[xaxis][iplane,:,:]
                y = self.parent.dbavg[yaxis][iplane,:,:]

                vel_avg = avgfunc(self.parent.dbavg)[iplane,:,:]

                if wake_meandering_stats_file != None:
                    wake_meandering_stats = pd.read_csv(wake_meandering_stats_file[iplaneiter])
                    xc = wake_meandering_stats[xaxis+'c_mean'][0]
                    yc = wake_meandering_stats[yaxis+'c_mean'][0]
                else:
                    if self.actiondict['xc'] == None: 
                        grid = self.extract_1d_from_meshgrid(x)
                        xc = (grid[-1]+grid[0])/2.0
                    if self.actiondict['yc'] == None: 
                        grid = self.extract_1d_from_meshgrid(y)
                        yc = (grid[-1]+grid[0])/2.0

                Routside = ((x-xc)**2 + (y-yc)**2) > Radius**2
                masked_vel = np.ma.array(vel_avg,mask=Routside)
                rotor_avg[iplane] = masked_vel.mean()
                planeloc[iplane] = plane_loc
                print("Rotor Average Velocity at x = ",plane_loc,": ",rotor_avg[iplane])

            # Write the data to the CSV file
            if not savefile==None:
                # Write the data to the CSV file
                with open(savefile, mode='w') as file:
                    if not natural_axes:
                        file.write("x_location, rotor_avg_vel\n")  # Write header row
                    else:
                        file.write("a3_location, rotor_avg_vel\n")  # Write header row
                    for iplane in iplanes:
                        file.write("{:.15f}, {:.15f}\n".format(planeloc[iplane], rotor_avg[iplane]))
            return 

    @registeraction(actionlist)
    class compute_wake_thickness():
        actionname = 'wakeThickness'
        blurb      = 'Computes the wake displacement and momentum thickness'
        required   = False
        actiondefs = [
                {'key':'iplane'             , 'required':True  , 'default': [0] , 'help':'List of iplane values'},
                {'key':'noturbine_pkl_file' , 'required':False , 'default':None , 'help':'pickle file containing rotor planes for the case with no turbine'},
                {'key':'U_inf'              , 'required':False , 'default':None , 'help':'constant value for U_inf for cases with uniform inflow'},
                {'key':'savefile'           , 'required':False , 'default':None , 'help':'csv filename to save results'},
        ]

        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent
            print('Initialized '+self.actionname+' inside '+parent.name)
            return

        def calcDelta(self, x, y, U, Uinf):
            xvec = x[0,:]
            yvec = y[:,0]
            deltaInt = 1.0 - U/Uinf
            delta = np.trapz(np.trapz(deltaInt, yvec, axis=0), xvec, axis=0)
            return delta

        def calcTheta(self, x, y, U, Uinf):
            xvec = x[0,:]
            yvec = y[:,0]
            thetaInt = U/Uinf*(1.0 - U/Uinf)
            theta = np.trapz(np.trapz(thetaInt, yvec, axis=0), xvec, axis=0)
            return theta

        def execute(self):
            print(f'Executing {self.actionname}')
            iplanes = self.actiondict['iplane']
            savefile = self.actiondict['savefile']
            rotor_avg = {}
            xloc = {}
            delta = {}
            theta = {}
            if(self.actiondict['noturbine_pkl_file'] is not None):
                RP_noturb = loadpickle(self.actiondict['noturbine_pkl_file'])
            if not isinstance(iplanes, list): iplanes = [iplanes,]
            for iplane in iplanes:
                iplane = int(iplane)
                x = self.parent.dbavg['x'][iplane,0,0]
                y = self.parent.dbavg['y'][iplane,:,:]
                z = self.parent.dbavg['z'][iplane,:,:]
                Uh_turbine = np.sqrt(self.parent.dbavg['velocityx_avg'][iplane,:,:]**2 + self.parent.dbavg['velocityy_avg'][iplane,:,:]**2)
                if(self.actiondict['noturbine_pkl_file'] is not None):
                    Uh_noturbine = np.sqrt(RP_noturb['velocityx_avg'][iplane,:,:]**2+RP_noturb['velocityy_avg'][iplane,:,:]**2)
                elif self.actiondict['U_inf'] is not None:
                    Uh_noturbine = self.actiondict['U_inf']
                delta[iplane] = self.calcDelta(y, z, Uh_turbine, Uh_noturbine)
                theta[iplane] = self.calcTheta(y, z, Uh_turbine, Uh_noturbine)
                xloc[iplane] = x
                print(f"Wake thickness at x = {x}: displacement thickness = {delta[iplane]}, momentum thickness = {theta[iplane]}")

            # Write the data to the CSV file
            if not savefile == None:
                with open(savefile, mode='w') as file:
                    file.write("x_location, displacement_thickness, momentum_thickness\n")  # Write header row
                    for iplane in iplanes:
                        file.write("{:.15f}, {:.15f}, {:.15f}\n".format(xloc[iplane], delta[iplane], theta[iplane]))
            return

    @registeraction(actionlist)
    class contourplot(contourplottemplate):
        actionname = 'contourplot'
        def __init__(self, parent, inputs):
            super().__init__(parent, inputs)
            self.plotdb = self.parent.dbavg
            return

    @registeraction(actionlist)
    class interpolate(interpolatetemplate):
        """
        Add the default interpolation template
        """
        actionname = 'interpolate'
        def __init__(self, parent, inputs):
            super().__init__(parent, inputs)
            self.interpdb = self.parent.dbavg
            return

    @registeraction(actionlist)
    class circavg(circavgtemplate):
        """
        Add the default circumferential average template
        """
        actioname = 'circavg'
        def __init__(self, parent, inputs):
            super().__init__(parent, inputs)
            self.interpdb = self.parent.dbavg
            return
        
