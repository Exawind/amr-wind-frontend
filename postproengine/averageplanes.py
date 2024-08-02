# Get the location where this script is being run
import sys, os
scriptpath = os.path.dirname(os.path.realpath(__file__))
basepath   = os.path.dirname(scriptpath)
# Add any possible locations of amr-wind-frontend here
amrwindfedirs = ['../',
                 basepath]
for x in amrwindfedirs: sys.path.insert(1, x)

from postproengine import registerplugin, mergedicts, registeraction
import postproamrwindsample_xarray as ppsamplexr
import postproamrwindsample as ppsample
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from postproengine import interpolatetemplate, circavgtemplate

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
        
    ]
    actionlist = {}                    # Dictionary for holding sub-actions
    example = """
avgplanes:
  - name: avg_smallXYplane
    ncfile:
    - /lustre/orion/cfd162/world-shared/lcheung/AdvancedControlsWakes/Runs/LowWS_LowTI.Frontier/oneturb_7x2/rundir_baseline/post_processing/XY_35000.nc
    - /lustre/orion/cfd162/world-shared/lcheung/AdvancedControlsWakes/Runs/LowWS_LowTI.Frontier/oneturb_7x2/rundir_baseline/post_processing/XY_50000.nc
    - /lustre/orion/cfd162/world-shared/lcheung/AdvancedControlsWakes/Runs/LowWS_LowTI.Frontier/oneturb_7x2/rundir_baseline/post_processing/XY_65000.nc    
    - /lustre/orion/cfd162/world-shared/lcheung/AdvancedControlsWakes/Runs/LowWS_LowTI.Frontier/oneturb_7x2/rundir_baseline/post_processing/XY_77500.nc
    tavg: [17800, 18500]
    plot:
      plotfunc: 'lambda u, v, w: np.sqrt(u**2 + v**2)'
      title: 'AVG horizontal velocity'
      xaxis: x           # Which axis to use on the abscissa 
      yaxis: y           # Which axis to use on the ordinate 
      iplane: 1    
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
            varnames = plane['varnames']
            #Get all times if not specified 
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
                self.dbavg  = ppsamplexr.avgPlaneXR(filelist, tavg, varnames=varnames, groupname=group,
                                                    includeattr=True, savepklfile=pklfile, verbose=verbose)
            
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
            {'key':'iplane',   'required':True,  'help':'List of iplane values',  'default':[0,]},
            {'key':'Diam', 'required':True,  'help':'Turbine Diameter',  'default':0},
            {'key':'zc',       'required':False,  'help':'Center of rotor disk in z',  'default':None},
            {'key':'yc',       'required':False,  'help':'Center of rotor disk in y',  'default':None},
            {'key':'savefile',  'required':False,  'default':None,'help':'csv filename to save results'},
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent
            print('Initialized '+self.actionname+' inside '+parent.name)
            return

        def execute(self):
            print('Executing '+self.actionname)
            yc = self.actiondict['yc']
            zc = self.actiondict['zc']
            Radius =  self.actiondict['Diam']/2.0
            iplanes = self.actiondict['iplane']
            savefile = self.actiondict['savefile']
            rotor_avg = {}
            xloc = {}
            if not isinstance(iplanes, list): iplanes = [iplanes,]
            for iplane in iplanes:
                iplane = int(iplane)
                x = self.parent.dbavg['x'][iplane,0,0]
                y = self.parent.dbavg['y'][iplane,:,:]
                z = self.parent.dbavg['z'][iplane,:,:]
                vel_avg = self.parent.dbavg['velocityx'+'_avg'][iplane,:,:]
                if yc == None: yc = (y[-1]+y[0])/2.0
                if zc == None: zc = (z[-1]+z[0])/2.0
                Routside = ((y-yc)**2 + (z-zc)**2) > Radius**2
                masked_vel = np.ma.array(vel_avg,mask=Routside)
                rotor_avg[iplane] = masked_vel.mean()
                xloc[iplane] = x
                print("Rotor Average Velocity at x = ",x,": ",rotor_avg[iplane])

            # Write the data to the CSV file
            if not savefile==None:
                # Write the data to the CSV file
                with open(savefile, mode='w') as file:
                    file.write("x_location, rotor_avg_vel\n")  # Write header row
                    for iplane in iplanes:
                        file.write("{:.15f}, {:.15f}\n".format(xloc[iplane], rotor_avg[iplane]))
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
    class plot():
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
        {'key':'clevels',   'required':False,  'default':'np.linspace(0, 12, 121)',
         'help':'Color levels (eval expression)',},
        {'key':'xlabel',    'required':False,  'default':'X [m]',
         'help':'Label on the X-axis', },
        {'key':'ylabel',    'required':False,  'default':'Y [m]',
         'help':'Label on the Y-axis', },
        {'key':'title',     'required':False,  'default':'',
         'help':'Title of the plot',},
        {'key':'plotfunc',  'required':False,  'default':'lambda u, v, w: u',
         'help':'Function to plot (lambda expression)',},
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent
            print('Initialized '+self.actionname+' inside '+parent.name)
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
                plotq = plotfunc(self.parent.dbavg['velocityx_avg'], self.parent.dbavg['velocityy_avg'], self.parent.dbavg['velocityz_avg'])
                c=plt.contourf(self.parent.dbavg[xaxis][iplane,:,:], 
                            self.parent.dbavg[yaxis][iplane,:,:], plotq[iplane, :, :], 
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
        
