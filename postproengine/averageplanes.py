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

"""
Plugin for creating instantaneous planar images

See README.md for details on the structure of classes here
"""

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
        {'key':'xaxis',    'required':False,  'default':'x',
        'help':'Which axis to use on the abscissa', },
        {'key':'yaxis',    'required':False,  'default':'y',
        'help':'Which axis to use on the ordinate', },
        {'key':'savepklfile', 'required':False,  'default':'',
        'help':'Name of pickle file to save results', },
        {'key':'group',   'required':False,  'default':None,
         'help':'Which group to pull from netcdf file', },
        {'key':'varnames',  'required':False,  'default':['velocityx', 'velocityy', 'velocityz'],
         'help':'Variables to extract from the netcdf file',},        
        
    ]
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
        for plane in self.yamldictlist:
            tavg     = plane['tavg']
            ncfile   = plane['ncfile']
            group    = plane['group']
            xaxis    = plane['xaxis']
            yaxis    = plane['yaxis']
            yaxis    = plane['yaxis']
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
            self.dbavg  = ppsamplexr.avgPlaneXR(filelist, tavg, varnames=varnames, groupname=group, savepklfile=pklfile, verbose=verbose)
            # Do any sub-actions required for this task
            for a in self.actionlist:
                action = self.actionlist[a]
                # Check to make sure required actions are there
                if action.required and (action.actionname not in self.yamldictlist[0].keys()):
                    # This is a problem, stop things
                    raise ValueError('Required action %s not present'%action.actionname)
                if action.actionname in self.yamldictlist[0].keys():
                    actionitem = action(self, self.yamldictlist[0][action.actionname])
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

