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
    name      = "averagedplanes"
    # Description of task
    blurb     = "Average netcdf sample planes"
    inputdefs = [
        # -- Execute parameters ----
        {'key':'name',     'required':True,  'default':'',
         'help':'An arbitrary name',},
        {'key':'ncfile',   'required':True,  'default':'',
        'help':'NetCDF sampling file', },
        {'key':'tavg',    'required':True,  'default':[],
            'help':'Which times to average over', }, 
        {'key':'xaxis',    'required':False,  'default':'x',
        'help':'Which axis to use on the abscissa', },
        {'key':'yaxis',    'required':False,  'default':'y',
        'help':'Which axis to use on the ordinate', },
        {'key':'savepklfile', 'required':False,  'default':'',
        'help':'Name of pickle file to save results', },
        {'key':'group',   'required':False,  'default':None,
         'help':'Which group to pull from netcdf file', },
        
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
            pklfile  = plane['pklfile']
            
            #Get all times if not specified 
            if tavg==[]:
                timevec = ppsample.getVar(ppsample.loadDataset(ncfile), 'time')
                tavg = [timevec[0],timevec[-1]]

            # Load the plane
            self.dbavg  = ppsamplexr.avgPlaneXR(ncfile, tavg, varnames=varnames, groupname=group, savepklfile=pklfile, verbose=verbose)
        
            # Do any sub-actions required for this task
            for a in self.actionlist:
                action = self.actionlist[a]
                # Check to make sure required actions are there
                if action.required and (action.actionname not in yamldict.keys()):
                    # This is a problem, stop things
                    raise ValueError('Required action %s not present'%action.actionname)
                if action.actionname in yamldict.keys():
                    actionitem = action(self, yamldict[action.actionname])
                    actionitem.execute()
        return 

    # --- Inner classes for action list ---
    @registeraction(actionlist)
    class compute_rotor_averaged_velocity():
        actionname = 'rotorAvgVel'
        blurb      = 'Computes the rotor averaged velocity'
        required   = True
        actiondefs = [
            {'key':'iplane',   'required':True,  'help':'List of iplane values',  'default':[0,]},
            {'key':'Diam', 'required':True,  'help':'Turbine Diameter',  'default':0},
            {'key':'zc',       'required':False,  'help':'Center of rotor disk in z',  'default':None},
            {'key':'yc',       'required':False,  'help':'Center of rotor disk in y',  'default':None},
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            print('Initialized '+self.actionname+' inside '+parent.name)
            print(self.actiondict)
            return

        def execute(self):
            print('Executing: compute rotor averaged velocity')
            yc = self.actiondict['yc']
            zc = self.actiondict['zc']
            Radius =  self.actiondict['Diam']/2.0
            iplanes = self.actiondict['iplane']
            rotor_avg = {}
            for iplane in iplanes:
                iplane = int(iplane)
                x = dbavg['x'][iplane,0,0]
                y = dbavg['y'][iplane,:,:]
                z = dbavg['z'][iplane,:,:]
                vel_avg = self.dbavg['velocityx'+'_avg'][iplane,:,:]
                if yc == None: yc = (y[-1]+y[0])/2.0
                if zc == None: zc = (z[-1]+z[0])/2.0
                Routside = ((y-yc)**2 + (z-zc)**2) > Radius**2
                masked_vel = np.ma.array(vel_avg,mask=Routside)
                rotor_avg[iplane] = masked_vel.mean()
                if verbose:
                    print("Rotor Average Velocity at x = ",x,": ",rotor_avg[iplane])
            return rotor_avg

    @registeraction(actionlist)
    class plotRotorAvgPlane():
        actionname = 'plotRotorAvgPlane'
        blurb      = 'Plot rotor averaged planes'
        required   = True
        actiondefs = [
        {'key':'dpi',       'required':False,  'default':125,
         'help':'Figure resolution', },
        {'key':'figsize',   'required':False,  'default':[12,3],
         'help':'Figure size (inches)', },
        {'key':'savefile',  'required':False,  'default':'',
         'help':'Filename to save the picture', },
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            print('Initialized '+self.actionname+' inside '+parent.name)
            print(self.actiondict)
            return

        def execute(self):
            print('Executing: plot rotor averaged plane')
            figsize  = self.actiondict['figsize']
            fig, ax = plt.subplots(1,1,figsize=(figsize[0],figsize[1]), dpi=dpi)
            xaxis    = self.actiondict['xaxis']
            yaxis    = self.actiondict['yaxis']
            dpi      = self.actiondict['dpi']
            iplanes  = self.actiondict['iplane']
            savefile = self.actiondict['savefile']
            for iplane in iplanes:
                c=plt.contourf(self.dbavg[xaxis][iplane,:,:], 
                            self.dbavg[yaxis][iplane,:,:], self.dbavg['velocityx_avg'][iplane, :, :], 
                            levels=121,cmap='coolwarm', extend='both')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3%", pad=0.05)
                cbar=fig.colorbar(c, ax=ax, cax=cax)
                cbar.ax.tick_params(labelsize=7)
                ax.set_xlabel(xaxis)
                ax.set_ylabel(yaxis)
                ax.axis('scaled')

                if len(savefile)>0:
                    savefname = savefile.format(iplane=iplane)
                    if verbose: print('Saving '+savefname)
                    plt.savefig(savefname)
            return

