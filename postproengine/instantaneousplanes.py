# Get the location where this script is being run
import sys, os
scriptpath = os.path.dirname(os.path.realpath(__file__))
basepath   = os.path.dirname(scriptpath)
# Add any possible locations of amr-wind-frontend here
amrwindfedirs = ['../',
                 basepath]
for x in amrwindfedirs: sys.path.insert(1, x)

from postproengine import registerplugin, mergedicts, registeraction
from postproengine import compute_axis1axis2_coords
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
class postpro_instantaneousplanes():
    """
    Make plots of instantaneous planes
    """
    # Name of task (this is same as the name in the yaml
    name      = "instantaneousplanes"
    # Description of task
    blurb     = "Make instantaneous plots from netcdf sample planes"
    inputdefs = [
        # -- Required parameters ----
        {'key':'name',     'required':True,  'default':'',
         'help':'An arbitrary name',},
        {'key':'ncfile',   'required':True,  'default':'',
        'help':'NetCDF sampling file', },
        {'key':'iters',    'required':True,  'default':2,
        'help':'Which iterations to pull from netcdf file', },
        {'key':'iplane',   'required':True,  'default':0,
        'help':'Which plane to pull from netcdf file', },
        {'key':'xaxis',    'required':True,  'default':'x',
        'help':'Which axis to use on the abscissa', },
        {'key':'yaxis',    'required':True,  'default':'y',
        'help':'Which axis to use on the ordinate', },
        # --- optional parameters ---
        {'key':'times',    'required':False,  'default':[],
         'help':'Which times to pull from netcdf file (overrides netCDF)', },        
        {'key':'group',   'required':False,  'default':None,
         'help':'Which group to pull from netcdf file', },
        {'key':'title',     'required':False,  'default':'',
         'help':'Title of the plot',},
        {'key':'varnames',  'required':False,  'default':['velocityx', 'velocityy', 'velocityz'],
         'help':'Variables to extract from the netcdf file',},        
        {'key':'plotfunc',  'required':False,  'default':'lambda u, v, w: np.sqrt(u**2 + v**2)',
         'help':'Function to plot (lambda expression)',},
        {'key':'clevels',   'required':False,  'default':'np.linspace(0, 12, 121)',
         'help':'Color levels (eval expression)',},
        {'key':'xlabel',    'required':False,  'default':'X [m]',
         'help':'Label on the X-axis', },
        {'key':'ylabel',    'required':False,  'default':'Y [m]',
         'help':'Label on the Y-axis', },
        {'key':'dpi',       'required':False,  'default':125,
         'help':'Figure resolution', },
        {'key':'figsize',   'required':False,  'default':[12,3],
         'help':'Figure size (inches)', },
        {'key':'savefile',  'required':False,  'default':'',
         'help':'Filename to save the picture', },
        
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
        for planeiter , plane in enumerate(self.yamldictlist):
            iters    = plane['iters']
            iplane   = plane['iplane']
            ncfile   = plane['ncfile']
            xaxis    = plane['xaxis']
            yaxis    = plane['yaxis']
            
            # Load optional quantities
            times    = plane['times']
            group    = plane['group']
            varnames = plane['varnames']
            plotfunc = eval(plane['plotfunc'])
            title    = plane['title']
            clevels  = eval(plane['clevels'])
            xlabel   = plane['xlabel']
            ylabel   = plane['ylabel']
            savefile = plane['savefile']
            dpi      = plane['dpi']
            figsize  = plane['figsize']
            
            # Get the times instead
            if len(times)>0:
                timevec = ppsample.getVar(ppsample.loadDataset(ncfile), 'time')
                find_nearest = lambda a, a0: np.abs(np.array(a) - a0).argmin()
                iters = [find_nearest(timevec, t) for t in times]
            
            # Load the plane
            db  = ppsamplexr.getPlaneXR(ncfile, iters, varnames, groupname=group, verbose=verbose, gettimes=True, includeattr=True)

            # Convert to native axis1/axis2 coordinates if necessary
            if ('a1' in [xaxis, yaxis]) or \
               ('a2' in [xaxis, yaxis]) or \
               ('a3' in [xaxis, yaxis]):
                compute_axis1axis2_coords(db)
            
            # Loop through each time instance and plot
            for iplot, i in enumerate(iters):
                time  = db['times'][iplot]
                fig, ax = plt.subplots(1,1,figsize=(figsize[0],figsize[1]), dpi=dpi)
                plotq = plotfunc(db['velocityx'][i], db['velocityy'][i], db['velocityz'][i])
                c=plt.contourf(db[xaxis][iplane,:,:], db[yaxis][iplane,:,:], plotq[iplane, :, :], levels=clevels, cmap='coolwarm', extend='both')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3%", pad=0.05)
                cbar=fig.colorbar(c, ax=ax, cax=cax)
                cbar.ax.tick_params(labelsize=7)
                
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(eval("f'{}'".format(title)))
                ax.axis('scaled')
                if len(savefile)>0:
                    savefname = savefile.format(time=time, iplane=iplane)
                    if verbose: print('Saving '+savefname)
                    plt.savefig(savefname)
        
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

    # --- Inner classes for action list ---
    # This is not needed at the moment, keeping here just in case
    #@registeraction(actionlist)
    class action1():
        actionname = 'action1'
        blurb      = 'A description of action'
        required   = True
        actiondefs = [
            {'key':'name',     'required':True,  'help':'An arbitrary name',  'default':''},
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent
            print('Initialized '+self.actionname+' inside '+parent.name)
            print(self.actiondict)
            return

        def execute(self):
            print('Executing ' + self.actionname)
            return

