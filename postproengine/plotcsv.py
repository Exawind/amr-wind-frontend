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
import pandas as pd

"""
Plugin for creating line plots of csv files

See README.md for details on the structure of classes here
"""

@registerplugin
class postpro_plotcsv():
    """
    Make plots of instantaneous planes
    """
    # Name of task (this is same as the name in the yaml
    name      = "plotcsv"
    # Description of task
    blurb     = "Make plots of csv files"
    inputdefs = [
        # --- Required parameters ----
        {'key':'name',     'required':True,  'default':'',
         'help':'An arbitrary name',},
        {'key':'csvfiles', 'required':True,  'default':'',
         'help':'A list of dictionaries containing csv files',},        
        # --- optional parameters ----
        {'key':'dpi',       'required':False,  'default':125,
         'help':'Figure resolution', },
        {'key':'figsize',   'required':False,  'default':[12,3],
         'help':'Figure size (inches)', },
        {'key':'savefile',  'required':False,  'default':'',
         'help':'Filename to save the picture', },
        {'key':'xlabel',    'required':False,  'default':'Time [s]',
         'help':'Label on the X-axis', },
        {'key':'ylabel',    'required':False,  'default':'',
         'help':'Label on the Y-axis', },
        {'key':'xscale',    'required':False,  'default':'linear',
         'help':'Scale on the X-axis (options: linear/log/symlog/logit)', },
        {'key':'yscale',    'required':False,  'default':'linear',
         'help':'Scale on the Y-axis (options: linear/log/symlog/logit)', },
        {'key':'title',     'required':False,  'default':'',
         'help':'Title of the plot',},
        {'key':'legendopts', 'required':False,  'default':{},
         'help':'Dictionary with legend options',},
        {'key':'postplotfunc', 'required':False,  'default':'',
         'help':'Function to call after plot is created. Function should have arguments func(fig, ax)',},
        {'key':'figname',    'required':False,  'default':None,
         'help':'Name/number of figure to create plot in'},
        {'key':'axesnum',    'required':False,  'default':None,
         'help':'Which subplot axes to create plot in'},

    ]
    actionlist = {}                    # Dictionary for holding sub-actions
    example = """
plotcsv:
  - name: plotfiles
    xlabel: 'Time'
    ylabel: 'Power'
    title: 'Turbine power'
    legendopts: {'loc':'upper left'}
    csvfiles:
    - {'file':'T0.csv', 'xcol':'Time', 'ycol':'GenPwr', 'lineopts':{'color':'r', 'lw':2, 'label':'T0'}}

Note that the csvfiles dictionary list can also include xscalefunc and yscalefunc lambda functions 
to manipulate x and y inputs.  For instance,
   'xscalefunc':'lambda x:x-72.5'
shifts the x data by 72.5.  Similarly, 
 'yscalefunc':'lambda y:y*2.0'
Multiples y by 2.0.  If ycol is the empty string '', then the lambda function input is the whole dataframe.  
This allows you to provide the function
 'yscalefunc':'lambda y:y["BldPitch1"]+["BldPitch1"]'
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
        for plotitemiter , plotitem in enumerate(self.yamldictlist):
            csvfiles = plotitem['csvfiles']
            savefile = plotitem['savefile']
            dpi      = plotitem['dpi']
            figsize  = plotitem['figsize']
            xlabel   = plotitem['xlabel']
            ylabel   = plotitem['ylabel']
            xscale   = plotitem['xscale']
            yscale   = plotitem['yscale']
            title    = plotitem['title']
            legendopts = plotitem['legendopts']
            postplotfunc = plotitem['postplotfunc']
            figname  = plotitem['figname']
            axesnum  = None if plotitem['axesnum'] is None else plotitem['axesnum']

            if (figname is not None) and (axesnum is not None):
                fig     = plt.figure(figname)
                allaxes = fig.get_axes()
                ax      = allaxes[axesnum]
            else:
                fig, ax = plt.subplots(1,1,figsize=(figsize[0],figsize[1]), dpi=dpi)

            for csvitem in csvfiles:
                fname    = csvitem['file']
                xcol     = csvitem['xcol']
                ycol     = csvitem['ycol']
                lineopts = csvitem['lineopts'] if 'lineopts' in csvitem else {}
                xscalefunc = csvitem['xscalefunc'] if 'xscalefunc' in csvitem else 'lambda x: x'
                yscalefunc = csvitem['yscalefunc'] if 'yscalefunc' in csvitem else 'lambda y: y'
                xscalef  = eval(xscalefunc)
                yscalef  = eval(yscalefunc)

                varnames = [xcol, ycol]

                if len(ycol)>0:
                    self.df  = pd.read_csv(fname, comment='#', usecols=lambda col: any(keyword in col for keyword in varnames))
                    yplot = yscalef(np.array(self.df[ycol]))
                else:
                    self.df  = pd.read_csv(fname, comment='#')
                    yplot = yscalef(self.df)
                xplot = xscalef(np.array(self.df[xcol]))
                ax.plot(xplot, yplot, **lineopts)

            # Set up axes and labels
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            ax.set_title(title)
            ax.legend(**legendopts)

            # Run any post plot functions
            if len(postplotfunc)>0:
                modname = postplotfunc.split('.')[0]
                funcname = postplotfunc.split('.')[1]
                func = getattr(sys.modules[modname], funcname)
                func(fig, ax)

            # Save the figure
            if len(savefile)>0:
                savefname = savefile
                if verbose: print('Saving '+savefname)
                plt.savefig(savefname)
            

            # Do any sub-actions required for this task
            for a in self.actionlist:
                action = self.actionlist[a]
                # Check to make sure required actions are there
                if action.required and (action.actionname not in self.yamldictlist[plotitemiter].keys()):
                    # This is a problem, stop things
                    raise ValueError('Required action %s not present'%action.actionname)
                if action.actionname in self.yamldictlist[plotitemiter].keys():
                    actionitem = action(self, self.yamldictlist[plotitemiter][action.actionname])
                    actionitem.execute()
        return
