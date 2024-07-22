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
        {'key':'title',     'required':False,  'default':'',
         'help':'Title of the plot',},
        {'key':'legendopts', 'required':False,  'default':{},
         'help':'Dictionary with legend options',},
        {'key':'postplotfunc', 'required':False,  'default':'',
         'help':'Function to call after plot is created. Function should have arguments func(fig, ax)',},                
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
        for plotitem in self.yamldictlist:
            csvfiles = plotitem['csvfiles']
            savefile = plotitem['savefile']
            dpi      = plotitem['dpi']
            figsize  = plotitem['figsize']
            xlabel   = plotitem['xlabel']
            ylabel   = plotitem['ylabel']
            title    = plotitem['title']
            legendopts = plotitem['legendopts']
            postplotfunc = plotitem['postplotfunc']
            
            fig, ax = plt.subplots(1,1,figsize=(figsize[0],figsize[1]), dpi=dpi)

            for csvitem in csvfiles:
                fname    = csvitem['file']
                xcol     = csvitem['xcol']
                ycol     = csvitem['ycol']
                lineopts = csvitem['lineopts'] if 'lineopts' in csvitem else {}
                
                varnames = [xcol, ycol]
                self.df  = pd.read_csv(fname, comment='#', usecols=lambda col: any(keyword in col for keyword in varnames))
                ax.plot(self.df[xcol], self.df[ycol], **lineopts)

            # Set up axes and labels
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
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
                savefname = savefile.format(time=time, iplane=iplane)
                if verbose: print('Saving '+savefname)
                plt.savefig(savefname)
            

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
