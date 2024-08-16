# Get the location where this script is being run
import sys, os
scriptpath = os.path.dirname(os.path.realpath(__file__))
basepath   = os.path.dirname(scriptpath)
utilpath   = os.path.join(basepath, "utilities")
# Add any possible locations of amr-wind-frontend here
amrwindfedirs = ['../',
                 utilpath,
                 basepath]
for x in amrwindfedirs: sys.path.insert(1, x)

from postproengine import registerplugin, mergedicts, registeraction
import windspectra
import postproamrwindsample_xarray as ppsamplexr


def avgSpectraNCfile(ncfilename, ptlist, group, timeindices, avgbins=[], verbose=True):
    # Initialize dicts
    Suu_avg=None
    Svv_avg=None
    Sww_avg=None
    avgUlong=None
    all_ulongavgs = []
    # First extract all of the data from the netcdf file
    db = ppsamplexr.getPlanePtsXR(ncfilename, timeindices, ptlist, groupname=group, verbose=verbose, gettimes=True)
    t  = np.array(db['times'])
    for ipt, pt in enumerate(ptlist):
        u = np.array(db[pt]['velocityx'])
        v = np.array(db[pt]['velocityy'])
        w = np.array(db[pt]['velocityz'])
        ulong, ulat = windspectra.convertUxytoLongLat(u,v)
        all_ulongavgs.append(np.mean(ulong))
        f, Suu      = windspectra.avgWindSpectra(t, ulong, avgbins)
        f, Svv      = windspectra.avgWindSpectra(t, ulat,  avgbins)
        f, Sww      = windspectra.avgWindSpectra(t, w,     avgbins)        
        if ipt == 0:
            lent    = len(t)
            favg    = f
            Suu_avg = Suu
            Svv_avg = Svv
            Sww_avg = Sww
        else:
            Suu_avg = Suu_avg + Suu
            Svv_avg = Svv_avg + Svv
            Sww_avg = Sww_avg + Sww    
    Npts = len(ptlist)
    Suu_avg = Suu_avg/Npts
    Svv_avg = Svv_avg/Npts
    Sww_avg = Sww_avg/Npts
    return favg, Suu_avg, Svv_avg, Sww_avg, np.mean(all_ulongavgs)

@registerplugin
class windspectra():
    """
    Example task
    """
    name      = "windspectra"                # Name of task (this is same as the name in the yaml)
    blurb     = "Calculate the wind spectra in time"  # Description of task
    inputdefs = [
        # --- Required inputs ---
        {'key':'name',     'required':True,  'help':'An arbitrary name',  'default':''},
        {'key':'ncfile',   'required':True,  'help':'NetCDF file',        'default':''},
        {'key':'group',    'required':True,  'help':'Group name in netcdf file',  'default':''},
        {'key':'pointlocationfunction', 'required':True,  'default':'',
         'help':'Function to call to generate point locations. Function should have no arguments and return a list of (i,j,k) indices',},

        # --- Optional inputs ---
        {'key':'avgbins',  'required':False, 'help':'Averaging time windows', 'default':[]},
    ]
    actionlist = {}                    # Dictionary for holding sub-actions
    example = """
"""

    # --- Stuff required for main task ---
    def __init__(self, inputs):
        self.yamldictlist = []
        inputlist = inputs if isinstance(inputs, list) else [inputs]
        for indict in inputlist:
            self.yamldictlist.append(mergedicts(indict, self.inputdefs))
        print('Initialized '+self.name)
        print(self.yamldictlist)
        print(self.actionlist)
        return
    
    def execute(self):
        # Do any task-related here
        if verbose: print('Running '+self.name)
        for iplane, plane in enumerate(self.yamldictlist):
            # Run any necessary stuff for this task
            ncfile   = plane['ncfile']
            group    = plane['group']
            avgbins  = plane['avgbins']
            
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
