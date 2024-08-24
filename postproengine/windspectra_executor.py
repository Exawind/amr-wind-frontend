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
import numpy as np
import pandas as pd
import postproamrwindabl as ppabl

def avgSpectraNCfile(ncfilename, ptlist, group, timeindices, avgbins=[], verbose=True):
    # Initialize dicts
    Suu_avg=None
    Svv_avg=None
    Sww_avg=None
    avgUlong=None
    all_ulongavgs = []
    # First extract all of the data from the netcdf file
    # TODO: allow pltlist to interp points
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
class windspectra_executor():
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
        {'key':'csvfile',  'required':True,  'help':'Filename to save spectra to',  'default':''},

        # --- Optional inputs ---
        {'key':'timeindices',      'required':False, 'help':'Which indices to use from netcdf file', 'default':[]},
        {'key':'avgbins',          'required':False, 'help':'Averaging time windows', 'default':[]},
        {'key':'thirdoctaveband',  'required':False, 'help':'Use 1/3 octave band averaging', 'default':False},
        {'key':'normalize',        'required':False, 'help':'Normalize the output spectra f and U^2', 'default':True},
    ]
    actionlist = {}                    # Dictionary for holding sub-actions
    example = """
windspectra:
- name: spectra1
  ncfile: /lustre/orion/cfd162/scratch/lcheung/sampling_80000.nc
  group: p_bot
  pointlocationfunction: spectrapoints.getptlist
  csvfile: spectra1.csv
  kaimal:
    ustarsource: ablstatsfile
    ablstatsfile:  /lustre/orion/cfd162/scratch/lcheung/abl_statistics80000.nc
    avgt: [20000, 25000]
    #ustar: 0.289809
    csvfile: kaimal1.csv
    z: 26.5
"""

    # --- Stuff required for main task ---
    def __init__(self, inputs, verbose=False):
        self.yamldictlist = []
        inputlist = inputs if isinstance(inputs, list) else [inputs]
        for indict in inputlist:
            self.yamldictlist.append(mergedicts(indict, self.inputdefs))
        print('Initialized '+self.name)
        #print(self.yamldictlist)
        #print(self.actionlist)
        return
    
    def execute(self, verbose=False):
        # Do any task-related here
        if verbose: print('Running '+self.name)
        for iplane, plane in enumerate(self.yamldictlist):
            # Run any necessary stuff for this task
            ncfile   = plane['ncfile']
            group    = plane['group']
            pointlocfunc    = plane['pointlocationfunction']
            thirdoctaveband = plane['thirdoctaveband']

            timeindices  = plane['timeindices']
            avgbins      = plane['avgbins']
            csvfile      = plane['csvfile']
            normalize    = plane['normalize']

            # Get the point locations from the udf
            modname  = pointlocfunc.split('.')[0]
            funcname = pointlocfunc.split('.')[1]
            func = getattr(sys.modules[modname], funcname)
            ptlist = func()
            
            # Extract the spectra outputs
            favg, Suu_avg, Svv_avg, Sww_avg, avgUlong = avgSpectraNCfile(ncfile, ptlist, group,
                                                                         timeindices, avgbins=[], verbose=verbose)

            # Save average velocity
            self.avgUlong = avgUlong
            
            # Convert to 1/3-octave band
            if thirdoctaveband:
                # TODO: do 1/3-octave band averaging
                self.favg_save = favg
                self.Suu_save = Suu_avg
                self.Svv_save = Svv_avg
                self.Sww_save = Sww_avg
            else:
                self.favg_save = favg
                self.Suu_save = Suu_avg
                self.Svv_save = Svv_avg
                self.Sww_save = Sww_avg

            # Normalize
            if normalize:
                self.Suu_save = self.favg_save*self.Suu_save/self.avgUlong**2
                self.Svv_save = self.favg_save*self.Svv_save/self.avgUlong**2
                self.Sww_save = self.favg_save*self.Sww_save/self.avgUlong**2
                
            #print(favg)
            #print(Suu_avg)

            # Save data to csv file
            dfcsv = pd.DataFrame()
            dfcsv['f']   = self.favg_save
            dfcsv['Suu'] = self.Suu_save
            dfcsv['Svv'] = self.Svv_save
            dfcsv['Sww'] = self.Sww_save
            dfcsv.to_csv(csvfile, index=False, sep=',')
            
            # Do any sub-actions required for this task
            for a in self.actionlist:
                action = self.actionlist[a]
                # Check to make sure required actions are there
                if action.required and (action.actionname not in yamldict.keys()):
                    # This is a problem, stop things
                    raise ValueError('Required action %s not present'%action.actionname)
                if action.actionname in self.yamldictlist[iplane].keys():
                    actionitem = action(self, self.yamldictlist[iplane][action.actionname])
                    actionitem.execute()
        return

    @registeraction(actionlist)
    class kaimal():
        actionname = 'kaimal'
        blurb      = 'Get the Kaimal spectra'
        required   = False
        actiondefs = [
            {'key':'csvfile',       'required':True,  'default':'',  'help':'CSV file to save Kaimal spectra to', },
            {'key':'ustarsource',   'required':True,  'help':'Source of ustar information (Options: specified/ablstatsfile)',  'default':'specifiied'},
            {'key':'ustar',         'required':False, 'help':'Group name in netcdf file',  'default':0.0},
            {'key':'z',             'required':True,  'help':'z-height for Kaimal spectra',  'default':0.0},
            {'key':'ablstatsfile',  'required':False, 'default':'',  'help':'NetCDF abl statistics file', },
            {'key':'avgt',          'required':False, 'default':[],  'help':'Average time over ustar', },
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent
            print('Initialized '+self.actionname+' inside '+parent.name)
            return

        
        def execute(self):
            print('Executing '+self.actionname)
            csvfile  = self.actiondict['csvfile']
            z        = self.actiondict['z']
            ustarsource = self.actiondict['ustarsource']
            
            # Get the ustar value
            if ustarsource == 'specified':
                ustar        = self.actiondict['ustar']
            else:
                ablstatsfile = self.actiondict['ablstatsfile']
                avgt         = self.actiondict['avgt']
                ncdat = ppabl.loadnetcdffile(ablstatsfile)
                ustar = ppabl.timeAvgScalar(ncdat, 'ustar', avgt)
                print('ustar = ',ustar)
                
            Suu_Kai = windspectra.getKaimal(self.parent.favg_save, z, self.parent.avgUlong)
            Svv_Kai = windspectra.getKaimal(self.parent.favg_save, z, self.parent.avgUlong, params=windspectra.vKaimalconst)
            Sww_Kai = windspectra.getKaimal(self.parent.favg_save, z, self.parent.avgUlong, params=windspectra.wKaimalconst)

            # Normalize the results
            Suu_Kai = ustar**2*Suu_Kai/self.parent.avgUlong**2
            Svv_Kai = ustar**2*Svv_Kai/self.parent.avgUlong**2
            Sww_Kai = ustar**2*Sww_Kai/self.parent.avgUlong**2

            # write the results to a CSV file
            dfcsv = pd.DataFrame()
            dfcsv['f']   = self.parent.favg_save
            dfcsv['Suu'] = Suu_Kai
            dfcsv['Svv'] = Svv_Kai
            dfcsv['Sww'] = Sww_Kai
            dfcsv.to_csv(csvfile, index=False, sep=',')
            
            return
            
