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
import correlate as corr
import postproamrwindsample_xarray as ppsamplexr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections            import OrderedDict

# Load ruamel or pyyaml as needed
try:
    import ruamel.yaml
    #yaml = ruamel.yaml.YAML(typ='unsafe', pure=True)
    yaml = ruamel.yaml.YAML(typ='rt')
    #print("# Loaded ruamel.yaml")
    useruamel=True
    loaderkwargs = {}
    dumperkwargs = {}    
    Loader=yaml.load
    #print("Done ruamel.yaml")
except:
    import yaml as yaml
    print("# Loaded yaml")
    useruamel=False
    loaderkwargs = {}
    dumperkwargs = {'default_flow_style':False }
    Loader=yaml.safe_load

if useruamel:
    from ruamel.yaml.comments import CommentedMap 
    def comseq(d):
        """
        Convert OrderedDict to CommentedMap
        """
        if isinstance(d, OrderedDict):
            cs = CommentedMap()
            for k, v in d.items():
                cs[k] = comseq(v)
            return cs
        return d

@registerplugin
class correlate_executor():
    """
    Example task
    """
    name      = "correlate"                # Name of task (this is same as the name in the yaml)
    blurb     = "Calculate the two-point correlation and integral lengthscale"  # Description of task
    inputdefs = [
        # --- Required inputs ---
        {'key':'name',     'required':True,  'help':'An arbitrary name',  'default':''},
        {'key':'ncfile',   'required':True,  'help':'NetCDF file',        'default':''},
        {'key':'group',    'required':True,  'help':'Group name in netcdf file',  'default':''},
        {'key':'timerange','required':True,  'help':'Time range to evaluate the correlation over',  'default':[]},
        {'key':'iplane',   'required':True,  'help':'Plane number to probe',  'default':0},
        {'key':'probelength', 'required':True,  'help':'Probe length',  'default':0},
        {'key':'probelocationfunction', 'required':True,  'default':'',
         'help':'Function to call to generate point locations. Function should have no arguments and return a list of (i,j,k) indices',},
        {'key':'plotprobept', 'required':False,  'help':'Make a plot of the probe locations',  'default':False},
        {'key':'saveprefix', 'required':False,  'help':'Filename prefix for all saved files',  'default':''},
    ]
    actionlist = {}                    # Dictionary for holding sub-actions
    example = """
```yaml
correlate:
  - name: two-point correlation (AMR-Wind)
    ncfile: /lustre/orion/cfd162/scratch/lcheung/sampling_80000.nc
    group: p_hub
    timerange: [20000, 21000]
    iplane: 0
    probelength: 1000
    probelocationfunction: spectrapoints.probelocations
    plotprobept: True
    saveprefix: correlation
    integrallengthscale:
      savefile: lengthscale.yaml
```
    
Note that in spectrapoints.py, the probelocations function is defined as:
```python
def probelocations(s=1):
    import numpy as np
    ds = 10
    startx = np.arange(100,200,ds)
    starty = np.arange(100,200,ds)[::s]
    startp = []
    yoffset=0
    [[startp.append([x,y+yoffset*iy,0]) for x in startx] for iy, y in enumerate(starty)]
    return startp
```
"""

    # --- Stuff required for main task ---
    def __init__(self, inputs, verbose=False):
        self.yamldictlist = []
        inputlist = inputs if isinstance(inputs, list) else [inputs]
        for indict in inputlist:
            self.yamldictlist.append(mergedicts(indict, self.inputdefs))
        print('Initialized '+self.name)
        return

    def execute(self, verbose=False):
        # Do any task-related here
        if verbose: print('Running '+self.name)
        for iplanenum, plane in enumerate(self.yamldictlist):
            # Run any necessary stuff for this task
            ncfile   = plane['ncfile']
            group    = plane['group']
            timerange= plane['timerange']
            iplane   = plane['iplane']
            probelength   = plane['probelength']
            probelocfunc  = plane['probelocationfunction']
            plotprobept   = plane['plotprobept']
            saveprefix    = plane['saveprefix']

            # Get the point locations from the udf
            modname  = probelocfunc.split('.')[0]
            funcname = probelocfunc.split('.')[1]
            func = getattr(sys.modules[modname], funcname)

            # Compute the average field
            avgdat, headers = corr.avgNCplaneXR(ncfile, timerange, group, iplane, verbose=verbose)

            # Compute the wind direction and WS
            ws, winddir           = corr.getavgwind(avgdat, headers, 0)
            print('WS   = '+repr(ws))
            print('Wdir = '+repr(winddir))
            winddirORIG = winddir + 0.0
            
            # Make the longitudinal probe list
            if (winddir>270): s=-1
            else:             s=+1
            startp     = func(s=s)
            plistLONG  = corr.makeprobeline(startp, round(winddir,2), probelength, avgdat)
            Nlong      = len(plistLONG)
            #print("Len(plist)=%i"%Nlong)
            if len(plistLONG[0])<3:
                raise ValueError('len(plistLONG[0])=%i'%len(plistLONG[0]))
            
            # Make the lateral probe list
            winddir = round(winddirORIG+90.0, 2)
            print(winddir)
            if (winddir>270): s=-1
            else:             s=+1
            startp2     = func(s=s)
            plistLAT = corr.makeprobeline(startp2, winddir, probelength, avgdat)
            Nlat     = len(plistLAT)
            #print("Len(plist)=%i"%Nlat)
            if len(plistLAT[0])<3:
                raise ValueError('len(plistLAT[0])=%i'%len(plistLAT[0]))
            
            # Plot the probe points
            if plotprobept:
                # LONG
                plt.figure(dpi=125)
                corr.plotprobes(plistLONG, avgdat, headers, verbose=False, skip=3)
                plt.title('Longitudinal')
                plt.axis('equal')
                # LAT
                plt.figure(dpi=125)
                corr.plotprobes(plistLAT, avgdat, headers, verbose=False, skip=3)
                plt.title('Lateral')
                plt.axis('equal')

            ij   = [0,0]
            plist        = plistLONG + plistLAT
            allf, allRij = corr.makeRij(ij, plist, [], True, None, iplane, group,
                                        avgdat = avgdat, headers=headers, timerange=timerange, ncfilename=ncfile,
                                        verbose=verbose, skip=10)

            print()
            # Split it back into LONG/LAT
            avgRijLong   = np.mean(allRij[:Nlong], axis=0)
            avgRijLat    = np.mean(allRij[Nlong:], axis=0)
            xi = allf[Nlong]

            self.avgRijLong = avgRijLong
            self.avgRijLat  = avgRijLat
            self.xi         = xi

            # Save the files
            if len(saveprefix)>0:
                # write the average results to a CSV file
                dfcsv = pd.DataFrame()
                dfcsv['xi']      = self.xi
                dfcsv['RijLong'] = self.avgRijLong
                dfcsv['RijLat']  = self.avgRijLat
                dfcsv.to_csv(saveprefix+'.AVG_Rij.csv', index=False, sep=',')
                
                # write full results to a dat file
                np.savetxt(saveprefix+'.xi.dat', self.xi)
                np.savetxt(saveprefix+'.ALL_RijLong.dat', allRij[:Nlong])
                np.savetxt(saveprefix+'.ALL_RijLat.dat' , allRij[Nlong:])
                
                        
            # Do any sub-actions required for this task
            for a in self.actionlist:
                action = self.actionlist[a]
                # Check to make sure required actions are there
                if action.required and (action.actionname not in yamldict.keys()):
                    # This is a problem, stop things
                    raise ValueError('Required action %s not present'%action.actionname)
                if action.actionname in self.yamldictlist[iplanenum].keys():
                    actionitem = action(self, self.yamldictlist[iplanenum][action.actionname])
                    actionitem.execute()

        # Done with executor
        return


    @registeraction(actionlist)
    class integrallengthscale():
        actionname = 'integrallengthscale'
        blurb      = 'Calculate the integral lengthscale'
        required   = False
        actiondefs = [
            {'key':'savefile',       'required':True,  'default':'',  'help':'YAML file to save the results to', },
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent
            print('Initialized '+self.actionname+' inside '+parent.name)
            return

        def execute(self):
            print('Executing '+self.actionname)
            savefile  = self.actiondict['savefile']

            # Calculate and save the data
            try:
                LONGlengthscale = corr.calclengthscale(self.parent.xi, self.parent.avgRijLong-0.0)
            except:
                LONGlengthscale = 0.0
            print('LONG lengthscale = %f'%LONGlengthscale)

            try:
                LATlengthscale = corr.calclengthscale(self.parent.xi, self.parent.avgRijLat)
            except:
                LATlengthscale = 0.0
            print('LAT lengthscale  = %f'%LATlengthscale)
            savedict = OrderedDict({'longitudinal':float(LONGlengthscale),
                                    'lateral':float(LATlengthscale)})
            outfile = open(savefile, 'w')
            yaml.dump(comseq(savedict), outfile, **dumperkwargs)
            outfile.close()
            return
