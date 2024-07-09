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
import time
import struct

"""
Plugin for creating instantaneous planar images

See README.md for details on the structure of classes here
"""

@registerplugin
class postpro_dataconverts():
    """
    Convert AMR-Wind planes to different file types 
    """
    # Name of task (this is same as the name in the yaml)
    name      = "convert"
    # Description of task
    blurb     = "Converts netcdf sample planes to different file formats"
    inputdefs = [
        # -- Execute parameters ----
        {'key':'name',     'required':True,  'default':'',
         'help':'An arbitrary name',},
        {'key':'ncfile',   'required':True,  'default':'',
        'help':'NetCDF sampling file', },
        {'key':'group',   'required':False,  'default':None,
         'help':'Which group to pull from netcdf file', },
        {'key':'iterrange',    'required':True,  'default':[0,1],
        'help':'Range of iterations to pull from netcdf file (inclusive)', },
        {'key':'times',    'required':False,  'default':None,
         'help':'Which times to pull from netcdf file (overrides iters). An empty list, [], indicates use all times.', },        
        
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
            #Get all times if not specified 
            iters    = plane['iterrange']
            ncfile   = plane['ncfile']
            times    = plane['times']
            group    = plane['group']
            varnames = ['velocityx', 'velocityy', 'velocityz']

            # Get the times instead
            iters = np.arange(iters[0],iters[1]+1)
            notRead = True
            if not times == None:
                timevec = ppsample.getVar(ppsample.loadDataset(ncfile), 'time')
                if times == []:
                    iters = np.arange(0,len(timevec))
                    # Load the plane
                    output_dt = (timevec[-1]-timevec[0])/(len(timevec)-1)
                    if not groupname == None:
                        self.db = ppsamplexr.getFullPlaneXR(ncfile, len(timevec), output_dt,groupname=group)
                        notRead = False
                else:
                    find_nearest = lambda a, a0: np.abs(np.array(a) - a0).argmin()
                    iters = [find_nearest(timevec, t) for t in times]
            
            if notRead:
                # Load the plane
                self.db = ppsamplexr.getPlaneXR(ncfile, iters, varnames, groupname=group, verbose=verbose, gettimes=True)

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
    class convert_to_bts():
        actionname = 'bts'
        blurb      = 'Converts data to bts files'
        required   = False
        actiondefs = [
            {'key':'iplane',   'required':True,  'help':'Index of x location to read',  'default':0},
            {'key':'yhh',       'required':True,  'help':'Hub height location in y',  'default':None},
            {'key':'zhh',       'required':True,  'help':'Hub height location in z',  'default':None},
            {'key':'btsfile',   'required':True,  'default':None,'help':'bts file name to save results'},
            {'key':'ID',        'required':False, 'default':8,'help':'bts file ID. 8="periodic", 7="non-periodic"'},
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent
            print('Initialized '+self.actionname+' inside '+parent.name)
            return




        def execute(self):
            def bts_write(btsdict, filename):
                """
                Write to bts file based on openfast-toolbox

                """
                nDim, nt, ny, nz = btsdict['u'].shape
                if 'uTwr' not in btsdict.keys() :
                    btsdict['uTwr']=np.zeros((3,nt,0))
                if 'ID' not in btsdict.keys() :
                    btsdict['ID']=7

                _, _, nTwr = btsdict['uTwr'].shape
                tsTwr  = btsdict['uTwr']
                ts     = btsdict['u']
                intmin = -32768
                intrng = 65535
                off    = np.empty((3), dtype    = np.float32)
                scl    = np.empty((3), dtype    = np.float32)
                info = 'Generated by TurbSimFile on {:s}.'.format(time.strftime('%d-%b-%Y at %H:%M:%S', time.localtime()))
                # Calculate scaling, offsets and scaling data
                out    = np.empty(ts.shape, dtype=np.int16)
                outTwr = np.empty(tsTwr.shape, dtype=np.int16)
                for k in range(3):
                    all_min, all_max = ts[k].min(), ts[k].max()
                    if nTwr>0:
                        all_min=min(all_min, tsTwr[k].min())
                        all_max=max(all_max, tsTwr[k].max())
                    if all_min == all_max:
                        scl[k] = 1
                    else:
                        scl[k] = intrng / (all_max-all_min)
                    off[k]    = intmin - scl[k] * all_min
                    out[k]    = (ts[k]    * scl[k] + off[k]).astype(np.int16)
                    outTwr[k] = (tsTwr[k] * scl[k] + off[k]).astype(np.int16)
                z0 = btsdict['z'][0]
                dz = btsdict['z'][1]- btsdict['z'][0]
                dy = btsdict['y'][1]- btsdict['y'][0]
                dt = btsdict['t'][1]- btsdict['t'][0]
                dt = np.around(dt , decimals=7)

                # Providing estimates of uHub and zHub even if these fields are not used
                zHub = btsdict['zRef']
                uHub = btsdict['uRef']
                bHub = True

                with open(filename, mode='wb') as f:            
                    f.write(struct.pack('<h4l', btsdict['ID'], nz, ny, nTwr, nt))
                    f.write(struct.pack('<6f', dz, dy, dt, uHub, zHub, z0)) # NOTE uHub, zHub maybe not used
                    f.write(struct.pack('<6f', scl[0],off[0],scl[1],off[1],scl[2],off[2]))
                    f.write(struct.pack('<l' , len(info)))
                    f.write(info.encode())
                    try:
                        for it in np.arange(nt):
                            f.write(out[:,it,:,:].tobytes(order='F'))
                            f.write(outTwr[:,it,:].tobytes(order='F'))
                    except:
                        for it in np.arange(nt):
                            f.write(out[:,it,:,:].tostring(order='F'))
                            f.write(outTwr[:,it,:].tostring(order='F'))

            print('Executing '+self.actionname)
            ts = {}

            yloc   = self.actiondict['yhh']
            zloc   = self.actiondict['zhh']
            iplane = self.actiondict['iplane']
            btsfile = self.actiondict['btsfile']
            ID  = self.actiondict['ID']
            x=np.asarray(self.parent.db['x'].data)
            y=np.asarray(self.parent.db['y'].data)
            z=np.asarray(self.parent.db['z'].data)
            x=x[:,0,0]
            y=y[0,0,:]
            z=z[0,:,0]
            xloc = x[iplane]
            nt = len(self.parent.db['times'])
            ny = len(y)
            nz = len(z)
            xind = np.where(x == xloc)[0]
            yind = np.where(y == yloc)[0]
            zind = np.where(z == zloc)[0]
            ts["u"]          = np.ndarray((3,nt,ny,nz)) 
            t = np.array(self.parent.db['times'])
            tsteps = np.array(self.parent.db['timesteps'])
            for titer , tval in enumerate(tsteps):
                ts['u'][0,titer,:,:] = np.swapaxes(np.array(self.parent.db['velocityx'][tval]),1,2)[xind,:,:]
                ts['u'][1,titer,:,:] = np.swapaxes(np.array(self.parent.db['velocityy'][tval]),1,2)[xind,:,:]
                ts['u'][2,titer,:,:] = np.swapaxes(np.array(self.parent.db['velocityz'][tval]),1,2)[xind,:,:]
            ts['t']  = np.round(t,decimals=7)
            ts['y']  = y - np.mean(y) # y always centered on 0
            ts['z']  = z
            ts['ID'] = ID
            ts['zRef'] = zloc
            ts['uRef'] = float(np.mean(ts['u'][0,:,yind,zind]))
            print("Writing to bts file: ",btsfile)
            bts_write(ts,btsfile)

            return 

