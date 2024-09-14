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
import numpy.linalg as linalg
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import struct
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
import gzip

"""
Plugin for creating instantaneous planar images

See README.md for details on the structure of classes here
"""

def strdecode(v):
    if sys.version_info[0] < 3:
        return v.decode('utf-8')
    else:
        try:
            return str(v, encoding='utf-8')
        except:
            return v

def readheader(filename):
    fname, fext = os.path.splitext(filename)
    headers=[]
    # Else just use the one file headers
    if ((fext == '.gz') or (fext == '.GZ')):
        with gzip.open(filename) as fp:
            timestring = fp.readline().strip().split()[1]
            headerline = str(strdecode(fp.readline()))
            #print(headerline.replace("#",""))
            headerstr = headerline.replace("#","").strip().split()
            headers.extend(headerstr[:])
    else:
        with open(filename) as fp:
            timestring = fp.readline().strip().split()[1]
            headerstr = fp.readline().replace("#","").strip().split()
            headers.extend(headerstr[:])
    time=float(strdecode(timestring).replace(",",""))
    return time, headers


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
        {'key':'filelist',   'required':True,  'default':'',
        'help':'NetCDF sampling file', },
        {'key':'trange',    'required':True,  'default':None,
         'help':'Which times to pull from netcdf file, e.g., [tstart,tend]', },        
        
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
        self.verbose=verbose
        if verbose: print('Running '+self.name)
        # Loop through and create plots
        for planeiter , plane in enumerate(self.yamldictlist):
            #Get all times if not specified 
            self.filelist= plane['filelist']
            self.times    = plane['trange']

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
    @registeraction(actionlist)
    class convert_to_bts():
        actionname = 'bts'
        blurb      = 'Converts data to bts files'
        required   = False
        actiondefs = [
            {'key':'iplane',   'required':True,  'help':'Index of x location to read',  'default':0},
            {'key':'yhh',       'required':True,  'help':'Location in flow to use as hub height location in y',  'default':None},
            {'key':'zhh',       'required':True,  'help':'Location in flow to use as hub height location in z',  'default':None},
            {'key':'btsfile',   'required':True,  'default':None,'help':'bts file name to save results'},
            {'key':'ID',        'required':False, 'default':8,'help':'bts file ID. 8="periodic", 7="non-periodic"'},
            {'key':'turbine_height','required':False,  'help':'Height of the turbine (if different than zc)',  'default':None},
            {'key':'group',   'required':False,  'default':None,
            'help':'Which group to pull from netcdf file', },
            {'key':'diam','required':False, 'default':None,'help':'Diameter for computing rotor averaged velocity'},
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
            diam   = self.actiondict['diam']
            turbine_height = self.actiondict['turbine_height']
            if turbine_height == None: turbine_height = zloc
            iplane = self.actiondict['iplane']
            btsfile = self.actiondict['btsfile']
            group    = self.actiondict['group']
            ID  = self.actiondict['ID']

            varnames = ['velocityx', 'velocityy', 'velocityz']

            # Load the plane
            self.db = ppsamplexr.getPlaneXR(self.parent.filelist, [0,1], varnames, groupname=group, verbose=False,includeattr=True,gettimes=True,timerange=self.parent.times)
            XX = np.array(self.db['x'])
            YY = np.array(self.db['y'])
            ZZ = np.array(self.db['z'])

            flow_index  = -np.ones(3)
            for i in range(0,3):
                if (sum(self.db['axis'+str(i+1)]) != 0):
                    flow_index[i] = np.nonzero(self.db['axis'+str(i+1)])[0][0]
            for i in range(0,3):
                if flow_index[i] == -1:
                    flow_index[i] = 3 - (flow_index[i-1] + flow_index[i-2])

            streamwise_index = 2-np.where(flow_index == 0)[0][0]
            lateral_index = 2-np.where(flow_index == 1)[0][0]
            vertical_index = 2-np.where(flow_index == 2)[0][0]

            slices = [slice(None)] * 3
            slices[lateral_index] = 0
            slices[vertical_index] = 0
            x = XX[tuple(slices)]

            slices = [slice(None)] * 3
            slices[streamwise_index] = 0
            slices[vertical_index] = 0
            y = YY[tuple(slices)]

            slices = [slice(None)] * 3
            slices[streamwise_index] = 0
            slices[lateral_index] = 0
            z = ZZ[tuple(slices)]

            xloc = x[iplane]
            xind = np.where(x == xloc)[0]
            yind = np.where(y == yloc)[0]
            zind = np.where(z == zloc)[0]

            y0_dist = abs(yloc-y[0])
            y1_dist = abs(yloc-y[-1])
            y_box_size = 2*min(y0_dist,y1_dist)

            if y0_dist == min(y0_dist,y1_dist): 
                y_box_ind = np.argmin(abs(y - y_box_size - y[0]))
                y = y[0:y_box_ind+1] 
            if y1_dist == min(y0_dist,y1_dist): 
                y_box_ind = np.argmin(abs(y - (y[-1] - y_box_size)))
                y = y[y_box_ind:] 


            bot_ind = np.argmin(abs(z-(zloc - turbine_height)))
            z = z[bot_ind:]
            nt = len(self.db['times'])
            ny = len(y)
            nz = len(z)
            ts["u"]          = np.ndarray((3,nt,ny,nz)) 
            permutation = [streamwise_index, lateral_index, vertical_index]
            t = np.array(self.db['times'])
            tsteps = np.array(self.db['timesteps'])
            uRef = np.ndarray(nt)
            for titer , tval in enumerate(tsteps):
                if y0_dist == y1_dist: 
                    ts['u'][0,titer,:,:] = np.transpose(np.array(self.db['velocityx'][tval]),permutation)[iplane,:,bot_ind:]
                    ts['u'][1,titer,:,:] = np.transpose(np.array(self.db['velocityy'][tval]),permutation)[iplane,:,bot_ind:]
                    ts['u'][2,titer,:,:] = np.transpose(np.array(self.db['velocityz'][tval]),permutation)[iplane,:,bot_ind:]
                elif y0_dist == min(y0_dist,y1_dist): 
                    ts['u'][0,titer,:,:] = np.transpose(np.array(self.db['velocityx'][tval]),permutation)[iplane,0:y_box_ind+1,bot_ind:]
                    ts['u'][1,titer,:,:] = np.transpose(np.array(self.db['velocityy'][tval]),permutation)[iplane,0:y_box_ind+1,bot_ind:]
                    ts['u'][2,titer,:,:] = np.transpose(np.array(self.db['velocityz'][tval]),permutation)[iplane,0:y_box_ind+1,bot_ind:]
                else:
                    ts['u'][0,titer,y_box_ind:,bot_ind:] = np.transpose(np.array(self.db['velocityx'][tval]),permutation)[iplane,y_box_ind:,bot_ind:]
                    ts['u'][1,titer,y_box_ind:,bot_ind:] = np.transpose(np.array(self.db['velocityy'][tval]),permutation)[iplane,y_box_ind:,bot_ind:]
                    ts['u'][2,titer,y_box_ind:,bot_ind:] = np.transpose(np.array(self.db['velocityz'][tval]),permutation)[iplane,y_box_ind:,bot_ind:]

                interpolator = RegularGridInterpolator((y, z), ts['u'][0,titer, :, :])
                uRef[titer]  = interpolator((yloc, zloc))

            Radius = diam/2.0
            YY,ZZ = np.meshgrid(y,z,indexing='ij')
            Routside = ((YY-yloc)**2 + (ZZ-zloc)**2) > Radius**2
            vel_avg = np.mean(ts['u'][0,:,:,:],axis=0)
            masked_vel = np.ma.array(vel_avg,mask=Routside)
            print("Rotor averged velocity: ",masked_vel.mean())
            ts['t']  = np.round(t,decimals=7)
            ts['y']  = y - np.mean(y) # y always centered on 0
            ts['z']  = z
            ts['ID'] = ID
            ts['zRef'] = zloc
            #ts['uRef'] = float(np.mean(ts['u'][0,:,yind,zind]))
            ts['uRef'] = float(np.mean(uRef))
            print("Reference velocity: ",ts['uRef'])

            print("Writing to bts file: ",btsfile)
            bts_write(ts,btsfile)

            return 

    # --- Inner classes for action list ---
    @registeraction(actionlist)
    class convert_nalu_wind_to_amr_wind():
        actionname = 'nalu_to_amr'
        blurb      = 'Converts a list of planes from Nalu-Wind to AMR-Wind'
        required   = False
        actiondefs = [
            {'key':'savefile',  'required':True,  'default':None,'help':'Name of AMR-Wind file'},
            {'key':'coordfile','required':False, 'default':None,'help':'Nalu-Wind coordinate file'},
            {'key':'groupname','required':False, 'default':'plane', 'help':'netCDF group name'},
        ]
        """
        convert:
        name: Wake YZ plane
        filelist: '/pscratch/kbrown1/GE2.8-127_Stable2_AWCOFF_bugfix/sliceData/YZslice/YZslice_01.00D_*_4.dat.gz'
        trange: [200,1300]

        nalu_to_amr: 
            savefile: test.nc
            coordfile: '/pscratch/kbrown1/GE2.8-127_Stable2_AWCOFF_bugfix/sliceData/YZslice/YZslice_01.00D_coordXYZ.dat.gz'
        """

        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent
            print('Initialized '+self.actionname+' inside '+parent.name)
            return

        def execute(self):
            outFile  = self.actiondict['savefile']
            coordfile = self.actiondict['coordfile']
            groupname = self.actiondict['groupname']

            filelist = ppsamplexr.getFileList(self.parent.filelist)

            # Load the coordinates
            if coordfile is None:
                coorddat = np.loadtxt(filelist[0])[:, :6]
            else:
                coorddat = np.loadtxt(coordfile)

            Numk      = int(max(coorddat[:,0]))+1
            Numj      = int(max(coorddat[:,1]))+1
            Numi      = int(max(coorddat[:,2]))+1

            # Calculate the origin and plane axes
            corner    = coorddat[(coorddat[:,1]==0)&(coorddat[:,2]==0),:][0][3:6]

            # Compute axis1 and axis2
            dxrow     = coorddat[(coorddat[:,1]==0)&(coorddat[:,2]==1),:][0][3:6]
            dyrow     = coorddat[(coorddat[:,1]==1)&(coorddat[:,2]==0),:][0][3:6]
            dX    = linalg.norm(np.array(dxrow)-np.array(corner))
            dY    = linalg.norm(np.array(dyrow)-np.array(corner))
            axis1 = (np.array(dxrow)-np.array(corner))*(Numi - 1)
            axis2 = (np.array(dyrow)-np.array(corner))*(Numj - 1)

            # Compute axis3 and offsets
            if Numk > 1:
                dzrow     = coorddat[(coorddat[:,0]==1)&(coorddat[:,1]==0)&(coorddat[:,2]==0),:][0][3:6]
                axis3     = (np.array(dzrow)-np.array(corner))*(Numk - 1)
                offsets   = []
                for ik in range(Numk):
                    dzoffset = coorddat[(coorddat[:,0]==(ik+1))&(coorddat[:,1]==0)&(coorddat[:,2]==0),:][0][3:6]
                    zoffset  = np.array(dzoffset)-np.array(corner)
                    offsets.append(linalg.norm(zoffset))
            else:
                axis3  = np.cross(axis1, axis2)
                offsets = [0.0]
            # Normalize and make them array
            axis3  = axis3/np.linalg.norm(axis3)
            offsets = np.array(offsets)
            dfs = []
            times = []
            for i, fname in enumerate(filelist):
                if self.parent.verbose: ppsamplexr.progress(i+1, len(filelist))
                time, headers = readheader(fname)
                times.append(time)
                df = np.loadtxt(fname)
                dfs.append(df)
            if self.parent.verbose: print()

            # Store Nalu data in AMR style plane sampler netcdf
            times = np.array(times)
            num_points = len(coorddat)
            num_time_steps = len(times)

            coordinates = np.zeros((num_points,3))
            velocityx   = np.zeros((num_time_steps,num_points))
            velocityy   = np.zeros((num_time_steps,num_points))
            velocityz   = np.zeros((num_time_steps,num_points))

            # Locate the columns with velocity
            ivelocityx = headers.index('velocity_probe[0]')
            ivelocityy = headers.index('velocity_probe[1]')
            ivelocityz = headers.index('velocity_probe[2]')

            for time_step, df in enumerate(dfs):
                velocityx[time_step,:] = df[:, ivelocityx]
                velocityy[time_step,:] = df[:, ivelocityy]
                velocityz[time_step,:] = df[:, ivelocityz]

            coordinates[:,0] = coorddat[:,3]
            coordinates[:,1] = coorddat[:,4]
            coordinates[:,2] = coorddat[:,5]

            # Write out the netcdf file
            ncFile = Dataset(outFile,'w')
            ncFile.title="AMR-Wind data from Nalu output"
            ncFile.createDimension('num_time_steps',num_time_steps)
            ncFile.createDimension('ndim',3)
            time_nc = ncFile.createVariable('time', times[0].dtype, ('num_time_steps',))
            time_nc[:] = times

            AMR_group_name = groupname
            grp = ncFile.createGroup(AMR_group_name)
            grp.createDimension('num_points',num_points)
            coordinates_nc = grp.createVariable('coordinates', coordinates[0,0].dtype, ('num_points','ndim'))
            velocityx_nc = grp.createVariable('velocityx', velocityx[0,0].dtype, ('num_time_steps','num_points'))
            velocityy_nc = grp.createVariable('velocityy', velocityy[0,0].dtype, ('num_time_steps','num_points'))
            velocityz_nc = grp.createVariable('velocityz', velocityz[0,0].dtype, ('num_time_steps','num_points'))

            grp.sampling_type='PlaneSampler'

            grp.ijk_dims=np.array([int(Numi),int(Numj),int(Numk)])

            coordinates_nc[:] = coordinates

            velocityx_nc[:] = velocityx
            velocityy_nc[:] = velocityy
            velocityz_nc[:] = velocityz

            grp.origin = corner

            grp.axis1  = axis1
            grp.axis2  = axis2
            grp.axis3  = axis3

            grp.offsets = offsets

            if self.parent.verbose:
                print(ncFile,flush=True)
                for grp in ncFile.groups.items():
                    print(grp,flush=True)
            ncFile.close()
            if self.parent.verbose:
                print('wrote '+outFile)

            return 

