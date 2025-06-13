# Get the location where this script is being run
import sys, os
scriptpath = os.path.dirname(os.path.realpath(__file__))
basepath   = os.path.dirname(scriptpath)
# Add any possible locations of amr-wind-frontend here
amrwindfedirs = ['../',
                 basepath]
for x in amrwindfedirs: sys.path.insert(1, x)

from postproengine import registerplugin, mergedicts, registeraction
from postproengine import compute_axis1axis2_coords, get_mapping_xyz_to_axis1axis2, extract_1d_from_meshgrid
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
Plugin for converting AMR-Wind data to/from other formats

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
    Convert AMR-Wind planes to/from different file types 
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
    example = """
```yaml
convert:
name: plane2bts
filelist: /gpfs/lcheung/HFM/exawind-benchmarks/NREL5MW_ALM_BD_noturb/post_processing/rotorplaneDN_30000.nc
trange: [15000.0,15600.0]
bts:
    #iplane            : [0,1,2,3,4] #comment out to use all iplanes
    #xc                : 1797.5 #use midplane
    yc                : 90.0
    btsfile           : test_{iplane}.bts
    ID                : 8
    turbine_height    : 90.0
    group             : T0_rotorplaneDN
    diam              : 127.0
    xaxis             : 'a1'
    yaxis             : 'a2'
    varnames: ['velocitya1','velocitya2','velocitya3']
```
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
            {'key':'iplane',   'required':False,  'help':'Index of x location to read',  'default':None},
            {'key':'xc',       'required':False,  'help':'Location in flow to center plane on in the abscissa.',  'default':None},
            {'key':'yc',       'required':False,  'help':'Location to center plane on in the ordinate.',  'default':None},
            {'key':'btsfile',   'required':True,  'default':None,'help':'bts file name to save results'},
            {'key':'ID',        'required':False, 'default':8,'help':'bts file ID. 8="periodic", 7="non-periodic"'},
            {'key':'turbine_height','required':True,  'help':'Height of the turbine.',  'default':None},
            {'key':'group',   'required':False,  'default':None,
            'help':'Which group to pull from netcdf file', },
            {'key':'diam','required':True, 'default':None,'help':'Diameter for computing rotor averaged velocity'},
            {'key':'xaxis',    'required':False,  'default':'y','help':'Which axis to use on the abscissa', },
            {'key':'yaxis',    'required':False,  'default':'z','help':'Which axis to use on the ordinate', },
            {'key':'varnames',  'required':False,  'default':['velocityx', 'velocityy', 'velocityz'],
            'help':'Variables to extract from the netcdf file',},        
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent
            print('Initialized '+self.actionname+' inside '+parent.name)
            return

        def execute(self):

            def get_plane_data(ncfile,varnames,group,trange,iplanes,xaxis,yaxis):
                db = ppsamplexr.getPlaneXR(ncfile,[],varnames,groupname=group,verbose=0,includeattr=True,gettimes=True,timerange=trange)
                if iplanes == None: 
                    if isinstance(db['offsets'], np.ndarray):
                        iplanes = list(range(len(db['offsets'])))
                    else:
                        iplanes = [0,]

                if not isinstance(iplanes, list): iplanes = [iplanes,]

                if ('a1' in [xaxis, yaxis]) or ('a2' in [xaxis, yaxis]) or ('a3' in [xaxis, yaxis]):
                    compute_axis1axis2_coords(db,rot=0)
                    R = get_mapping_xyz_to_axis1axis2(db['axis1'],db['axis2'],db['axis3'],rot=0)
                    origin = db['origin']
                    origina1a2a3 = R@db['origin']
                    offsets = db['offsets']
                    offsets = [offsets] if (not isinstance(offsets, list)) and (not isinstance(offsets,np.ndarray)) else offsets

                xc = np.zeros(len(iplanes))
                YY = np.array(db[xaxis])
                ZZ = np.array(db[yaxis])

                t = np.asarray(np.array(db['times']).data)
                udata = {}
                for iplaneiter, iplane in enumerate(iplanes):
                    if ('a1' in [xaxis, yaxis]) or ('a2' in [xaxis, yaxis]) or ('a3' in [xaxis, yaxis]):
                        xc[iplaneiter] = origina1a2a3[-1] + offsets[iplane]
                    else:
                        xc[iplaneiter] = db['x'][iplane,0,0]
                    y,axisy = extract_1d_from_meshgrid(YY[iplane,:,:])
                    z,axisz = extract_1d_from_meshgrid(ZZ[iplane,:,:])
                    permutation = [0,axisy+1,axisz+1]
                    udata[iplane] = np.zeros((len(t),len(y),len(z),3))
                    for i,tstep in enumerate(db['timesteps']):
                        if ('velocitya' in varnames[0]) or ('velocitya' in varnames[1]) or ('velocitya' in varnames[2]):
                            ordered_data = np.transpose(np.array(db['velocitya3'][tstep]),permutation)
                            udata[iplane][i,:,:,0] = ordered_data[iplane,:,:]

                            ordered_data = np.transpose(np.array(db['velocity'+xaxis][tstep]),permutation)
                            udata[iplane][i,:,:,1] = ordered_data[iplane,:,:]

                            ordered_data = np.transpose(np.array(db['velocity'+yaxis][tstep]),permutation)
                            udata[iplane][i,:,:,2] = ordered_data[iplane,:,:]
                        else:
                            ordered_data = np.transpose(np.array(db['velocityx'][tstep]),permutation)
                            udata[iplane][i,:,:,0] = ordered_data[iplane,:,:]

                            ordered_data = np.transpose(np.array(db['velocityy'][tstep]),permutation)
                            udata[iplane][i,:,:,1] = ordered_data[iplane,:,:]

                            ordered_data = np.transpose(np.array(db['velocityz'][tstep]),permutation)
                            udata[iplane][i,:,:,2] = ordered_data[iplane,:,:]

                return udata , xc, y , z , t, iplanes 



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
            ycenter   = self.actiondict['xc']
            zcenter   = self.actiondict['yc']
            diam   = self.actiondict['diam']
            turbine_height = self.actiondict['turbine_height']
            if turbine_height == None: turbine_height = zloc
            iplane = self.actiondict['iplane']
            btsfile = self.actiondict['btsfile']
            group    = self.actiondict['group']
            ID  = self.actiondict['ID']
            xaxis    = self.actiondict['xaxis']
            yaxis    = self.actiondict['yaxis']
            varnames = self.actiondict['varnames']

            # Load the plane
            udata,xcs,y,z,times,iplanes = get_plane_data(self.parent.filelist,varnames,group,self.parent.times,iplane,xaxis,yaxis)

            #use midplane value if not specified
            if ycenter == None:
                ycenter = (y[-1]+y[0])/2.0

            if zcenter == None:
                zcenter = (z[-1]+z[0])/2.0

            #allow for lateral offsets in center of turbsim planes
            y0_dist = abs(ycenter-y[0])
            y1_dist = abs(ycenter-y[-1])
            y_box_size = 2*min(y0_dist,y1_dist)
            if y0_dist == min(y0_dist,y1_dist): 
                y_box_ind = np.argmin(abs(y - y_box_size - y[0]))
                y = y[0:y_box_ind+1] 
            if y1_dist == min(y0_dist,y1_dist): 
                y_box_ind = np.argmin(abs(y - (y[-1] - y_box_size)))
                y = y[y_box_ind:] 

            bot_ind = np.argmin(abs(z-(zcenter - turbine_height)))
            z = z[bot_ind:]
            nt = len(times)
            ny = len(y)
            nz = len(z)
            t = np.array(times)

            for iplane in iplanes:
                ts = {}
                ts["u"] = np.ndarray((3,nt,ny,nz)) 
                uRef = np.ndarray(nt)
                for titer , tval in enumerate(t):
                    if y0_dist == y1_dist: 
                        ts['u'][0,titer,:,:] = udata[iplane][titer,:,bot_ind:,0]
                        ts['u'][1,titer,:,:] = udata[iplane][titer,:,bot_ind:,1]
                        ts['u'][2,titer,:,:] = udata[iplane][titer,:,bot_ind:,2]
                    elif y0_dist == min(y0_dist,y1_dist): 
                        ts['u'][0,titer,:,:] = udata[iplane][titer,0:y_box_ind+1,bot_ind:,0]
                        ts['u'][1,titer,:,:] = udata[iplane][titer,0:y_box_ind+1,bot_ind:,1]
                        ts['u'][2,titer,:,:] = udata[iplane][titer,0:y_box_ind+1,bot_ind:,2]
                    else:
                        ts['u'][0,titer,y_box_ind:,bot_ind:] = udata[iplane][titer,y_box_ind:,bot_ind:,0]
                        ts['u'][1,titer,y_box_ind:,bot_ind:] = udata[iplane][titer,y_box_ind:,bot_ind:,1]
                        ts['u'][2,titer,y_box_ind:,bot_ind:] = udata[iplane][titer,y_box_ind:,bot_ind:,2]

                    interpolator = RegularGridInterpolator((y, z), ts['u'][0,titer, :, :])
                    uRef[titer]  = interpolator((ycenter, zcenter))

                Radius = diam/2.0
                YY,ZZ = np.meshgrid(y,z,indexing='ij')
                Routside = ((YY-ycenter)**2 + (ZZ-zcenter)**2) > Radius**2
                vel_avg = np.mean(ts['u'][0,:,:,:],axis=0)
                masked_vel = np.ma.array(vel_avg,mask=Routside)
                print("Rotor Average Velocity at x = ",xcs[iplane],": ",masked_vel.mean())
                ts['t']  = np.round(t,decimals=7)
                ts['y']  = y - np.mean(y) # y always centered on 0
                ts['z']  = z
                ts['ID'] = ID
                ts['zRef'] = zcenter
                ts['uRef'] = float(np.mean(uRef))
                print("Reference velocity: ",ts['uRef'], " at lateral and vertical locations: ",ycenter," and ",zcenter)

                savefname = btsfile.format(iplane=iplane)
                print("Writing to bts file: ",savefname)
                bts_write(ts,savefname)
                print()

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

