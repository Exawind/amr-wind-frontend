# Get the location where this script is being run
import sys, os
scriptpath = os.path.dirname(os.path.realpath(__file__))
basepath   = os.path.dirname(scriptpath)
# Add any possible locations of amr-wind-frontend here
amrwindfedirs = ['../',
                 basepath]
for x in amrwindfedirs: sys.path.insert(1, x)

from postproengine import registerplugin, mergedicts, registeraction, contourplottemplate
from postproengine import compute_axis1axis2_coords, get_mapping_xyz_to_axis1axis2

from postproengine import extract_1d_from_meshgrid
import postproamrwindsample_xarray as ppsamplexr
import postproamrwindsample as ppsample
from postproengine import interpolatetemplate, circavgtemplate 
from postproengine import compute_axis1axis2_coords
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.ticker as ticker
import math

"""
Plugin for post processing control volumes

See README.md for details on the structure of classes here
"""

@registerplugin
class postpro_controlvolume():
    """
    Postprocess control volumes
    """
    # Name of task (this is same as the name in the yaml)
    name      = "controlvolume"
    # Description of task
    blurb     = "Control volume analysis "
    inputdefs = [
        # -- Execute parameters ----
        {'key':'name',     'required':True,  'default':'',
         'help':'An arbitrary name',},

        {'key':'Uinf','required':True,'default':None,'help':'U inflow',},
        {'key':'diam','required':True,'default':None,'help':'Turbine diameter',},
        {'key':'axis','required':True,'default':None,'help':'Order of axis',},
        {'key':'center','required':True,'default':None,'help':'Center of control volume',},
        {'key':'rho','required':True,'default':1.25,'help':'Density',},
        {'key':'box_dims','required':True,'default':None,'help':'Dimensions of control volume in turbine diameters',},
        {'key':'bot_avg_file','required':True,'default':'','help':'Bot avg pkl file',},
        {'key':'bot_rs_file','required':True,'default':'','help':'Bot rs pkl file',},
        #{'key':'varnames','required':False,'default':['velocityx', 'velocityy', 'velocityz'],'help':'Variable names ',},
        {'key':'top_avg_file','required':True,'default':'','help':'top avg pkl file',},
        {'key':'top_rs_file','required':True,'default':'','help':'top rs pkl file',},
        {'key':'lft_avg_file','required':True,'default':'','help':'lft avg pkl file',},
        {'key':'lft_rs_file','required':True,'default':'','help':'lft rs pkl file',},
        {'key':'rht_avg_file','required':True,'default':'','help':'rht avg pkl file',},
        {'key':'rht_rs_file','required':True,'default':'','help':'rht rs pkl file',},
        {'key':'x_avg_files','required':True,'default':'','help':'x avg pkl files',},
        {'key':'x_rs_files','required':True,'default':'','help':'x rs pkl files',},
        {'key':'savepklfile', 'required':False,  'default':'',
        'help':'Name of pickle file to save results', },

    ]
    actionlist = {}                    # Dictionary for holding sub-actions
    example = """
controlvolume:
  name: Streamwise CV
  Uinf: 9.03
  diam: 240
  center: [2280.0,1000.0,150.0]
  axis: ['x','y','z']
  rho: 1.2456
  box_dims: [6,1,1]
  body_force: [0.00014295185866400572, 0.0008354682029301641, 0.0]

  bot_avg_file: '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_avg_XY.pkl'
  bot_rs_file:  '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_rs_XY.pkl'

  top_avg_file: '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_avg_XY.pkl'
  top_rs_file:  '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_rs_XY.pkl'

  lft_avg_file: '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_avg_XZl.pkl'
  lft_rs_file:  '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_rs_XZl.pkl'

  rht_avg_file: '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_avg_XZr.pkl'
  rht_rs_file:  '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_rs_XZr.pkl'

  x_avg_files:
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_avg_YZwake1.pkl'
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_avg_YZwake2.pkl'
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_avg_YZwake3.pkl'
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_avg_YZwake4.pkl'
  x_rs_files:
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_rs_YZwake1.pkl'
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_rs_YZwake2.pkl'
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_rs_YZwake3.pkl'
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline/postpro_rs_YZwake4.pkl'

  table:

  plot_totals:
    savefile: 'test_cv_total.png'

  plot_contributions:
    savefile: 'test_cv_contributions.png'
    """
    # --- Stuff required for main task ---
    def __init__(self, inputs, verbose=False):
        self.yamldictlist = []
        inputlist = inputs if isinstance(inputs, list) else [inputs]
        for indict in inputlist:
            self.yamldictlist.append(mergedicts(indict, self.inputdefs))
        if verbose: print('Initialized '+self.name)
        return

    def loadpkl(self,pklfiles):

        # Load the files into a dictionary  
        dd_avg = defaultdict(list) # initialize dictionary
        for pklfile in pklfiles:
            with open(pklfile, 'rb') as fp:
                d = pickle.load(fp)
                axis_info = {} 
                axis_info['axis1'] = d['axis1']
                axis_info['axis2'] = d['axis2']
                axis_info['axis3'] = d['axis3']
                axis_info['origin'] = d['origin']
            for key, value in d.items():
                dd_avg[key].append(value)
        dd_avg = dict(dd_avg)

        # Flatten the lists from each offset/group into single 3D numpy arrays
        dd2_avg = {}
        for index, key in enumerate(dd_avg.keys()):
            # print(key)
            if np.array(dd_avg[key][0]).ndim == 3:
                dd2_avg[key] = []
                for i in range(len(dd_avg[key])):
                    if i==0:
                        dd2_avg[key] = dd_avg[key][i]
                    else:
                        dd2_avg[key] = np.concatenate((dd2_avg[key],dd_avg[key][i]),axis=0)

        return dd2_avg, axis_info

    def doubleIntegral(self,val,xgrid,ygrid,flag=False):
        integral = np.trapz(np.trapz(val, x=ygrid,axis=1),x=xgrid,axis=0)
        if flag: print(val[0,0])
        return integral

    def tripleIntegral(self,val,xgrid,ygrid,zgrid):
        integral = np.trapz(np.trapz(np.trapz(val, x=zgrid,axis=2),x=ygrid,axis=1),x=xgrid,axis=0)
        return integral

    def Merge(self,dict1, dict2):
        res = {**dict1, **dict2}
        return res

    def load_avg_rs_files(self,avg_file,rs_file,axis,boxCenter,boxDimensions):

        dd2_avg, axis_info = self.loadpkl(avg_file)
        dd2_rs, _  = self.loadpkl(rs_file)
        dd2 = self.Merge(dd2_avg,dd2_rs)

        XX = np.array(dd2[axis[0]])
        YY = np.array(dd2[axis[1]])
        ZZ = np.array(dd2[axis[2]])
        x,axisx = extract_1d_from_meshgrid(XX[0,:,:])
        y,axisy = extract_1d_from_meshgrid(YY[0,:,:])
        z,axisz = extract_1d_from_meshgrid(ZZ[0,:,:])

        if axisx == -1: axis3_label = axis[0]
        if axisy == -1: axis3_label = axis[1]
        if axisz == -1: axis3_label = axis[2]

        #unique in the direction of offset
        slice_result = dd2[axis3_label][:, 0, 0] 
        sortedd, newOrder = np.unique(slice_result, return_index=True)
        for index, key in enumerate(dd2.keys()):
            dd2[key] = dd2[key][newOrder, :, :]

        #crop to control volume
        indicesStreamnormal = (dd2[axis[0]]>=boxCenter[0]-boxDimensions[0]/2) & (dd2[axis[0]]<=boxCenter[0]+boxDimensions[0]/2) & (dd2[axis[1]]>=boxCenter[1]-boxDimensions[1]/2) & (dd2[axis[1]]<=boxCenter[1]+boxDimensions[1]/2) & (dd2[axis[2]]>=boxCenter[2]-boxDimensions[2]/2) & (dd2[axis[2]]<=boxCenter[2]+boxDimensions[2]/2)
        indicesStreamnormal_dim = np.where(indicesStreamnormal)

        dd3 = {}
        for index, key in enumerate(dd2.keys()):
            if np.array(dd2[key]).ndim == 3:
                dd3[key] = []
                temp = dd2[key][indicesStreamnormal]
                dd3[key] = np.reshape(temp,(len(set(indicesStreamnormal_dim[0])),len(set(indicesStreamnormal_dim[1])),len(set(indicesStreamnormal_dim[2]))))

        #crop axis3 direction for dd2
        dd2_slice = dd2[axis3_label][:, 0, 0] 
        dd3_slice = dd3[axis3_label][:, 0, 0] 
        _, indices_dd2, _ = np.intersect1d(dd2_slice, dd3_slice,return_indices=True)
        for index, key in enumerate(dd2.keys()):
            dd2[key] = dd2[key][indices_dd2,:,:]

        return dd2,dd3,axis_info

    def are_aligned(self,v1, v2):
        return np.all(np.cross(v1, v2) == 0)

    def get_label_and_ind_in_dir(self,axis_info,axis_dir,dd,axis):
        n1 = axis_info['axis1']/np.linalg.norm(axis_info['axis1'])
        n2 = axis_info['axis2']/np.linalg.norm(axis_info['axis2'])
        n3 = axis_info['axis3']/np.linalg.norm(axis_info['axis3'])

        #ijk dims flipped 
        if self.are_aligned(n1,axis_dir): axis_ind = 2
        if self.are_aligned(n2,axis_dir): axis_ind = 1
        if self.are_aligned(n3,axis_dir): axis_ind = 0

        XX = np.array(dd[axis[0]])
        YY = np.array(dd[axis[1]])
        ZZ = np.array(dd[axis[2]])
        xvec,axisx = extract_1d_from_meshgrid(XX[0,:,:])
        yvec,axisy = extract_1d_from_meshgrid(YY[0,:,:])
        zvec,axisz = extract_1d_from_meshgrid(ZZ[0,:,:])

        if axisx+1 == axis_ind:
            axis_label = axis[0]

        if axisy+1 == axis_ind:
            axis_label = axis[1]

        if axisz+1 == axis_ind:
            axis_label = axis[2]

        return axis_label , axis_ind

    def execute(self, verbose=False):
        if verbose: print('Running '+self.name)
        # Loop through and create plots
        for iplane, plane in enumerate(self.yamldictlist):
            Uinf = plane['Uinf']
            diam = plane['diam']
            axis = plane['axis']
            center_position = plane['center']            
            rho = plane['rho']
            boxDimensions = np.asarray(plane['box_dims'])
            boxDimensions*=diam
            #varnames = plane['varnames']            
            body_force = np.asarray(plane['body_force'])
            savepklfile = plane['savepklfile']

            rot_time_period = -(2.0 * 2.0*np.pi / 0.007524699)*Uinf 
            coriolis_factor = 2.0 * 2.0*np.pi / rot_time_period

            boxCenter = [center_position[0]+boxDimensions[0]/2, center_position[1], center_position[2]]

            corr_mapping = {
                'velocityx': 'u',
                'velocityy': 'v',
                'velocityz': 'w',
                'velocitya1': 'ua1',
                'velocitya2': 'ua2',
                'velocitya3': 'ua3'
            }

            print("Loading YZ planes...",end='',flush=True)
            x_avg_files   = plane['x_avg_files']            
            x_rs_files    = plane['x_rs_files']            
            dd2_YZ_coarse,dd3_YZ_coarse,axis_info_YZ  = self.load_avg_rs_files(x_avg_files,x_rs_files,axis,boxCenter,boxDimensions)
            print("Done")

            print("Loading XY planes...",end='',flush=True)
            bot_avg_file = plane['bot_avg_file']
            bot_rs_file  = plane['bot_rs_file']
            bot_iplane   = 0 #domain will get cropped so this is always 0

            top_avg_file = plane['top_avg_file']            
            top_rs_file  = plane['top_rs_file']            
            top_iplane   = -1 #domain will get cropped so this is always -1

            dd2_XY,dd3_XY,axis_info_XY  = self.load_avg_rs_files([bot_avg_file,top_avg_file],[bot_rs_file,top_rs_file],axis,boxCenter,boxDimensions)
            print("Done")

            print("Loading XZ planes...",end='',flush=True)
            lft_avg_file = plane['lft_avg_file']            
            lft_rs_file  = plane['lft_rs_file']            
            lft_iplane   = 0 #domain will get cropped so this is always 0

            rht_avg_file = plane['rht_avg_file']            
            rht_rs_file  = plane['rht_rs_file']            
            rht_iplane   = -1 #domain will get cropped so this is always -1

            dd2_XZ,dd3_XZ,axis_info_XZ  = self.load_avg_rs_files([lft_avg_file,rht_avg_file],[lft_rs_file,rht_rs_file],axis,boxCenter,boxDimensions)
            print("Done")

            print("Interpolating YZ data in x...",end='',flush=True)

            dd3 = dd3_YZ_coarse
            XX = np.array(dd3_YZ_coarse[axis[0]])
            YY = np.array(dd3_YZ_coarse[axis[1]])
            ZZ = np.array(dd3_YZ_coarse[axis[2]])
            xvec,axisx = extract_1d_from_meshgrid(XX[0,:,:])
            yvec,axisy = extract_1d_from_meshgrid(YY[0,:,:])
            zvec,axisz = extract_1d_from_meshgrid(ZZ[0,:,:])
            if axisx == -1: axis3_label = axis[0]
            if axisy == -1: axis3_label = axis[1]
            if axisz == -1: axis3_label = axis[2]

            xvec = dd3_YZ_coarse[axis3_label][:,0,0] #in direction of streamwise plane 

            #streamwise label may be different for XZ and XY
            streamwise_label_XY , streamwise_ind_XY = self.get_label_and_ind_in_dir(axis_info_XY,axis_info_YZ['axis3'],dd3_XY,axis)
            streamwise_label_XZ , streamwise_ind_XZ = self.get_label_and_ind_in_dir(axis_info_XZ,axis_info_YZ['axis3'],dd3_XZ,axis)
            #XZ and XY labels in YZ axis1 (called vertical here)
            vertical_label_XY , vertical_ind_XY = self.get_label_and_ind_in_dir(axis_info_XY,axis_info_YZ['axis2'],dd3_XY,axis)
            vertical_label_XZ , vertical_ind_XZ = self.get_label_and_ind_in_dir(axis_info_XZ,axis_info_YZ['axis2'],dd3_XZ,axis)
            #XZ and XY labels in YZ axis2 (called lateral here)
            lateral_label_XY , lateral_ind_XY = self.get_label_and_ind_in_dir(axis_info_XY,axis_info_YZ['axis1'],dd3_XY,axis)
            lateral_label_XZ , lateral_ind_XZ = self.get_label_and_ind_in_dir(axis_info_XZ,axis_info_YZ['axis1'],dd3_XZ,axis)


            xvecnew,_ = extract_1d_from_meshgrid(dd3_XZ[streamwise_label_XZ][0,:,:])
            indicesWithinRange = np.where((xvecnew >= np.min(xvec)) & (xvecnew <= np.max(xvec)))
            xnew = xvecnew[indicesWithinRange] 

            # Interpolate in x
            dd3_YZ = {}
            for index, key in enumerate(dd3.keys()):
                if np.array(dd3[key]).ndim == 3:
                    arr = dd3[key]
                    dd3_YZ[key] = np.zeros((len(xnew),arr.shape[1],arr.shape[2]))
                    for j in range(dd3[key].shape[1]):
                        for k in range(dd3[key].shape[2]):
                            dd3_YZ[key][:,j,k] = np.interp(xnew,xvec,arr[:,j,k]) 

            
            lateral_label_YZ , lateral_ind_YZ = self.get_label_and_ind_in_dir(axis_info_YZ,axis_info_YZ['axis1'],dd3_YZ,axis)
            streamwise_label_YZ , streamwise_ind_YZ = self.get_label_and_ind_in_dir(axis_info_YZ,axis_info_YZ['axis3'],dd3_YZ,axis)
            vertical_label_YZ , vertical_ind_YZ = self.get_label_and_ind_in_dir(axis_info_YZ,axis_info_YZ['axis2'],dd3_YZ,axis)

            #permute to YZ data ordering and reduce size to match limits of YZ grid (i.e., cut out x/D locations just behind the rotor where there is no YZ data) 
            permutation = (streamwise_ind_XY,vertical_ind_XY,lateral_ind_XY)
            for index, key in enumerate(dd3_XY.keys()):
                if np.array(dd3_XY[key]).ndim == 3:
                    dd3_XY[key] = np.transpose(dd3_XY[key],permutation)
                    dd3_XY[key] = np.squeeze(dd3_XY[key][indicesWithinRange,:,:])

            permutation = (streamwise_ind_XZ,vertical_ind_XZ,lateral_ind_XZ)
            for index, key in enumerate(dd3_XZ.keys()):
                if np.array(dd3_XZ[key]).ndim == 3:
                    dd3_XZ[key] = np.transpose(dd3_XZ[key],permutation)
                    dd3_XZ[key] = np.squeeze(dd3_XZ[key][indicesWithinRange,:,:])

            # print("XY INFO: ",)
            # print("X: ",dd3_XY[streamwise_label_XY][:,0,0],streamwise_label_XY,streamwise_ind_XY)
            # print("Y: ",dd3_XY[lateral_label_XY][0,0,:],lateral_label_XY,lateral_ind_XY)
            # print("Z: ",dd3_XY[vertical_label_XY][0,:,0],vertical_label_XY,vertical_ind_XY)

            # print()
            # print("XZ INFO: ",)
            # print("X: ",dd3_XZ[streamwise_label_XZ][:,0,0],streamwise_label_XZ,streamwise_ind_XZ)
            # print("Y: ",dd3_XZ[lateral_label_XZ][0,0,:],lateral_label_XZ,lateral_ind_XZ)
            # print("Z: ",dd3_XZ[vertical_label_XZ][0,:,0],vertical_label_XZ,vertical_ind_XZ)

            # print()
            # print("YZ INFO: ",)
            # print("X: ",dd3_YZ[streamwise_label_YZ][:,0,0],streamwise_label_YZ,streamwise_ind_YZ)
            # print("Y: ",dd3_YZ[lateral_label_YZ][0,0,:],lateral_label_YZ,lateral_ind_YZ)
            # print("Z: ",dd3_YZ[vertical_label_YZ][0,:,0],vertical_label_YZ,vertical_ind_YZ)

            print("Done")

            print("Calculating streamwise gradients...",end='',flush=True)

            streamwise_velocity_label = 'velocity' + streamwise_label_XY + '_avg'
            dd3_XY['grad_px_derived_avg']        = np.gradient(dd3_XY['p_avg'],dd3_XY[streamwise_label_XY][:,0,0],axis=0)
            dd3_XY['grad_velocity0_derived_avg'] = np.gradient(dd3_XY[streamwise_velocity_label],dd3_XY[streamwise_label_XY][:,0,0],axis=0)
            dd3_XY['grad_velocity1_derived_avg'] = np.gradient(dd3_XY[streamwise_velocity_label],dd3_XY[lateral_label_XY][0,0,:],axis=2)
            dd3_XY['grad_velocity2_derived_avg'] = np.gradient(dd3_XY[streamwise_velocity_label],dd3_XY[vertical_label_XY][0,:,0],axis=1)

            streamwise_velocity_label = 'velocity' + streamwise_label_XZ + '_avg'
            dd3_XZ['grad_px_derived_avg']        = np.gradient(dd3_XZ['p_avg'],dd3_XZ[streamwise_label_XZ][:,0,0],axis=0)
            dd3_XZ['grad_velocity0_derived_avg'] = np.gradient(dd3_XZ[streamwise_velocity_label],dd3_XZ[streamwise_label_XZ][:,0,0],axis=0)
            dd3_XZ['grad_velocity1_derived_avg'] = np.gradient(dd3_XZ[streamwise_velocity_label],dd3_XZ[lateral_label_XZ][0,0,:],axis=2)
            dd3_XZ['grad_velocity2_derived_avg'] = np.gradient(dd3_XZ[streamwise_velocity_label],dd3_XZ[vertical_label_XY][0,:,0],axis=1)

            streamwise_velocity_label = 'velocity' + streamwise_label_YZ + '_avg'
            dd3_YZ['grad_px_derived_avg']        = np.gradient(dd3_YZ['p_avg'],dd3_YZ[streamwise_label_YZ][:,0,0],axis=0)
            dd3_YZ['grad_velocity0_derived_avg'] = np.gradient(dd3_YZ[streamwise_velocity_label],dd3_YZ[streamwise_label_YZ][:,0,0],axis=0)
            dd3_YZ['grad_velocity1_derived_avg'] = np.gradient(dd3_YZ[streamwise_velocity_label],dd3_YZ[lateral_label_YZ][0,0,:],axis=2)
            dd3_YZ['grad_velocity2_derived_avg'] = np.gradient(dd3_YZ[streamwise_velocity_label],dd3_YZ[vertical_label_YZ][0,:,0],axis=1)

            print("Done")

            # calculate useful parameters
            print("Calculating remaining transport terms...",end='',flush=True)
            streamPos = dd3_YZ[streamwise_label_YZ][:,0,0]-center_position[0]
            numStreamPos = len(streamPos)
            minStreamPosIncrement = np.min(np.diff(streamPos))

            streamwise_velocity_label = 'velocity' + streamwise_label_XY 
            lateral_velocity_label = 'velocity' + lateral_label_XY 
            vertical_velocity_label = 'velocity' + vertical_label_XY 
            streamwise_streamwise_label = corr_mapping[streamwise_velocity_label] + corr_mapping[streamwise_velocity_label]
            streamwise_lateral_label = corr_mapping[streamwise_velocity_label] + corr_mapping[lateral_velocity_label]
            streamwise_vertical_label = corr_mapping[streamwise_velocity_label] + corr_mapping[vertical_velocity_label]
            dd3_XY['u_avg_cubed'] = dd3_XY[streamwise_velocity_label+'_avg']**3# units of m^3/s^3
            dd3_XY['u_avg_squared_v_avg'] = dd3_XY[streamwise_velocity_label+'_avg']**2*dd3_XY[lateral_velocity_label+'_avg'] # units of m^3/s^3
            dd3_XY['u_avg_squared_w_avg'] = dd3_XY[streamwise_velocity_label+'_avg']**2*dd3_XY[vertical_velocity_label+'_avg'] # units of m^3/s^3
            dd3_XY['P_x'] = -dd3_XY[streamwise_streamwise_label + '_avg']*dd3_XY['grad_velocity0_derived_avg'] - dd3_XY[streamwise_lateral_label+'_avg']*dd3_XY['grad_velocity1_derived_avg'] - dd3_XY[streamwise_vertical_label + '_avg']*dd3_XY['grad_velocity2_derived_avg'] # units of m^3/s^3 (ordering of gradients from AMR from 0-8 is dudx dudy dudz dvdx dvdy dvz dwdx dwdy dwdz)
            dd3_XY['1_over_rho_u_avg_dp_dx_avg'] = (1/rho)*dd3_XY[streamwise_velocity_label+'_avg']*dd3_XY['grad_px_derived_avg'] # units of m^2/s^3
            dd3_XY['coriolis_x'] = -coriolis_factor*dd3_XY[lateral_velocity_label+'_avg']*dd3_XY[streamwise_velocity_label+'_avg'] # units of m^2/s^3
            dd3_XY['body_force'] = body_force[0]*dd3_XY[streamwise_velocity_label+'_avg'] # units of m^2/s^3? (see note in the input section at the top of this ipynb)

            streamwise_velocity_label = 'velocity' + streamwise_label_XZ 
            lateral_velocity_label = 'velocity' + lateral_label_XZ 
            vertical_velocity_label = 'velocity' + vertical_label_XZ 
            streamwise_streamwise_label = corr_mapping[streamwise_velocity_label] + corr_mapping[streamwise_velocity_label]
            streamwise_lateral_label = corr_mapping[streamwise_velocity_label] + corr_mapping[lateral_velocity_label]
            streamwise_vertical_label = corr_mapping[streamwise_velocity_label] + corr_mapping[vertical_velocity_label]
            dd3_XZ['u_avg_cubed'] = dd3_XZ[streamwise_velocity_label+'_avg']**3# units of m^3/s^3
            dd3_XZ['u_avg_squared_v_avg'] = dd3_XZ[streamwise_velocity_label+'_avg']**2*dd3_XZ[lateral_velocity_label+'_avg'] # units of m^3/s^3
            dd3_XZ['u_avg_squared_w_avg'] = dd3_XZ[streamwise_velocity_label+'_avg']**2*dd3_XZ[vertical_velocity_label+'_avg'] # units of m^3/s^3
            dd3_XZ['P_x'] = -dd3_XZ[streamwise_streamwise_label + '_avg']*dd3_XZ['grad_velocity0_derived_avg'] - dd3_XZ[streamwise_lateral_label+'_avg']*dd3_XZ['grad_velocity1_derived_avg'] - dd3_XZ[streamwise_vertical_label + '_avg']*dd3_XZ['grad_velocity2_derived_avg'] # units of m^3/s^3 (ordering of gradients from AMR from 0-8 is dudx dudy dudz dvdx dvdy dvz dwdx dwdy dwdz)
            dd3_XZ['1_over_rho_u_avg_dp_dx_avg'] = (1/rho)*dd3_XZ[streamwise_velocity_label+'_avg']*dd3_XZ['grad_px_derived_avg'] # units of m^2/s^3
            dd3_XZ['coriolis_x'] = -coriolis_factor*dd3_XZ[lateral_velocity_label+'_avg']*dd3_XZ[streamwise_velocity_label+'_avg'] # units of m^2/s^3
            dd3_XZ['body_force'] = body_force[0]*dd3_XZ[streamwise_velocity_label+'_avg'] # units of m^2/s^3? (see note in the input section at the top of this ipynb)

            streamwise_velocity_label = 'velocity' + streamwise_label_YZ 
            lateral_velocity_label = 'velocity' + lateral_label_YZ 
            vertical_velocity_label = 'velocity' + vertical_label_YZ 
            streamwise_streamwise_label = corr_mapping[streamwise_velocity_label] + corr_mapping[streamwise_velocity_label]
            streamwise_lateral_label = corr_mapping[streamwise_velocity_label] + corr_mapping[lateral_velocity_label]
            streamwise_vertical_label = corr_mapping[streamwise_velocity_label] + corr_mapping[vertical_velocity_label]
            dd3_YZ['u_avg_cubed'] = dd3_YZ[streamwise_velocity_label+'_avg']**3# units of m^3/s^3
            dd3_YZ['u_avg_squared_v_avg'] = dd3_YZ[streamwise_velocity_label+'_avg']**2*dd3_YZ[lateral_velocity_label+'_avg'] # units of m^3/s^3
            dd3_YZ['u_avg_squared_w_avg'] = dd3_YZ[streamwise_velocity_label+'_avg']**2*dd3_YZ[vertical_velocity_label+'_avg'] # units of m^3/s^3
            dd3_YZ['P_x'] = -dd3_YZ[streamwise_streamwise_label + '_avg']*dd3_YZ['grad_velocity0_derived_avg'] - dd3_YZ[streamwise_lateral_label+'_avg']*dd3_YZ['grad_velocity1_derived_avg'] - dd3_YZ[streamwise_vertical_label + '_avg']*dd3_YZ['grad_velocity2_derived_avg'] # units of m^3/s^3 (ordering of gradients from AMR from 0-8 is dudx dudy dudz dvdx dvdy dvz dwdx dwdy dwdz)
            dd3_YZ['1_over_rho_u_avg_dp_dx_avg'] = (1/rho)*dd3_YZ[streamwise_velocity_label+'_avg']*dd3_YZ['grad_px_derived_avg'] # units of m^2/s^3
            dd3_YZ['coriolis_x'] = -coriolis_factor*dd3_YZ[lateral_velocity_label+'_avg']*dd3_YZ[streamwise_velocity_label+'_avg'] # units of m^2/s^3
            dd3_YZ['body_force'] = body_force[0]*dd3_YZ[streamwise_velocity_label+'_avg'] # units of m^2/s^3? (see note in the input section at the top of this ipynb)


            print("Done")

            ## calculate LHS####
            print("Calculating LHS (out)...",end='',flush=True)
            df_out = pd.DataFrame(index=streamPos/diam)

            ## mean-flow convection
            df_out = df_out.assign(P_mean=[None] * len(df_out))
            for i in range(numStreamPos):
                qoi = 'u_avg_cubed'
                val = dd3_YZ[qoi][i,:,:]
                lateral_grid  = dd3_YZ[lateral_label_YZ][i,0,:]
                vertical_grid = dd3_YZ[vertical_label_YZ][i,:,0]
                df_out.iloc[i, df_out.columns.get_loc('P_mean')]  = 0.5*self.doubleIntegral(val,vertical_grid,lateral_grid)

            ## turb transport
            df_out = df_out.assign(P_turb=[None] * len(df_out))
            for i in range(numStreamPos):
                streamwise_velocity_label = 'velocity' + streamwise_label_YZ 
                #qoi = 'u_avg_uu_avg'
                qoi = corr_mapping[streamwise_velocity_label] + '_avg_' + corr_mapping[streamwise_velocity_label] + corr_mapping[streamwise_velocity_label] + '_avg'
                val = dd3_YZ[qoi][i,:,:]
                lateral_grid  = dd3_YZ[lateral_label_YZ][i,0,:]
                vertical_grid = dd3_YZ[vertical_label_YZ][i,:,0]
                df_out.iloc[i, df_out.columns.get_loc('P_turb')]  = self.doubleIntegral(val,vertical_grid,lateral_grid)

            df_out['P'] = df_out['P_mean'] + df_out['P_turb']
            print("Done")
            #####################

            ## calculate RHS####
            print("Calculating RHS (in)...",end='',flush=True)
            df_in = pd.DataFrame(index=streamPos/diam)

            ## mean-flow convection
            # front
            df_in = df_in.assign(P_mean_fr=[None] * len(df_in))
            for i in range(numStreamPos):
                qoi = 'u_avg_cubed'
                #val = dd3_YZ[qoi][i,:,:]
                # TODO: WHY DOES KEN ONLY USE 0 HERE? 
                val = dd3_YZ[qoi][0,:,:]
                lateral_grid  = dd3_YZ[lateral_label_YZ][i,0,:]
                vertical_grid = dd3_YZ[vertical_label_YZ][i,:,0]
                df_in.iloc[i, df_in.columns.get_loc('P_mean_fr')] = 0.5*self.doubleIntegral(val,vertical_grid,lateral_grid)

            # left
            df_in = df_in.assign(P_mean_left=[None] * len(df_in))
            for i in range(numStreamPos):
                dim0_index = slice(0,i+1,1) # inner integral limits
                qoi = 'u_avg_squared_v_avg'
                val = dd3_XZ[qoi][dim0_index,:,lft_iplane]
                streamwise_grid  = dd3_XZ[streamwise_label_XZ][dim0_index,0,lft_iplane]
                vertical_grid = dd3_XZ[vertical_label_XZ][dim0_index,:,lft_iplane]
                df_in.iloc[i, df_in.columns.get_loc('P_mean_left')] = 0.5*self.doubleIntegral(val,streamwise_grid,vertical_grid)
            # right
            df_in = df_in.assign(P_mean_right=[None] * len(df_in))
            for i in range(numStreamPos):
                dim0_index = slice(0,i+1,1) # inner integral limits
                qoi = 'u_avg_squared_v_avg'
                val = dd3_XZ[qoi][dim0_index,:,rht_iplane]
                streamwise_grid  = dd3_XZ[streamwise_label_XZ][dim0_index,0,rht_iplane]
                vertical_grid = dd3_XZ[vertical_label_XZ][dim0_index,:,rht_iplane]
                df_in.iloc[i, df_in.columns.get_loc('P_mean_right')] = -0.5*self.doubleIntegral(val,streamwise_grid,vertical_grid)
            # bottom
            df_in = df_in.assign(P_mean_bot=[None] * len(df_in))
            for i in range(numStreamPos):
                dim0_index = slice(0,i+1,1) # inner integral limits
                qoi = 'u_avg_squared_w_avg'
                val = dd3_XY[qoi][dim0_index,bot_iplane,:]
                streamwise_grid  = dd3_XY[streamwise_label_XY][dim0_index,bot_iplane,0]
                lateral_grid = dd3_XY[lateral_label_XY][i,bot_iplane,:]
                df_in.iloc[i, df_in.columns.get_loc('P_mean_bot')] = 0.5*self.doubleIntegral(val,streamwise_grid,lateral_grid)

            # top
            df_in = df_in.assign(P_mean_top=[None] * len(df_in))
            for i in range(numStreamPos):
                dim0_index = slice(0,i+1,1) # inner integral limits
                qoi = 'u_avg_squared_w_avg'
                val = dd3_XY[qoi][dim0_index,top_iplane,:]
                streamwise_grid  = dd3_XY[streamwise_label_XY][dim0_index,top_iplane,0]
                lateral_grid = dd3_XY[lateral_label_XY][i,top_iplane,:]
                df_in.iloc[i, df_in.columns.get_loc('P_mean_top')] = -0.5*self.doubleIntegral(val,streamwise_grid,lateral_grid)
            ## turb transport
            # front
            df_in = df_in.assign(P_turb_fr=[None] * len(df_in))
            for i in range(numStreamPos):
                qoi = 'u_avg_uu_avg'
                #val = dd3_YZ[qoi][i,:,:]
                # TODO: WHY DOES KEN ONLY USE 0 HERE? 
                val = dd3_YZ[qoi][0,:,:]
                lateral_grid  = dd3_YZ[lateral_label_YZ][i,0,:]
                vertical_grid = dd3_YZ[vertical_label_YZ][i,:,0]
                df_in.iloc[i, df_in.columns.get_loc('P_turb_fr')] = self.doubleIntegral(val,vertical_grid,lateral_grid)

            # left
            df_in = df_in.assign(P_turb_left=[None] * len(df_in))
            for i in range(numStreamPos):
                dim0_index = slice(0,i+1,1) # inner integral limits
                qoi = 'u_avg_uv_avg'
                val = dd3_XZ[qoi][dim0_index,:,lft_iplane]
                streamwise_grid  = dd3_XZ[streamwise_label_XZ][dim0_index,0,lft_iplane]
                vertical_grid = dd3_XZ[vertical_label_XZ][i,:,lft_iplane]
                df_in.iloc[i, df_in.columns.get_loc('P_turb_left')] = self.doubleIntegral(val,streamwise_grid,vertical_grid)

            # right
            df_in = df_in.assign(P_turb_right=[None] * len(df_in))
            for i in range(numStreamPos):
                dim0_index = slice(0,i+1,1) # inner integral limits
                qoi = 'u_avg_uv_avg'
                val = dd3_XZ[qoi][dim0_index,:,rht_iplane]
                streamwise_grid  = dd3_XZ[streamwise_label_XZ][dim0_index,0,rht_iplane]
                vertical_grid = dd3_XZ[vertical_label_XZ][i,:,rht_iplane]
                df_in.iloc[i, df_in.columns.get_loc('P_turb_right')] = -self.doubleIntegral(val,streamwise_grid,vertical_grid)
                
            # bottom
            df_in = df_in.assign(P_turb_bot=[None] * len(df_in))
            for i in range(numStreamPos):
                dim0_index = slice(0,i+1,1) # inner integral limits
                qoi = 'u_avg_uw_avg'
                val = dd3_XY[qoi][dim0_index,bot_iplane,:]
                streamwise_grid  = dd3_XY[streamwise_label_XY][dim0_index,bot_iplane,0]
                lateral_grid = dd3_XY[lateral_label_XY][i,bot_iplane,:]
                df_in.iloc[i, df_in.columns.get_loc('P_turb_bot')] = self.doubleIntegral(val,streamwise_grid,lateral_grid)

            # top
            df_in = df_in.assign(P_turb_top=[None] * len(df_in))
            for i in range(numStreamPos):
                dim0_index = slice(0,i+1,1) # inner integral limits
                qoi = 'u_avg_uw_avg'
                val = dd3_XY[qoi][dim0_index,top_iplane,:]
                streamwise_grid  = dd3_XY[streamwise_label_XY][dim0_index,top_iplane,0]
                lateral_grid = dd3_XY[lateral_label_XY][i,top_iplane,:]
                df_in.iloc[i, df_in.columns.get_loc('P_turb_top')] = -self.doubleIntegral(val,streamwise_grid,lateral_grid)
                
            ## production
            df_in = df_in.assign(P_prod=[None] * len(df_in))
            for i in range(numStreamPos):
                qoi = 'P_x'
                dim0_index = slice(0,i+1,1) # inner integral limits
                streamwise_grid  = dd3_YZ[streamwise_label_YZ][dim0_index,0,0]
                lateral_grid  = dd3_YZ[lateral_label_YZ][i,0,:]
                vertical_grid = dd3_YZ[vertical_label_YZ][i,:,0]
                val = dd3_YZ[qoi][dim0_index,:,:]
                df_in.iloc[i, df_in.columns.get_loc('P_prod')] = -self.tripleIntegral(val,streamwise_grid,vertical_grid,lateral_grid)
                
            ## pressure
            df_in = df_in.assign(P_pres=[None] * len(df_in))
            for i in range(numStreamPos):
                qoi = '1_over_rho_u_avg_dp_dx_avg'
                dim0_index = slice(0,i+1,1) # inner integral limits
                streamwise_grid  = dd3_YZ[streamwise_label_YZ][dim0_index,0,0]
                lateral_grid  = dd3_YZ[lateral_label_YZ][i,0,:]
                vertical_grid = dd3_YZ[vertical_label_YZ][i,:,0]
                val = dd3_YZ[qoi][dim0_index,:,:]
                df_in.iloc[i, df_in.columns.get_loc('P_pres')] = -self.tripleIntegral(val,streamwise_grid,vertical_grid,lateral_grid)
            ## coriolis
            df_in = df_in.assign(P_cori=[None] * len(df_in))
            for i in range(numStreamPos):
                qoi = 'coriolis_x'
                dim0_index = slice(0,i+1,1) # inner integral limits
                streamwise_grid  = dd3_YZ[streamwise_label_YZ][dim0_index,0,0]
                lateral_grid  = dd3_YZ[lateral_label_YZ][i,0,:]
                vertical_grid = dd3_YZ[vertical_label_YZ][i,:,0]
                val = dd3_YZ[qoi][dim0_index,:,:]
                df_in.iloc[i, df_in.columns.get_loc('P_cori')] = self.tripleIntegral(val,streamwise_grid,vertical_grid,lateral_grid)

            ## body force
            df_in = df_in.assign(P_bodf=[None] * len(df_in))
            for i in range(numStreamPos):
                qoi = 'body_force'
                dim0_index = slice(0,i+1,1) # inner integral limits
                streamwise_grid  = dd3_YZ[streamwise_label_YZ][dim0_index,0,0]
                lateral_grid  = dd3_YZ[lateral_label_YZ][i,0,:]
                vertical_grid = dd3_YZ[vertical_label_YZ][i,:,0]
                val = dd3_YZ[qoi][dim0_index,:,:]
                df_in.iloc[i, df_in.columns.get_loc('P_bodf')] = self.tripleIntegral(val,streamwise_grid,vertical_grid,lateral_grid)

            print("Done")

            df_in['P_mean_lr'] = df_in['P_mean_left'] + df_in['P_mean_right']
            df_in['P_mean_tb'] = df_in['P_mean_top'] + df_in['P_mean_bot']
            df_in['P_mean'] = df_in['P_mean_fr'] + df_in['P_mean_lr'] + df_in['P_mean_tb']
            df_in['P_turb_lr'] = df_in['P_turb_left'] + df_in['P_turb_right']
            df_in['P_turb_tb'] = df_in['P_turb_top'] + df_in['P_turb_bot']
            df_in['P_turb'] = df_in['P_turb_fr'] + df_in['P_turb_lr'] + df_in['P_turb_tb']
            df_in['P'] = df_in['P_mean'] + df_in['P_turb'] + df_in['P_prod'] + df_in['P_pres'] + df_in['P_cori'] + df_in['P_bodf']
            df_in['P_reduced'] = df_in['P_mean'] + df_in['P_turb'] + df_in['P_prod'] + df_in['P_pres']

            self.df_in = df_in
            self.df_out = df_out
            self.Uinf = Uinf
            self.boxDimensions=boxDimensions

            if len(savepklfile)>0:
                directory, file_name = os.path.split(savepklfile)
                os.makedirs(directory, exist_ok=True)

                with open(savepklfile, 'wb') as f:
                    pickle.dump(df_in, f)
                    pickle.dump(df_out, f)

            # Do any sub-actions required for this task
            for a in self.actionlist:
                action = self.actionlist[a]
                # Check to make sure required actions are there
                if action.required and (action.actionname not in self.yamldictlist[0].keys()):
                    # This is a problem, stop things
                    raise ValueError('Required action %s not present'%action.actionname)
                if action.actionname in self.yamldictlist[iplane].keys():
                    actionitem = action(self, self.yamldictlist[iplane][action.actionname])
                    actionitem.execute()
        return 

    # --- Inner classes for action list ---
    @registeraction(actionlist)
    class print_table():
        actionname = 'table'
        blurb      = 'Print table of results from control volume analysis'
        required   = False
        actiondefs = [
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent
            print('Initialized '+self.actionname+' inside '+parent.name)
            return

        def execute(self):
            print('Executing '+self.actionname)

            sigfigs = 3
            referencePlane_xOverD = 0.1
            normalization = self.parent.Uinf**3*(self.parent.boxDimensions[1]*self.parent.boxDimensions[2])

            # rhs
            index = -1
            print()
            print("RHS")
            print("--------")
            columns_to_plot = ['P_mean', 'P_turb']
            print((self.parent.df_out[columns_to_plot].iloc[index]/normalization).apply(lambda value: round(value, -int(math.floor(math.log10(abs(value)))) + sigfigs)))
            print()

            # lhs
            index = -1
            print("LHS")
            print("--------")
            columns_to_plot = ['P_mean', 'P_turb', 'P_prod', 'P_pres', 'P_cori', 'P_bodf']
            print((self.parent.df_in[columns_to_plot].iloc[index]/normalization).apply(lambda value: round(value, -int(math.floor(math.log10(abs(value)))) + sigfigs)))
            print()


            # residual
            index = -1
            columns_to_plot = ['P']
            print("RESIDUAL")
            print("--------")
            print(((self.parent.df_out[columns_to_plot].iloc[index]-self.parent.df_in[columns_to_plot].iloc[index])/normalization).apply(lambda value: round(value, -int(math.floor(math.log10(abs(value)))) + sigfigs)))
            print()


    # --- Inner classes for action list ---
    @registeraction(actionlist)
    class plot_totals():
        actionname = 'plot_totals'
        blurb      = 'Plot totals from control volume'
        required   = False
        actiondefs = [
            {'key':'savefile','required':False,'default':None,'help':'filename to save plot'},
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent
            print('Initialized '+self.actionname+' inside '+parent.name)
            return

        def execute(self):
            print('Executing '+self.actionname)
            savefilename = self.actiondict['savefile']

            sigfigs = 3
            referencePlane_xOverD = 0.1
            normalization = self.parent.Uinf**3*(self.parent.boxDimensions[1]*self.parent.boxDimensions[2])

            fig, axs = plt.subplots(1,figsize=(12,7), sharex=True)
            plt.plot((self.parent.df_out['P']-self.parent.df_out['P'][referencePlane_xOverD])/normalization,label='$\phi_{out,total}$')
            plt.plot((self.parent.df_in['P']-self.parent.df_in['P'][referencePlane_xOverD])/normalization,label='$\phi_{in,total}$')
            plt.xlabel('$x/D$ [-]')
            plt.ylabel('$\phi \; U_{\inf}^{-3} \; D^{-2}$ [-]')
            plt.legend()
            fig.savefig(savefilename)

    @registeraction(actionlist)
    class plot_contributions():
        actionname = 'plot_contributions'
        blurb      = 'Plot contributions from control volume'
        required   = False
        actiondefs = [
            {'key':'savefile','required':False,'default':None,'help':'filename to save plot'},
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent
            print('Initialized '+self.actionname+' inside '+parent.name)
            return

        def execute(self):
            print('Executing '+self.actionname)
            savefilename = self.actiondict['savefile']

            sigfigs = 3
            referencePlane_xOverD = 0.1
            normalization = self.parent.Uinf**3*(self.parent.boxDimensions[1]*self.parent.boxDimensions[2])
            fsize=16

            legend_labels = ['$\phi_{mean, left}$', '$\phi_{mean, right}$', '$\phi_{mean, top}$', '$\phi_{mean, bot}$', '$\phi_{turb., left}$', '$\phi_{turb., right}$', '$\phi_{turb., top}$', '$\phi_{turb., bot}$', '$\phi_{prod.}$', '$\phi_{pres.}$']
            legend2a_labels = ['$\phi_{total}$']

            blue_shades = [(0.2, 0.4, 0.6), (0.3, 0.5, 0.7), (0.4, 0.6, 0.8), (0.5, 0.7, 0.9)]#, (0.6, 0.8, 1.0)]
            orange_shades = [(1.0, 0.5, 0.1), (1.0, 0.6, 0.2), (1.0, 0.7, 0.3), (1.0, 0.8, 0.4)]#, (1.0, 0.9, 0.5)]
            red_shade = [(0.6, 0.2, 0.0)]
            gray_shade = [(135/256,142/256,151/256)]

            df = (self.parent.df_in.sub(self.parent.df_in.loc[referencePlane_xOverD]))/normalization

            fig, ax1 = plt.subplots(figsize = (20,7),dpi=125)
            ax2 = plt.twiny(ax = ax1)
            indices_to_plot = np.where((df.index % 1 == 0) | (df.index % 1 == 0.25)| (df.index % 1 == 0.5) | (df.index % 1 == 0.75))[0]
            columns_to_plot = ['P_mean_left', 'P_mean_right', 'P_mean_top', 'P_mean_bot', 'P_turb_left', 'P_turb_right', 'P_turb_top', 'P_turb_bot', 'P_prod', 'P_pres'] # , 'P_cori', 'P_bodf'
            df.iloc[indices_to_plot].plot(y=columns_to_plot, kind='bar', color=blue_shades + orange_shades + red_shade + gray_shade, stacked=True, ax = ax1, label=legend_labels)
            df.iloc[indices_to_plot]['P_reduced'].plot(secondary_y=False, ax=ax2, color='black')
            ax1.axhline(y = 0, color = (0.5,0.5,0.5), linestyle = '-', zorder = -1)
            handles, labels = ax1.get_legend_handles_labels()
            order = [7,6,5,4,3,2,1,0,8,9]
            ax1.set_ylim([-0.1, 0.2])
            ax2.set_ylim([-0.1, 0.2])
            #ax1.yaxis.set_tick_params(labelsize=fsize) 
            ax1.legend([handles[idx] for idx in order],[labels[idx] for idx in order],bbox_to_anchor=(1, 1.15),fontsize=fsize)
            ax2.legend(legend2a_labels, loc=2,fontsize=fsize)
            spacing = df.iloc[indices_to_plot].index[1]-df.iloc[indices_to_plot].index[0]
            ax2.set_xlim([df.iloc[indices_to_plot].index[0]-spacing/2, df.iloc[indices_to_plot].index[-1] + spacing/2])
            ax2.xaxis.set_tick_params(labeltop=False, top=False)
            xticks = ax1.get_xticks()
            xticklabels = ['' if i % 2 == 0 else str((i+1)/4) for i in xticks]
            ax1.set_xticklabels(xticklabels,fontsize=fsize)
            ax1.set_xlabel('$x/D$ [-]', labelpad=15,fontsize=fsize)
            ax1.text(-0.1, 0.65, 'Gain', transform=ax1.transAxes, rotation=90, va='center',fontsize=fsize)
            ax1.text(-0.1, 0.17, 'Loss', transform=ax1.transAxes, rotation=90, va='center',fontsize=fsize)
            #plt.gcf().set_size_inches(13, 8)
            fig.savefig(savefilename)
