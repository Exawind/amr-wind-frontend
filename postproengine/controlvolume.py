# Get the location where this script is being run
import sys, os
scriptpath = os.path.dirname(os.path.realpath(__file__))
basepath   = os.path.dirname(scriptpath)
# Add any possible locations of amr-wind-frontend here
amrwindfedirs = ['../',
                 basepath]
for x in amrwindfedirs: sys.path.insert(1, x)

from postproengine import registerplugin, mergedicts, registeraction, contourplottemplate
from postproengine import compute_axis1axis2axis3_coords,get_mapping_xyz_to_axis1axis2

from postproengine import extract_1d_from_meshgrid
import postproamrwindsample_xarray as ppsamplexr
import postproamrwindsample as ppsample
from postproengine import interpolatetemplate, circavgtemplate 
from postproengine import compute_axis1axis2_coords
import pickle
import re
import numpy as np
import scipy.interpolate as si
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

        {'key':'inflow_velocity_XYZ','required':True,'default':None,'help':'Inflow velocity from incflo.velocity setting in AMR-Wind (XYZ coordinates)',},
        {'key':'diam','required':True,'default':None,'help':'Turbine diameter',},
        {'key':'box_center_XYZ','required':False,'default':None,'help':'Center of control volume (XYZ coordinates)',},
        {'key':'box_fr_center_XYZ','required':False,'default':None,'help':'Center of control volume on front face (XYZ coordinates)',},
        {'key':'box_fr_streamwise_offset','required':False,'default':0,'help':'Streamwise offset when specifying front face of control volume, in turbine diameters',},
        {'key':'rho','required':True,'default':1.25,'help':'Density',},
        {'key':'latitude','required':True,'default':0,'help':'Latitude',},
        {'key':'body_force_XYZ','required':True,'default':None,'help':'Body force from AMR-Wind input file (XYZ coordinates)',},
        {'key':'streamwise_box_size','required':True,'default':None,'help':'Streamwise dimension of control volume in turbine diameters ',},
        {'key':'lateral_box_size','required':True,'default':None,'help':'Lateral dimension of control volume in turbine diameters ',},
        {'key':'vertical_box_size','required':True,'default':None,'help':'Vertical dimension of control volume in turbine diameters ',},
        {'key':'compute_pressure_gradient','required':False,'default':True,'help':'To approximate the streamwise pressure gradient based on finite different between streamwise planes',},
        {'key':'bot_avg_file','required':True,'default':'','help':'Bot avg pkl file',},
        {'key':'bot_rs_file','required':True,'default':'','help':'Bot rs pkl file',},
        {'key':'varnames','required':False,'default':['velocityx', 'velocityy', 'velocityz'],'help':'Variable names ',},
        {'key':'top_avg_file','required':True,'default':'','help':'top avg pkl file',},
        {'key':'top_rs_file','required':True,'default':'','help':'top rs pkl file',},
        {'key':'lft_avg_file','required':True,'default':'','help':'lft avg pkl file',},
        {'key':'lft_rs_file','required':True,'default':'','help':'lft rs pkl file',},
        {'key':'rht_avg_file','required':True,'default':'','help':'rht avg pkl file',},
        {'key':'rht_rs_file','required':True,'default':'','help':'rht rs pkl file',},
        {'key':'streamwise_avg_files','required':True,'default':'','help':'streamwise avg pkl files',},
        {'key':'streamwise_rs_files','required':True,'default':'','help':'streamwise rs pkl files',},
        {'key':'savepklfile', 'required':False,  'default':'',
        'help':'Name of pickle file to save results', },

    ]
    actionlist = {}                    # Dictionary for holding sub-actions
    example = """
controlvolume:
  name: Streamwise CV
  inflow_velocity_XYZ: [9.03,0,0]
  diam: 240
  latitude: 39.55
  box_center_XYZ: [3000.0,1000.0,150.0]
  #box_fr_center_XYZ: [2280.0,1000.0,150.0]
  #box_fr_streamwise_offset: 0
  streamwise_box_size: 6
  lateral_box_size: 1
  vertical_box_size: 1
  rho: 1.2456
  body_force_XYZ: [0.00014295185866400572, 0.0008354682029301641, 0.0]
  varnames: ['velocitya1','velocitya2','velocitya3']
  compute_pressure_gradient: True
  savepklfile: 'Control_Volume_a123.pkl'
  #varnames: ['velocityx','velocityy','velocityz']

  bot_avg_file: '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_avg_XY.pkl'
  bot_rs_file:  '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_rs_XY.pkl'

  top_avg_file: '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_avg_XY.pkl'
  top_rs_file:  '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_rs_XY.pkl'

  lft_avg_file: '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_avg_XZl.pkl'
  lft_rs_file:  '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_rs_XZl.pkl'

  rht_avg_file: '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_avg_XZr.pkl'
  rht_rs_file:  '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_rs_XZr.pkl'

  streamwise_avg_files:
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_avg_YZwake1.pkl'
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_avg_YZwake2.pkl'
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_avg_YZwake3.pkl'
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_avg_YZwake4.pkl'
  streamwise_rs_files:
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_rs_YZwake1.pkl'
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_rs_YZwake2.pkl'
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_rs_YZwake3.pkl'
    - '/nscratch/kbrown1/Advanced_Control/AMR/Turbine_Runs/One_Turb/MedWS_LowTI/postpro/Data/Baseline_test_pp/postpro_rs_YZwake4.pkl'

  print_table:

  plot_totals:
    savefile: 'test_cv_total_a1a2a3.png'

  plot_contributions:
    savefile: 'test_cv_contributions_a1a2a3.png'
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

                compute_axis1axis2axis3_coords(d,0)
                R = get_mapping_xyz_to_axis1axis2(d['axis1'],d['axis2'],d['axis3'],rot=0)
                axis_info['axis1'] = R[0,:]
                axis_info['axis2'] = R[1,:]
                axis_info['axis3'] = R[2,:]
                origina1a2a3 = R@d['origin']

                d['a1'] = d['a1'] + origina1a2a3[0]
                d['a2'] = d['a2'] + origina1a2a3[1]
                d['a3'] = d['a3'] + origina1a2a3[2]
                
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

    def load_avg_rs_files(self,avg_file,rs_file):

        dd2_avg, axis_info = self.loadpkl(avg_file)
        dd2_rs, _  = self.loadpkl(rs_file)
        dd2 = self.Merge(dd2_avg,dd2_rs)

        #unique in the direction of offset
        slice_result = dd2['a3'][:, 0, 0] 
        sortedd, newOrder = np.unique(slice_result, return_index=True)
        for index, key in enumerate(dd2.keys()):
            dd2[key] = dd2[key][newOrder, :, :]

        return dd2, axis_info

    def permute_data(self,dd3,permutation):
        for index, key in enumerate(dd3.keys()):
            if np.array(dd3[key]).ndim == 3:
                dd3[key] = np.transpose(dd3[key],permutation)
                #dd3[key] = np.squeeze(dd3[key][indicesWithinRange,:,:])
        return dd3

    def rotate_data(self,quant,axis,axis_info):
        axis_number1 = re.search(r'\d+', axis[0]).group()
        axis_number2 = re.search(r'\d+', axis[1]).group()
        axis_number3 = re.search(r'\d+', axis[2]).group()
        R = get_mapping_xyz_to_axis1axis2(axis_info['axis'+axis_number1],axis_info['axis'+axis_number2],axis_info['axis'+axis_number3],rot=0)
        return R@quant

    def flow_aligned_gradient(self,dpdx,dpdy,dpdz,axis,axis_info):
        axis_number1 = re.search(r'\d+', axis[0]).group()
        axis_number2 = re.search(r'\d+', axis[1]).group()
        axis_number3 = re.search(r'\d+', axis[2]).group()


        # transformation from x,y,z to streamwise, vertical, lateral direction
        #[as,av,al] = R[x,y,z]
        #ai = Rij xj
        R = get_mapping_xyz_to_axis1axis2(axis_info['axis'+axis_number1],axis_info['axis'+axis_number2],axis_info['axis'+axis_number3],rot=0)

        Rinv = np.linalg.inv(R) #jacobian Rji^{-1} = dxj/dai

        #dp/dai = dp/dxj * dxj/dai
        #       = dp/dxj * Rji^{-1} 

        #gradients in axis[0] direction -- (streamwise)
        dpda0 = dpdx * Rinv[0,0] + dpdy * Rinv[1,0] + dpdz * Rinv[2,0] 

        #gradients in axis[1] direction -- (vertical)
        dpda1 = dpdx * Rinv[0,1] + dpdy * Rinv[1,1] + dpdz * Rinv[2,1] 

        #gradients in axis[2] direction -- (lateral)
        dpda2 = dpdx * Rinv[0,2] + dpdy * Rinv[1,2] + dpdz * Rinv[2,2] 

        return dpda0,dpda1,dpda2

    def interpolate_axis(self,dd3_coarse,xvec,xnew,axis):
        dd3 = {}
        interp_method ='linear'
        for index, key in enumerate(dd3_coarse.keys()):
            if np.array(dd3_coarse[key]).ndim == 3:
                arr = dd3_coarse[key]
                if axis == 0:
                    dd3[key] = np.zeros((len(xnew),arr.shape[1],arr.shape[2]))
                    for j in range(dd3_coarse[key].shape[1]):
                        for k in range(dd3_coarse[key].shape[2]):
                            if np.all(np.isin(xnew, xvec)):
                                dd3[key][:,j,k] = arr[np.searchsorted(xvec, xnew),j,k]
                            else:
                                dd3[key][:,j,k] = si.interpn((xvec,),arr[:,j,k],xnew,bounds_error=False,method=interp_method,fill_value=None) 
                                #dd3[key][:,j,k] = np.interp(xnew,xvec,arr[:,j,k])
                elif axis == 1:
                    dd3[key] = np.zeros((arr.shape[0],len(xnew),arr.shape[2]))
                    for j in range(dd3_coarse[key].shape[0]):
                        for k in range(dd3_coarse[key].shape[2]):
                            if np.all(np.isin(xnew, xvec)):
                                dd3[key][j,:,k] = arr[j,np.searchsorted(xvec, xnew),k]
                            else:
                                dd3[key][j,:,k] = si.interpn((xvec,),arr[j,:,k],xnew,bounds_error=False,method=interp_method,fill_value=None) 
                                #dd3[key][j,:,k] = np.interpn(xnew,xvec,arr[j,:,k])
                elif axis == 2:
                    dd3[key] = np.zeros((arr.shape[0],arr.shape[1],len(xnew)))
                    for j in range(dd3_coarse[key].shape[0]):
                        for k in range(dd3_coarse[key].shape[1]):
                            if np.all(np.isin(xnew, xvec)):
                                dd3[key][j,k,:] = arr[j,k,np.searchsorted(xvec, xnew)]
                            else:
                                dd3[key][j,k,:] = si.interpn((xvec,),arr[j,k,:],xnew,bounds_error=False,method=interp_method,fill_value=None) 
                                #dd3[key][j,k,:] = np.interpn(xnew,xvec,arr[j,k,:])
        return dd3

    def add_control_volume_boundary(self,dd3,xvec,axis,boxCenter,boxDimensions):
        start = boxCenter[axis] - boxDimensions[axis]/2.0
        end   = boxCenter[axis] + boxDimensions[axis]/2.0
        xvecorg = np.copy(xvec)

        interp = False
        if (start not in xvec): 
            start_index = np.searchsorted(xvec,start)
            xvec = np.insert(xvec, start_index, start)
            interp = True

        if (end not in xvec): 
            end_index = np.searchsorted(xvec,end)
            xvec = np.insert(xvec, end_index, end)
            interp = True

        if interp:
            dd3 = self.interpolate_axis(dd3,xvecorg,xvec,axis)

        return dd3

    def get_control_volume_boundaries(self,grid,boxCenter,boxDimensions):
        minval = boxCenter - boxDimensions/2.0
        maxval = boxCenter + boxDimensions/2.0
        minind = np.argmin(abs(grid-minval))
        maxind = np.argmin(abs(grid-maxval))
        return minind,maxind

    def update_control_volume_center_dims(self,s_grid,v_grid,l_grid,boxCenter,boxDimensions):
        s_min , s_max = self.get_control_volume_boundaries(s_grid,boxCenter[0],boxDimensions[0])
        v_min , v_max = self.get_control_volume_boundaries(v_grid,boxCenter[1],boxDimensions[1])
        l_min,  l_max = self.get_control_volume_boundaries(l_grid,boxCenter[2],boxDimensions[2])

        boxCenter = np.zeros(3)
        boxCenter[0] = (s_grid[s_min] + s_grid[s_max])/2.0
        boxCenter[1] = (v_grid[v_min] + v_grid[v_max])/2.0
        boxCenter[2] = (l_grid[l_min] + l_grid[l_max])/2.0

        boxDimensions = np.zeros(3)
        boxDimensions[0] = s_grid[s_max] - s_grid[s_min]
        boxDimensions[1] = v_grid[v_max] - v_grid[v_min]
        boxDimensions[2] = l_grid[l_max] - l_grid[l_min]

        return boxCenter, boxDimensions

    def crop_data_axis(self,dd2,axis_label,axis,boxCenter,boxDimensions):
        if axis == 0:
            mask = (dd2[axis_label][:,0,0] >= ((boxCenter[0]-boxDimensions[0]/2.0))) & (dd2[axis_label][:,0,0] <= ((boxCenter[0]+boxDimensions[0]/2.0)))
        elif axis == 1:
            mask = (dd2[axis_label][0,:,0] >= ((boxCenter[1]-boxDimensions[1]/2.0))) & (dd2[axis_label][0,:,0] <= ((boxCenter[1]+boxDimensions[1]/2.0)))
        elif axis == 2:
            mask = (dd2[axis_label][0,0,:] >= ((boxCenter[2]-boxDimensions[2]/2.0))) & (dd2[axis_label][0,0,:] <= ((boxCenter[2]+boxDimensions[2]/2.0)))

        mask_ind = np.argwhere(mask)[:,0]

        dd3 = {}
        for index, key in enumerate(dd2.keys()):
            if np.array(dd2[key]).ndim == 3:
                if axis == 0:
                    dd3[key] = dd2[key][mask,:,:]
                if axis == 1:
                    dd3[key] = dd2[key][:,mask,:]
                if axis == 2:
                    dd3[key] = dd2[key][:,:,mask]

        return dd3

    def crop_data(self,dd2,axis,axis_info,boxCenter,boxDimensions):
        streamwise_mask = (dd2[axis[0]][:,0,0] >= ((boxCenter[0]-boxDimensions[0]/2.0))) & (dd2[axis[0]][:,0,0] <= ((boxCenter[0]+boxDimensions[0]/2.0)))
        vertical_mask   = (dd2[axis[1]][0,:,0] >= ((boxCenter[1]-boxDimensions[1]/2.0))) & (dd2[axis[1]][0,:,0] <= ((boxCenter[1]+boxDimensions[1]/2.0)))
        lateral_mask    = (dd2[axis[2]][0,0,:] >= ((boxCenter[2]-boxDimensions[2]/2.0))) & (dd2[axis[2]][0,0,:] <= ((boxCenter[2]+boxDimensions[2]/2.0)))

        streamwise_indices = np.argwhere(streamwise_mask)[:,0]
        vertical_indices   = np.argwhere(vertical_mask)[:,0]
        lateral_indices    = np.argwhere(lateral_mask)[:,0]

        streamwise_indices = streamwise_indices[:, np.newaxis, np.newaxis]  
        vertical_indices = vertical_indices[np.newaxis, :, np.newaxis]     
        lateral_indices = lateral_indices[np.newaxis, np.newaxis, :]

        dd3 = {}
        for index, key in enumerate(dd2.keys()):
            if np.array(dd2[key]).ndim == 3:
                dd3[key] = dd2[key][streamwise_indices,vertical_indices,lateral_indices]

        return dd3

        axis_number1 = re.search(r'\d+', axis[0]).group()
        axis_number2 = re.search(r'\d+', axis[1]).group()
        axis_number3 = re.search(r'\d+', axis[2]).group()
        axis3_label = 'a3'
        if int(axis_number1) == 3: 
            axis3_ind   = 0
            dd2_slice = dd2[axis3_label][:, 0, 0] 
            dd3_slice = dd3[axis3_label][:, 0, 0] 
        if int(axis_number2) == 3: 
            axis3_ind   = 1
            dd2_slice = dd2[axis3_label][0, :, 0] 
            dd3_slice = dd3[axis3_label][0, :, 0] 
        if int(axis_number3) == 3: 
            axis3_ind   = 2
            dd2_slice = dd2[axis3_label][0, 0, :] 
            dd3_slice = dd3[axis3_label][0, 0, :] 

        #crop offset direction for dd2
        _, indices_dd2, _ = np.intersect1d(dd2_slice, dd3_slice,return_indices=True)
        for index, key in enumerate(dd2.keys()):
            if int(axis_number1) == 3: 
                dd2[key] = dd2[key][indices_dd2,:,:]
            if int(axis_number2) == 3: 
                dd2[key] = dd2[key][:,indices_dd2,:]
            if int(axis_number3) == 3: 
                dd2[key] = dd2[key][:,:,indices_dd2]

        return dd2,dd3

    def are_aligned(self,v1, v2):
        return np.all(np.cross(v1, v2) == 0)

    def get_label_and_ind_in_dir(self,axis_info,axis_dir,dd):
        n1 = axis_info['axis1']/np.linalg.norm(axis_info['axis1'])
        n2 = axis_info['axis2']/np.linalg.norm(axis_info['axis2'])
        n3 = axis_info['axis3']/np.linalg.norm(axis_info['axis3'])

        #ijk dims flipped 
        if self.are_aligned(n1,axis_dir): 
            axis_ind = 2
            axis_label = 'a1'
        elif self.are_aligned(n2,axis_dir): 
            axis_ind = 1
            axis_label = 'a2'
        elif self.are_aligned(n3,axis_dir): 
            axis_ind = 0
            axis_label = 'a3'
        else:
            print("Error: Domain not aligned...exiting")
            sys.exit()

        return axis_label , axis_ind

    def sort_rs_labels(self,string1,string2):
        strings = (string1, string2)
        sorted_strings = sorted(strings)
        result = ''.join(sorted_strings)
        return result

    def execute(self, verbose=False):
        if verbose: print('Running '+self.name)
        # Loop through and create plots
        for iplane, plane in enumerate(self.yamldictlist):
            inflow_velocity_XYZ = np.asarray(plane['inflow_velocity_XYZ'])
            diam = plane['diam']
            rho = plane['rho']
            latitude = plane['latitude']
            compute_pressure_gradient = plane['compute_pressure_gradient']
            #axis = plane['axis']

            streamwise_box_size = plane['streamwise_box_size']
            lateral_box_size = plane['lateral_box_size']
            vertical_box_size = plane['vertical_box_size']
            boxDimensions = np.asarray([streamwise_box_size,vertical_box_size,lateral_box_size])
            boxDimensions*=diam
            varnames = plane['varnames']            
            body_force_XYZ = np.asarray(plane['body_force_XYZ'])
            savepklfile = plane['savepklfile']
            boxCenter_XYZ = np.asarray(plane['box_center_XYZ'])

            front_specified = False
            if None in boxCenter_XYZ:
                boxFrCenter_XYZ = np.asarray(plane['box_fr_center_XYZ'])
                boxFrOffset     = plane['box_fr_streamwise_offset']*diam
                if None in boxFrCenter_XYZ:
                    print("Error: Must specify box center coordinates...exiting")
                    sys.exit()
                else:
                    front_specified = True

            corr_mapping = {
                'velocityx': 'u',
                'velocityy': 'v',
                'velocityz': 'w',
                'velocitya1': 'ua1',
                'velocitya2': 'ua2',
                'velocitya3': 'ua3'
            }

            print("Loading streamwise flow planes...",end='',flush=True)
            x_avg_files   = plane['streamwise_avg_files']            
            x_rs_files    = plane['streamwise_rs_files']            
            dd2_YZ_coarse,axis_info_YZ  = self.load_avg_rs_files(x_avg_files,x_rs_files)
            print("Done")

            print("Loading vertical flow planes...",end='',flush=True)
            bot_avg_file = plane['bot_avg_file']
            bot_rs_file  = plane['bot_rs_file']
            bot_iplane   = 0 #domain will get cropped so this is always 0

            top_avg_file = plane['top_avg_file']            
            top_rs_file  = plane['top_rs_file']            
            top_iplane   = -1 #domain will get cropped so this is always -1

            dd2_XY,axis_info_XY  = self.load_avg_rs_files([bot_avg_file,top_avg_file],[bot_rs_file,top_rs_file])
            print("Done")

            print("Loading lateral flow planes...",end='',flush=True)
            lft_avg_file = plane['lft_avg_file']            
            lft_rs_file  = plane['lft_rs_file']            
            lft_iplane   = 0 #domain will get cropped so this is always 0

            rht_avg_file = plane['rht_avg_file']            
            rht_rs_file  = plane['rht_rs_file']            
            rht_iplane   = -1 #domain will get cropped so this is always -1

            dd2_XZ,axis_info_XZ  = self.load_avg_rs_files([lft_avg_file,rht_avg_file],[lft_rs_file,rht_rs_file])
            print("Done")

            streamwise_dir = axis_info_YZ['axis3'] #Define streamwise direction as normal to YZ data
            vertical_dir   = axis_info_XY['axis3'] #Define vertical direction as normal to XY data
            lateral_dir    = axis_info_XZ['axis3'] #Define lateral direction as normal to XZ data

            #determine labels in streamwise direction
            streamwise_label_YZ , streamwise_ind_YZ = self.get_label_and_ind_in_dir(axis_info_YZ,streamwise_dir,dd2_YZ_coarse)
            streamwise_label_XY , streamwise_ind_XY = self.get_label_and_ind_in_dir(axis_info_XY,streamwise_dir,dd2_XY)
            streamwise_label_XZ , streamwise_ind_XZ = self.get_label_and_ind_in_dir(axis_info_XZ,streamwise_dir,dd2_XZ)

            #determine labels in vertical direction
            vertical_label_YZ , vertical_ind_YZ = self.get_label_and_ind_in_dir(axis_info_YZ,vertical_dir,dd2_YZ_coarse)
            vertical_label_XY , vertical_ind_XY = self.get_label_and_ind_in_dir(axis_info_XY,vertical_dir,dd2_XY)
            vertical_label_XZ , vertical_ind_XZ = self.get_label_and_ind_in_dir(axis_info_XZ,vertical_dir,dd2_XZ)

            #determine labels in lateral direction
            lateral_label_YZ , lateral_ind_YZ = self.get_label_and_ind_in_dir(axis_info_YZ,lateral_dir,dd2_YZ_coarse)
            lateral_label_XY , lateral_ind_XY = self.get_label_and_ind_in_dir(axis_info_XY,lateral_dir,dd2_XY)
            lateral_label_XZ , lateral_ind_XZ = self.get_label_and_ind_in_dir(axis_info_XZ,lateral_dir,dd2_XZ)

            axis_YZ = [streamwise_label_YZ,vertical_label_YZ,lateral_label_YZ]
            axis_XY = [streamwise_label_XY,vertical_label_XY,lateral_label_XY]
            axis_XZ = [streamwise_label_XZ,vertical_label_XZ,lateral_label_XZ]

            #permute data to streamwise, vertical, lateral
            print("Permuting data to streamwise, vertical, lateral...",end='',flush=True)

            permutation = (streamwise_ind_XY,vertical_ind_XY,lateral_ind_XY)
            dd2_XY = self.permute_data(dd2_XY,permutation)

            permutation = (streamwise_ind_XZ,vertical_ind_XZ,lateral_ind_XZ)
            dd2_XZ = self.permute_data(dd2_XZ,permutation)

            permutation = (streamwise_ind_YZ,vertical_ind_YZ,lateral_ind_YZ)
            dd2_YZ_coarse = self.permute_data(dd2_YZ_coarse,permutation)

            body_force_YZ = self.rotate_data(body_force_XYZ,axis_YZ,axis_info_YZ)
            body_force_XZ = self.rotate_data(body_force_XYZ,axis_XZ,axis_info_XZ)
            body_force_XY = self.rotate_data(body_force_XYZ,axis_XY,axis_info_XY)

            inflow_velocity_YZ = self.rotate_data(inflow_velocity_XYZ,axis_YZ,axis_info_YZ)
            inflow_velocity_XZ = self.rotate_data(inflow_velocity_XYZ,axis_XZ,axis_info_XZ)
            inflow_velocity_XY = self.rotate_data(inflow_velocity_XYZ,axis_XY,axis_info_XY)

            #See: https://github.com/Exawind/amr-wind/blob/f336699e709f4c4c4255adea3c86d2203a3ddd54/amr-wind/equation_systems/icns/source_terms/CoriolisForcing.cpp
            rot_time_period = 86164.091
            omega = 2 * np.pi / rot_time_period
            corfac = 2.0 * omega
            rad_latitude = latitude * (np.pi/180)
            sinphi = np.sin(rad_latitude)
            cosphi = np.sin(rad_latitude)

            #check if inflow velocity is horizontal
            if abs(inflow_velocity_XYZ[2]<=1e-11):
                fac = 0.0
            else:
                fac = 1.0
                print("Warning: Vertical coriolis forces may not be accounted for")

            ax = corfac * (inflow_velocity_XYZ[1] * fac * inflow_velocity_XYZ[2] * cosphi)
            ay = -corfac * inflow_velocity_XYZ[0] * sinphi
            az = fac * corfac * inflow_velocity_XYZ[0] * cosphi
            coriolis_forcing_XYZ = np.array([ax,ay,az]) 
            coriolis_forcing_YZ = self.rotate_data(coriolis_forcing_XYZ,axis_YZ,axis_info_YZ)
            coriolis_forcing_XZ = self.rotate_data(coriolis_forcing_XYZ,axis_XZ,axis_info_XZ)
            coriolis_forcing_XY = self.rotate_data(coriolis_forcing_XYZ,axis_XY,axis_info_XY)

            if front_specified:
                boxFrCenter = self.rotate_data(boxFrCenter_XYZ,axis_YZ,axis_info_YZ)
                boxCenter = boxFrCenter
                boxCenter[0] = boxFrCenter[0] + boxDimensions[0]/2.0 + boxFrOffset
            else:
                boxCenter = self.rotate_data(boxCenter_XYZ,axis_YZ,axis_info_YZ)

            if not compute_pressure_gradient:
                dpdx = dd2_YZ_coarse['grad_px_avg']
                dpdy = dd2_YZ_coarse['grad_py_avg']
                dpdz = dd2_YZ_coarse['grad_pz_avg']
                dpds_YZ,_ ,_ = self.flow_aligned_gradient(dpdx,dpdy,dpdz,axis_YZ,axis_info_YZ)
                dd2_YZ_coarse['grad_px_derived_avg'] = dpds_YZ
            print("Done")

            #Get nearest control volume in flow-aligned coordinates based on available planes
            print("Geting data inside control volume...",end='',flush=True)
            boxCenter, boxDimensions = self.update_control_volume_center_dims(dd2_YZ_coarse['a3'][:,0,0],dd2_XY['a3'][0,:,0],dd2_XZ['a3'][0,0,:],boxCenter,boxDimensions)

            #Refine streamwise grid
            xvec_YZ = dd2_YZ_coarse[streamwise_label_YZ][:,0,0] 
            xvec_XZ = dd2_XZ[streamwise_label_XZ][:,0,0]
            xvec_XY = dd2_XY[streamwise_label_XY][:,0,0]
            dx_XZ = xvec_XZ[1] - xvec_XZ[0]
            dx_XY = xvec_XY[1] - xvec_XY[0]
            dx = max(dx_XZ,dx_XY) 
            xmin = boxCenter[0] - boxDimensions[0]/2.0
            xmax = boxCenter[0] + boxDimensions[0]/2.0
            xnew = np.arange(xmin-dx,xmax+dx+dx,dx)

            dd3_YZ = dd2_YZ_coarse
            dd3_XZ = dd2_XZ
            dd3_XY = dd2_XY

            #Check vertical control volume boundaries and
            #interpolate if necessary
            zvec_YZ = dd3_YZ[vertical_label_YZ][0,:,0] 
            zvec_XZ = dd3_XZ[vertical_label_XZ][0,:,0]
            zvec_XY = dd3_XY[vertical_label_XY][0,:,0]
            dd3_YZ = self.add_control_volume_boundary(dd3_YZ,zvec_YZ,1,boxCenter,boxDimensions)
            dd3_XZ = self.add_control_volume_boundary(dd3_XZ,zvec_XZ,1,boxCenter,boxDimensions)
            dd3_XY = self.add_control_volume_boundary(dd3_XY,zvec_XY,1,boxCenter,boxDimensions)

            #Compute vertical velocity gradients
            if not any('velocitya' in v for v in varnames):
                streamwise_flow_YZ = 'x'
            else:
                streamwise_flow_YZ = streamwise_label_YZ
            streamwise_velocity_label = 'velocity' + streamwise_flow_YZ + '_avg'
            dd3_YZ['grad_velocity2_derived_avg'] = np.gradient(dd3_YZ[streamwise_velocity_label],dd3_YZ[vertical_label_YZ][0,:,0],axis=1)

            #Crop data in vertical direction
            dd3_YZ = self.crop_data_axis(dd3_YZ,vertical_label_YZ,1,boxCenter,boxDimensions)
            dd3_XZ = self.crop_data_axis(dd3_XZ,vertical_label_XZ,1,boxCenter,boxDimensions)
            dd3_XY = self.crop_data_axis(dd3_XY,vertical_label_XY,1,boxCenter,boxDimensions)

            #Check lateral control volume boundaries and
            #interpolate if necessary
            yvec_YZ = dd3_YZ[lateral_label_YZ][0,0,:] 
            yvec_XZ = dd3_XZ[lateral_label_XZ][0,0,:]
            yvec_XY = dd3_XY[lateral_label_XY][0,0,:]
            dd3_YZ = self.add_control_volume_boundary(dd3_YZ,yvec_YZ,2,boxCenter,boxDimensions)
            dd3_XZ = self.add_control_volume_boundary(dd3_XZ,yvec_XZ,2,boxCenter,boxDimensions)
            dd3_XY = self.add_control_volume_boundary(dd3_XY,yvec_XY,2,boxCenter,boxDimensions)

            #Compute lateral velocity gradients
            if not any('velocitya' in v for v in varnames):
                streamwise_flow_YZ = 'x'
            else:
                streamwise_flow_YZ = streamwise_label_YZ
            streamwise_velocity_label = 'velocity' + streamwise_flow_YZ + '_avg'
            dd3_YZ['grad_velocity1_derived_avg'] = np.gradient(dd3_YZ[streamwise_velocity_label],dd3_YZ[lateral_label_YZ][0,0,:],axis=2)

            #Crop data in vertical directions
            dd3_YZ = self.crop_data_axis(dd3_YZ,lateral_label_YZ,2,boxCenter,boxDimensions)
            dd3_XZ = self.crop_data_axis(dd3_XZ,lateral_label_XZ,2,boxCenter,boxDimensions)
            dd3_XY = self.crop_data_axis(dd3_XY,lateral_label_XY,2,boxCenter,boxDimensions)

            #Interpolate all planes to same streamwise grid
            dd3_YZ = self.interpolate_axis(dd3_YZ,xvec_YZ,xnew,axis=0)
            dd3_XZ = self.interpolate_axis(dd3_XZ,xvec_XZ,xnew,axis=0)
            dd3_XY = self.interpolate_axis(dd3_XY,xvec_XY,xnew,axis=0)

            #Compute streamwise gradients
            if not any('velocitya' in v for v in varnames):
                streamwise_flow_YZ = 'x'
            else:
                streamwise_flow_YZ = streamwise_label_YZ
            streamwise_velocity_label = 'velocity' + streamwise_flow_YZ + '_avg'
            dd3_YZ['grad_velocity0_derived_avg'] = np.gradient(dd3_YZ[streamwise_velocity_label],dd3_YZ[streamwise_label_YZ][:,0,0],axis=0)
            if compute_pressure_gradient:
                dd3_YZ['grad_px_derived_avg'] = np.gradient(dd3_YZ['p_avg'],dd3_YZ[streamwise_label_YZ][:,0,0],axis=0)

            #Crop data in streamwise direction
            dd3_YZ = self.crop_data_axis(dd3_YZ,streamwise_label_YZ,0,boxCenter,boxDimensions)
            dd3_XZ = self.crop_data_axis(dd3_XZ,streamwise_label_XZ,0,boxCenter,boxDimensions)
            dd3_XY = self.crop_data_axis(dd3_XY,streamwise_label_XY,0,boxCenter,boxDimensions)

            #3D crop. More efficient for interpolation to do each direction 
            #dd3_YZ = self.crop_data(dd3_YZ,axis_YZ,axis_info_YZ,boxCenter,boxDimensions)
            #dd3_XY = self.crop_data(dd3_XY,axis_XY,axis_info_XY,boxCenter,boxDimensions)
            #dd3_XZ = self.crop_data(dd3_XZ,axis_XZ,axis_info_XZ,boxCenter,boxDimensions)

            #DEBUGGING PRINT STATEMENTS
            #print()
            #print("XY INFO: ",)
            #print("Vertical dir: ",vertical_dir)
            #print("X: ",streamwise_label_XY)
            #print("Y: ",lateral_label_XY)
            #print("Z: ",vertical_label_XY)
            #print("X: ",dd3_XY[streamwise_label_XY][:,0,0],streamwise_label_XY,streamwise_ind_XY)
            #print("Y: ",dd3_XY[lateral_label_XY][0,0,:],lateral_label_XY,lateral_ind_XY)
            #print("Z: ",dd3_XY[vertical_label_XY][0,:,0],vertical_label_XY,vertical_ind_XY)

            #print()
            #print("XZ INFO: ",)
            #print("Lateral dir: ",lateral_dir)
            #print("X: ",streamwise_label_XZ)
            #print("Y: ",lateral_label_XZ)
            #print("Z: ",vertical_label_XZ)
            #print("X: ",dd3_XZ[streamwise_label_XZ][:,0,0],streamwise_label_XZ,streamwise_ind_XZ)
            #print("Y: ",dd3_XZ[lateral_label_XZ][0,0,:],lateral_label_XZ,lateral_ind_XZ)
            #print("Z: ",dd3_XZ[vertical_label_XZ][0,:,0],vertical_label_XZ,vertical_ind_XZ)

            #print()
            #print("YZ INFO: ",)
            #print("Streamwise dir: ",streamwise_dir)
            #print("X: ",streamwise_label_YZ)
            #print("Y: ",lateral_label_YZ)
            #print("Z: ",vertical_label_YZ)
            #print("X: ",dd3_YZ[streamwise_label_YZ][:,0,0],streamwise_label_YZ,streamwise_ind_YZ)
            #print("Y: ",dd3_YZ[lateral_label_YZ][0,0,:],lateral_label_YZ,lateral_ind_YZ)
            #print("Z: ",dd3_YZ[vertical_label_YZ][0,:,0],vertical_label_YZ,vertical_ind_YZ)

            print()
            print("--> BOX CENTER [Streamwise, Vertical, Lateral]: ",boxCenter)
            print("--> BOX DIMS [Streamwise, Vertical, Lateral]: ",boxDimensions)
            print("--> BOX CENTER + DIM/2.0 [Streamwise, Vertical, Lateral]: ",boxCenter+boxDimensions/2.0)
            print("--> BOX CENTER - DIM/2.0 [Streamwise, Vertical, Lateral]: ",boxCenter-boxDimensions/2.0)

            print("Done")

            # calculate useful parameters
            if not any('velocitya' in v for v in varnames):
                streamwise_label_XY = 'x'
                streamwise_label_YZ = 'x'
                streamwise_label_XZ = 'x'

                lateral_label_XY = 'y'
                lateral_label_YZ = 'y'
                lateral_label_XZ = 'y'

                vertical_label_XY = 'z'
                vertical_label_YZ = 'z'
                vertical_label_XZ = 'z'

            print("Calculating remaining transport terms...",end='',flush=True)
            streamPos = dd3_YZ[streamwise_label_YZ][:,0,0]-(boxCenter[0]-boxDimensions[0]/2.0)
            numStreamPos = len(streamPos)
            minStreamPosIncrement = np.min(np.diff(streamPos))

            streamwise_velocity_label = 'velocity' + streamwise_label_XY 
            lateral_velocity_label = 'velocity' + lateral_label_XY 
            vertical_velocity_label = 'velocity' + vertical_label_XY 
            streamwise_streamwise_label = self.sort_rs_labels(corr_mapping[streamwise_velocity_label],corr_mapping[streamwise_velocity_label])
            streamwise_lateral_label = self.sort_rs_labels(corr_mapping[streamwise_velocity_label],corr_mapping[lateral_velocity_label])
            streamwise_vertical_label = self.sort_rs_labels(corr_mapping[streamwise_velocity_label],corr_mapping[vertical_velocity_label])
            dd3_XY['u_avg_cubed'] = dd3_XY[streamwise_velocity_label+'_avg']**3# units of m^3/s^3
            dd3_XY['u_avg_squared_v_avg'] = dd3_XY[streamwise_velocity_label+'_avg']**2*dd3_XY[lateral_velocity_label+'_avg'] # units of m^3/s^3
            dd3_XY['u_avg_squared_w_avg'] = dd3_XY[streamwise_velocity_label+'_avg']**2*dd3_XY[vertical_velocity_label+'_avg'] # units of m^3/s^3

            streamwise_velocity_label = 'velocity' + streamwise_label_XZ 
            lateral_velocity_label = 'velocity' + lateral_label_XZ 
            vertical_velocity_label = 'velocity' + vertical_label_XZ 
            streamwise_streamwise_label = self.sort_rs_labels(corr_mapping[streamwise_velocity_label],corr_mapping[streamwise_velocity_label])
            streamwise_lateral_label = self.sort_rs_labels(corr_mapping[streamwise_velocity_label],corr_mapping[lateral_velocity_label])
            streamwise_vertical_label = self.sort_rs_labels(corr_mapping[streamwise_velocity_label],corr_mapping[vertical_velocity_label])
            dd3_XZ['u_avg_cubed'] = dd3_XZ[streamwise_velocity_label+'_avg']**3# units of m^3/s^3
            dd3_XZ['u_avg_squared_v_avg'] = dd3_XZ[streamwise_velocity_label+'_avg']**2*dd3_XZ[lateral_velocity_label+'_avg'] # units of m^3/s^3
            dd3_XZ['u_avg_squared_w_avg'] = dd3_XZ[streamwise_velocity_label+'_avg']**2*dd3_XZ[vertical_velocity_label+'_avg'] # units of m^3/s^3

            streamwise_velocity_label = 'velocity' + streamwise_label_YZ 
            lateral_velocity_label = 'velocity' + lateral_label_YZ 
            vertical_velocity_label = 'velocity' + vertical_label_YZ 
            streamwise_streamwise_label = self.sort_rs_labels(corr_mapping[streamwise_velocity_label],corr_mapping[streamwise_velocity_label])
            streamwise_lateral_label = self.sort_rs_labels(corr_mapping[streamwise_velocity_label],corr_mapping[lateral_velocity_label])
            streamwise_vertical_label = self.sort_rs_labels(corr_mapping[streamwise_velocity_label],corr_mapping[vertical_velocity_label])
            dd3_YZ['u_avg_cubed'] = dd3_YZ[streamwise_velocity_label+'_avg']**3# units of m^3/s^3
            dd3_YZ['u_avg_squared_v_avg'] = dd3_YZ[streamwise_velocity_label+'_avg']**2*dd3_YZ[lateral_velocity_label+'_avg'] # units of m^3/s^3
            dd3_YZ['u_avg_squared_w_avg'] = dd3_YZ[streamwise_velocity_label+'_avg']**2*dd3_YZ[vertical_velocity_label+'_avg'] # units of m^3/s^3
            dd3_YZ['P_x'] = -dd3_YZ[streamwise_streamwise_label + '_avg']*dd3_YZ['grad_velocity0_derived_avg'] - dd3_YZ[streamwise_lateral_label+'_avg']*dd3_YZ['grad_velocity1_derived_avg'] - dd3_YZ[streamwise_vertical_label + '_avg']*dd3_YZ['grad_velocity2_derived_avg'] # units of m^3/s^3 (ordering of gradients from AMR from 0-8 is dudx dudy dudz dvdx dvdy dvz dwdx dwdy dwdz)
            dd3_YZ['1_over_rho_u_avg_dp_dx_avg'] = (1/rho)*dd3_YZ[streamwise_velocity_label+'_avg']*dd3_YZ['grad_px_derived_avg'] # units of m^2/s^3
            dd3_YZ['coriolis_x'] = coriolis_forcing_YZ[2]*dd3_YZ[lateral_velocity_label+'_avg']*dd3_YZ[streamwise_velocity_label+'_avg'] # units of m^2/s^3
            dd3_YZ['body_force'] = body_force_YZ[0]*dd3_YZ[streamwise_velocity_label+'_avg'] # units of m^2/s^3? (see note in the input section at the top of this ipynb)

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
                #qoi = 'u_avg_uu_avg'
                streamwise_velocity_label = 'velocity' + streamwise_label_YZ 
                qoi = corr_mapping[streamwise_velocity_label] + '_avg_' + corr_mapping[streamwise_velocity_label] + corr_mapping[streamwise_velocity_label] + '_avg'
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
                #qoi = 'u_avg_uv_avg'
                streamwise_velocity_label = 'velocity' + streamwise_label_XZ
                lateral_velocity_label = 'velocity' + lateral_label_XZ
                qoi = corr_mapping[streamwise_velocity_label] + '_avg_' + self.sort_rs_labels(corr_mapping[streamwise_velocity_label],corr_mapping[lateral_velocity_label]) + '_avg'
                val = dd3_XZ[qoi][dim0_index,:,lft_iplane]
                streamwise_grid  = dd3_XZ[streamwise_label_XZ][dim0_index,0,lft_iplane]
                vertical_grid = dd3_XZ[vertical_label_XZ][i,:,lft_iplane]
                df_in.iloc[i, df_in.columns.get_loc('P_turb_left')] = self.doubleIntegral(val,streamwise_grid,vertical_grid)

            # right
            df_in = df_in.assign(P_turb_right=[None] * len(df_in))
            for i in range(numStreamPos):
                dim0_index = slice(0,i+1,1) # inner integral limits
                #qoi = 'u_avg_uv_avg'
                streamwise_velocity_label = 'velocity' + streamwise_label_XZ
                lateral_velocity_label = 'velocity' + lateral_label_XZ
                qoi = corr_mapping[streamwise_velocity_label] + '_avg_' + self.sort_rs_labels(corr_mapping[streamwise_velocity_label],corr_mapping[lateral_velocity_label]) + '_avg'
                val = dd3_XZ[qoi][dim0_index,:,rht_iplane]
                streamwise_grid  = dd3_XZ[streamwise_label_XZ][dim0_index,0,rht_iplane]
                vertical_grid = dd3_XZ[vertical_label_XZ][i,:,rht_iplane]
                df_in.iloc[i, df_in.columns.get_loc('P_turb_right')] = -self.doubleIntegral(val,streamwise_grid,vertical_grid)
                
            # bottom
            df_in = df_in.assign(P_turb_bot=[None] * len(df_in))
            for i in range(numStreamPos):
                dim0_index = slice(0,i+1,1) # inner integral limits
                #qoi = 'u_avg_uw_avg'
                streamwise_velocity_label = 'velocity' + streamwise_label_XY
                vertical_velocity_label = 'velocity' + vertical_label_XY
                qoi = corr_mapping[streamwise_velocity_label] + '_avg_' + self.sort_rs_labels(corr_mapping[streamwise_velocity_label],corr_mapping[vertical_velocity_label]) + '_avg'

                val = dd3_XY[qoi][dim0_index,bot_iplane,:]
                streamwise_grid  = dd3_XY[streamwise_label_XY][dim0_index,bot_iplane,0]
                lateral_grid = dd3_XY[lateral_label_XY][i,bot_iplane,:]
                df_in.iloc[i, df_in.columns.get_loc('P_turb_bot')] = self.doubleIntegral(val,streamwise_grid,lateral_grid)

            # top
            df_in = df_in.assign(P_turb_top=[None] * len(df_in))
            for i in range(numStreamPos):
                dim0_index = slice(0,i+1,1) # inner integral limits
                #qoi = 'u_avg_uw_avg'
                streamwise_velocity_label = 'velocity' + streamwise_label_XY
                vertical_velocity_label = 'velocity' + vertical_label_XY
                qoi = corr_mapping[streamwise_velocity_label] + '_avg_' + self.sort_rs_labels(corr_mapping[streamwise_velocity_label],corr_mapping[vertical_velocity_label]) + '_avg'
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
            self.Uinf = inflow_velocity_YZ[0]
            self.boxDimensions=boxDimensions

            if len(savepklfile)>0:
                directory, file_name = os.path.split(savepklfile)
                if directory!='':
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
        actionname = 'print_table'
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
            referencePlane_xOverD = self.parent.df_in.index[0]
            normalization = self.parent.Uinf**3*(self.parent.boxDimensions[1]*self.parent.boxDimensions[2])

            # rhs
            index = -1
            print()
            print("RHS")
            print("--------")
            columns_to_plot = ['P_mean', 'P_turb']
            for col in columns_to_plot:
                print(col,"\t",self.parent.df_out[col].iloc[index]/normalization)
            print()

            # lhs
            index = -1
            print("LHS")
            print("--------")
            columns_to_plot = ['P_mean', 'P_turb', 'P_prod', 'P_pres', 'P_cori', 'P_bodf']
            for col in columns_to_plot:
                print(col,"\t",self.parent.df_in[col].iloc[index]/normalization)
            print()

            index = -1
            print("Reduces")
            print("--------")
            columns_to_plot = ['P_reduced',]
            #print((self.parent.df_in[columns_to_plot].iloc[index]/normalization).apply(lambda value: round(value, -int(math.floor(math.log10(abs(value)))) + sigfigs)))
            print((self.parent.df_in[columns_to_plot].iloc[index].values[0]/normalization))
            print()

            # residual
            index = -1
            columns_to_plot = ['P']
            print("RESIDUAL")
            print("--------")
            total_out = self.parent.df_out[columns_to_plot].iloc[index]
            total_in  = self.parent.df_in[columns_to_plot].iloc[index]
            diff = total_out - total_in
            #print((diff/normalization).apply(lambda value: round(value, -int(math.floor(math.log10(abs(value)))) + sigfigs)))
            print("total out: ",total_out.values[0]/normalization)
            print("total in: " ,total_in.values[0]/normalization)
            print("Residual: ",diff.values[0]/normalization)
            print()

            index = -1
            columns_to_plot = ['P']
            print("Relative Error: (Total_In - Total_Out)/Total_Out")
            print("--------")
            total_out = self.parent.df_out[columns_to_plot].iloc[index]
            total_in  = self.parent.df_in[columns_to_plot].iloc[index]
            diff = (total_in - total_out)/total_out
            print(diff.values[0])
            #print((diff).apply(lambda value: round(value, -int(math.floor(math.log10(abs(value)))) + sigfigs)),"%")
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
            referencePlane_xOverD = self.parent.df_in.index[0]
            normalization = self.parent.Uinf**3*(self.parent.boxDimensions[1]*self.parent.boxDimensions[2])

            fig, axs = plt.subplots(1,figsize=(12,7), sharex=True)
            plt.plot((self.parent.df_out['P']-self.parent.df_out['P'][referencePlane_xOverD])/normalization,label='$\phi_{out,total}$')
            plt.plot((self.parent.df_in['P']-self.parent.df_in['P'][referencePlane_xOverD])/normalization,label='$\phi_{in,total}$')
            plt.xlabel('$x/D$ [-]')
            plt.ylabel('$\phi \; U_{\inf}^{-3} \; D^{-2}$ [-]')
            plt.ylim([-0.1,0.1])
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
            referencePlane_xOverD = self.parent.df_in.index[0]
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
            #ax1.set_ylim([-0.1, 0.2])
            #ax2.set_ylim([-0.1, 0.2])
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
            #ax1.text(-0.1, 0.65, 'Gain', transform=ax1.transAxes, rotation=90, va='center',fontsize=fsize)
            #ax1.text(-0.1, 0.17, 'Loss', transform=ax1.transAxes, rotation=90, va='center',fontsize=fsize)
            #plt.gcf().set_size_inches(13, 8)
            fig.savefig(savefilename)
