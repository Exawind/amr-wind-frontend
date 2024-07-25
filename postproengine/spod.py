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
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import amrwind_frontend as amrwind
from itertools import product

def plot_radial(Ur,theta,r,rfact=1.4,cmap='coolwarm',newfig=True,vmin=None,vmax=None,ax=None):
    if newfig == True:
        fig, ax = plt.subplots(subplot_kw={'projection':'polar'})
    else:
        ax = ax

    LR = r[-1]
    #ax=plt.subplot(111,polar=True)
    #im = ax.pcolormesh(theta,r,Ur,cmap='coolwarm',vmin=0,vmax=8)
    if vmin == None or vmax == None:
        im = ax.pcolormesh(theta,r,Ur,cmap=cmap)
    else:
        im = ax.pcolormesh(theta,r,Ur,cmap=cmap,vmin=vmin,vmax=vmax)
    #im = ax.pcolormesh(theta,r,Ur,cmap='jet')
    #cbar = plt.colorbar(im,orientation='horizontal')
    #cbar = plt.colorbar(im,orientation='vertical')

    # ---- mod here ---- #
    #ax.set_theta_zero_location("N")  # theta=0 at the top
    #ax.set_theta_direction(-1)  # theta increasing clockwise
    ax.set_rmax(LR)
    #ax.set_yticks([0, LR/4, LR/2, 3*LR/4,LR])  # less radial ticks
    ax.set_yticks([0,LR/1.4, LR])  # less radial ticks
    #ax.set_rlabel_position(-22.5)  # get radial labels away from plotted line
    ax.set_yticklabels(["$0$","$R$","1.4$R$"])
    ax.grid(True)
    # skoet the locations of the angular gridlines
    #lines, labels = thetagrids(range(45, 360, 90))

    # set the locations and labels of the angular gridlines
    lines, labels = plt.thetagrids([0,45,90,135,180,225,270,315], ("$90^\circ$","$45^\circ$","$0^\circ$","$315^\circ$","$270^\circ$","$225^\circ$","$180^\circ$","$135^\circ$"))
    #ax.spines['polar'].set_visible(False)
    return im


def read_cart_data(ncfile,group,trange,iplanes):
    udata = {}
    xc = {}
    timevec = ppsample.getVar(ppsample.loadDataset(ncfile), 'time')
    varnames = ['velocityx','velocityy','velocityz']
    if trange==[]:
        # load entire file
        db = ppsamplexr.getFullPlaneXR(ncfile,len(timevec),timevec[1]-timevec[0],groupname=group)
        y = np.asarray(db['y'].data)
        z = np.asarray(db['z'].data)
        t = np.asarray(np.array(db['times']).data)
        for iplane in iplanes:
            xc[iplane] = np.asarray(db['x'].data[iplane])
            udata[iplane] = np.zeros((len(t),len(z),len(y),3))
            udata[iplane][:,:,:,0] = np.array(db['velocityx'][:,iplane,:,:])
            udata[iplane][:,:,:,1] = np.array(db['velocityy'][:,iplane,:,:])
            udata[iplane][:,:,:,2] = np.array(db['velocityz'][:,iplane,:,:])
    else:
        find_nearest = lambda a, a0: np.abs(np.array(a) - a0).argmin()
        iters = [find_nearest(timevec, t) for t in trange]
        iters = np.arange(iters[0],iters[1]+1)
        db = ppsamplexr.getPlaneXR(ncfile,iters,varnames,groupname=group,verbose=0,includeattr=True,gettimes=True)
        YY = np.array(db['y'])
        ZZ = np.array(db['z'])

        flow_index  = -np.ones(3)
        for i in range(0,3):
            if (sum(db['axis'+str(i+1)]) != 0):
                flow_index[i] = np.nonzero(db['axis'+str(i+1)])[0][0]
        for i in range(0,3):
            if flow_index[i] == -1:
                flow_index[i] = 3 - (flow_index[i-1] + flow_index[i-2])

        streamwise_index = 2-np.where(flow_index == 0)[0][0]
        lateral_index = 2-np.where(flow_index == 1)[0][0]
        vertical_index = 2-np.where(flow_index == 2)[0][0]

        slices = [slice(None)] * 3
        slices[streamwise_index] = 0
        slices[vertical_index] = 0
        y = YY[tuple(slices)]

        slices = [slice(None)] * 3
        slices[streamwise_index] = 0
        slices[lateral_index] = 0
        z = ZZ[tuple(slices)]

        permutation = [streamwise_index, vertical_index, lateral_index]

        t = np.asarray(np.array(db['times']).data)
        for iplane in iplanes:
            xc[iplane] = np.asarray(db['x'][iplane][0,0])
            udata[iplane] = np.zeros((len(t),len(z),len(y),3))
            for i,tstep in enumerate(iters):
                ordered_data = np.transpose(np.array(db['velocityx'][tstep]),permutation)
                udata[iplane][i,:,:,0] = ordered_data[iplane,:,:]

                ordered_data = np.transpose(np.array(db['velocityy'][tstep]),permutation)
                udata[iplane][i,:,:,1] = ordered_data[iplane,:,:]

                ordered_data = np.transpose(np.array(db['velocityz'][tstep]),permutation)
                udata[iplane][i,:,:,2] = ordered_data[iplane,:,:]

    return udata , y , z , t 

def interpolate_radial_to_cart(U,rr,theta,YY,ZZ,offsety,offsetz):
    YR = np.sqrt(np.square((YY-offsety)) + np.square((ZZ-offsetz)))
    ZR = np.arctan2((ZZ-offsetz),(YY-offsety))
    ZR[ZR < 0] += 2 * np.pi
    grid = (YR,ZR)
    positions = np.vstack(list(map(np.ravel,grid))).T
    U_interp = sp.interpolate.interpn((rr,theta),U,positions,method='linear',bounds_error=False,fill_value = 0)
    U_interp = np.reshape(U_interp,(YR.shape[0],ZR.shape[1]))
    return U_interp

def interpolate_cart_to_radial(U,yy,zz,RR,TT,offsety,offsetz):
    YY_interp = RR * np.cos(TT) + offsety
    ZZ_interp = RR * np.sin(TT) + offsetz
    grid = (ZZ_interp,YY_interp)
    positions = np.vstack(list(map(np.ravel,grid))).T
    U_interp = sp.interpolate.interpn((zz,yy),U,positions,method='linear')
    U_interp = np.reshape(U_interp,(RR.shape[0],TT.shape[0]))
    return U_interp

def welch_fft(x,fs=1.0,nperseg=256,return_onesided=True,subtract_mean=True):
    """
    Calling welch() via scipy only returns the power spectral densitiy. 
    This returns the actual Fourier coefficient following a similar procedure
    """
    overlap = nperseg//2
    if return_onesided:
        wsfreq = np.fft.rfftfreq(nperseg,d=1/fs)
    else:
        wsfreq = np.fft.fftfreq(nperseg,d=1/fs)
    
    #segement the signal 
    segments = [x[i:i+nperseg] for i in range(0,len(x)-nperseg+1,nperseg-overlap)]

    #applying hamming window to each segment 
    window = sp.signal.get_window('hamming',nperseg)
    windowed_segments = [segment * window for segment in segments]

    #subtract the mean for each segement and apply the fft 
    if subtract_mean:
        mean_subtracted_segments = [segment - np.mean(segment) for segment in windowed_segments]
    else:
        mean_subtracted_segments = windowed_segments

    fft_segments = np.array([np.fft.fft(segments) for segments in mean_subtracted_segments])
    if return_onesided:
      fft_segments = fft_segments[:,0:int(nperseg/2)+1]
    return wsfreq,fft_segments

def get_angular_wavenumbers(N,L,negative=False):
    k = np.zeros(N)
    counter = 0
    if negative:
        for i in range(0,N//2):
            k[counter]  = 2*np.pi / L * -i
            counter += 1
        for i in range(0,N//2):
            k[counter]  = 2*np.pi / L * (N/2-i)
            counter += 1
    else:
        for i in range(0,int(N/2)+1):
            k[counter] = 2*np.pi/L * i 
            counter += 1
        for i in range(1,int(N/2)):
            k[counter] = -2*np.pi/L*(N/2-i) 
            counter += 1
    return k

def form_weighting_matrix_simpsons_rule(r):
    Nr = len(r)
    dr = r[1]-r[0] # assume uniform grid in r 
    W = np.zeros((Nr,Nr))
    #this case is even so we do trap rule for last two points
    if Nr % 2 == 0:
        for i in range(0,Nr-1):
            if i == 0 or i== Nr-2:
                W[i,i] =  dr * r[i] * 1.0/3.0
            elif i % 2 == 0:
                W[i,i] = dr * r[i] * 2.0 / 3.0  
            else:
                W[i,i] = dr * r[i] * 4.0 / 3.0  
        W[-1,-1] = dr * 1.0/2.0 * r[i] 
        W[-2,-2] += dr * 1.0/2.0 * r[i] 
    #for an odd number of points do simpsons rule over entire radial domain
    else:
        for i in range(0,Nr):
            if i == 0 or i == Nr-1:
                W[i,i] =  dr * r[i] * 1.0/3.0
            elif i % 2 == 0:
                W[i,i] = dr * r[i] * 2.0 / 3.0  
            else:
                W[i,i] = dr * r[i] * 4.0 / 3.0  
    
    return W


"""
Plugin for computing SPOD eigenvectors and eigenvalues from streamwise planes

See README.md for details on the structure of classes here
"""

@registerplugin
class postpro_spod():
    """
    SPOD of plane data
    """
    # Name of task (this is same as the name in the yaml)
    name      = "spod"
    # Description of task
    blurb     = "Compute SPOD eigenvectors and eigenvalues"
    inputdefs = [
        # -- Execute parameters ----
        {'key':'name',     'required':True,  'default':'',
         'help':'An arbitrary name',},
        {'key':'ncfile',   'required':True,  'default':'',
        'help':'NetCDF sampling file', },
        {'key':'trange',    'required':False,  'default':[],
            'help':'Which times to average over', }, 
        {'key':'group',   'required':False,  'default':None,
         'help':'Which group to pull from netcdf file', },
        {'key':'nperseg',   'required':True,  'default':256,
         'help':'Number of snapshots per segment to specify number of blocks.', },
        {'key':'yc',   'required':True,  'default':None,
         'help':'Lateral wake center', },
        {'key':'zc',   'required':True,  'default':None,
         'help':'Vertical wake center', },
        {'key':'wake_meandering_stats_file','required':False,  'default':None,
         'help':'The lateral and vertical wake center will be read from yc_mean and zc_mean columns of this file, overriding yc and zc.', },
        {'key':'LR_factor',   'required':False,  'default':1.4,
         'help':'Factor of blade-radius to define the radial domain extent.'},
        {'key':'NR',   'required':False,  'default':256,
         'help':'Number of points in the radial direction.'},
        {'key':'NTheta',   'required':False,  'default':256,
         'help':'Number of points in the azimuthal direction.'},
        {'key':'remove_temporal_mean',   'required':False,  'default':True,
         'help':'Boolean to remove temporal mean from SPOD.'},
        {'key':'remove_azimuthal_mean',   'required':False,  'default':False,
         'help':'Boolean to remove azimuthal mean from SPOD.'},
        {'key':'iplane',       'required':False,  'default':[0,],
         'help':'i-index of planes to postprocess', },
        {'key':'correlations','required':False,  'default':['U',],
            'help':'List of correlations to include in SPOD. Separate U,V,W components with dash. Examples: U-V-W, U,V,W,V-W ', },
        {'key':'output_dir',  'required':False,  'default':'./','help':'Directory to save results'},
        {'key':'savepklfile', 'required':False,  'default':'',
        'help':'Name of pickle file to save results', },
        {'key':'loadpklfile', 'required':False,  'default':'',
        'help':'Name of pickle file to load to perform actions', },
        {'key':'compute_eigen_vectors', 'required':False,  'default':True,
        'help':'Boolean to compute eigenvectors or just eigenvalues', },
        {'key':'sort', 'required':False,  'default':True,
        'help':'Boolean to included sorted wavenumber and frequency indices by eigenvalue', },
        {'key':'save_num_modes', 'required':False,  'default':None,
        'help':'Number of eigenmodes to save, ordered by eigenvalue. Modes will be save in array of shape (save_num_mods,NR).', },
        
    ]
    example = """
spod:
  name: Wake YZ plane
  ncfile: ./YZslice_00.50D_456.00s_1556.00s_cl00.nc
  group: xslice
  trange: [456.00,1556.50]
  nperseg: 256
  diam: 127
  yc: 375.0
  zc: 90
  wake_meandering_stats_file: ./data_converter/wake_meandering_baseline_area_1p4R/wake_meandering/wake_stats_00.50D.csv
  correlations:
    - U
    - V
    - W
    - U-V-W
    - V-W
  output_dir: ./test
  savepklfile: test.pkl
  #loadpklfile: ./test/test.pkl

  plot_eigvals:
    num: 11
    savefile: ./test/test_eigvals.png
    correlations:
      - U
    Uinf: 6.4

  plot_eigmodes:
    num: 2
    St: 0.3
    Uinf: 6.4
    savefile: ./test/test_eigmode.png
    correlations:
      - U
    """
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
        print('Running '+self.name)
        # Loop through and create plots
        for plane in self.yamldictlist:
            iplanes               = plane['iplane']
            trange                = plane['trange']
            ncfile                = plane['ncfile']
            self.diam             = plane['diam']
            group                 = plane['group']
            LR_factor             = plane['LR_factor']
            NR                    = plane['NR']
            NTheta                = plane['NTheta']
            ycenter               = plane['yc']
            zcenter               = plane['zc']
            nperseg               = plane['nperseg']
            correlations          = plane['correlations']
            remove_temporal_mean  = plane['remove_temporal_mean']
            remove_azimuthal_mean = plane['remove_azimuthal_mean']
            savefile              = plane['savepklfile']
            loadpklfile           = plane['loadpklfile']
            output_dir            = plane ['output_dir']
            compute_eigen_vectors = plane ['compute_eigen_vectors']
            sort                  =  plane ['sort']
            wake_meandering_stats_file = plane['wake_meandering_stats_file']
            save_num_modes        = plane['save_num_modes']

            if not loadpklfile:
                #Get all times if not specified 
                if not isinstance(iplanes, list): iplanes = [iplanes,]
                if not isinstance(correlations, list): correlations= [correlations,]

                print("--> Reading in cartesian velocities")
                udata_cart,y,z,times = read_cart_data(ncfile,group,trange,iplanes)
                tsteps = range(len(times))

                """
                Define polar grid 
                """
                LR = (self.diam/2.0)*LR_factor #specify radial extent for POD analysis 
                LTheta = 2 * np.pi
                r = np.linspace(0,LR,NR)
                theta = np.linspace(0,LTheta,NTheta+1)[0:-1] #periodic grid in theta
                RR, TT = np.meshgrid(r,theta,indexing='ij')
                dr = r[1]-r[0]
                dtheta = theta[1]-theta[0]
                components = ['velocityx','velocityy','velocityz']

                print("--> Interpolating cartesian data to polar coordinates")

                if wake_meandering_stats_file != None:
                    wake_meandering_stats = pd.read_csv(wake_meandering_stats_file)
                    ycenter = wake_meandering_stats['yc_mean'][0]
                    zcenter = wake_meandering_stats['zc_mean'][0]
                    print(ycenter)
                    print(zcenter)
                    print("--> Read in mean wake centers from ",wake_meandering_stats_file,". yc = ",ycenter,", zc = ",zcenter,".")

                udata_polar = np.zeros((NR,NTheta,len(tsteps),len(components)))
                for titer , t in enumerate(tsteps):
                    for compind in range(len(components)):
                        if zcenter-LR < 0:
                            print("Error: zcenter - LR negative. Exiting")
                            sys.exit()
                        udata_polar[:,:,titer,compind]  = interpolate_cart_to_radial(udata_cart[0][titer,:,:,compind],y,z,RR,TT,ycenter,zcenter)

                Nkt = int(nperseg/2) + 1
                time_segments = np.array([times[i:i+nperseg] for i in range(0,len(times)-nperseg+1,nperseg-nperseg//2)])
                dt = times[1]-times[0]
                w = np.hamming(nperseg)
                NB = time_segments.shape[0] #number of blocks 
                LT = time_segments[0][-1]-time_segments[0][0]
                angfreq = get_angular_wavenumbers(nperseg,LT)
                angfreq = angfreq[0:Nkt]

                print("--> Fourier transforming in time (number of blocks = "+str(NB) + ")")
                udata_that = np.zeros((NR,NTheta,NB,Nkt,3),dtype=complex)
                for rind in np.arange(0,len(r)):
                    for thetaind in np.arange(0,len(theta)):
                        for compind,comp in enumerate(components):
                            temp_signal = udata_polar[rind,thetaind,:,compind]
                            if remove_temporal_mean:
                                temp_signal = temp_signal - np.mean(temp_signal) #subtract out temporal mean
                            tfreq , tfft = welch_fft(temp_signal,fs=1/dt,nperseg=nperseg,subtract_mean=remove_azimuthal_mean)
                            udata_that[rind,thetaind,:,:,compind] = tfft

                print("--> Fourier transforming in Theta")
                NkTheta = int(NTheta)
                udata_rhat = np.zeros((NR,NkTheta,NB,Nkt,len(components)),dtype=complex)
                for rind in np.arange(0,len(r)):
                    for block in np.arange(0,NB):
                        for ktind in np.arange(0,Nkt):
                            for compind,comp in enumerate(components):
                                temp_signal = udata_that[rind,:,block,ktind,compind] 
                                if remove_azimuthal_mean:
                                    temp_signal = temp_signal - np.mean(temp_signal) 
                                udata_rhat[rind,:,block,ktind,compind] = np.fft.fft(temp_signal)

                del udata_that #don't need this anymore 
                thetafreq = np.fft.fftfreq(NTheta,d=dtheta)
                ktheta  = get_angular_wavenumbers(NTheta,LTheta,negative=True)

                """
                Compute the POD via the SVD for each ktheta and kt of interest
                """

                if compute_eigen_vectors:
                    self.POD_modes       = {corr: 0 for corr in correlations}
                self.POD_eigenvalues = {corr: 0 for corr in correlations}

                if sort:
                    self.sorted_inds  = {corr: {} for corr in correlations}

                W1D = form_weighting_matrix_simpsons_rule(r)
                scaling_factor_k = dt / (sum(np.hamming(nperseg)*NB)) 

                corr_dict = {'U': 0, 'V': 1, 'W': 2}
                for corr in correlations:
                    print("--> Computing SPOD for correlations: ",corr)
                    comp_corr = corr.split('-')
                    corr_inds = [corr_dict[corr.upper()] for corr in comp_corr]

                    W = np.kron(np.eye(len(corr_inds)),W1D)
                    Wsqrt = np.sqrt(W)
                    Wsqrtinv = np.zeros_like(Wsqrt)
                    for i in range(NR*len(corr_inds)):
                        if (Wsqrt[i,i]==0):
                            Wsqrtinv[i,i] = 0
                        else:
                            Wsqrtinv[i,i] = 1.0 / Wsqrt[i,i]
                    if compute_eigen_vectors:
                        self.POD_modes[corr]       = np.zeros((NR,len(ktheta),len(tfreq),NB,len(corr_inds)),dtype=complex)
                    self.POD_eigenvalues[corr] = np.zeros((len(ktheta),len(tfreq),NB),dtype=complex)

                    for ktheta_ind , ktheta_val in enumerate(ktheta):
                        for tfreq_ind , tfreq_val in enumerate(tfreq):

                            POD_Mat = np.zeros((NR*len(corr_inds),NB),dtype=complex)
                            for corr_ind_iter , corr_ind in enumerate(corr_inds):
                                POD_Mat[corr_ind_iter*NR:NR*(corr_ind_iter+1),0:NB] = np.copy(udata_rhat[:,ktheta_ind,:,tfreq_ind,corr_ind])
                            POD_Mat = np.sqrt(scaling_factor_k) * np.dot(Wsqrt,POD_Mat)
                            
                            if compute_eigen_vectors:
                                lsvd, ssvd, rsvd = np.linalg.svd(POD_Mat,full_matrices=False,compute_uv=True)
                                eigmodes = np.zeros((NR,NB,len(corr_inds)),dtype=complex)

                                temp  = np.dot(Wsqrtinv,lsvd)
                                for corr_ind_iter , corr_ind in enumerate(corr_inds):
                                    eigmodes[:,:,corr_ind_iter] = temp[corr_ind_iter*NR:NR*(corr_ind_iter+1),:]

                                self.POD_modes[corr][:,ktheta_ind,tfreq_ind,:,:]   = eigmodes

                            else:
                                ssvd = np.linalg.svd(POD_Mat,full_matrices=False,compute_uv=False)

                            eigval = ssvd**2
                            self.POD_eigenvalues[corr][ktheta_ind,tfreq_ind,:] = eigval

                    if sort:
                        sorted_eigind = np.argsort(np.abs(self.POD_eigenvalues[corr]),axis=None)[::-1]
                        self.sorted_inds[corr]['ktheta'] , self.sorted_inds[corr]['angfreq'], self.sorted_inds[corr]['block'] = np.unravel_index(sorted_eigind,self.POD_eigenvalues[corr][:,:,:].shape)


                if len(savefile)>0:
                    self.variables = {}
                    self.variables['angfreq'] = angfreq
                    self.variables['ktheta']   = ktheta
                    self.variables['blocks']   = np.arange(0,NB)
                    self.variables['r']        = r
                    self.variables['theta']    = theta
                    self.variables['y']        = y
                    self.variables['z']        = z
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    savefile = os.path.join(output_dir, savefile)
                    print("--> Saving to: ",savefile)
                    objects = [] 
                    objects.append(self.POD_eigenvalues)
                    objects.append(self.variables)
                    if sort:
                        objects.append(self.sorted_inds)


                    if compute_eigen_vectors:
                        if save_num_modes == None:
                            objects.append(self.POD_modes)
                        else:
                            save_modes = {}
                            for corr in correlations:
                                save_modes[corr] = np.zeros((save_num_modes,NR))
                                for mode in range(numModes_to_store):
                                    save_modes[corr][mode,:] = POD_modes[corr][:,POD_sorted_inds[corr]['ktheta'][mode],POD_sorted_inds[corr]['angfreq'][mode],POD_sorted_inds[awc][dist]['block'][mode]]
                            objects.append(save_modes)


                    with open(savefile, 'wb') as f:
                        for obj in objects:
                            pickle.dump(obj, f)

            if loadpklfile:
                print("--> Loading from: ",loadpklfile)
                with open(loadpklfile, 'rb') as f:
                    self.POD_eigenvalues = pickle.load(f)
                    self.variables = pickle.load(f)
                    if sort:
                        self.sorted_inds = pickle.load(f)
                    if compute_eigen_vectors:
                        self.POD_modes = pickle.load(f)

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

    @registeraction(actionlist)
    class plot_eigvals():
        actionname = 'plot_eigvals'
        blurb      = 'Plots the leading eigenvalues and corresponding wavenumber and frequencies'
        required   = False
        actiondefs = [
        {'key':'num',   'required':False,  'default':10,
         'help':'Number of eigenvalues to plot', },
        {'key':'figsize',   'required':False,  'default':[16,4],
         'help':'Figure size (inches)', },
        {'key':'savefile',  'required':False,  'default':'',
         'help':'Filename to save the picture', },
        {'key':'title',     'required':False,  'default':'',
         'help':'Title of the plot',},
        {'key':'dpi',       'required':False,  'default':125,
         'help':'Figure resolution', },
        {'key':'correlations','required':False,  'default':['U',],
         'help':'List of correlations to plot', },
        {'key':'Uinf','required':True,  'default':0,
         'help':'Velocity for compute strouhal frequency', },
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent
            print('Initialized '+self.actionname+' inside '+parent.name)
            return

        def execute(self):
            print('Executing '+self.actionname)
            figsize  = self.actiondict['figsize']
            dpi      = self.actiondict['dpi']
            savefile = self.actiondict['savefile']
            title    = self.actiondict['title']
            num      = self.actiondict['num']
            Uinf     = self.actiondict['Uinf']
            correlations = self.actiondict['correlations']
            if not isinstance(correlations, list): correlations= [correlations,]
            fig, axs = plt.subplots(len(correlations),3,figsize=(figsize[0],figsize[1]), dpi=dpi)
            axs = np.reshape(axs, (len(correlations), 3))
            fig.suptitle(eval("f'{}'".format(title)))
            for corr_iter, corr in enumerate(correlations):
                ax = axs[corr_iter,:]

                ax[0].set_xlabel("Eigenvalue Index")
                ax[1].set_xlabel("Eigenvalue Index")
                ax[2].set_xlabel("Eigenvalue Index")

                ax[0].set_ylabel("Normalized Eigenvalue")
                ax[1].set_ylabel("$\kappa_\Theta$")
                ax[2].set_ylabel("$\omega D/U 2 \pi$")

                max_eigs = self.parent.POD_eigenvalues[corr][self.parent.sorted_inds[corr]['ktheta'][0:num],self.parent.sorted_inds[corr]['angfreq'][0:num],self.parent.sorted_inds[corr]['block'][0:num]]
                max_eig = np.abs(max_eigs[0])
                ax[0].semilogy(np.arange(0,num),np.abs(max_eigs)/max_eig)
                ax[0].scatter(np.arange(0,num),np.abs(max_eigs)/max_eig)

                ax[1].scatter(np.arange(0,num),self.parent.variables['ktheta'][self.parent.sorted_inds[corr]['ktheta'][0:num]])
                ax[2].scatter(np.arange(0,num),self.parent.variables['angfreq'][self.parent.sorted_inds[corr]['angfreq'][0:num]]*self.parent.diam/(Uinf * 2 * np.pi))


            if len(savefile)>0:
                plt.savefig(savefile)

            return

    @registeraction(actionlist)
    class plot_eigmodes():
        actionname = 'plot_eigmodes'
        blurb      = 'Plots leading eigenmodes'
        required   = False
        actiondefs = [
        {'key':'num',   'required':False,  'default':1,
         'help':'Number of eigenvectors to include in reconstruction', },
        {'key':'figsize',   'required':False,  'default':[16,4],
         'help':'Figure size (inches)', },
        {'key':'savefile',  'required':False,  'default':'',
         'help':'Filename to save the picture', },
        {'key':'title',     'required':False,  'default':'',
         'help':'Title of the plot',},
        {'key':'dpi',       'required':False,  'default':125,
         'help':'Figure resolution', },
        {'key':'correlations','required':False,  'default':['U',],
         'help':'List of correlations to plot', },
        {'key':'Uinf','required':False,  'default':0,
         'help':'Velocity for compute strouhal frequency', },
        {'key':'St','required':False,  'default':None,
         'help':'Plot leading eigenmodes at fixed Strouhal frequency', },
        {'key':'itime','required':False,  'default':0,
         'help':'Time iteration to plot', },
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent
            print('Initialized '+self.actionname+' inside '+parent.name)
            return

        def execute(self):
            print('Executing '+self.actionname)
            figsize  = self.actiondict['figsize']
            dpi      = self.actiondict['dpi']
            savefile = self.actiondict['savefile']
            title    = self.actiondict['title']
            numModes = self.actiondict['num']
            Uinf     = self.actiondict['Uinf']
            itime    = self.actiondict['itime']
            St       = self.actiondict['St']
            correlations = self.actiondict['correlations']
            if not isinstance(correlations, list): correlations= [correlations,]

            for corr in correlations:
                eig_modes = self.parent.POD_modes[corr]
                shape = self.parent.POD_modes[corr].shape
                mode_rhat = np.zeros((shape[0],shape[1],shape[2],shape[4]),dtype=complex)

                inds = np.arange(numModes)
                scaling = self.parent.diam/(Uinf * 2 * np.pi)
                if not St == None:
                    for i in range(0,numModes):
                        if i == 0:
                            ind = 0
                        else:
                            ind = int(inds[i-1] + 1)

                        st_index = np.argmin(np.abs(self.parent.variables['angfreq'] * scaling - St))
                        cont_flag = False
                        if not self.parent.sorted_inds[corr]['angfreq'][ind] == st_index: cont_flag = True
                        while cont_flag:
                            ind += 1
                            st_index = np.argmin(np.abs(self.parent.variables['angfreq'] * scaling - St))
                            if self.parent.sorted_inds[corr]['angfreq'][ind] == st_index: cont_flag = False
                        inds[i]=int(ind)
                for i in range(0,numModes):
                    ind = int(inds[i])
                    ktheta_ind = self.parent.sorted_inds[corr]['ktheta'][ind]
                    angfreq_ind = self.parent.sorted_inds[corr]['angfreq'][ind]
                    block_ind = self.parent.sorted_inds[corr]['block'][ind]
                    eigval_factor = self.parent.POD_eigenvalues[corr][ktheta_ind,angfreq_ind,block_ind]
                    print("---> Adding mode for ktheta = ",self.parent.variables['ktheta'][ktheta_ind]," and St = ", self.parent.variables['angfreq'][angfreq_ind] * scaling)
                    mode_rhat[:,ktheta_ind,angfreq_ind,:] += eigval_factor*self.parent.POD_modes[corr][:,ktheta_ind,angfreq_ind,block_ind,:]

                mode_r = np.fft.irfft(np.fft.ifft(mode_rhat,axis=1),axis=2)

                #scaling = self.parent.diam/(Uinf * 2 * np.pi)
                fig,ax = plt.subplots(1,1,sharey=False,sharex=False,figsize=(12,10),subplot_kw={'projection':'polar'})
                val = mode_r[:,:,itime,0]
                theta = self.parent.variables['theta']
                r = self.parent.variables['r']
                plot_radial(val,theta,r,cmap='jet',newfig=False,ax=ax)
            
            if len(savefile)>0:
                plt.savefig(savefile)

            return
