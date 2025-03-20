# Get the location where this script is being run
import sys, os
scriptpath = os.path.dirname(os.path.realpath(__file__))
basepath   = os.path.dirname(scriptpath)
# Add any possible locations of amr-wind-frontend here
amrwindfedirs = ['../',
                 basepath]
for x in amrwindfedirs: sys.path.insert(1, x)

from postproengine import registerplugin, mergedicts, registeraction
from postproengine import compute_axis1axis2_coords, get_mapping_xyz_to_axis1axis2
import postproamrwindsample_xarray as ppsamplexr
import postproamrwindsample as ppsample
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd
import numpy as np
import matplotlib
import scipy as sp
import scipy.signal
import time
import matplotlib.pyplot as plt
from itertools import product

def get_mode_number(numModes,sorted_inds,variables,corr,Uinf,diam,St=None,ktheta=None,tol=None):
    inds = [] 
    ktheta_vals = variables['ktheta']
    angfreq_vals = variables['angfreq']
    if (ktheta == None) and (St == None):
        for i in range(0,numModes):
            inds.append(int(i))
    #if specific ktheta but any St
    elif (not ktheta == None) and (St == None):
        if not isinstance(ktheta,list): ktheta=[ktheta,]
        for i in range(0,numModes):
            ind = 0
            ktheta_index = np.argmin(np.abs(variables['ktheta'] - ktheta[i]))
            cont_flag = False
            if tol == None:
                if not sorted_inds[corr]['ktheta'][ind] == ktheta_index : cont_flag = True
            else:
                if np.abs(ktheta_vals[sorted_inds[corr]['ktheta'][ind]] - ktheta[i]) > tol : cont_flag = True

            if ind in inds: cont_flag = True
            while cont_flag:
                ind += 1
                if tol == None:
                    ktheta_index = np.argmin(np.abs(variables['ktheta'] - ktheta[i]))
                    if sorted_inds[corr]['ktheta'][ind] == ktheta_index : cont_flag = False
                else:
                    if np.abs(ktheta_vals[sorted_inds[corr]['ktheta'][ind]] - ktheta[i]) <= tol : cont_flag = False
                if ind in inds: cont_flag = True

            inds.append(int(ind))

    #if specific St but any ktheta
    elif (not St == None) and (ktheta == None):
        scaling = diam/(Uinf * 2 * np.pi)
        if not isinstance(ktheta,list): ktheta=[ktheta,]
        for i in range(0,numModes):
            ind = 0
            cont_flag = False
            if tol == None:
                st_index = np.argmin(np.abs(variables['angfreq'] * scaling - St))
                if not sorted_inds[corr]['angfreq'][ind] == st_index: cont_flag = True
            else:
                if np.abs(angfreq_vals[sorted_inds[corr]['angfreq'][ind]] * scaling - St) > tol : cont_flag = True

            if ind in inds: cont_flag = True
            while cont_flag:
                ind += 1
                if tol == None:
                    st_index = np.argmin(np.abs(variables['angfreq'] * scaling - St))
                    if sorted_inds[corr]['angfreq'][ind] == st_index: cont_flag = False
                else:
                    if np.abs(angfreq_vals[sorted_inds[corr]['angfreq'][ind]] * scaling - St) <= tol : cont_flag = False
                if ind in inds: cont_flag = True
            inds.append(int(ind))

    #if specific St and specific ktheta
    elif (not St == None) and (not ktheta == None):
        scaling = diam/(Uinf * 2 * np.pi)
        if not isinstance(ktheta,list): ktheta=[ktheta]
        for i in range(0,numModes):
            ind = 0
            cont_flag = False
            if tol == None:
                ktheta_index = np.argmin(np.abs(variables['ktheta'] - ktheta[i]))
                st_index = np.argmin(np.abs(variables['angfreq'] * scaling - St))
                if (not sorted_inds[corr]['ktheta'][ind] == ktheta_index) or (not sorted_inds[corr]['angfreq'][ind] == st_index) : cont_flag = True
            else:
                if (np.abs(ktheta_vals[sorted_inds[corr]['ktheta'][ind]] - ktheta[i]) > tol ) or \
                (np.abs(angfreq_vals[sorted_inds[corr]['angfreq'][ind]] * scaling - St) > tol) : cont_flag = True

            if ind in inds: cont_flag = True
            while cont_flag:
                ind += 1
                if tol == None:
                    ktheta_index = np.argmin(np.abs(variables['ktheta'] - ktheta[i]))
                    st_index = np.argmin(np.abs(variables['angfreq'] * scaling - St))
                    if (sorted_inds[corr]['ktheta'][ind] == ktheta_index) and (sorted_inds[corr]['angfreq'][ind] == st_index) : cont_flag = False
                else:
                    if (np.abs(ktheta_vals[sorted_inds[corr]['ktheta'][ind]] - ktheta[i]) <= tol ) and \
                    (np.abs(angfreq_vals[sorted_inds[corr]['angfreq'][ind]] * scaling - St) <= tol) : cont_flag = False
                if ind in inds: cont_flag = True
            inds.append(int(ind))
    return inds

def reshape_stack_velocities(U,NR,NB,corr):
    corr_dict = {'U': 0, 'V': 1, 'W': 2}
    comp_corr = corr.split('-')
    corr_inds = [corr_dict[corr.upper()] for corr in comp_corr]
    shape = U.shape
    U_stack = np.zeros((NR*len(corr_inds),NB),dtype=complex)
    for corr_ind_iter , corr_ind in enumerate(corr_inds):
        U_stack[corr_ind_iter*NR:NR*(corr_ind_iter+1),0:NB] = np.copy(U[:,:,corr_ind])

    return U_stack

def reshape_stack_velocities_single_block(U,NR,corr):
    corr_dict = {'U': 0, 'V': 1, 'W': 2}
    comp_corr = corr.split('-')
    corr_inds = [corr_dict[corr.upper()] for corr in comp_corr]
    shape = U.shape
    U_stack = np.zeros((NR*len(corr_inds)),dtype=complex)
    for corr_ind_iter , corr_ind in enumerate(corr_inds):
        U_stack[corr_ind_iter*NR:NR*(corr_ind_iter+1)] = np.copy(U[:,corr_ind_iter])

    return U_stack

def reshape_unstack_velocities(U,NR,NB,corr):
    corr_dict = {'U': 0, 'V': 1, 'W': 2}
    comp_corr = corr.split('-')
    corr_inds = [corr_dict[corr.upper()] for corr in comp_corr]
    U_unstack = np.zeros((NR,NB,len(corr_inds)),dtype=complex)
    for corr_ind_iter , corr_ind in enumerate(corr_inds):
        U_unstack[:,:,corr_ind_iter] = np.copy(U[corr_ind_iter*NR:NR*(corr_ind_iter+1),:])
    return  U_unstack

def inner_product(U,V,W):
    """
    Computes the inner product <U,V>_x = U^H W V

    """
    #return np.conjugate(U).T @ W @ V
    return np.dot(np.dot(np.conj(U).T,W),V)

def compute_projection_coefficient(u,POD_modes,corr,NR,NB,W):
    """
    Compute projection coefficient given mode and signal at specific ktheta and freq index
    """
    u_stacked = reshape_stack_velocities(u,NR,NB,corr) #NR*corr for each block NB
    reconstructed_signal = np.zeros((u_stacked.shape[0],1),dtype=complex)
    proj_coeff = np.zeros(NB,dtype=complex)
    for block_ind in range(0,NB):
        eigenvector = POD_modes
        eigmode_stacked = reshape_stack_velocities_single_block(eigenvector,NR,corr) #NR*corr for each eigenmode NB
        proj_coeff[block_ind] = inner_product(eigmode_stacked,u_stacked[:,block_ind],W)
    return proj_coeff

def reconstruct_r_ktheta_f(u,modes,eig_inds,POD_modes,sorted_inds,variables,corr,W,POD_proj_coeff = None):
    NTheta  = len(variables['theta'])
    NR      = len(variables['r'])
    angfreq = variables['angfreq'] 
    ktheta  = variables['ktheta'] 
    NB = len(variables['blocks'])
    if modes == -1: #reconstruct all modes
        modes = list(range(len(sorted_inds[corr]['ktheta'])))
    else:
        if not isinstance(modes, list): modes = [modes,]
    if eig_inds == -1:
        eig_inds = list(range(NB))
    else:
        if not isinstance(eig_inds, list): eig_inds = [eig_inds,]

    #shape NR, NTheta, NB, Nkt, corr
    mode_rhat = np.zeros((NR,NTheta,NB,len(angfreq),POD_modes[corr].shape[-1]),dtype=complex)

    for mode_iter , mode in enumerate(modes):
        ktheta_ind  = sorted_inds[corr]['ktheta'][mode]
        angfreq_ind = sorted_inds[corr]['angfreq'][mode]
        #print("Reconstructing ktheta = ",ktheta[ktheta_ind], ", angfreq = ",angfreq[angfreq_ind], ", mode = ",mode," (",mode_iter,"/",len(modes),")")

        if POD_proj_coeff == None:
            u_stacked = reshape_stack_velocities(u[:,ktheta_ind,:,angfreq_ind,:],NR,NB,corr) #NR*corr for each block NB
            reconstructed_signal = np.zeros((u_stacked.shape[0],1),dtype=complex)
        for block_ind in range(0,NB):
            #for eig_ind in range(0,NB):
            for eig_ind in eig_inds:
                if POD_proj_coeff == None:
                    #If reading SPOD results from pkl file with sorted eigenvectors/eigenvalues
                    if POD_modes[corr].shape[0] != NR: 
                        eigenvector = POD_modes[corr][mode,:,:] 
                    else:
                        eigenvector = POD_modes[corr][:,ktheta_ind,angfreq_ind,eig_ind,:] 

                    eigmode_stacked = reshape_stack_velocities_single_block(eigenvector,NR,corr) #NR*corr for each eigenmode NB
                    proj_coeff = inner_product(eigmode_stacked,u_stacked[:,block_ind],W)
                    reconstructed_signal[:,0] = proj_coeff*eigmode_stacked
                    mode_rhat[:,ktheta_ind,block_ind,angfreq_ind,:] += reshape_unstack_velocities(reconstructed_signal,NR,1,corr)[:,0,:]
                else:
                    #If reading SPOD results from pkl file with sorted eigenvectors/eigenvalues
                    if POD_modes[corr].shape[0] != NR: 
                        eigenvector = POD_modes[corr][mode,:,:] 
                        proj_coeff = POD_proj_coeff[corr][mode,eig_ind] #projection of u onto POD mode
                    else:
                        eigenvector = POD_modes[corr][:,ktheta_ind,angfreq_ind,eig_ind,:] 
                        proj_coeff = POD_proj_coeff[corr][ktheta_ind,angfreq_ind,block_ind,eig_ind] #projection of u onto POD mode
                    mode_rhat[:,ktheta_ind,block_ind,angfreq_ind,:] += proj_coeff * eigenvector

    return mode_rhat


def transform_to_realspace(u_r_ktheta_f,variables,numSteps,dt,corr,components=None,nowindow=False):
    NR      = len(variables['r'])
    NTheta  = len(variables['ktheta']) 
    angfreq = variables['angfreq'] 
    Nblocks = len(variables['blocks'])
    nperseg = variables['nperseg']
    overlap = variables['noverlap']
    r = variables['r']

    NCorr = u_r_ktheta_f.shape[-1]
    u_r_theta_t = np.zeros((NR,NTheta,numSteps,NCorr))

    #inverse Fourier transform in theta using ifft
    u_r_theta_f = np.fft.ifft(u_r_ktheta_f,axis=1)

    #inverse fourier transform in time using isft
    if components==None: components = range(NCorr)
    for r in range(NR):
        for theta in range(NTheta):
            for comp in components:
                Zxx = np.zeros((len(angfreq),Nblocks),dtype=complex)
                for block_ind in range(Nblocks):
                    Zxx[:,block_ind] = u_r_theta_f[r,theta,block_ind,:,comp]/Nblocks
                if nowindow:
                    real_signal = np.fft.irfft(Zxx[:,0],n=numSteps)
                else:
                    t, real_signal = compute_istft(Zxx,fs=1.0/dt,nperseg=nperseg,noverlap=overlap,window='hamming')
                u_r_theta_t[r,theta,:,comp] = real_signal[:numSteps]
    return u_r_theta_t

def reconstruct_r_theta_t_flow_istfft(inds,numSteps,dt,nperseg,overlap,sorted_inds,variables,POD_proj_coeff,POD_modes,corr,components=None):
    NTheta  = len(variables['theta'])
    NR      = len(variables['r'])
    angfreq = variables['angfreq'] 
    numModes = len(inds)
    shape = POD_modes[corr].shape
    Nblocks = len(variables['blocks'])

    mode_rhat = np.zeros((NR,NTheta,len(angfreq),shape[-1]),dtype=complex) 
    angfreq_ind_list = []
    for i in range(0,numModes):
        # Get ktheta index, angfreq index, and block index associated with mode number
        ind = inds[i]
        ktheta_ind  = sorted_inds[corr]['ktheta'][ind]
        angfreq_ind = sorted_inds[corr]['angfreq'][ind]
        block_ind   = sorted_inds[corr]['block'][ind]
        angfreq_ind_list.append(angfreq_ind)
        #If reading SPOD results from pkl file with sorted eigenvectors/eigenvalues
        if POD_modes[corr].shape[0] != NR: 
            proj_coeff  = POD_proj_coeff[corr][ind]                                      #projection of u onto POD mode
            mode_rhat[:,ktheta_ind,angfreq_ind,:] += proj_coeff*POD_modes[corr][ind] #reconstruction of fourier mode by the ith POD mode

        #else, continuing from executor
        else:
            proj_coeff  = POD_proj_coeff[corr][ktheta_ind,angfreq_ind,block_ind] #projection of u onto POD mode
            mode_rhat[:,ktheta_ind,angfreq_ind,:] += proj_coeff*POD_modes[corr][:,ktheta_ind,angfreq_ind,block_ind,:] #reconstruction of fourier mode by the ith POD mode

    #fourier transform in theta 
    mode_r_that = np.fft.ifft(mode_rhat,axis=1) 

    #fourier transform in time
    mode_r = np.zeros((NR,NTheta,numSteps,shape[-1])) 
    if components==None: components = range(mode_r_that.shape[-1])
    for r in range(mode_r_that.shape[0]):
        for theta in range(mode_r_that.shape[1]):
            for comp in components:
                Zxx = np.zeros((len(angfreq),Nblocks),dtype=complex)
                for angfreq_ind in angfreq_ind_list:
                    Zxx[angfreq_ind,:] = mode_r_that[r,theta,angfreq_ind,comp]
                t, real_signal = compute_istft(Zxx,fs=1.0/dt,nperseg=nperseg,noverlap=overlap,window='hamming')
                mode_r[r,theta,:,comp] = real_signal[:numSteps]

    return mode_r


def reconstruct_flow(inds,numSteps,dt,sorted_inds,variables,POD_proj_coeff,POD_modes,corr):
    NTheta  = len(variables['theta'])
    NR      = len(variables['r'])
    angfreq = variables['angfreq'] 
    angfreq_full = 2*np.pi*np.fft.rfftfreq(numSteps,dt) #angular frequencies for full time window
    numModes = len(inds)
    shape = POD_modes[corr].shape

    mode_rhat = np.zeros((NR,NTheta,len(angfreq_full),shape[-1]),dtype=complex) 

    for i in range(0,numModes):
        # Get ktheta index, angfreq index, and block index associated with mode number
        ind = inds[i]
        ktheta_ind  = sorted_inds[corr]['ktheta'][ind]
        angfreq_ind = sorted_inds[corr]['angfreq'][ind]
        angfreq_full_ind = np.argmin(abs(angfreq[angfreq_ind] - angfreq_full)) #nearest angular frequency in full time window
        block_ind   = sorted_inds[corr]['block'][ind]

        #If reading SPOD results from pkl file with sorted eigenvectors/eigenvalues
        if POD_modes[corr].shape[0] != NR: 
            proj_coeff  = POD_proj_coeff[corr][ind]                                      #projection of u onto POD mode
            mode_rhat[:,ktheta_ind,angfreq_full_ind,:] += proj_coeff*POD_modes[corr][ind] #reconstruction of fourier mode by the ith POD mode

        #else, continuing from executor
        else:
            proj_coeff  = POD_proj_coeff[corr][ktheta_ind,angfreq_ind,block_ind] #projection of u onto POD mode
            mode_rhat[:,ktheta_ind,angfreq_full_ind,:] += proj_coeff*POD_modes[corr][:,ktheta_ind,angfreq_ind,block_ind,:] #reconstruction of fourier mode by the ith POD mode

    #inverse fourier transform in theta and then time. 
    mode_r = np.fft.irfft(np.fft.ifft(mode_rhat,axis=1),axis=2,n=numSteps)  #note, the entire time window is included in the reconstruction, not just a single block. 
    return mode_r

def plot_radial(Ur,theta,r,rfact=1.4,cmap='coolwarm',newfig=True,vmin=None,vmax=None,ax=None,colorbar=True):
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
    if colorbar:
        cbar = plt.colorbar(im,orientation='horizontal')
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

def extract_1d_from_meshgrid(Z):
    # Check along axis 0
    unique_rows = np.unique(Z, axis=0)
    if unique_rows.shape[0] == 1:
        return unique_rows[0],1

    # Check along axis 1
    unique_cols = np.unique(Z, axis=1)
    if unique_cols.shape[1] == 1:
        return unique_cols[:, 0],0


def read_cart_data(ncfile,varnames,group,trange,iplanes,xaxis,yaxis):

    db = ppsamplexr.getPlaneXR(ncfile,[0,1],varnames,groupname=group,verbose=0,includeattr=True,gettimes=True,timerange=trange)
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
        permutation = [0,axisz+1,axisy+1]
        udata[iplane] = np.zeros((len(t),len(z),len(y),3))
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

def compute_stft(x, fs=1.0, nperseg=256, noverlap=None, window='hamming',subtract_mean=False):
    """
    Compute the Short-Time Fourier Transform (STFT) of a signal using scipy.

    Parameters:
    x (numpy array): The input signal.
    fs (float): Sampling frequency of the signal.
    nperseg (int): Length of each segment.
    noverlap (int): Number of points to overlap between segments.
    window (str or tuple or array_like): Desired window to use.

    Returns:
    f, t, Zxx: STFT of the signal.
    f: frequencies
    t: time bins
    Zxx: Complex fourier coefficients for each frequencies per time bin
    """
    f, t, Zxx = scipy.signal.stft(x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)

    if subtract_mean:
        Zxx[0,:] = 0+0j #zero out the zero frequency for each block to remove temporal mean 

    return f, t, Zxx.swapaxes(0,1)

def compute_istft(Zxx, fs=1.0, nperseg=256, noverlap=None, window='hamming'):
    """
    Compute the Inverse Short-Time Fourier Transform (ISTFT) to reconstruct the signal using scipy.

    Parameters:
    Zxx (numpy array): STFT of the signal.
    fs (float): Sampling frequency of the signal.
    nperseg (int): Length of each segment.
    noverlap (int): Number of points to overlap between segments.
    window (str or tuple or array_like): Desired window to use.

    Returns:
    t, x: Reconstructed signal.
    """
    t, x = scipy.signal.istft(Zxx, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    return t, x

def fft_one_block(x,fs=1.0,overlap = 0, nowindow=False,return_onesided=True,subtract_mean=True):
    """
    Computed rFFT in time with one block and no damping
    """
    #segement the signal 
    nperseg = len(x)
    if return_onesided:
        wsfreq = np.fft.rfftfreq(nperseg,d=1/fs)
    else:
        wsfreq = np.fft.fftfreq(nperseg,d=1/fs)
    
    segments = [x[i:i+nperseg] for i in range(0,len(x)-nperseg+1,nperseg-overlap)]

    if nowindow:
        windowed_segments = segments
    else:
        window = np.hamming(nperseg)
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
    #window = sp.signal.get_window('hamming',nperseg)
    window = np.hamming(nperseg)
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
        {'key':'nperseg',   'required':False,  'default':None,
         'help':'Number of snapshots per segment to specify number of blocks. Default is 1 block.', },
        {'key':'xc',   'required':False,  'default':None,
         'help':'Wake center on xaxis', },
        {'key':'yc',   'required':False,  'default':None,
         'help':'Wake center on yaxis', },
        {'key':'xaxis',    'required':False,  'default':'y',
        'help':'Which axis to use on the abscissa', },
        {'key':'yaxis',    'required':False,  'default':'z',
        'help':'Which axis to use on the ordinate', },
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
        {'key':'iplane',       'required':False,  'default':None,
         'help':'List of i-index of plane to postprocess', },
        {'key':'correlations','required':False,  'default':['U',],
            'help':'List of correlations to include in SPOD. Separate U,V,W components with dash. Examples: U-V-W, U,V,W,V-W ', },
        {'key':'output_dir',  'required':False,  'default':'./','help':'Directory to save results'},
        {'key':'savepklfile', 'required':False,  'default':'',
        'help':'Name of pickle file to save results', },
        {'key':'loadpklfile', 'required':False,  'default':None,
        'help':'Name of pickle file to load to perform actions', },
        {'key':'compute_eigen_vectors', 'required':False,  'default':True,
        'help':'Boolean to compute eigenvectors or just eigenvalues', },
        {'key':'sort', 'required':False,  'default':True,
        'help':'Boolean to included sorted wavenumber and frequency indices by eigenvalue', },
        {'key':'save_num_modes', 'required':False,  'default':None,
        'help':'Number of eigenmodes to save, ordered by eigenvalue. Modes will be save in array of shape (save_num_mods,NR).', },
        {'key':'cylindrical_velocities', 'required':False,  'default':False,
        'help':'Boolean to use cylindrical velocity components instead of cartesian. If True U->U_x, V->U_r, W->U_\Theta', },
        {'key':'save_all_proj_coeff', 'required':False,  'default':False,
        'help':'Boolean to precompute and store the projection coefficients of the original signal onto all POD eigenvectors. Will significantly slow down SPOD computation. Use save_num_modes to only store a subset', },
        {'key':'varnames',  'required':False,  'default':['velocityx', 'velocityy', 'velocityz'],
         'help':'Variables to extract from the netcdf file',},        
        {'key':'verbose',  'required':False,  'default':True,
         'help':'Print extra information.',},        
        {'key':'nowindow',  'required':False,  'default':False,
         'help':'Do not window time fourier transform with single block (e.g.,, for periodic signals in time).',},        
        
    ]
    example = """
```yaml
spod:
  name: Wake YZ plane
  ncfile: /lustre/orion/cfd162/world-shared/lcheung/ALCC_Frontier_WindFarm/farmruns/LowWS_LowTI/ABL_ALM_10x10/rundir_AWC0/post_processing/rotor_*.nc
  iplane:
    - 7
    - 8
  group: T00_rotor
  trange: [25400,26000]
  nperseg: 256
  diam: 240
  #xc: 480
  yc: 150
  NR: 128
  NTheta: 128
  LR_factor: 1.2
  xaxis: 'a1'
  yaxis: 'a2'
  varnames: ['velocitya1','velocitya2','velocitya3']
  wake_meandering_stats_file:
    - ./T00_wake_meandering/wake_stats_7.csv
    - ./T00_wake_meandering/wake_stats_8.csv
  cylindrical_velocities: False
  correlations:
    - U
  output_dir: ./T00_spod_results/
  savepklfile: spod_{iplane}.pkl
  save_num_modes: 100
  verbose: True
  #loadpklfile: ./test/test.pkl

  plot_eigvals:
    num: 11
    savefile: ./eigvals_{iplane}.png
    correlations:
      - U
    Uinf: 6.5

  plot_eigmodes:
    num: 1
    Uinf: 6.5
    savefile: ./eigmode_{iplane}.png
    correlations:
      - U
```
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
        for planeiter, plane in enumerate(self.yamldictlist):
            iplanes               = plane['iplane']
            self.trange           = plane['trange']
            ncfile                = plane['ncfile']
            self.diam             = plane['diam']
            group                 = plane['group']
            LR_factor             = plane['LR_factor']
            NR                    = plane['NR']
            NTheta                = plane['NTheta']
            ycenter               = plane['xc']
            zcenter               = plane['yc']
            self.nperseg          = plane['nperseg']
            correlations          = plane['correlations']
            self.remove_temporal_mean  = plane['remove_temporal_mean']
            remove_azimuthal_mean = plane['remove_azimuthal_mean']
            savefile              = plane['savepklfile']
            loadpklfile           = plane['loadpklfile']
            self.output_dir       = plane ['output_dir']
            compute_eigen_vectors = plane ['compute_eigen_vectors']
            sort                  =  plane ['sort']
            wake_center_files = plane['wake_meandering_stats_file']
            save_num_modes        = plane['save_num_modes']
            save_proj_coeff       = plane['save_all_proj_coeff']
            self.cylindrical_velocities= plane['cylindrical_velocities']
            self.xaxis    = plane['xaxis']
            self.yaxis    = plane['yaxis']
            self.varnames = plane['varnames']
            self.verbose = plane['verbose']
            self.nowindow = plane['nowindow']


            #Get all times if not specified 
            #if iplanes == None: iplanes = list(range(len(self.db['offsets'])))
            #if not isinstance(iplanes, list): iplanes = [iplanes,]
            if not isinstance(correlations, list): correlations= [correlations,]
            if not wake_center_files == None and not isinstance(wake_center_files, list): wake_center_files = [wake_center_files,]
            if self.verbose:
                print("--> Reading in velocity data",flush=True)
            udata_cart,xcs,y,z,self.times,iplanes = read_cart_data(ncfile,self.varnames,group,self.trange,iplanes,self.xaxis,self.yaxis)
            #file = 'ucart_data_pulse.pkl'
            #file = 'ucart_data.pkl'
            # with open(file,'wb') as f:
            #     pickle.dump(udata_cart,f)
            #     pickle.dump(xcs,f)
            #     pickle.dump(y,f)
            #     pickle.dump(z,f)
            #     pickle.dump(self.times,f)
            #     pickle.dump(iplanes,f)
            # sys.exit()
            # with open(file, 'rb') as f:
            #     udata_cart = pickle.load(f)
            #     xcs = pickle.load(f)
            #     y = pickle.load(f)
            #     z = pickle.load(f)
            #     self.times = pickle.load(f)
            #     iplanes  = pickle.load(f)
            if wake_center_files != None and len(wake_center_files) != len(iplanes):
                print("Error: len(wake_center_files) != len(iplanes). Exiting.",flush=True)
                sys.exit()

            for iplaneiter, iplane in enumerate(iplanes):
                print("--> Working on iplane: ",iplane," of ",iplanes,flush=True)
                self.iplane = iplane

                tsteps = range(len(self.times))
                dt = self.times[1]-self.times[0]

                """
                Define polar grid 
                """
                LR = (self.diam/2.0)*LR_factor #specify radial extent for POD analysis 
                LTheta = 2 * np.pi
                r = np.linspace(0,LR,NR)
                theta = np.linspace(0,LTheta,NTheta+1)[0:-1] #periodic grid in theta
                self.RR, self.TT = np.meshgrid(r,theta,indexing='ij')
                dr = r[1]-r[0]
                dtheta = theta[1]-theta[0]
                components = ['velocityx','velocityy','velocityz']

                if self.verbose:
                    print("--> Interpolating cartesian data to polar coordinates",flush=True)

                if wake_center_files != None:
                    wake_meandering_stats_file = wake_center_files[iplaneiter]
                    wake_meandering_stats = pd.read_csv(wake_meandering_stats_file)
                    ycenter = wake_meandering_stats[self.xaxis + 'c_mean'][0]
                    zcenter = wake_meandering_stats[self.yaxis + 'c_mean'][0]
                    if self.verbose:
                        print("--> Read in mean wake centers from ",wake_meandering_stats_file+". "+self.xaxis+"c = "+str(ycenter)+", "+self.yaxis+"c = "+str(zcenter)+".",flush=True)
                else:
                    if plane['xc'] == None: 
                        ycenter = (y[-1]+y[0])/2.0
                        if self.verbose:
                            print("--> Centering on middle of xaxis: ",ycenter,flush=True)
                    if plane['yc'] == None: 
                        zcenter = (z[-1]+z[0])/2.0
                        if self.verbose:
                            print("--> Centering on middle of yaxis: ",zcenter,flush=True)

                self.udata_polar = np.zeros((NR,NTheta,len(tsteps),len(components)))
                for titer , t in enumerate(tsteps):
                    for compind in range(len(components)):
                        if zcenter-LR < 0:
                            print("Error: zcenter - LR negative. Exiting",flush=True)
                            print("zcenter: ",zcenter,", LR: ",LR,", zcenter-LR: ",zcenter-LR,flush=True)
                            sys.exit()
                        self.udata_polar[:,:,titer,compind]  = interpolate_cart_to_radial(udata_cart[iplane][titer,:,:,compind],y,z,self.RR,self.TT,ycenter,zcenter)

                if self.cylindrical_velocities==True:
                    if self.verbose:
                        print("--> Transforming to cylindrical velocity components",flush=True)
                    for titer , t in enumerate(tsteps):
                        v_vel = np.copy(self.udata_polar[:,:,titer,1])
                        w_vel = np.copy(self.udata_polar[:,:,titer,2])

                        self.udata_polar[:,:,titer,1] = v_vel * np.cos(self.TT) + w_vel * np.sin(self.TT)
                        self.udata_polar[:,:,titer,2] = -v_vel * np.sin(self.TT) + w_vel * np.cos(self.TT)

                if self.nperseg==None:
                    self.nperseg = len(self.times)
                    print("nperseg: ",self.nperseg,flush=True)

                if self.nowindow:
                    self.overlap = 0 
                    NB = int(1.0)
                    tfreq ,  _ = fft_one_block(self.udata_polar[0,0,:,0],fs=1.0/dt,overlap=0,nowindow=self.nowindow,subtract_mean=self.remove_temporal_mean)
                else:
                    self.overlap = self.nperseg//2
                    tfreq , self.time_segments, _ = compute_stft(self.udata_polar[0,0,:,0],fs=1.0/dt,nperseg=self.nperseg,noverlap=self.overlap,subtract_mean=False)
                    NB = self.time_segments.shape[0] #number of blocks 
                angfreq = tfreq * 2 * np.pi #compute angular frequencies 
                Nkt = len(angfreq)
                #LT = time_segments[0][-1]-time_segments[0][0]
                #LT = self.times[int(time_segments[1])] - self.times[time_segments[0]]
                #angfreq = get_angular_wavenumbers(self.nperseg,LT)
                #angfreq = angfreq[0:Nkt]

                if self.verbose:
                    print("--> Fourier transforming in time (number of blocks = "+str(NB) + ")",flush=True)
                self.udata_that = np.zeros((NR,NTheta,NB,Nkt,3),dtype=complex)
                for rind in np.arange(0,len(r)):
                    for thetaind in np.arange(0,len(theta)):
                        for compind,comp in enumerate(components):
                            temp_signal = self.udata_polar[rind,thetaind,:,compind]
                            if self.remove_temporal_mean:
                                temp_signal = temp_signal - np.mean(temp_signal) #subtract out global mean of signal
                            #tfreq , tfft = welch_fft(temp_signal,fs=1/dt,nperseg=self.nperseg,subtract_mean=self.remove_temporal_mean)
                            if self.nowindow:
                                tfreq , tfft = fft_one_block(temp_signal,fs=1.0/dt,overlap=0,nowindow=self.nowindow,subtract_mean=self.remove_temporal_mean)
                            else:
                                #_ , _ , tfft = compute_stft(temp_signal,fs=1.0/dt,nperseg=self.nperseg,noverlap=self.overlap,subtract_mean=self.remove_temporal_mean)
                                _ , _ , tfft = compute_stft(temp_signal,fs=1.0/dt,nperseg=self.nperseg,noverlap=self.overlap,subtract_mean=False)
                            self.udata_that[rind,thetaind,:,:,compind] = tfft

                if self.verbose:
                    print("--> Fourier transforming in Theta",flush=True)
                NkTheta = int(NTheta)
                self.udata_rhat = np.zeros((NR,NkTheta,NB,Nkt,len(components)),dtype=complex)
                for rind in np.arange(0,NR):
                    for block in np.arange(0,NB):
                        for ktind in np.arange(0,Nkt):
                            for compind,comp in enumerate(components):
                                temp_signal = self.udata_that[rind,:,block,ktind,compind] 
                                if remove_azimuthal_mean:
                                    temp_signal = temp_signal - np.mean(temp_signal) 
                                self.udata_rhat[rind,:,block,ktind,compind] = np.fft.fft(temp_signal)

                #del udata_that #don't need this anymore 
                thetafreq = np.fft.fftfreq(NTheta,d=dtheta)
                ktheta  = get_angular_wavenumbers(NTheta,LTheta,negative=True)

                if loadpklfile==None:

                    """
                    Compute the POD via the SVD for each ktheta and kt of interest
                    """

                    if compute_eigen_vectors:
                        self.POD_modes       = {corr: 0 for corr in correlations}
                        if save_proj_coeff:
                            self.POD_proj_coeff  = {corr: 0 for corr in correlations}
                    self.POD_eigenvalues = {corr: 0 for corr in correlations}

                    if sort:
                        self.sorted_inds  = {corr: {} for corr in correlations}

                    self.W1D = form_weighting_matrix_simpsons_rule(r)
                    scaling_factor_k = dt / (sum(np.hamming(self.nperseg)**2)*NB) 
                    #self.W1D = np.eye(len(r))
                    #scaling_factor_k = 1.0
                    #scaling_factor_k = 1.0/NB


                    corr_dict = {'U': 0, 'V': 1, 'W': 2}
                    for corr in correlations:
                        if self.verbose:
                            print("--> Computing SPOD for correlations: ",corr,flush=True)
                        comp_corr = corr.split('-')
                        corr_inds = [corr_dict[corr.upper()] for corr in comp_corr]

                        W = np.kron(np.eye(len(corr_inds)),self.W1D)
                        Wsqrt = np.sqrt(W)
                        Wsqrtinv = np.zeros_like(Wsqrt)
                        for i in range(NR*len(corr_inds)):
                            if (Wsqrt[i,i]==0):
                                Wsqrtinv[i,i] = 0
                            else:
                                Wsqrtinv[i,i] = 1.0 / Wsqrt[i,i]
                        if compute_eigen_vectors:
                            self.POD_modes[corr]       = np.zeros((NR,len(ktheta),len(tfreq),NB,len(corr_inds)),dtype=complex)

                            #First NB is block, second is eigvalue. To reconstruct the original signal, each block needs
                            #to be projected onto each eigenvector
                            if save_proj_coeff:
                                self.POD_proj_coeff[corr]  = np.zeros((len(ktheta),len(tfreq),NB,NB),dtype=complex)

                        self.POD_eigenvalues[corr] = np.zeros((len(ktheta),len(tfreq),NB),dtype=complex)

                        for ktheta_ind , ktheta_val in enumerate(ktheta):
                            for tfreq_ind , tfreq_val in enumerate(tfreq):
                                POD_Mat = reshape_stack_velocities(self.udata_rhat[:,ktheta_ind,:,tfreq_ind,:],NR,NB,corr)
                                #POD_Mat = np.zeros((NR*len(corr_inds),NB),dtype=complex)
                                #for corr_ind_iter , corr_ind in enumerate(corr_inds):
                                    #POD_Mat[corr_ind_iter*NR:NR*(corr_ind_iter+1),0:NB] = np.copy(self.udata_rhat[:,ktheta_ind,:,tfreq_ind,corr_ind])
                                POD_Mat_scaled = np.sqrt(scaling_factor_k) * np.dot(Wsqrt,POD_Mat)
                                
                                if compute_eigen_vectors:
                                    lsvd, ssvd, rsvd = np.linalg.svd(POD_Mat_scaled,full_matrices=False,compute_uv=True)
                                    #eigmodes = np.zeros((NR,NB,len(corr_inds)),dtype=complex)
                                    eigmodes_stacked = np.dot(Wsqrtinv,lsvd)
                                    eigmodes = reshape_unstack_velocities(eigmodes_stacked,NR,NB,corr)
                                    self.POD_modes[corr][:,ktheta_ind,tfreq_ind,:,:] = np.copy(eigmodes)

                                    if save_proj_coeff:
                                        for block_ind in range(NB): #each block of the original signal
                                            for eig_ind in range(NB): #each eigvenvector 
                                                self.POD_proj_coeff[corr][ktheta_ind,tfreq_ind,block_ind,eig_ind] = \
                                                    inner_product(eigmodes_stacked[:,eig_ind],POD_Mat[:,block_ind],W)
                                else:
                                    ssvd = np.linalg.svd(POD_Mat_scaled,full_matrices=False,compute_uv=False)
                                eigval = ssvd**2
                                self.POD_eigenvalues[corr][ktheta_ind,tfreq_ind,:] = eigval
                        if sort:
                            sorted_eigind = np.argsort(np.abs(self.POD_eigenvalues[corr]),axis=None)[::-1]
                            self.sorted_inds[corr]['ktheta'] , self.sorted_inds[corr]['angfreq'], self.sorted_inds[corr]['block'] = np.unravel_index(sorted_eigind,self.POD_eigenvalues[corr][:,:,:].shape)


                    self.variables = {}
                    self.variables['angfreq']  = angfreq
                    self.variables['ktheta']   = ktheta
                    self.variables['blocks']   = np.arange(0,NB)
                    self.variables['r']        = r
                    self.variables['theta']    = theta
                    self.variables['y']        = y
                    self.variables['z']        = z
                    self.variables['x']        = xcs[iplaneiter]
                    self.variables['ycenter']  = ycenter
                    self.variables['zcenter']  = zcenter
                    self.variables['nperseg']  = self.nperseg
                    self.variables['noverlap'] = self.overlap
                    self.variables['times']    = self.times
                    if len(savefile)>0:
                        if not os.path.exists(self.output_dir):
                            os.makedirs(self.output_dir)
                        savefname = savefile.format(iplane=iplane)
                        savefilename = os.path.join(self.output_dir, savefname)
                        if self.verbose:
                            print("--> Saving to: ",savefilename,flush=True)
                        objects = [] 
                        objects.append(self.POD_eigenvalues)
                        objects.append(self.variables)
                        if sort:
                            objects.append(self.sorted_inds)


                        if compute_eigen_vectors:
                            if save_num_modes == None:
                                objects.append(self.POD_modes)
                                if save_proj_coeff:
                                    objects.append(self.POD_proj_coeff)
                            else:
                                save_modes      = {}
                                save_proj_coeff = {}
                                corr_dict = {'U': 0, 'V': 1, 'W': 2}
                                for corr in correlations:
                                    comp_corr = corr.split('-')
                                    corr_inds = [corr_dict[corr.upper()] for corr in comp_corr]
                                    W1D = form_weighting_matrix_simpsons_rule(r)
                                    W = np.kron(np.eye(len(corr_inds)),W1D)
                                    save_modes[corr] = np.zeros((save_num_modes,NR,len(corr.split('-'))),dtype=complex)
                                    save_proj_coeff[corr] = np.zeros((save_num_modes,NB),dtype=complex)
                                    for mode in range(save_num_modes):
                                        ktheta_ind  = self.sorted_inds[corr]['ktheta'][mode]
                                        angfreq_ind = self.sorted_inds[corr]['angfreq'][mode]
                                        eig_ind     = self.sorted_inds[corr]['block'][mode]
                                        save_modes[corr][mode,:,:] = self.POD_modes[corr][:,ktheta_ind,angfreq_ind,eig_ind,:]
                                        save_proj_coeff[corr][mode,:] = compute_projection_coefficient(self.udata_rhat[:,ktheta_ind,:,angfreq_ind,:],save_modes[corr][mode,:,:],corr,NR,NB,W)
                                objects.append(save_modes)
                                objects.append(save_proj_coeff)
                        with open(savefilename, 'wb') as f:
                            for obj in objects:
                                pickle.dump(obj, f)

                if loadpklfile!=None:
                    print("--> Loading from: ",loadpklfile,flush=True)
                    with open(loadpklfile, 'rb') as f:
                        self.POD_eigenvalues = pickle.load(f)
                        self.variables = pickle.load(f)
                        if sort:
                            self.sorted_inds = pickle.load(f)
                        if compute_eigen_vectors:
                            self.POD_modes      = pickle.load(f)
                            try:
                                self.POD_proj_coeff = pickle.load(f)
                            except:
                                self.POD_proj_coeff = None

                # Do any sub-actions required for this task for each plane
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
                if self.parent.verbose:
                    print("Leading ktheta: ",self.parent.variables['ktheta'][self.parent.sorted_inds[corr]['ktheta'][0:num]])
                    print("Leading wD/U2pi: ",self.parent.variables['angfreq'][self.parent.sorted_inds[corr]['angfreq'][0:num]]*self.parent.diam/(Uinf * 2 * np.pi))


            if len(savefile)>0:
                savefname = savefile.format(iplane=self.parent.iplane)
                savefilename = os.path.join(self.parent.output_dir, savefname)
                plt.savefig(savefilename)

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

            NR      = len(self.parent.variables['r'])
            NTheta  = len(self.parent.variables['theta'])
            angfreq = self.parent.variables['angfreq']
            numSteps = len(self.parent.times)
            angfreq_full = 2*np.pi*np.fft.rfftfreq(numSteps,self.parent.times[1]-self.parent.times[0])

            for corr in correlations:
                eig_modes = self.parent.POD_modes[corr]
                shape = self.parent.POD_modes[corr].shape

                mode_rhat = np.zeros((NR,NTheta,len(angfreq_full),shape[-1]),dtype=complex) 

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
                    angfreq_full_ind = np.argmin(abs(angfreq[angfreq_ind] - angfreq_full))
                    block_ind = self.parent.sorted_inds[corr]['block'][ind]

                    if self.parent.POD_modes[corr].shape[0] != NR: ##loaded save_modes
                        proj_coeff  = self.parent.POD_proj_coeff[corr][ind]
                        mode_rhat[:,ktheta_ind,angfreq_full_ind,:] += proj_coeff*self.parent.POD_modes[corr][ind]
                    else:
                        proj_coeff  = self.parent.POD_proj_coeff[corr][ktheta_ind,angfreq_ind,block_ind]
                        mode_rhat[:,ktheta_ind,angfreq_full_ind,:] += proj_coeff*self.parent.POD_modes[corr][:,ktheta_ind,angfreq_ind,block_ind,:]
                    print("---> Adding mode for ktheta = ",self.parent.variables['ktheta'][ktheta_ind]," and St = ", self.parent.variables['angfreq'][angfreq_ind] * scaling)

                mode_r = np.fft.irfft(np.fft.ifft(mode_rhat,axis=1),axis=2,n=numSteps)

                #scaling = self.parent.diam/(Uinf * 2 * np.pi)
                fig,ax = plt.subplots(1,1,sharey=False,sharex=False,figsize=(12,10),subplot_kw={'projection':'polar'})
                val = mode_r[:,:,itime,0]
                theta = self.parent.variables['theta']
                r = self.parent.variables['r']
                plot_radial(val,theta,r,cmap='jet',newfig=False,ax=ax)
            
            if len(savefile)>0:
                savefname = savefile.format(iplane=self.parent.iplane)
                savefilename = os.path.join(self.parent.output_dir, savefname)
                plt.savefig(savefilename)

            return

    @registeraction(actionlist)
    class radial_shear_stress_flux():
        actionname = 'radial_shear_stress_flux'
        blurb      = 'Compute radial shear stress flux contribution from individual streamwise SPOD modes'
        required   = False
        actiondefs = [
        {'key':'number_of_modes',   'required':False,  'default':1,
         'help':'Number of individual eigenvectors to include in reconstruction, sorted by eigenvalues', },
        {'key':'savefile',  'required':False,  'default':'',
         'help':'Filename to save results', },
        {'key':'correlations','required':False,  'default':['U',],
         'help':'List of correlations', },
        {'key':'decompose_radial_velocity','required':False,  'default':False,
         'help':'Boolean to apply SPOD decomposition to radial velocity in addition to streamwise velocity. Radial velocity must be included in correlations.', },
        {'key':'r','required':True,  'default':0,
            'help':'Radius value to compute radial shear stress flux.', },
        {'key':'Uinf','required':False, 'default':None,
            'help':'Inflow velocity for defining Strouhal number.', },
        {'key':'ktheta_list','required':False,  'default':None,
         'help':'List of kthetas to including in reconstruction. Override number of modes. ', },
        {'key':'St_list','required':False,  'default':None,
         'help':'List of Strouhal numbers to including in reconstruction. Override number of modes. ', },
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent
            print('Initialized '+self.actionname+' inside '+parent.name)
            return

        def evaluate_at_r(self,u_r,r,reval):
            evaluated_value = np.interp(reval, r, u_r)
            return evaluated_value 

        def integrate_in_r(self,u,r,r1,r2):
            idx1 = np.searchsorted(r, r1)
            idx2 = np.searchsorted(r, r2)
            # Slice the arrays to get the range of interest
            r_range = r[idx1:idx2+1]
            u_range = u[idx1:idx2+1]
            # Perform the integration using Simpson's rule
            integral_simps = scipy.integrate.simps(u_range, r_range)
            return integral_simps


        def execute(self):
            print('Executing '+self.actionname)
            numModes              = self.actiondict['number_of_modes']
            savefile = self.actiondict['savefile']
            correlations = self.actiondict['correlations']
            Uinf         = self.actiondict['Uinf']
            rval         = self.actiondict['r']
            ktheta_list  = self.actiondict['ktheta_list']
            St_list      = self.actiondict['St_list']
            decompose_radial_velocity = self.actiondict['decompose_radial_velocity']
            if not isinstance(correlations, list): correlations= [correlations,]
            if not ktheta_list  == None:
                if not isinstance(ktheta_list,list): St_list=[St_list,]
            if not St_list == None:
                if not isinstance(St_list,list): St_list=[St_list,]

            ### Convert to cylindrical velocity 
            numSteps = len(self.parent.times)
            dt = self.parent.times[1]-self.parent.times[0]
            if self.parent.cylindrical_velocities == False:
                radial_velocity = np.zeros_like(self.parent.udata_polar[:,:,:,1])
                print("--> Transforming to cylindrical velocity components")
                for titer in range(0,numSteps):
                    v_vel = np.copy(self.parent.udata_polar[:,:,titer,1])
                    w_vel = np.copy(self.parent.udata_polar[:,:,titer,2])

                    #Radial velocity 
                    radial_velocity[:,:,titer] = v_vel * np.cos(self.parent.TT) + w_vel * np.sin(self.parent.TT)

                    #Azimuthal velocity -- not needed 
                    #self.parent.udata_polar[:,:,titer,2] = -v_vel * np.sin(self.parent.TT) + w_vel * np.cos(self.parent.TT)
            else:
                radial_velocity = self.parent.udata_polar[:,:,:,1]

            NTheta  = len(self.parent.variables['theta'])
            NR      = len(self.parent.variables['r'])
            r = self.parent.variables['r']
            W1D = form_weighting_matrix_simpsons_rule(r)
            ktheta  = self.parent.variables['ktheta']
            angfreq = self.parent.variables['angfreq'] #angular frequencies for each block 

            db = {corr: {} for corr in correlations}
            db_velxr = {} 
            db_total = {} 

            ### Compute time averaged streamwise velocity 
            streamwise_velocity = self.parent.udata_polar[:,:,:,0]
            velocityx_avg  = np.mean(streamwise_velocity,axis=2)
            velocityx_fluc = streamwise_velocity - np.mean(streamwise_velocity,axis=2,keepdims=True) 


            ### Compute radial fluctuations
            velocityr_avg  = np.mean(radial_velocity,axis=2,keepdims=True)
            velocityr_fluc = radial_velocity - velocityr_avg
            #velocityy_fluc = self.parent.udata_polar[:,:,:,1] - velocityy_avg
            #velocityz_fluc = self.parent.udata_polar[:,:,:,2] - velocityz_avg

            db_velxr['velocityx_avg'] = velocityx_avg
            db_velxr['velocityr_avg'] = velocityr_avg

            uxur_avg = np.mean(velocityx_fluc * velocityr_fluc,axis=2)
            db_total['uxur_avg'] = uxur_avg 
            db_total['ux_avg_uxur_avg'] = velocityx_avg * uxur_avg 
            db_total['int_neg_ux_avg_uxur_avg'] = self.evaluate_at_r(np.mean(-db_total['ux_avg_uxur_avg'],axis=1),r,rval)
            print("int_neg_ux_avg_uxur_avg",db_total['int_neg_ux_avg_uxur_avg'],flush=True)

            corr_dict = {'U': 0, 'V': 1, 'W': 2}
            for corr in correlations:
                comp_corr = corr.split('-')
                corr_inds = [corr_dict[corr.upper()] for corr in comp_corr]
                if not 0 in corr_inds:
                    print("Error: must included U correlations to decompose streamwise velocity...exiting.")
                    sys.exit()
                W = np.kron(np.eye(len(corr_inds)),W1D)
                db[corr] = {}
                shape = self.parent.POD_modes[corr].shape
                #loop over leading modes in order of eigenvalues
                if ktheta_list == None or St_list == None: 
                    mode_list = range(0,numModes)
                else: 
                    mode_list = []
                    for mode_iter in range(len(ktheta_list)):
                        mode_number = get_mode_number(1,self.parent.sorted_inds,self.parent.variables,corr,Uinf,self.parent.diam,St=St_list[mode_iter],ktheta=ktheta_list[mode_iter],tol=None)
                        mode_list.append(mode_number[0])
                for mode_ind in mode_list:
                    db[corr][mode_ind] = {}
                    ktheta_ind  = self.parent.sorted_inds[corr]['ktheta'][mode_ind]
                    angfreq_ind = self.parent.sorted_inds[corr]['angfreq'][mode_ind]
                    eig_ind = self.parent.sorted_inds[corr]['block'][mode_ind]
                    eig_val = self.parent.POD_eigenvalues[corr][ktheta_ind,angfreq_ind,eig_ind]
                    print("Reconstructing ktheta = ",ktheta[ktheta_ind], ", angfreq = ",angfreq[angfreq_ind],", eig ind = ", eig_ind , ", mode number = ",mode_ind, ", eigenvalue = ",eig_val,flush=True)

                    u_r_ktheta_f = reconstruct_r_ktheta_f(self.parent.udata_rhat,mode_ind,eig_ind,self.parent.POD_modes,self.parent.sorted_inds,self.parent.variables,corr,W)

                    #u_r_theta_t dims (NR,NTheta,numSteps,NCorr)
                    u_r_theta_t = transform_to_realspace(u_r_ktheta_f,self.parent.variables,numSteps,dt,corr,components=None,nowindow=self.parent.nowindow)

                    #subtract global mean of recustructed signal. 
                    velocityx_fluc_mode = u_r_theta_t[:,:,:,0] - np.mean(u_r_theta_t[:,:,:,0],axis=2,keepdims=True) 
                    uxmodeur_avg = np.mean(velocityx_fluc_mode * velocityr_fluc,axis=2)
                    #compute u'_x u'_r
                    db[corr][mode_ind]['uxmodeur_avg'] = uxmodeur_avg 

                    #Compute the radial shear stress flux u_avg * u'_x u'_r
                    db[corr][mode_ind]['ux_avg_uxmodeur_avg'] = velocityx_avg * uxmodeur_avg 

                    db[corr][mode_ind]['int_neg_ux_avg_uxmodeur_avg'] = self.evaluate_at_r(np.mean(-db[corr][mode_ind]['ux_avg_uxmodeur_avg'],axis=1),r,rval)

                    print("--> int_neg_ux_avg_uxmodeur_avg",db[corr][mode_ind]['int_neg_ux_avg_uxmodeur_avg'],flush=True)

                    if decompose_radial_velocity and 0 in corr_inds and 1 in corr_inds:
                        #subtract global mean of recustructed signal. 
                        try:
                            velocityr_fluc_mode = u_r_theta_t[:,:,:,1] - np.mean(u_r_theta_t[:,:,:,1],axis=2,keepdims=True) 
                            uxmodeurmode_avg = np.mean(velocityx_fluc_mode * velocityr_fluc_mode,axis=2)
                            #compute u'_x u'_r
                            db[corr][mode_ind]['uxmodeurmode_avg'] = uxmodeurmode_avg 

                            #Compute the radial shear stress flux u_avg * u'_x u'_r
                            db[corr][mode_ind]['ux_avg_uxmodeurmode_avg'] = velocityx_avg * uxmodeurmode_avg 

                            db[corr][mode_ind]['int_neg_ux_avg_uxmodeurmode_avg'] = self.evaluate_at_r(np.mean(-db[corr][mode_ind]['ux_avg_uxmodeurmode_avg'],axis=1),r,rval)

                            print("--> int_neg_ux_avg_uxmodeurmode_avg",db[corr][mode_ind]['int_neg_ux_avg_uxmodeurmode_avg'],flush=True)

                        except:
                            print("Radial velocity must be included in list of correlations...exiting.")
                            return
                print("\n",flush=True)


            #save the results
            if len(savefile)>0:
                savefname = savefile.format(iplane=self.parent.iplane)
                savefilename = os.path.join(self.parent.output_dir, savefname)
                with open(savefilename, 'wb') as f:
                    pickle.dump(db, f)
                    pickle.dump(db_total, f)
                    pickle.dump(db_velxr,f)
            return

    @registeraction(actionlist)
    class unit_tests():
        actionname = 'unit_tests'
        blurb      = 'Run SPOD unit test suite'
        required   = False
        actiondefs = [
        {'key':'correlations','required':False,  'default':['U',],
         'help':'List of correlations', },
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent
            print('Initialized '+self.actionname+' inside '+parent.name)
            return

        def execute(self):
            print('Executing '+self.actionname)
            correlations = self.actiondict['correlations']
            if not isinstance(correlations, list): correlations= [correlations,]


            ### Convert to cylindrical velocity 
            numSteps = len(self.parent.times)
            dt = self.parent.times[1]-self.parent.times[0]

            """
            First unit test is reconstructing a 1D signal using the stfft functionality
            """
            # Example usage
            print()
            print("Running unit test: Short Time FFT reconstrution of 1D signal")
            fs = 1.0/dt
            nperseg = self.parent.nperseg
            overlap = self.parent.overlap
            window='hamming'
            original_signal = np.random.randn(numSteps)
            # Compute the STFT
            f, t, Zxx = compute_stft(original_signal, fs=fs, nperseg=nperseg, noverlap=overlap,subtract_mean = False,window=window)
            Zxx = Zxx.swapaxes(0,1) #need to swap for istft
            # Compute the ISTFT to reconstruct the signal
            _, reconstructed_signal = compute_istft(Zxx, fs=fs, nperseg=nperseg, noverlap=overlap,window=window)
            # Check the maximum absolute difference
            max_difference = np.max(np.abs(original_signal - reconstructed_signal[:len(original_signal)]))
            np.testing.assert_array_almost_equal(max_difference, 1e-12)
            print(f"--> Maximum difference: {max_difference:.6e}")
            print()


            """
            #Second unit test is testing the orthogonality of the eigenvectors 
            """
            print("Running unit test: Orthogonality of eigenvectors")
            components = [0,] 
            NTheta  = len(self.parent.variables['theta'])
            r       = self.parent.variables['r']
            NR      = len(r)
            angfreq = self.parent.variables['angfreq'] #angular frequencies for each block 
            NB      = len(self.parent.variables['blocks'])
            mode = 0 #just pick out the leading mode for now
            corr_dict = {'U': 0, 'V': 1, 'W': 2}
            for corr in correlations:
                comp_corr = corr.split('-')
                corr_inds = [corr_dict[corr.upper()] for corr in comp_corr]
                W1D = form_weighting_matrix_simpsons_rule(r)
                W = np.kron(np.eye(len(corr_inds)),W1D)
                #numModes = len(self.parent.sorted_inds[corr]['ktheta'])
                ktheta_ind  = self.parent.sorted_inds[corr]['ktheta'][mode]
                angfreq_ind = self.parent.sorted_inds[corr]['angfreq'][mode]

                # Test orthogonality of eigenvectors
                if self.parent.POD_modes[corr].shape[0] == NR: 
                    eigmodes_stacked = reshape_stack_velocities(self.parent.POD_modes[corr][:,ktheta_ind,angfreq_ind,:,:],NR,NB,corr)
                    max_difference = 0.0
                    for i in range(NB):
                        psi1 = eigmodes_stacked[:,i]
                        for j in range(NB):
                            if i != j:
                                psi2 = eigmodes_stacked[:,j]
                                inner_test = np.abs(inner_product(psi1,psi2,W))
                                np.testing.assert_array_almost_equal(inner_test, 1e-12)
                                max_difference = max(inner_test,max_difference)
                    print(f"--> Maximum difference: {max_difference:.6e}")
                    print()
                else:
                    print("Skipping test (rerun without savefile to activate test)")
                    print()

            """
            #Third unit test is reconstructing the streamwise velocity in r for ktheta and f 
            """
            print("Running unit test: reconstructing streamwise velocity in r for ktheta and f ")
            components = [0,] 
            NTheta  = len(self.parent.variables['theta'])
            r       = self.parent.variables['r']
            NR      = len(r)
            angfreq = self.parent.variables['angfreq'] #angular frequencies for each block 
            NB      = len(self.parent.variables['blocks'])
            modes   = 0 
            corr_dict = {'U': 0, 'V': 1, 'W': 2}
            for corr in correlations:
                ktheta_ind  = self.parent.sorted_inds[corr]['ktheta'][modes]
                angfreq_ind = self.parent.sorted_inds[corr]['angfreq'][modes]
                comp_corr = corr.split('-')
                corr_inds = [corr_dict[corr.upper()] for corr in comp_corr]
                W1D = form_weighting_matrix_simpsons_rule(r)
                W = np.kron(np.eye(len(corr_inds)),W1D)
                u = self.parent.udata_rhat
                eig_inds = -1 #reconstruct using all eigenvectors

                ux_r_ktheta_f = reshape_stack_velocities(u[:,ktheta_ind,:,angfreq_ind,:],NR,NB,corr)
                #mode_r = reconstruct_r_ktheta_f(u,modes,self.parent.POD_modes,self.parent.sorted_inds,self.parent.variables,corr,W,self.parent.POD_proj_coeff)
                mode_r = reconstruct_r_ktheta_f(u,modes,eig_inds,self.parent.POD_modes,self.parent.sorted_inds,self.parent.variables,corr,W)
                mode_r = reshape_stack_velocities(mode_r[:,ktheta_ind,:,angfreq_ind,:],NR,NB,corr)
                max_difference = 0
                for block_ind in range(NB):
                    diff_vec = ux_r_ktheta_f[:,block_ind] - mode_r[:,block_ind]
                    diff = np.abs(inner_product(diff_vec,diff_vec,W))
                    max_difference = max(diff,max_difference)
                print(f"--> Maximum difference: {max_difference:.6e}")
                print()

            """
            #Fourth unit test is reconstructing the full streamwise velocity in r in theta and time
            """
            print("Running unit test: reconstructing streamwise velocity in realspace")
            components = [0,] 
            NTheta  = len(self.parent.variables['theta'])
            r       = self.parent.variables['r']
            NR      = len(r)
            angfreq = self.parent.variables['angfreq'] #angular frequencies for each block 
            NB      = len(self.parent.variables['blocks'])
            modes   = -1 #all modes 
            eig_inds = -1 #reconstruct using all eigenvectors
            corr_dict = {'U': 0, 'V': 1, 'W': 2}
            numSteps = len(self.parent.times)
            for corr in correlations:
                ktheta_ind  = self.parent.sorted_inds[corr]['ktheta'][mode]
                angfreq_ind = self.parent.sorted_inds[corr]['angfreq'][mode]
                comp_corr = corr.split('-')
                corr_inds = [corr_dict[corr.upper()] for corr in comp_corr]
                W1D = form_weighting_matrix_simpsons_rule(r)
                W = np.kron(np.eye(len(corr_inds)),W1D)
                #self.udata_polar = np.zeros((NR,NTheta,len(tsteps),len(components)))
                u_r_theta_t_orig = self.parent.udata_polar - np.mean(self.parent.udata_polar,axis=2,keepdims=True)
                u_r_ktheta_f = reconstruct_r_ktheta_f(self.parent.udata_rhat,modes,eig_inds,self.parent.POD_modes,self.parent.sorted_inds,self.parent.variables,corr,W)
                #np.save('u_r_ktheta_f.npy', u_r_ktheta_f)
                #u_r_ktheta_f = np.load('u_r_ktheta_f.npy')
                u_r_theta_t = transform_to_realspace(u_r_ktheta_f,self.parent.variables,numSteps,dt,corr,components=None,nowindow=self.parent.nowindow)
                max_difference = 0
                for theta_iter in range(NTheta):
                    for time_iter in range(numSteps):
                        ux_r_theta_t_orig    = reshape_stack_velocities_single_block(u_r_theta_t_orig[:,theta_iter,time_iter,:],NR,corr)
                        ux_r_theta_t_stacked = reshape_stack_velocities_single_block(u_r_theta_t[:,theta_iter,time_iter,:],NR,corr)
                        diff_vec = ux_r_theta_t_stacked - ux_r_theta_t_orig
                        diff = np.abs(inner_product(diff_vec,diff_vec,W))
                        max_difference = max(diff,max_difference)
                print(f"--> Maximum difference: {max_difference:.6e}")
                print()
            return
