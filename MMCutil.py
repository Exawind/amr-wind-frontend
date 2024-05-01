#!/usr/bin/env python
# Script to create a wrf profile netCDF file

from netCDF4 import Dataset
import numpy as     np
import sys
from scipy import interpolate

import time
from functools import partial

# See https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def progress(count, total, suffix=''):
    """
    print out a progressbar
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()

def AMRcellcenters(z0, z1, Nz):
    """
    Get the cell centers on the AMR-Wind grid
    """
    dz = (z1-z0)/Nz
    return np.linspace(z0+0.5*dz,z1-0.5*dz,Nz)    

def makeVelArray(prob_hi, prob_lo, n, velfunc, verbose=False):
    getdx = lambda phi, plo, n: (np.array(phi)-np.array(plo))/n
    # Define empty array
    u     = np.zeros((n[0], n[1], n[2]))
    # Get the dx
    dx    = getdx(prob_hi, prob_lo, n)
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                if verbose:
                    count = k+1 + j*n[2] + i*n[1]*n[2] 
                    progress(count, n[0]*n[1]*n[2])
                x = prob_lo[0] + (i+0.5)*dx[0] 
                y = prob_lo[1] + (j+0.5)*dx[1]
                z = prob_lo[2] + (k+0.5)*dx[2]
                u[i,j,k] = velfunc(x,y,z)
    return u

def makeVelArrayZvec(prob_hi, prob_lo, n, velfunc, verbose=False):
    getdx = lambda phi, plo, n: (np.array(phi)-np.array(plo))/n
    # Define empty array
    u     = np.zeros((n[0], n[1], n[2]))
    # Get the dx
    dx    = [getdx(prob_hi[i], prob_lo[i], n[i]) for i in range(len(n))]
    uvec  = np.zeros(n[2])
    for k in range(n[2]):
        i = 0
        j = 0
        x = prob_lo[0] + (i+0.5)*dx[0] 
        y = prob_lo[1] + (j+0.5)*dx[1]
        z = prob_lo[2] + (k+0.5)*dx[2]
        uvec[k] = velfunc(x,y,z)
    for i in range(n[0]):
        if verbose:
            progress(i+1, n[0])
            #count = i*n[1]*n[2]
            #progress(count+1, n[0]*n[1]*n[2])
        for j in range(n[1]):
            u[i,j,:] = uvec
    if verbose: print('')
    return u

def writeUVW_NC(ncfilename, Nx, Ny, Nz, Ux, Uy, Uz, ndim=3):
    """
    Write an initial condition netcdf file
    """
    rootgrp = Dataset(ncfilename, "w", format="NETCDF4")
    
    rootgrp.created_on = time.ctime(time.time())
    rootgrp.title      = "AMR-Wind initial conditions"

    nc_ndim            = rootgrp.createDimension("ndim", ndim)
    nc_nx              = rootgrp.createDimension("nx",   Nx)
    nc_ny              = rootgrp.createDimension("ny",   Ny)
    nc_nz              = rootgrp.createDimension("nz",   Nz)

    threedim           = ("nx", "ny", "nz",)
    nc_uvel            = rootgrp.createVariable("uvel", "f8", threedim)
    nc_vvel            = rootgrp.createVariable("vvel", "f8", threedim)
    nc_wvel            = rootgrp.createVariable("wvel", "f8", threedim)

    nc_uvel[:,:,:]     = Ux
    nc_vvel[:,:,:]     = Uy
    nc_wvel[:,:,:]     = Uz

    # Close the file
    rootgrp.close()

# def InterpVelocity(z1, v1, x, y, z):
#     """
#     Assumes that the velocity is only a function of z
#     """
#     return np.interp(z, z1, v1)

def makeIC_zonly(prob_lo, prob_hi, n_cell, ufunc, vfunc, wfunc, ncfilename,
                 verbose=False):
    uvel = makeVelArrayZvec(prob_hi, prob_lo, n_cell, ufunc, verbose=verbose)
    vvel = makeVelArrayZvec(prob_hi, prob_lo, n_cell, vfunc, verbose=verbose)
    wvel = makeVelArrayZvec(prob_hi, prob_lo, n_cell, wfunc, verbose=verbose)
    writeUVW_NC(ncfilename, n_cell[0], n_cell[1], n_cell[2], uvel, vvel, wvel)
    return


def makeIC_fromMMC(prob_lo, prob_hi, n_cell, udata, vdata, Tdata,
                   MMCtime, zMMC, ncfilename, tstart, verbose=False):
    """
    Creates the initial condition based on the MMC profile data
    provided
    """
    interpfunc = lambda zdat, fdat, x, y, z: np.interp(z, zdat, fdat)
    # Get the initial profiles from MMC data
    uinit, vinit, Tinit = [], [], []
    for zi, z in enumerate(zMMC):
        uinit.append(np.interp(tstart, MMCtime, udata[:,zi]))
        vinit.append(np.interp(tstart, MMCtime, vdata[:,zi]))
        Tinit.append(np.interp(tstart, MMCtime, Tdata[:,zi]))

    # Get the inital profile functions
    ufunc = partial(interpfunc, zMMC, uinit)
    vfunc = partial(interpfunc, zMMC, vinit)
    wfunc = lambda x, y, z: 0.0

    uvel = makeVelArrayZvec(prob_hi, prob_lo, n_cell, ufunc, verbose=verbose)
    vvel = makeVelArrayZvec(prob_hi, prob_lo, n_cell, vfunc, verbose=verbose)
    wvel = makeVelArrayZvec(prob_hi, prob_lo, n_cell, wfunc, verbose=verbose)
    writeUVW_NC(ncfilename, n_cell[0], n_cell[1], n_cell[2], uvel, vvel, wvel)

    # Get the temperature inputs
    Theights = ' '.join([str(x) for x in zMMC])
    Ttemps   = ' '.join([str(x) for x in Tinit])
    return Theights, Ttemps

def makeMMCforcing(probe_lo, probe_hi, n_cell, udata, vdata, Tdata,
                   fluxdata, MMCtime, zMMC, ncfilename, sign=-1):
    """
    Write the MMC forcing netCDF file
    """
    # Construct the AMR-Wind grid at the cell centers
    zamr      = AMRcellcenters(probe_lo[2], probe_hi[2], n_cell[2])

    # Write the netcdf file with WRF forcing
    rootgrp = Dataset(ncfilename, "w", format="NETCDF4")
    print(rootgrp.data_model)

    # Create the heights
    heights   = zamr
    nheight   = rootgrp.createDimension("nheight", len(heights))
    ncheights = rootgrp.createVariable("heights", "f8", ("nheight",))
    ncheights[:] = heights

    # Create the times
    times     = MMCtime
    ntime     = rootgrp.createDimension("ntime", len(times))
    nctimes   = rootgrp.createVariable("times", "f8", ("ntime",))
    nctimes[:] = times

    # Write the datasizes
    datasize  = len(heights)*len(times)
    dsize     = rootgrp.createDimension("datasize", datasize)
    print("Wrote heights and times")

    # Add momentum u profiles
    nc_momu     = rootgrp.createVariable("momentum_u", "f8", 
                                         ("ntime", "nheight",))
    for i in range(len(times)):
        nc_momu[i,:] = np.interp(zamr, zMMC, udata[i,:])

    # Add momentum v profiles
    nc_momv     = rootgrp.createVariable("momentum_v", "f8", 
                                         ("ntime", "nheight",))
    for i in range(len(times)):
        nc_momv[i,:] = np.interp(zamr, zMMC, vdata[i,:])
    print("Wrote momentum profiles")

    # Add the temperature profiles
    nc_temp     = rootgrp.createVariable("temperature", "f8", 
                                         ("ntime", "nheight",))
    for i in range(len(times)):
        nc_temp[i,:] = np.interp(zamr, zMMC, Tdata[i,:])
    print("Wrote temperature profiles")

    # Add the temperature fluxes
    tflux       = fluxdata
    nc_tflux    = rootgrp.createVariable("tflux", "f8", ("ntime",))
    # Negative sign because SOWFA convention is opposite AMR-Wind
    nc_tflux[:] = sign*tflux   
    print("Wrote tflux profiles")

    # Close the file
    rootgrp.close()
    print("Done")

    return

def testFunctions(ICfilename, MMCfilename):
    """
    Quick test of MMC functions
    """
    ufunc = lambda x, y, z: 1.0
    vfunc = lambda x, y, z: 0.0
    wfunc = lambda x, y, z: 0.0

    probe_lo = [0, 0, 0]
    probe_hi = [100, 100, 100]
    n_cell   = [10, 10, 10]
    
    zMMC     = [0, 100, 200]
    times    = [0, 1000]
    
    udat = np.array([[1, 1, 1], 
                     [1, 1, 1] ])
    vdat = np.array([[0, 0, 0], 
                     [0, 0, 0] ])
    Tdat = np.array([[300, 300, 300], 
                     [300, 300, 300] ])
    fluxdat = np.array([0, 0])

    makeIC_zonly(probe_lo, probe_hi, n_cell, ufunc, vfunc, wfunc, ICfilename, 
                 verbose=True)

    makeMMCforcing(probe_lo, probe_hi, n_cell, udat, vdat, Tdat, fluxdat,
                   times, zMMC, MMCfilename)
    
    return

# ----------------------------------
if __name__ == "__main__":
    ICfilename  = 'test.nc'
    MMCfilename = 'MMC.nc'
    
    testFunctions(ICfilename, MMCfilename)
