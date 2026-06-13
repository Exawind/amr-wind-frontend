#!/usr/bin/env python
# Script to create an initial condition netCDF file from a plt file

from netCDF4 import Dataset
import numpy as     np
import sys
import yt
import argparse
import time

def loadplt(pltdir):
    ds = yt.load(pltdir, 
                 unit_system="mks",
                 units_override={"length_unit": (1.0, "m"), 
                                 "mass_unit": (1.0, "kg"), 
                                 "velocity_unit": (1.0, "m/s"), 
                                 "time_unit": (1.0, "s")})
    
    return ds

def get_coveringgrid_vars(ds, varlist, maxlevel=0):
    cg = ds.covering_grid(maxlevel,
                          ds.domain_left_edge,
                          ds.domain_dimensions)
    outdict = {}
    for v in varlist:
        outdict[v] = cg[v].to_ndarray()
    return outdict

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
    return

def erfplt2nc(pltdir, ncfile,
              varlist = ['x_velocity', 'y_velocity', 'z_velocity'],
              input_ds=None):
    if input_ds is None:
        ds = loadplt(pltdir)
    else:
        ds = input_ds     
    dim     = ds.domain_dimensions
    erfv    = get_coveringgrid_vars(ds, varlist, maxlevel=0)
    writeUVW_NC(ncfile, dim[0], dim[1], dim[2],
                erfv[varlist[0]],
                erfv[varlist[1]],
                erfv[varlist[2]],
                #erfv['x_velocity'],
                #erfv['y_velocity'],
                #erfv['z_velocity'],
    )
    return ds

def avgtemp(pltdir, Tvar, input_ds=None):
    if input_ds is None:
        ds = loadplt(pltdir)
    else:
        ds = input_ds
    cg = get_coveringgrid_vars(ds, ['z', Tvar], maxlevel=0)
    
    zm = cg['z']
    Tm = cg[Tvar]
    zvec = zm[0,0,:]
    avgTvec = []
    for i in range(len(zvec)):
        avgT = np.mean(Tm[:,:,i])
        avgTvec.append(avgT)
        #print(zvec[i], avgT)
    return zvec, avgTvec, ds

# ========================================================================
# Main
# ========================================================================
if __name__ == "__main__":
    helpstring = 'Convert PLT file to netcdf file'
    # Handle arguments
    parser     = argparse.ArgumentParser(description=helpstring)
    parser.add_argument(
        "pltdir",
        help="Input plt directory",
        type=str,
    )
    parser.add_argument(
        "ncfile",
        help="Output netcdf file",
        type=str,
    )
    parser.add_argument(
        '--erfnames', 
        help="Use ERF variable names",
        default=False,
        action='store_true')
    parser.add_argument(
        '--avgT', 
        help="horizontally average temperature variable",
        default=False,
        action='store_true')

    erfvels = ['x_velocity', 'y_velocity', 'z_velocity']
    amrvels = ['velocityx',  'velocityy',  'velocityz']

    erfT    = 'theta'
    amrT    = 'temperature'
    
    # Load the options
    args      = parser.parse_args()
    pltdir    = args.pltdir
    ncfile    = args.ncfile
    erfnames  = args.erfnames
    avgT      = args.avgT

    if erfnames:
        vvars = erfvels
        Tvar  = erfT
    else:
        vvars = amrvels
        Tvar  = amrT

    # Write the netcdf file
    ds = erfplt2nc(pltdir, ncfile, varlist=vvars)
    if avgT:
        zvec, avgTvec, _ = avgtemp(pltdir, Tvar, input_ds=ds)
        # print out the vectors
        print()
        print('ABL.temperature_heights = ' + ' '.join([str(z) for z in zvec]))
        print()
        print('ABL.temperature_values = ' +' '.join([str(T) for T in avgTvec]))
