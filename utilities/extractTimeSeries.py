#!/usr/bin/env python
#
# Copyright (c) 2022, Alliance for Sustainable Energy
#
# This software is released under the BSD 3-clause license. See LICENSE file
# for more details.
#

import sys
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import os.path
import argparse
import mmap

def findMatchingPt(ptlist, p, eps):
    for ipt, xpt in enumerate(ptlist):
        if (np.linalg.norm(np.array(xpt)-np.array(p)))<eps: return ipt
    # Error out
    raise Exception("error in findMatchingPt") 
        
def getTimeSeries(ncdat, xvec, yvec, zvec, group, 
                  savestring='', useindices=False, verbose=True,
                  extravars=[]):
    """
    Extract the velocity time-history multiple x, y, z locations from
    planar probe data.
    """
    #Nt     = ncdat.dimensions['num_time_steps'].size
    Npts   = ncdat[group].dimensions['num_points'].size
    allpts = ncdat[group].variables['coordinates']

    t      = ncdat['time'][:]
    allvx  = ncdat[group].variables['velocityx']
    allvy  = ncdat[group].variables['velocityy']
    allvz  = ncdat[group].variables['velocityz']
    allextra = {}
    for v in extravars:
        allextra[v] = ncdat[group].variables[v]
    Navg   = 0
    zerocol = np.zeros(len(t))
    all_ulongavgs = []
    returndat = {}
    for x in xvec:
        for y in yvec:
            for z in zvec:
                if useindices:
                    Nijk   = ncdat[group].ijk_dims
                    Ni     = Nijk[0]
                    Nj     = Nijk[1]
                    ipt    = int(x) + int(y)*Ni + int(z)*Ni*Nj
                    xyzpt = np.array([allpts[ipt, 0], 
                                      allpts[ipt, 1], 
                                      allpts[ipt, 2]])
                    icol   = x*np.ones(len(t))
                    jcol   = y*np.ones(len(t))
                    kcol   = z*np.ones(len(t))
                else:
                    xyzpt = np.array([x,y,z])
                    ipt   = findMatchingPt(allpts, xyzpt, 1.0E-6)
                    icol  = zerocol
                    jcol  = zerocol
                    kcol  = zerocol

                #print(allpts[ipt, :])
                u = allvx[:,ipt]
                if verbose: print("extracted u")
                v = allvy[:,ipt]
                if verbose: print("extracted v")
                w = allvz[:,ipt]
                if verbose: print("extracted w")
                extradat = {}
                for vvar in extravars:
                    extradat[vvar] = allextra[vvar][:,ipt]
                    if verbose: print("extracted "+vvar)
                xcol = xyzpt[0]*np.ones(len(t))
                ycol = xyzpt[1]*np.ones(len(t))
                zcol = xyzpt[2]*np.ones(len(t))
                savedat = np.vstack((t, 
                                     icol, jcol, kcol, 
                                     xcol, ycol, zcol, 
                                     u, v, w))
                for vvar in extravars:
                    savedat = np.vstack((savedat, extradat[vvar]))
                returndat[(x,y,z)] = savedat
                if savestring != '':
                    fname=savestring%(x,y,z)
                    np.savetxt(fname, savedat.transpose())
                    if verbose: print("saved "+fname)
    return returndat


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    helpstring="""
Extract time-series from AMR-Wind netcdf file

Example:
    
    python extractTimeSeries.py post_processing/sampling60000.nc -x 0 100 200 -y 0 100 200 -z 20 -o spectra/point_%05.0f_%05.0f_%05.0f.dat
or
    python extractTimeSeries.py post_processing_dx2.5/sampling60000.nc -x 0 -y 1 -z 0 --group 'p_f' -i -o spectra/point_%05.0f_%05.0f_%05.0f.dat
    """

    # Parse arguments
    parser = argparse.ArgumentParser(description=helpstring, 
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "ncfile",
        help="netcdf file",
        type=str,
    )
    parser.add_argument(
        "-o",
        "--outfile",
        help="output file format",
        default='point_%5.0f_%5.0f_%5.0f.dat',
        type=str,
    )
    parser.add_argument(
        "-g",
        "--group",
        help="NetCDF group, default=p_f",
        default='p_f',
        type=str,
    )
    parser.add_argument(
        "-i",
        "--useindices",
        dest='useindices',
        help="Use indices instead of coordinates in x,y,z strings, default=False",
        default=False,
        action='store_true'
    )

    parser.add_argument(
        "-x",
        nargs='+',
        help="x points to extract",
        dest='xpts',
        required=True,
    )

    parser.add_argument(
        "-y",
        nargs='+',
        help="y points to extract",
        dest='ypts',
        required=True,
    )

    parser.add_argument(
        "-z",
        nargs='+',
        help="z points to extract",
        dest='zpts',
        required=True,
    )

    parser.add_argument(
        "-v",
        "--vars",
        nargs='+',
        default='',
        help="additional vars to extract (in addition to velocity)",
        dest='extravars',
    )
    parser.add_argument(
        "-m",
        "--mmap",
        dest='mmap',
        help="Load entire netcdf file in memory",
        default=False,
        action='store_true'
    )

    # Load the options
    args = parser.parse_args()
    filename  = args.ncfile
    savefile  = args.outfile
    group     = args.group
    useindices= args.useindices
    mmapopt   = args.mmap
    xpts      = [float(x) for x in args.xpts]
    ypts      = [float(y) for y in args.ypts]
    zpts      = [float(z) for z in args.zpts]
    extravars = args.extravars
    print("netcdf file    = "+filename)
    print("netcdf group   = "+group)
    print("output format  = "+savefile)
    print("using indices  = "+repr(useindices))
    print("xpts           = "+repr(xpts))
    print("ypts           = "+repr(ypts))    
    print("zpts           = "+repr(zpts))
    print("extra vars     = "+repr(extravars))
    print("mmap           = "+repr(mmapopt))

    print("Reading dataset from "+filename)
    if mmapopt:
        with open(filename, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ, flags=mmap.MAP_PRIVATE)
            ncread = mm.read()
        ncdata = Dataset('inmemory.nc', memory=ncread)
    else:
        ncdata = Dataset(filename, 'r')
    print("Done")
    getTimeSeries(ncdata, xpts, ypts, zpts, group, savestring=savefile,
                    useindices=useindices, extravars=extravars, verbose=True);
