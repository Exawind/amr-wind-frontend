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

# This is the header that goes at the top of all VTK files
vtkheader="""# vtk DataFile Version 3.0
vtk output
ASCII
DATASET STRUCTURED_GRID
"""

defaultvars = [{'name':'velocity', 
                'vars':['velocityx', 'velocityy', 'velocityz']}]

def findMatchingPt(ptlist, p, eps):
    for ipt, xpt in enumerate(ptlist):
        if (np.linalg.norm(np.array(xpt)-np.array(p)))<eps: return ipt
    # Error out
    raise Exception("error in findMatchingPt") 

def parsevarstring(varstring):
    vardict={}
    splitcolon = varstring.split(":")
    print(splitcolon)
    if len(splitcolon)==1:
        var = splitcolon[0].strip()
        vardict = {'name':var, 'vars':[var]}
    else:
        varname = splitcolon[0].strip()
        varlist = [x.strip() for x in splitcolon[1].split(',')]
        vardict = {'name':varname, 'vars':varlist}
    return vardict

def convertplane2vtk(ncdat, timevec, savefile, group, kplanes=[], 
                     verbose=True, extractvars=defaultvars):
    """
    Convert sample plane to vtk output
    """

    #extractvars = [{'name':'velocity', 
    #                'vars':['velocityx', 'velocityy', 'velocityz']}]
    #extractvars = ['velocityx', 'velocityy', 'velocityz']
    
    Npts   = ncdat[group].dimensions['num_points'].size
    allpts = ncdat[group].variables['coordinates']
    t      = ncdat['time'][:]

    Nijk   = ncdat[group].ijk_dims
    Ni     = Nijk[0]
    Nj     = Nijk[1]

    Npoints   = (Ni)*(Nj)
    Ncells    = (Ni-1)*(Nj-1)
    Nvars     = len(extractvars)

    # Get the list of time indexes
    tindexvec = []
    for time in timevec:
        idx = np.where(t==time)
        if len(idx[0])>0:
            tindexvec.append(idx[0][0])
        else:
            print(t)
            print("Time %f not found"%time)
            return
    if verbose:
        print("tindexvec: "+repr(tindexvec))

    if len(kplanes)<1: kplanes=[0]

    for k in kplanes:
        if verbose: print("Working on plane %i"%k)
        # Get the xyz points
        xyz    = []
        allidx = []
        for i in range(Ni):
            for j in range(Nj):
                ipt = i + j*Ni + k*Ni*Nj
                allidx.append(ipt)
                xyz.append([allpts[ipt,0], allpts[ipt,1], allpts[ipt,2]])
        if verbose: print("Extracted xyz")

        for titer, tindex in enumerate(tindexvec):
            if verbose: print("Time = "+repr(timevec[titer]))
            # Write the stuff
            filename=savefile.format(k=k,time=timevec[titer])
            if verbose: print("Writing "+filename)
            f = open(filename,"w")
            # Write the header and coordinates
            f.write(vtkheader)
            f.write("DIMENSIONS %i %i 1\n"%(Ni, Nj))
            f.write("POINTS %i float\n"%(Npoints))
            for row in xyz:
                f.write("%e %e %e\n"%(row[0], row[1], row[2]))
            f.write("CELL_DATA %i\n"%Ncells)
            f.write("POINT_DATA %i\n"%Npoints)
    
            # Write the variables
            f.write("FIELD FieldData %i\n"%Nvars)
            for var in extractvars:
                varcomp = len(var['vars'])
                varname = var['name']
                f.write("%s %i %i float\n"%(varname, varcomp, Npoints))
                allvardat = np.zeros((varcomp, len(allidx)))
                for iv, v in enumerate(var['vars']):
                    vardat  = ncdat[group].variables[v]
                    allvardat[iv, :] = vardat[tindex, allidx]
                for i in range(len(allidx)):
                    f.write(" ".join([repr(x) for x in allvardat[:, i] ]))
                    f.write("\n")
            # for var in extractvars:
            #     varcomp = 1   # Number of components in variable
            #     f.write("%s %i %i float\n"%(var, varcomp, Npoints))
            #     vardat  = ncdat[group].variables[var]
            #     vardatt = vardat[tindex, allidx]
            #     for v in vardatt:
            #         f.write("%s\n"%repr(v))
            f.close()
    return

# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":
        
    helpstring = 'Convert sample plane netcdf data to ASCII VTK format'
    # Handle arguments
    parser     = argparse.ArgumentParser(description=helpstring)
    parser.add_argument(
        "ncfile",
        help="netcdf file",
        type=str,
    )

    parser.add_argument(
        "-o",
        "--outfile",
        help="output vtk filename",
        default='plane.vtk',
        type=str,
        required=True,
    )

    parser.add_argument(
        "-g",
        "--group",
        help="NetCDF group, default=p_f",
        default='p_f',
        type=str,
    )

    parser.add_argument(
        "-t",
        "--time",
        help="Times to extract",
        dest='time',
        nargs='+',
        required=True,
    )

    parser.add_argument(
        "-k",
        "--kplanes",
        nargs='+',
        help="k plane indices to extract",
        dest='kplanes',
        required=True,
    )

    parser.add_argument(
        "--vars",
        nargs='+',
        help="variables to extract [DEFAULT = velocity:velocityx,velocityy,velocityz]",
        dest='varstrings',
        default='velocity:velocityx,velocityy,velocityz',
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest='verbose',
        help="Turn on verbose mode",
        default=False,
        action='store_true',
    )

    # Load the options
    args      = parser.parse_args()
    filename  = args.ncfile
    savefile  = args.outfile
    group     = args.group
    kplanes   = [int(k) for k in args.kplanes]
    times     = [float(t) for t in args.time]
    #tindex    = args.time
    verbose   = args.verbose
    print(args.varstrings)
    varstringlist =[args.varstrings] if not isinstance(args.varstrings, list) else args.varstrings
    vardict   = [parsevarstring(x) for x in varstringlist]

    print("netcdf file    = "+filename)
    print("netcdf group   = "+group)
    print("output vtk     = "+savefile)
    print("kplanes        = "+repr(kplanes))
    print("tindex         = "+repr(times))
    print("verbose        = "+repr(verbose))
    print(vardict)

    ncdat   = Dataset(filename, 'r')
    convertplane2vtk(ncdat, times, savefile, group, 
                     kplanes=kplanes, extractvars=vardict, verbose=verbose) 
