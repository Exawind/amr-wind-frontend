#!/usr/bin/env python
# Script to stitch netCDF files together

# Can this test script with 
# ./catNC.py /gpfs/lcheung/TCF/GreensFunctionValidation/UniformCt_AMRWind/UniformCt_freespace/processtest/Ct0.2/post_processing/*.nc -o out.nc  -g blockageplane centerline  --tlims 800 1200 --varlist velocityx

from netCDF4 import Dataset
import numpy as     np
import sys
import os.path
import argparse
import time

#import xarray as    xr  # (Add this later)

def addTime(t, timevec, timesubset):
    checktimesubset = lambda t, tlims: True if len(tlims)==0 else ((tlims[0]<=t) and (t<=tlims[1]))
    if t in timevec:
        return False
    else:
        return checktimesubset(t, timesubset)

def stitchtimes(filelist, timesubset=[]):
    """
    Construct a single time vector from individual netcdf files in the
    filelist.
    """
    mastertimevec = []
    includetimes  = []
    # Index of which file and which iter this time comes from
    timeindex     = []
    itime         = 0
    for fi, f in enumerate(filelist):
       ncdat = Dataset(f, 'r')
       time  = np.array(ncdat.variables['time'])
       includelist = [False]*len(time)
       for iter, t in enumerate(time): 
           if addTime(t, mastertimevec, timesubset):
               mastertimevec.append(t)
               includelist[iter] = True
               timeindex.append([fi, iter, itime])
               itime += 1
       includetimes.append(includelist)
       ncdat.close()
    return mastertimevec, timeindex

def openNCfile(ncfilename, timevec, ndim=3):
    # Write the netcdf file with WRF forcing
    rootgrp = Dataset(ncfilename, "w", format="NETCDF4")

    rootgrp.created_on = time.ctime(time.time())
    rootgrp.title      = "AMR-Wind data sampling output"

    nc_ndim            = rootgrp.createDimension("ndim", ndim)
    nc_num_time_steps  = rootgrp.createDimension("num_time_steps", None)

    nc_times     = rootgrp.createVariable("time", "f8", ("num_time_steps",))
    nc_times[:]  = timevec

    return rootgrp

def getGroups(f):
    ncdat = Dataset(f, 'r')
    groups = [k for k, g in ncdat.groups.items()]
    ncdat.close()
    return groups

def addGroup(rootgrp, group, filelist, timeindexlist, 
             includevars=[], verbose=False, spinner=False):
    # Get the basic dimensions
    num_time_steps  = rootgrp.dimensions["num_time_steps"].size
    ndim  = rootgrp.dimensions["ndim"].size

    # create a group in rootgrp
    dest_subgroup = rootgrp.createGroup(group)

    # Add the attributes
    firstfile = filelist[0]
    src_ncdat = Dataset(firstfile, 'r')
    for key, val in src_ncdat[group].__dict__.items():
        dest_subgroup.setncattr(key, val)

    # Add the dimensions
    src_dims = src_ncdat[group].dimensions
    for key, val in src_dims.items():
        dest_subgroup.createDimension(key, val.size)
    # Get the num_points
    num_points = None
    if 'num_points' in src_ncdat[group].dimensions:
        num_points = src_ncdat[group].dimensions['num_points'].size
    #print('num_points = '+repr(num_points))

    # Get the list of variables
    varlist = [k for k, g in src_ncdat[group].variables.items()]

    # Copy over coordinates
    if 'coordinates' in varlist:
        coords = dest_subgroup.createVariable("coordinates", "f8", 
                                              ("num_points","ndim",))
        coords[:,:] = src_ncdat[group].variables['coordinates'][:,:]
        varlist.remove('coordinates')

    # Close the data file (for the first file)
    src_ncdat.close()

    # Split the timeindexlist
    filetimeindex = {}
    for i in range(len(filelist)):
        filetimeindex[i] = list()
    for entry in timeindexlist:
        ifile = entry[0]
        filetimeindex[entry[0]] += [entry]

    # Check to see if a variable should be included
    usevar = lambda v, vlist: True if (vlist is None) else (v in vlist)

    # ------ Spinner stuff ------- 
    if spinner:
        # Copy over the points vector
        if verbose: print("Adding points")
        addVar_to_group('points', group, dest_subgroup, 
                        ("num_time_steps", "num_points","ndim",),
                        filelist, filetimeindex, arraysize=3)
        varlist.remove('points')
        # Copy over the rotor_angles_rad vector
        if usevar('rotor_angles_rad', includevars):
            if verbose: print("Adding rotor_angles_rad")
            addVar_to_group('rotor_angles_rad', group, dest_subgroup, 
                            ("num_time_steps", "nang",), 
                            filelist, filetimeindex)
            varlist.remove('rotor_angles_rad')        
        if usevar('rotor_hub_pos', includevars):
            if verbose: print("Adding rotor_hub_pos")
            addVar_to_group('rotor_hub_pos', group, dest_subgroup, 
                            ("num_time_steps", "ndim",), 
                            filelist, filetimeindex)
            varlist.remove('rotor_hub_pos')        
    # ----------------------------

    # Add the other variables
    for v in varlist:
        if (not usevar(v, includevars)):
            continue
        if verbose: print("Adding "+v)
        vdat = dest_subgroup.createVariable(v, "f8", 
                                            ("num_time_steps","num_points",))
        for ifile, fname in enumerate(filelist):
            if len(filetimeindex[ifile]) == 0: 
                continue
            # Loop through all times in filetimeindex
            src_ncdat = Dataset(fname, 'r')
            srcvar = src_ncdat[group].variables[v]
            for entry in filetimeindex[ifile]:
                ilocal  = entry[1]
                iglobal = entry[2]
                vdat[iglobal,:] = srcvar[ilocal,:]
            src_ncdat.close()
    return

def addVar_to_group(varname, groupname, dest_subgroup, dimlist, 
                    filelist, filetimeindex, arraysize=2):
    vdat = dest_subgroup.createVariable(varname, "f8", dimlist)
    for ifile, fname in enumerate(filelist):
        if len(filetimeindex[ifile]) == 0: 
            continue
        # Loop through all times in filetimeindex
        src_ncdat = Dataset(fname, 'r')
        srcvar = src_ncdat[groupname].variables[varname]
        for entry in filetimeindex[ifile]:
            ilocal  = entry[1]
            iglobal = entry[2]
            if arraysize==2:
                vdat[iglobal,:] = srcvar[ilocal,:]
            elif arraysize==3:
                vdat[iglobal,:,:] = srcvar[ilocal,:,:]
        src_ncdat.close()
    return


# ========================================================================
# Main
# ========================================================================
if __name__ == "__main__":
        
    helpstring = 'Concatenate sampling netcdf files together'
    # Handle arguments
    parser     = argparse.ArgumentParser(description=helpstring)
    parser.add_argument(
        "ncfile",
        help="Input netcdf file(s)",
        type=str,
        nargs='+',
    )
    parser.add_argument(
        "-g",
        "--groups",
        help="Which groups to include [Default: all groups]",
        type=str,
        nargs='+',
        required=False
    )
    parser.add_argument(
        "--varlist",
        help="Which variables to include [Default: all variables]",
        type=str,
        nargs='+',
        required=False
    )
    parser.add_argument(
        "--tlims",
        help="Only extract times within [tmin, tmax]",
        type=float,
        nargs=2,
        required=False
    )
    parser.add_argument(
        "-o",
        "--outfile",
        help="output netcdf filename",
        default='output.nc',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--spinner', 
        help="NetCDF is a spinner lidar file",
        default=False,
        action='store_true')
    parser.add_argument(
        '-v',
        '--verbose', 
        help="Turn on verbose",
        default=False,
        action='store_true')

    # Load the options
    args      = parser.parse_args()
    outfile   = args.outfile
    ncfiles   = args.ncfile
    verbose   = args.verbose
    spinner   = args.spinner
    varlist   = args.varlist
    tlims     = [] if args.tlims is None else args.tlims
    #print(ncfiles)
    #print(tlims)

    # Get the list of times
    mastertvec, timeindex = stitchtimes(ncfiles, timesubset=tlims)
    #print(timeindex)
    #print(mastertvec)
    #print(includelist)

    # Get the list of groups
    grouplist = getGroups(ncfiles[0])
    includegroups = grouplist if args.groups is None else args.groups
    #print(includegroups)

    # Open the netCDF file
    rootgrp   = openNCfile(outfile, mastertvec)

    # Loop through the groups
    for g in includegroups:
        if verbose: print("Adding group "+g)
        addGroup(rootgrp, g, ncfiles, timeindex, 
                 includevars=varlist, verbose=verbose, spinner=spinner)

    # Close the file
    rootgrp.close()
    print("Done")
