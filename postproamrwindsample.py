#!/usr/bin/env python
#
# Copyright (c) 2022, Alliance for Sustainable Energy
#
# This software is released under the BSD 3-clause license. See LICENSE file
# for more details.
#
#
#

import sys
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import mmap

def loadDataset(filename, usemmap=False):
    if usemmap:
        print("Loading entire file into memory...")
        with open(filename, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ, flags=mmap.MAP_PRIVATE)
            ncread = mm.read()
        return Dataset('inmemory.nc', memory=ncread)
    else:
        return Dataset(filename, 'r')

def getGroups(ncdat):
    return [k for k, g in ncdat.groups.items()]

def getGroupSampleType(ncdat, group):
    return ncdat[group].sampling_type

def getVarList(ncdat, group=None):
    if group is None:
        return [k for k, g in ncdat.variables.items()]
    else:
        return [k for k, g in ncdat[group].variables.items()]

def getVar(ncdat, var, group=None):
    if group is None:
        return ncdat.variables[var]
    else:
        return ncat[group].variables[var]

def getPlotAxis(xyz, plotaxis):
    if plotaxis.upper()=='S':
        # Calculate the plot axis
        saxis = [0]
        Npts = len(xyz[:,0])
        for i in range(Npts-1):
            dist=np.linalg.norm(xyz[i,:]-xyz[i+1,:]) + saxis[-1]
            saxis.append(dist)
        return saxis
    if plotaxis.upper()=='X':
        return xyz[:,0]
    if plotaxis.upper()=='Y':
        return xyz[:,1]
    if plotaxis.upper()=='Z':
        return xyz[:,2]


def getLineSampleAtTime(ncdat, group, varlist, it):
    nc      = ncdat[group]
    Npts    = nc.dimensions['num_points'].size
    xyz     = nc.variables['coordinates']

    linedat = {}
    for var in varlist:
        ncvar  = nc.variables[var]
        vardat = np.zeros(Npts)
        for i in range(Npts):
            vardat[i] = ncvar[it, i]
        linedat[var] = vardat
    return xyz, linedat

def getPlaneSampleAtTime(ncdat, group, var, itime, kplane):
    Nijk    = ncdat[group].ijk_dims
    allpts  = ncdat[group].variables['coordinates']
    vardat  = ncdat[group].variables[var]
    axis1   = ncdat[group].axis1
    axis2   = ncdat[group].axis2

    N1      = Nijk[0]
    N2      = Nijk[1]
    xmesh   = np.zeros((N1,N2))
    ymesh   = np.zeros((N1,N2))
    zmesh   = np.zeros((N1,N2))
    vmesh   = np.zeros((N1,N2))

    # Set up the s directions
    ds1     = np.linalg.norm(axis1)/(N1-1)
    ds2     = np.linalg.norm(axis2)/(N2-1)
    s1mesh  = np.zeros((N1,N2))
    s2mesh  = np.zeros((N1,N2))

    for i in range(N1):
        for j in range(N2):
            ipt = i + j*N1 + kplane*N1*N2
            x   = allpts[ipt,0]
            y   = allpts[ipt,1]
            z   = allpts[ipt,2]
            v   = vardat[itime, ipt]
            xmesh[i,j] = x
            ymesh[i,j] = y
            zmesh[i,j] = z
            vmesh[i,j] = v

            s1  = i*ds1
            s2  = j*ds2
            s1mesh[i,j] = s1
            s2mesh[i,j] = s2
    return xmesh, ymesh, zmesh, s1mesh, s2mesh, vmesh
