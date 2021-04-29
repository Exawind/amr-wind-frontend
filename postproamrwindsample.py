#!/usr/bin/env python
#
#

import sys
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

def loadDataset(filename):
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
