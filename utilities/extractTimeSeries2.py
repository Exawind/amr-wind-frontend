#!/usr/bin/env python

# Get the location where this script is being run
import sys, os
scriptpath = os.path.dirname(os.path.realpath(__file__))
basepath   = os.path.dirname(scriptpath)

# Add any possible locations of amr-wind-frontend here
amrwindfedirs = ['../',
                 basepath]
import sys, os, shutil
for x in amrwindfedirs: sys.path.insert(1, x)

# Load the libraries
import postproamrwindsample as ppsample
import numpy             as np
import xarray as xr
import argparse
from collections import OrderedDict

extractvar = lambda xrds, var, i : xrds[var][i,:].data.reshape(tuple(xrds.attrs['ijk_dims'][::-1]))

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

def extractpt(ncfile, ptlist, varlist=['velocityx','velocityy','velocityz'], timesubset=None, group=None, verbose=0):
    groups=ppsample.getGroups(ppsample.loadDataset(ncfile))
    g=groups[0] if group is None else group
    datadict=OrderedDict()
    with xr.open_dataset(ncfile, group=g) as ds:
        xm = ds['coordinates'].data[:,0].reshape(tuple(ds.attrs['ijk_dims'][::-1]), order='C')
        ym = ds['coordinates'].data[:,1].reshape(tuple(ds.attrs['ijk_dims'][::-1]), order='C')
        zm = ds['coordinates'].data[:,2].reshape(tuple(ds.attrs['ijk_dims'][::-1]), order='C')
        dtime=xr.open_dataset(ncfile)
        ds = ds.assign_coords(coords={'xm':(['x','y','z'], xm),
                                      'ym':(['x','y','z'], ym),
                                      'zm':(['x','y','z'], zm),
                                      'time':dtime['time'],
                                     })
        dtime.close()
        iplane = 0
        N=len(ds['time'])
        for pt in ptlist:
            ijkrev = pt[::-1]
            ptdict=OrderedDict()
            ptdict['time'] = []
            ptdict['x']    = float(xm[ijkrev])
            ptdict['y']    = float(ym[ijkrev])
            ptdict['z']    = float(zm[ijkrev])
            for v in varlist:
                ptdict[v]  = []
            datadict[pt] = ptdict
        tloop = np.arange(N) if timesubset is None else timesubset
        Nloop = len(tloop)   # Edit this to choose a subset
        for itime in tloop: 
            progress(itime+1, Nloop)
            for pt in ptlist:
                datadict[pt]['time'].append(float(ds['time'][itime]))
            for v in varlist:
                vvar = extractvar(ds, v, itime)
                for pt in ptlist:
                    ijkrev = pt[::-1]
                    datadict[pt][v].append(vvar[ijkrev])
        print()
    return datadict

def datadict2file(datadict, filetemplate, varlist=['velocityx','velocityy','velocityz'], verbose=False):
    for pt, data in datadict.items():
        t=data['time']
        N=len(t)
        icol   = pt[0]*np.ones(N)
        jcol   = pt[1]*np.ones(N)
        kcol   = pt[2]*np.ones(N)
        xcol   = data['x']*np.ones(N)
        ycol   = data['y']*np.ones(N)
        zcol   = data['z']*np.ones(N)
        savedat = np.vstack((t, 
                             icol, jcol, kcol, 
                             xcol, ycol, zcol, 
                             ))
        for vvar in varlist:
            savedat = np.vstack((savedat, data[vvar]))
        fname=filetemplate.format(x=data['x'],y=data['y'],z=data['z'])
        headers="time i j k x y z "+' '.join(varlist)
        np.savetxt(fname, savedat.transpose(),header=headers)
        if verbose: print("saved "+fname)
    return
