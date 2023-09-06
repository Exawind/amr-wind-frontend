# Load the libraries
import postproamrwindsample as ppsample
import numpy             as np

import xarray as xr

extractvar = lambda xrds, var, i : xrds[var][i,:].data.reshape(tuple(xrds.attrs['ijk_dims'][::-1]))

def getPlaneXR(ncfile, itimevec, varnames, vxvar='velocityx',
               vyvar='velocityy', vzvar='velocityz', groupname=None,
               verbose=0, includeattr=False):
    # Create a fresh db dictionary
    db = {}
    for v in varnames: db[v] = {}
    db['timesteps'] = []
    # Now load the ncfile data
    if groupname is None:
        groups= ppsample.getGroups(ppsample.loadDataset(ncfile))
        group = groups[0]
    else:
        group = groupname
    with xr.open_dataset(ncfile, group=group) as ds:
        reshapeijk = ds.attrs['ijk_dims'][::-1]
        xm = ds['coordinates'].data[:,0].reshape(tuple(reshapeijk))
        ym = ds['coordinates'].data[:,1].reshape(tuple(reshapeijk))
        zm = ds['coordinates'].data[:,2].reshape(tuple(reshapeijk))
        dtime=xr.open_dataset(ncfile)
        dtime.close()
        db['x'] = xm
        db['y'] = ym
        db['z'] = zm        
        for itime in itimevec:
            if verbose>0:
                print("extracting iter "+repr(itime))
            db['timesteps'].append(itime)
            for v in varnames:
                vvar = extractvar(ds, v, itime)
                db[v][itime] = vvar
            vx = extractvar(ds, vxvar, itime)
            vy = extractvar(ds, vyvar, itime)
            vz = extractvar(ds, vzvar, itime)
        if includeattr:
            for k, g in ds.attrs.items():
                db[k] = g
    return db
