# Load the libraries
import postproamrwindsample as ppsample
import numpy             as np
import sys
import xarray as xr

extractvar = lambda xrds, var, i : xrds[var][i,:].data.reshape(tuple(xrds.attrs['ijk_dims'][::-1]))

def getPlaneXR(ncfile, itimevec, varnames, groupname=None,
               verbose=0, includeattr=False, gettimes=False):
    # Create a fresh db dictionary
    db = {}
    for v in varnames: db[v] = {}
    db['timesteps'] = []
    timevec = None
    if gettimes:
        timevec = ppsample.getVar(ppsample.loadDataset(ncfile), 'time')
        db['times'] = []
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
            if gettimes:
                db['times'].append(float(timevec[itime]))
            for v in varnames:
                vvar = extractvar(ds, v, itime)
                db[v][itime] = vvar
            #vx = extractvar(ds, vxvar, itime)
            #vy = extractvar(ds, vyvar, itime)
            #vz = extractvar(ds, vzvar, itime)
        if includeattr:
            for k, g in ds.attrs.items():
                db[k] = g
    return db

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

def avgPlaneXR(ncfile, timerange,
               extrafuncs=[],
               varnames=['velocityx','velocityy','velocityz'],
               groupname=None, verbose=False, includeattr=False):
    """
    Compute the average of ncfile variables
    """
    suf='_avg'
    # Create a fresh db dictionary
    db = {}
    for v in varnames: db[v] = {}
    t1 = timerange[0]
    t2 = timerange[1]
    timevec = ppsample.getVar(ppsample.loadDataset(ncfile), 'time')
    filtertime=np.where((t1 <= np.array(timevec)) & (np.array(timevec) <= t2))
    Ntotal=len(filtertime[0])
    db['times'] = []
    # Now load the ncfile data
    if groupname is None:
        groups= ppsample.getGroups(ppsample.loadDataset(ncfile))
        group = groups[0]
    else:
        group = groupname
    db['group'] = group
    with xr.open_dataset(ncfile, group=group) as ds:
        reshapeijk = ds.attrs['ijk_dims'][::-1]
        xm = ds['coordinates'].data[:,0].reshape(tuple(reshapeijk))
        ym = ds['coordinates'].data[:,1].reshape(tuple(reshapeijk))
        zm = ds['coordinates'].data[:,2].reshape(tuple(reshapeijk))
        db['x'] = xm
        db['y'] = ym
        db['z'] = zm
        # Set up the initial mean fields
        zeroarray = extractvar(ds, varnames[0], 0)
        for v in varnames:
            db[v+suf] = np.full_like(zeroarray, 0.0)
        if len(extrafuncs)>0:
            for f in extrafuncs:
                db[f['name']+suf] = np.full_like(zeroarray, 0.0)
        Ncount = 0
        # Loop through and accumulate
        for itime, t in enumerate(timevec):
            if (t1 <= t) and (t <= t2):
                if verbose: progress(Ncount+1, Ntotal)
                db['times'].append(float(t))
                vdat = {}
                for v in varnames:
                    vdat[v] = extractvar(ds, v, itime)
                    db[v+suf] += vdat[v]
                if len(extrafuncs)>0:
                    for f in extrafuncs:
                        name = f['name']+suf
                        func = f['func']
                        db[name] += func(vdat)
                Ncount += 1
        # Normalize
        if Ncount > 0:
            for v in varnames:
                db[v+suf] /= float(Ncount)
            if len(extrafuncs)>0:
                for f in extrafuncs:
                    name = f['name']+suf
                    db[name] /= float(Ncount)
        if verbose: print()
        # include attributes
        if includeattr:
            for k, g in ds.attrs.items():
                db[k] = g
    return db

def MinMaxStd_PlaneXR(ncfile, timerange,
                      extrafuncs=[], avgdb = None,
                      varnames=['velocityx','velocityy','velocityz'], groupname=None,
                      verbose=False, includeattr=False):
    """
    Calculate the min, max, and standard deviation
    """
    bigval = sys.float_info.max
    smin = '_min'
    smax = '_max'
    sstd = '_std'
    savg = '_avg'
    db = {}
    if avgdb is None:
        db = avgPlaneXR(ncfile, timerange,
                        extrafuncs=extrafuncs,
                        varnames=varnames,
                        groupname=groupname, verbose=verbose, includeattr=includeattr)
    else:
        db.update(avgdb)
    group = db['group']
    timevec = ppsample.getVar(ppsample.loadDataset(ncfile), 'time')
    t1 = timerange[0]
    t2 = timerange[1]
    Ntotal=len(db['times'])
    with xr.open_dataset(ncfile, group=group) as ds:
        reshapeijk = ds.attrs['ijk_dims'][::-1]
        zeroarray = extractvar(ds, varnames[0], 0)
        # Set up the initial mean fields
        for v in varnames:
            db[v+sstd] =  np.full_like(zeroarray, 0.0)
            db[v+smax] =  np.full_like(zeroarray, -bigval)
            db[v+smin] =  np.full_like(zeroarray, bigval)
        if len(extrafuncs)>0:
            for f in extrafuncs:
                name = f['name']
                db[name+sstd] = np.full_like(zeroarray, 0.0)
                db[name+smax] = np.full_like(zeroarray, -bigval)
                db[name+smin] = np.full_like(zeroarray, bigval)
        Ncount = 0
        # Loop through and accumulate
        for itime, t in enumerate(timevec):
            if (t1 <= t) and (t <= t2):
                if verbose: progress(Ncount+1, Ntotal)
                vdat = {}
                for v in varnames:
                    vdat[v] = extractvar(ds, v, itime)        
                for v in varnames:
                    # Change max vals
                    filtermax = vdat[v] > db[v+smax]
                    db[v+smax][filtermax] = vdat[v][filtermax]
                    # Change min vals
                    filtermin = vdat[v] < db[v+smin]
                    db[v+smin][filtermin] = vdat[v][filtermin]
                    # Standard dev
                    db[v+sstd] += (vdat[v]-db[v+savg])*(vdat[v]-db[v+savg])
                if len(extrafuncs)>0:
                    for f in extrafuncs:
                        name = f['name']
                        func = f['func']
                        vinst= func(vdat)
                        vavg = db[name+savg]
                        # Change max vals
                        filtermax = vinst > db[name+smax]
                        db[name+smax][filtermax] = vinst[filtermax]
                        # Change min vals
                        filtermin = vinst < db[name+smin]
                        db[name+smin][filtermin] = vinst[filtermin]
                        # Standard dev
                        db[name+sstd] += (vinst-vavg)*(vinst-vavg)
                Ncount += 1
        # Normalize and sqrt std dev
        if Ncount > 0:
            for v in varnames:
                db[v+sstd] = np.sqrt(db[v+sstd]/float(Ncount))
            if len(extrafuncs)>0:
                for f in extrafuncs:
                    name = f['name']
                    db[name+sstd] = np.sqrt(db[name+sstd]/float(Ncount))
        if verbose: print()
    return db

def getLineXR(ncfile, itimevec, varnames, groupname=None,
              verbose=0, includeattr=False, gettimes=False):
    # Create a fresh db dictionary
    db = {}
    for v in varnames: db[v] = {}
    db['timesteps'] = []
    timevec = None
    if gettimes:
        timevec = ppsample.getVar(ppsample.loadDataset(ncfile), 'time')
        db['times'] = []
    # Now load the ncfile data
    if groupname is None:
        groups= ppsample.getGroups(ppsample.loadDataset(ncfile))
        group = groups[0]
    else:
        group = groupname
    with xr.open_dataset(ncfile, group=group) as ds:
        #reshapeijk = ds.attrs['ijk_dims'][::-1]
        xm = ds['coordinates'].data[:,0] #.reshape(tuple(reshapeijk))
        ym = ds['coordinates'].data[:,1] #.reshape(tuple(reshapeijk))
        zm = ds['coordinates'].data[:,2] #.reshape(tuple(reshapeijk))
        dtime=xr.open_dataset(ncfile)
        dtime.close()
        db['x'] = xm
        db['y'] = ym
        db['z'] = zm
        for itime in itimevec:
            if verbose>0:
                print("extracting iter "+repr(itime))
            db['timesteps'].append(itime)
            if gettimes:
                db['times'].append(float(timevec[itime]))
            for v in varnames:
                vvar = ds[v].data[itime,:]
                #vvar = extractvar(ds, v, itime)
                db[v][itime] = vvar
            #vx = extractvar(ds, vxvar, itime)
            #vy = extractvar(ds, vyvar, itime)
            #vz = extractvar(ds, vzvar, itime)
        if includeattr:
            for k, g in ds.attrs.items():
                db[k] = g
    return db
