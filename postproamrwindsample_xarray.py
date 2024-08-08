# Load the libraries
import postproamrwindsample as ppsample
import numpy             as np
import sys
import xarray as xr
import pickle
import matplotlib.pyplot as plt
import glob
import itertools
from postproengine import get_mapping_xyz_to_axis1axis2
from postproengine import apply_coordinate_transform
from mpl_toolkits.axes_grid1 import make_axes_locatable

extractvar = lambda xrds, var, i : xrds[var][i,:].data.reshape(tuple(xrds.attrs['ijk_dims'][::-1]))
nonan = lambda x, doreplace: np.nan_to_num(x) if doreplace else x

def getFileList(ncfileinput):
    ncfilelist = []
    if type(ncfileinput) is str:
        ncfilelist=list(glob.glob(ncfileinput))
    # elif type(ncfileinput) is not list:
    #     for ncfileiter, ncfile in enumerate(ncfileinput):
    #         files = glob.glob(ncfile[ncfileiter])
    #         for file in files:
    #             ncfilelist.append(file)
    else:
        for ncfileiter, ncfile in enumerate(ncfileinput):
            files = glob.glob(ncfile)
            for file in files:
                ncfilelist.append(file)

    return ncfilelist

def getPlaneXR(ncfileinput, itimevec, varnames, groupname=None,
               verbose=0, includeattr=False, gettimes=False,timerange=None):

    ncfilelist = getFileList(ncfileinput)

    # Create a fresh db dictionary
    db = {}
    for v in varnames: db[v] = {}
    db['timesteps'] = []

    timevec = []
    times   = []
    for ncfileiter,ncfile in enumerate(ncfilelist):
        times.append(ppsample.getVar(ppsample.loadDataset(ncfile), 'time')[:])
        timevec = np.concatenate((timevec, times[ncfileiter]))               
        timevec = np.unique(timevec)

    if timerange is not None:
        if len(timerange) != 2:
            print("Error: timerange must be an array of length 2, e.g., [t_start,t_end]. Exiting")
            sys.exit()
        find_nearest = lambda a, a0: np.abs(np.array(a) - a0).argmin()
        itimevec = [find_nearest(timevec, t) for t in timerange]
        itimevec = np.arange(itimevec[0],itimevec[1]+1)

    itime_processed = []
    for ncfileiter,ncfile in enumerate(ncfilelist):
        if gettimes:
            if ncfileiter == 0:
                db['times'] = []

        # Now load the ncfile data
        if groupname is None:
            groups= ppsample.getGroups(ppsample.loadDataset(ncfile))
            group = groups[0]
        else:
            group = groupname

        with xr.open_dataset(ncfile, group=group) as ds:
            if ncfileiter == 0:
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
                local_ind = np.where(np.isin(times[ncfileiter], timevec[itime]))[0]
                if itime not in itime_processed and len(local_ind)==1:
                    if verbose>0:
                        print("extracting iter "+repr(itime))
                    db['timesteps'].append(itime)
                    if gettimes:
                        db['times'].append(float(timevec[itime]))
                    for v in varnames:
                        vvar = extractvar(ds, v, local_ind)
                        db[v][itime] = vvar
                    #vx = extractvar(ds, vxvar, itime)
                    #vy = extractvar(ds, vyvar, itime)
                    #vz = extractvar(ds, vzvar, itime)
                    itime_processed.append(itime)
                else:
                    if verbose>0:
                        print("Already processed itime: ",itime)
            if ncfileiter == 0:
                if includeattr:
                    for k, g in ds.attrs.items():
                        db[k] = g
    return db

# See https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def progress(count, total, suffix='', digits=1):
    """
    print out a progressbar
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), digits)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s %s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()

def getPlanePtsXR(ncfile, itimevec, ptlist,
                  varnames=['velocityx', 'velocityy', 'velocityz'], 
                  groupname=None,
                  verbose=0, includeattr=False, gettimes=False):
    """
    Extract specific points from a plane given in the netcdf file
    """
    # Create a fresh db dictionary
    db = {}
    for pt in ptlist: 
        db[pt] = {}
        for v in varnames:
            db[pt][v] = []
    db['timesteps'] = []
    Ntimes = len(ppsample.getVar(ppsample.loadDataset(ncfile), 'time'))
    timevec = None
    if gettimes:
        timevec = ppsample.getVar(ppsample.loadDataset(ncfile), 'time')
        db['times'] = []
    if len(itimevec)==0:
        itimevec = list(range(Ntimes))
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
        for ind, itimeraw in enumerate(itimevec):
            itime = Ntimes+itimeraw if itimeraw<0 else itimeraw
            if verbose: progress(ind+1, len(itimevec))
            db['timesteps'].append(itime)
            if gettimes:
                db['times'].append(float(timevec[itime]))
            for v in varnames:
                vvar = extractvar(ds, v, itime)
                for pt in ptlist:
                    db[pt][v].append(vvar[pt])
        if includeattr:
            for k, g in ds.attrs.items():
                db[k] = g
    return db

def avgPlaneXR_OLD(ncfile, timerange,
               extrafuncs=[],
               varnames=['velocityx','velocityy','velocityz'],
               savepklfile='',
               groupname=None, verbose=False, includeattr=False, 
               replacenan=False):
    """
    Compute the average of ncfile variables
    """
    suf='_avg'
    # Create a fresh db dictionary
    db = {}
    #for v in varnames: db[v] = {}
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
                    vdat[v] = nonan(extractvar(ds, v, itime), replacenan)
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
        if len(savepklfile)>0:
            # Write out the picklefile
            dbfile = open(savepklfile, 'wb')
            pickle.dump(db, dbfile, protocol=2)
            dbfile.close()
    return db

def avgPlaneXR(ncfileinput, timerange,
               extrafuncs=[],
               varnames=['velocityx','velocityy','velocityz'],
               savepklfile='',
               groupname=None, verbose=False, includeattr=False, 
               replacenan=False,axis_rotation=0):
    """
    Compute the average of ncfile variables
    """
    ncfilelist = getFileList(ncfileinput)

    ncfile=ncfilelist[0]
    suf='_avg'

    #Apply transformation after computing cartesian average
    transform=False
    if varnames == ['velocitya1','velocitya2','velocitya3']:
        transform = True
        varnames = ['velocityx','velocityy','velocityz']

    # Create a fresh db dictionary
    db = {}
    eps = 1.0E-10
    t1 = timerange[0]-eps
    t2 = timerange[1]
    db['times'] = []
    # Now load the ncfile data
    if groupname is None:
        groups= ppsample.getGroups(ppsample.loadDataset(ncfile))
        group = groups[0]
    else:
        group = groupname
    db['group'] = group
    Ncount = 0
    for ncfile in ncfilelist:
        timevec     = ppsample.getVar(ppsample.loadDataset(ncfile), 'time')
        filtertime  = np.where((t1 <= np.array(timevec)) & (np.array(timevec) <= t2))
        Ntotal      = len(filtertime[0])
        if verbose:
            print("%s %i"%(ncfile, Ntotal))
            #print("%f %f"%(t1, t2))
        localNcount = 0
        with xr.open_dataset(ncfile, group=group) as ds:
            if 'x' not in ds:
                reshapeijk = ds.attrs['ijk_dims'][::-1]
                xm = ds['coordinates'].data[:,0].reshape(tuple(reshapeijk))
                ym = ds['coordinates'].data[:,1].reshape(tuple(reshapeijk))
                zm = ds['coordinates'].data[:,2].reshape(tuple(reshapeijk))
                db['x'] = xm
                db['y'] = ym
                db['z'] = zm
                db['axis1'] = ds.attrs['axis1']
                db['axis2'] = ds.attrs['axis2']
                db['axis3'] = ds.attrs['axis3']
            # Set up the initial mean fields
            zeroarray = extractvar(ds, varnames[0], 0)
            for v in varnames:
                if v+suf not in db:
                    db[v+suf] = np.full_like(zeroarray, 0.0)
            if len(extrafuncs)>0:
                for f in extrafuncs:
                    if f['name']+suf not in db:
                        db[f['name']+suf] = np.full_like(zeroarray, 0.0)
            # Loop through and accumulate
            for itime, t in enumerate(timevec):
                if (t1 < t) and (t <= t2):
                    t1 = t
                    if verbose: progress(localNcount+1, Ntotal)
                    db['times'].append(float(t))
                    vdat = {}
                    for v in varnames:
                        vdat[v] = nonan(extractvar(ds, v, itime), replacenan)
                        db[v+suf] += vdat[v]
                    if len(extrafuncs)>0:
                        for f in extrafuncs:
                            name = f['name']+suf
                            func = f['func']
                            db[name] += func(vdat)
                    Ncount += 1
                    localNcount += 1
            print()  # Done with this file
    # Normalize the result
    if Ncount > 0:
        for v in varnames:
            db[v+suf] /= float(Ncount)
        if len(extrafuncs)>0:
            for f in extrafuncs:
                name = f['name']+suf
                db[name] /= float(Ncount)

    if transform:
        R=get_mapping_xyz_to_axis1axis2(db['axis1'],db['axis2'],db['axis3'],rot=axis_rotation)
        #if not np.array_equal(R,np.eye(R.shape[0])):
        db['velocitya1_avg'],db['velocitya2_avg'],db['velocitya3_avg'] = apply_coordinate_transform(R,db['velocityx_avg'],db['velocityy_avg'],db['velocityz_avg'])
    else:
        db['velocitya1_avg'] = db['velocityx_avg']
        db['velocitya2_avg'] = db['velocityy_avg']
        db['velocitya3_avg'] = db['velocityz_avg']

    if verbose:
        print("Ncount = %i"%Ncount)
        print()
    # include attributes
    if includeattr:
        for k, g in ds.attrs.items():
            db[k] = g
    if len(savepklfile)>0:
        # Write out the picklefile
        dbfile = open(savepklfile, 'wb')
        pickle.dump(db, dbfile, protocol=2)
        dbfile.close()
    return db

def MinMaxStd_PlaneXR(ncfile, timerange,
                      extrafuncs=[], avgdb = None,
                      varnames=['velocityx','velocityy','velocityz'], savepklfile='',
                      groupname=None, verbose=False, includeattr=False):
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
        if verbose: print("Calculating averages")
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
        if verbose: print("Calculating min/max/std")
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
        if len(savepklfile)>0:
            # Write out the picklefile
            dbfile = open(savepklfile, 'wb')
            pickle.dump(db, dbfile, protocol=2)
            dbfile.close()        
    return db

def ReynoldsStress_PlaneXR_OLD(ncfile, timerange,
                           extrafuncs=[], avgdb = None,
                           varnames=['velocityx','velocityy','velocityz'],
                           savepklfile='', groupname=None, verbose=False, includeattr=False):
    """
    Calculate the reynolds stresses
    """
    savg = '_avg'
    corrlist = [
        # name   variable1   variable2
        ['uu_avg', 'velocityx', 'velocityx'],
        ['uv_avg', 'velocityx', 'velocityy'],
        ['uw_avg', 'velocityx', 'velocityz'],
        ['vv_avg', 'velocityy', 'velocityy'],
        ['vw_avg', 'velocityy', 'velocityz'],
        ['ww_avg', 'velocityz', 'velocityz'],
    ]
    db = {}
    if avgdb is None:
        if verbose: print("Calculating averages")
        db = avgPlaneXR(ncfile, timerange,
                        extrafuncs=extrafuncs,
                        varnames=varnames,
                        groupname=groupname, verbose=verbose, includeattr=includeattr)
    else:
        db.update(avgdb)
    group   = db['group']
    timevec = ppsample.getVar(ppsample.loadDataset(ncfile), 'time')
    t1      = timerange[0]
    t2      = timerange[1]
    Ntotal  = len(db['times'])
    with xr.open_dataset(ncfile, group=group) as ds:
        reshapeijk = ds.attrs['ijk_dims'][::-1]
        zeroarray = extractvar(ds, varnames[0], 0)
        # Set up the initial mean fields
        for corr in corrlist:
            suff = corr[0]
            db[suff] =  np.full_like(zeroarray, 0.0)
        Ncount = 0
        # Loop through and accumulate
        if verbose: print("Calculating reynolds-stress")
        for itime, t in enumerate(timevec):
            if (t1 <= t) and (t <= t2):
                if verbose: progress(Ncount+1, Ntotal)
                vdat = {}
                for v in varnames:
                    vdat[v] = extractvar(ds, v, itime)        
                for corr in corrlist:
                    name = corr[0]
                    v1   = corr[1]
                    v2   = corr[2]
                    # Standard dev
                    db[name] += (vdat[v1]-db[v1+savg])*(vdat[v2]-db[v2+savg])
                Ncount += 1
        # Normalize and sqrt std dev
        if Ncount > 0:
            for corr in corrlist:
                name = corr[0]
                db[name] = db[name]/float(Ncount)
        if verbose: print()
        if len(savepklfile)>0:
            # Write out the picklefile
            dbfile = open(savepklfile, 'wb')
            pickle.dump(db, dbfile, protocol=2)
            dbfile.close()
    return db

def ReynoldsStress_PlaneXR(ncfileinput, timerange,
                           extrafuncs=[], avgdb = None,
                           varnames=['velocityx','velocityy','velocityz'],
                           savepklfile='', groupname=None, verbose=False, includeattr=False,axis_rotation=0):
    """
    Calculate the reynolds stresses
    """
    ncfilelist = getFileList(ncfileinput)

    print('first ncfilelist ',ncfilelist)
    ncfile=ncfilelist[0]
    eps     = 1.0E-10
    t1      = timerange[0]-eps
    t2      = timerange[1]    
    savg = '_avg'

    corr_mapping = {
        'velocityx': 'u',
        'velocityy': 'v',
        'velocityz': 'w',
        'velocitya1': 'ua1',
        'velocitya2': 'ua2',
        'velocitya3': 'ua3'
    }

    combinations = itertools.combinations_with_replacement(varnames, 2)

    corrlist = [
        [f"{corr_mapping[var1]}{corr_mapping[var2]}_avg", var1, var2]
        for var1, var2 in combinations
    ]

    db = {}
    if avgdb is None:
        if verbose: print("Calculating averages")

        db = avgPlaneXR(ncfilelist, timerange,
                        extrafuncs=extrafuncs,
                        varnames=varnames,
                        groupname=groupname, verbose=verbose, includeattr=includeattr,axis_rotation=axis_rotation)
    else:
        db.update(avgdb)


    group   = db['group']
    Ncount = 0    
    for ncfile in ncfilelist:
        timevec     = ppsample.getVar(ppsample.loadDataset(ncfile), 'time')
        filtertime  = np.where((t1 <= np.array(timevec)) & (np.array(timevec) <= t2))
        Ntotal      = len(filtertime[0])
        if verbose:
            print("%s %i"%(ncfile, Ntotal))
        localNcount = 0
        with xr.open_dataset(ncfile, group=group) as ds:
            reshapeijk = ds.attrs['ijk_dims'][::-1]
            zeroarray = extractvar(ds, 'velocityx', 0)
            # Set up the initial mean fields
            for corr in corrlist:
                suff = corr[0]
                if suff not in db:
                    db[suff] =  np.full_like(zeroarray, 0.0)

            if any('velocitya' in v for v in varnames):
                R=get_mapping_xyz_to_axis1axis2(db['axis1'],db['axis2'],db['axis3'],rot=axis_rotation)
            # Loop through and accumulate
            if verbose: print("Calculating reynolds-stress")
            for itime, t in enumerate(timevec):
                if (t1 < t) and (t <= t2):
                    t1 = t
                    if verbose: progress(localNcount+1, Ntotal)
                    vdat = {}
                    for v in ['velocityx','velocityy','velocityz']:
                        vdat[v] = extractvar(ds, v, itime)        
                    if any('velocitya' in v for v in varnames):
                        vdat['velocitya1'],vdat['velocitya2'],vdat['velocitya3'] = apply_coordinate_transform(R,vdat['velocityx'],vdat['velocityy'],vdat['velocityz'])
                    for corr in corrlist:
                        name = corr[0]
                        v1   = corr[1]
                        v2   = corr[2]
                        # Standard dev
                        db[name] += (vdat[v1]-db[v1+savg])*(vdat[v2]-db[v2+savg])
                    Ncount += 1
                    localNcount += 1
            print()  # Done with this file
    # Normalize and sqrt std dev
    if Ncount > 0:
        for corr in corrlist:
            name = corr[0]
            db[name] = db[name]/float(Ncount)
    if verbose: print()
    if len(savepklfile)>0:
        # Write out the picklefile
        dbfile = open(savepklfile, 'wb')
        pickle.dump(db, dbfile, protocol=2)
        dbfile.close()
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

def getFullPlaneXR(ncfile, num_time_steps,output_dt, groupname,ordering=["x","z","y"]):
    """
    Read all planes in netcdf file

    Modified from openfast-toolbox

    """
    ds = xr.open_dataset(ncfile,group=groupname)    
    coordinates = {"x":(0,"axial"), "y":(1,"lateral"),"z":(2,"vertical")}
    c           = {}
    for coordinate,(i,desc) in coordinates.items():
        c[coordinate] = xr.IndexVariable( 
                                            dims=[coordinate],
                                            data=np.sort(np.unique(ds['coordinates'].isel(ndim=i))), 
                                            attrs={"description":"{0} coordinate".format(desc),"units":"m"}
                                        )
    c["times"] = xr.IndexVariable( 
                                dims=["times"],
                                data=ds.num_time_steps*output_dt,
                                attrs={"description":"time from start of simulation","units":"s"}
                             )    

    nt = len(c["times"])
    nx = len(c["x"])
    ny = len(c["y"])
    nz = len(c["z"])
    coordinates = {"x":(0,"axial","velocityx"), "y":(1,"lateral","velocityy"),"z":(2,"vertical","velocityz")}    
    v           = {}    
    for coordinate,(i,desc,u) in coordinates.items():        
        v[u] = xr.DataArray(np.reshape(getattr(ds,"velocity{0}".format(coordinate)).values,(nt,nx,nz,ny)), 
                                coords=c, 
                                dims=["times",ordering[0],ordering[1],ordering[2]],
                                name="{0} velocity".format(desc), 
                                attrs={"description":"velocity along {0}".format(coordinate),"units":"m/s"})

    ds = xr.Dataset(data_vars=v, coords=v[u].coords)           
    
    return ds 
