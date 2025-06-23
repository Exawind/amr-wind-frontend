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

def find_2nearest(a, a0):
    asort = np.argsort(np.abs(np.array(a)-a0))
    return asort[0], asort[1]

def interpfields(t1, t2, v1, v2):
    return (v2-v1)/(t2-t1)+v1

def getFileList(ncfileinput):
    ncfilelist = []
    if isinstance(ncfileinput, str):
        ncfilelist=list(sorted(glob.glob(ncfileinput)))
    elif not isinstance(ncfileinput, list):
        for ncfileiter, ncfile in enumerate(ncfileinput):
            files = sorted(glob.glob(ncfile[ncfileiter]))
            for file in files:
                ncfilelist.append(file)
    else:
        for ncfileiter, ncfile in enumerate(ncfileinput):
            files = sorted(glob.glob(ncfile))
            for file in files:
                ncfilelist.append(file)

    return ncfilelist

def maskTimeVector(t, extractbounds, tlimits, eps=0.0):
    """
    Given extractbounds tA <= t <= tB and 
          tlimits t1 <= t <= t2
    Find the times in time vector t which satisfy both limits
    """
    tmask = (extractbounds[0]-eps <= t) & (t <= extractbounds[1]+eps) & (tlimits[0]-eps <= t) & (t <= tlimits[1]+eps)
    return tmask

def replaceDuplicateTime(ncfile, ttarget, ireplace, fracdt=0.01):
    """
    See if there is any time in ncfile which matches ttarget

    matching criteria is if ttarget matches time to within fracdt*dt
    
    if ireplace = +1, then replaces time with the next time in sequence
    if ireplace = -1, then replaces time with the previous time in sequence
    """
    alltimes = ppsample.getVar(ppsample.loadDataset(ncfile), 'time')[:]
    i0, i1 = find_2nearest(alltimes, ttarget)
    t0, t1 = alltimes[i0], alltimes[i1]
    dt = np.abs(t1-t0)
    newtime = t0
    if np.abs(ttarget-t0) <= fracdt*dt:
        # Target time matches, need to replace the time
        newi = i0 + ireplace
        if (0 <= newi) and (newi < len(alltimes)):
            newtime = alltimes[newi]
            #print('%f matches!  need to replace with %f'%(ttarget, newtime))
        else:
            # hit an error, tried to replace with something outside
            raise ValueError(f'Error in replaceDuplicateTime.  newi={newi} is out of bounds')
    return newtime

def sortAndSpliceFileList(ncfilelist, splicepriority='laterfiles'):
    """
    Sorts a list of netcdf files so that it runs in ascending time order
    Also returns a list of times to extract from each file
    """
    # Check to make sure splicepriority is a correct option
    spliceoptions=['laterfiles','earlierfiles']
    if not (splicepriority in spliceoptions):
        raise ValueError(f'option splicepriority={splicepriority} is not one of '+repr(spliceoptions))
    
    # First run through the list and get the time extents for each file
    timebounds = []
    alltimesdict = {}
    for ncfile in ncfilelist:
        alltimes = ppsample.getVar(ppsample.loadDataset(ncfile), 'time')[:]
        timebounds.append([alltimes[0], alltimes[-1]])
        alltimesdict[ncfile] = alltimes

    # Now order the files based on the first time in timebounds
    ziplist = list(zip(ncfilelist, timebounds))
    sortedlist = sorted(ziplist, key=lambda x: x[1][0])
    #for x in sortedlist:
    #    print(x[0].split('/')[-1], x[1])

    sortedfilelist = [x[0] for x in sortedlist]
        
    # Now go through and get the extraction times for each file
    extractbounds = None
    if splicepriority=='laterfiles':
        # Go through the list in reverse order
        for l in sortedlist[::-1]:
            ltimes = l[1]
            ncfile = l[0]
            if extractbounds is None:
                extractbounds = [ltimes]
            else:
                prevbounds = extractbounds[-1]
                newmaxt = replaceDuplicateTime(ncfile, prevbounds[0], -1)
                extractbounds.append([ltimes[0], newmaxt])

        # flip the exactbounds
        extractbounds = extractbounds[::-1]
    else:
        # Run through the list in forward order
        for l in sortedlist:
            ltimes = l[1]
            ncfile = l[0]            
            if extractbounds is None:
                extractbounds = [ltimes]
            else:
                prevbounds = extractbounds[-1]
                newmint = replaceDuplicateTime(ncfile, prevbounds[1], +1)
                extractbounds.append([newmint, ltimes[1]])
    fullziplist = list(zip(sortedlist, extractbounds))

    # Create a list of time vectors from the netcdf files
    outtimevec = []
    for f in sortedfilelist:
        outtimevec.append(alltimesdict[f])
    return sortedfilelist, extractbounds, outtimevec

def getPlaneXR(ncfileinput, itimevec, varnames, groupname=None,
               verbose=0, includeattr=False, gettimes=False,timerange=None,times=None, axis_rotation=0):

    ncfilelist = getFileList(ncfileinput)
    ncfilelistsorted, extracttimes, timevecs = sortAndSpliceFileList(ncfilelist, splicepriority='laterfiles')

    # Create a fresh db dictionary
    db = {}
    for v in varnames: db[v] = {}
    db['timesteps'] = []

    find_nearest = lambda a, a0: np.abs(np.array(a) - a0).argmin()

    #Apply transformation after computing cartesian average
    transform=False
    if varnames == ['velocitya1','velocitya2','velocitya3']:
        transform = True
        varnames = ['velocityx','velocityy','velocityz']

    #concatenate timevecs in order without duplicates 
    all_timevecs = []
    for ncfileiter,ncfile in enumerate(ncfilelistsorted):
        timevec     = timevecs[ncfileiter]
        extracttime = extracttimes[ncfileiter]
        for time in timevec:
            if time >= extracttime[0] and time <= extracttime[1]:
                all_timevecs.append(time)
    all_timevecs = np.asarray(all_timevecs)

    if times is not None:
        for time in times:
            itimevec.append(np.argmin( np.abs( all_timevecs - time ) ))

    if timerange is not None:
        for itime, time in enumerate(all_timevecs):
            if time >= timerange[0] and time <= timerange[1]:
                itimevec.append(itime)

    itime_processed = []
    for ncfileiter,ncfile in enumerate(ncfilelistsorted):
        timevec     = timevecs[ncfileiter]
        extracttime = extracttimes[ncfileiter]
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
            if verbose>0:
                print("Extracting from ncfile: ",ncfile,ncfileiter)
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
                db['axis1'] = ds.attrs['axis1']
                db['axis2'] = ds.attrs['axis2']
                try:
                    db['axis3'] = ds.attrs['offset_vector']
                except:
                    db['axis3'] = ds.attrs['axis3']
                R=get_mapping_xyz_to_axis1axis2(db['axis1'],db['axis2'],db['axis3'],rot=axis_rotation)
            for itime in itimevec:
                time = all_timevecs[itime]
                if itime not in itime_processed and time >= extracttime[0] and time <= extracttime[1]:
                    local_ind = np.argmin(np.abs(time-timevec))
                    if verbose>0:
                        print("extracting iter "+repr(itime))
                    db['timesteps'].append(itime)
                    if gettimes:
                        db['times'].append(float(all_timevecs[itime]))
                    if not transform:
                        for v in varnames:
                            vvar = extractvar(ds, v, local_ind)
                            db[v][itime] = vvar
                    else:
                        vvarx = extractvar(ds, varnames[0], local_ind)
                        vvary = extractvar(ds, varnames[1], local_ind)
                        vvarz = extractvar(ds, varnames[2], local_ind)
                        db['velocitya1'][itime],db['velocitya2'][itime],db['velocitya3'][itime] = apply_coordinate_transform(R,vvarx,vvary,vvarz)
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


def avgPlaneXR(ncfileinput, timerange,
               extrafuncs=[],
               varnames=['velocityx','velocityy','velocityz'],
               savepklfile='',
               groupname=None, verbose=False, includeattr=False, 
               replacenan=False,axis_rotation=0):
    """
    Compute the average of ncfile variables
    """
    # make sure input is a list
    ncfilelist = getFileList(ncfileinput)
    ncfilelistsorted, extracttimes, timevecs = sortAndSpliceFileList(ncfilelist, splicepriority='laterfiles')
    ncfile=ncfilelist[0]
    suf='_avg'

    #Apply transformation after computing cartesian average
    natural_velocity_mapping = {
        'velocitya1': 'velocityx',
        'velocitya2': 'velocityy',
        'velocitya3': 'velocityz'
    }

    transform = False
    for viter, v in enumerate(varnames):
        for key, value in natural_velocity_mapping.items():
            if key in v:
                varnames[viter] = value
                transform = True
                break  

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

    for ncfileiter, ncfile in enumerate(ncfilelistsorted):
        timevec     = timevecs[ncfileiter]
        tmask       = maskTimeVector(timevec, extracttimes[ncfileiter], timerange, eps=0.0E-16) 
        Ntotal      = sum(tmask)
        
        if verbose:
            print("%s %i"%(ncfile, Ntotal))
            #print("%f %f"%(t1, t2))
        localNcount = 0
        with xr.open_dataset(ncfile, group=group) as ds:
            if verbose:
                print("Getting data from ncfile: ",ncfile)
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
                try:
                    db['axis3'] = ds.attrs['offset_vector']
                except:
                    db['axis3'] = ds.attrs['axis3']
                db['origin'] = ds.attrs['origin']
                db['offsets'] = ds.attrs['offsets']
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
                if tmask[itime]:
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
        db['velocitya1_avg'],db['velocitya2_avg'],db['velocitya3_avg'] = apply_coordinate_transform(R,db['velocityx_avg'],db['velocityy_avg'],db['velocityz_avg'])

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

def phaseAvgPlaneXR(ncfileinput, tstart, tend, tperiod,
                    extrafuncs=[],
                    varnames=['velocityx','velocityy','velocityz'],
                    savepklfile='',
                    groupname=None, verbose=False, includeattr=False,
                    replacenan=False,axis_rotation=0):
    """
    Compute the phase average of ncfile variables
    """
    # make sure input is a list
    ncfilelist = getFileList(ncfileinput)
    ncfile=ncfilelist[0]
    suf='_phavg'

    find_nearest = lambda a, a0: np.abs(np.array(a) - a0).argmin()

    #Apply transformation after computing cartesian average
    natural_velocity_mapping = {
        'velocitya1': 'velocityx',
        'velocitya2': 'velocityy',
        'velocitya3': 'velocityz'
    }

    transform = False
    for viter, v in enumerate(varnames):
        for key, value in natural_velocity_mapping.items():
            if key in v:
                varnames[viter] = value
                transform = True
                break

    # Create a fresh db dictionary
    db = {}
    eps = 1.0E-10
    tnow = tstart
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
        filtertime  = np.where((tnow <= np.array(timevec)) & (np.array(timevec) <= tend))
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
                db['origin'] = ds.attrs['origin']
                db['offsets'] = ds.attrs['offsets']
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
            while (tnow <= tend) and (tnow <= timevec[-1]):
                # Find the closest time to tnow
                i1, i2   = find_2nearest(timevec, tnow)
                ti1, ti2 = timevec[i1], timevec[i2]
                #print(tnow, ti1, ti2)
                #iclosest = find_nearest(timevec, tnow)
                #tclosest = timevec[iclosest]
                if verbose: progress(i1, len(timevec))

                # Add to db accumulation
                db['times'].append(float(tnow))
                vdat = {}
                for v in varnames:
                    v1      = nonan(extractvar(ds, v, i1), replacenan)
                    v2      = nonan(extractvar(ds, v, i2), replacenan)
                    vdat[v] = interpfields(ti1, ti2, v1, v2)
                    #vdat[v] = nonan(extractvar(ds, v, iclosest), replacenan)
                    db[v+suf] += vdat[v]
                if len(extrafuncs)>0:
                    for f in extrafuncs:
                        name = f['name']+suf
                        func = f['func']
                        db[name] += func(vdat)
                # increment counters
                Ncount += 1
                localNcount += 1
                tnow += tperiod
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
        db['velocitya1'+suf], db['velocitya2'+suf], db['velocitya3'+suf] = apply_coordinate_transform(R,
                                                                                                      db['velocityx'+suf],
                                                                                                      db['velocityy'+suf],
                                                                                                      db['velocityz'+suf])

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


def ReynoldsStress_PlaneXR(ncfileinput, timerange,
                           extrafuncs=[], avgdb = None,
                           varnames=['velocityx','velocityy','velocityz'],
                           savepklfile='', groupname=None, verbose=False, includeattr=False,axis_rotation=0):
    """
    Calculate the reynolds stresses
    """
    ncfilelist = getFileList(ncfileinput)
    ncfilelistsorted, extracttimes, timevecs = sortAndSpliceFileList(ncfilelist, splicepriority='laterfiles')
    ncfile=ncfilelist[0]

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

        orig_varnames = varnames[:]
        db = avgPlaneXR(ncfilelist, timerange,
                        extrafuncs=extrafuncs,
                        varnames=orig_varnames,
                        groupname=groupname, verbose=verbose, includeattr=includeattr,axis_rotation=axis_rotation)
    else:
        db.update(avgdb)

    group   = db['group']
    Ncount = 0    

    times_processed = []
    mindt = float('inf')
    times = []

    for ncfileiter, ncfile in enumerate(ncfilelistsorted):
        timevec     = timevecs[ncfileiter]
        tmask       = maskTimeVector(timevec, extracttimes[ncfileiter], timerange, eps=0.0E-16) 
        Ntotal      = sum(tmask)

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
                if tmask[itime]:
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

    if verbose:
        print("Ncount = %i"%Ncount)
        print()
    if len(savepklfile)>0:
        # Write out the picklefile
        dbfile = open(savepklfile, 'wb')
        pickle.dump(db, dbfile, protocol=2)
        dbfile.close()
    return db

def phaseAvgReynoldsStress1_PlaneXR(ncfileinput, tstart, tend, tperiod,
                                    extrafuncs=[], avgdb = None,
                                    varnames=['velocityx','velocityy','velocityz'], replacenan=False,
                                    savepklfile='', groupname=None, verbose=False, includeattr=False,axis_rotation=0):
    """
    Calculate the phase-averaged reynolds stresses
    
    Computes < (u_i - \overline{u_i})*(u_j - \overline{u_j}) >
    """
    ncfilelist = getFileList(ncfileinput)
    ncfile=ncfilelist[0]
    eps     = 1.0E-10
    tavg = '_avg'
    pavg = '_phavg'

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
        [f"{corr_mapping[var1]}{corr_mapping[var2]}{pavg}", var1, var2]
        for var1, var2 in combinations
    ]

    db = {}
    if avgdb is None:
        if verbose: print("Calculating averages")

        orig_varnames = varnames[:]
        db = avgPlaneXR(ncfilelist, [tstart, tend],
                        extrafuncs=extrafuncs,
                        varnames=orig_varnames,
                        groupname=groupname, verbose=verbose, includeattr=includeattr,axis_rotation=axis_rotation)
    else:
        db.update(avgdb)

    group   = db['group']
    Ncount = 0
    tnow = tstart
    for ncfile in ncfilelist:
        timevec     = ppsample.getVar(ppsample.loadDataset(ncfile), 'time')
        filtertime  = np.where((tnow <= np.array(timevec)) & (np.array(timevec) <= tend))
        Ntotal      = len(filtertime[0])
        if verbose:
            print("%s %i"%(ncfile, Ntotal))
            #print("%f %f"%(t1, t2))
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
            #if verbose: print("Calculating reynolds-stress")
            while (tnow <= tend) and (tnow <= timevec[-1]):
                # Find the closest time to tnow
                i1, i2   = find_2nearest(timevec, tnow)
                ti1, ti2 = timevec[i1], timevec[i2]
                #print(tnow, ti1, ti2)
                #iclosest = find_nearest(timevec, tnow)
                #tclosest = timevec[iclosest]
                if verbose: progress(i1, len(timevec))

                # Add to db accumulation
                db['times'].append(float(tnow))
                vdat = {}
                for v in varnames:
                    v1      = nonan(extractvar(ds, v, i1), replacenan)
                    v2      = nonan(extractvar(ds, v, i2), replacenan)
                    vdat[v] = interpfields(ti1, ti2, v1, v2)
                    if any('velocitya' in v for v in varnames):
                        vdat['velocitya1'],vdat['velocitya2'],vdat['velocitya3'] = apply_coordinate_transform(R,vdat['velocityx'],vdat['velocityy'],vdat['velocityz'])
                for corr in corrlist:
                    name = corr[0]
                    v1   = corr[1]
                    v2   = corr[2]
                    # Standard dev
                    db[name] += (vdat[v1]-db[v1+tavg])*(vdat[v2]-db[v2+tavg])
                # increment counters
                Ncount += 1
                localNcount += 1
                tnow += tperiod
            print()  # Done with this file

    # Normalize 
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

def avgLineXR(ncfileinput, timerange, varnames, extrafuncs=[], groupname=None,
              verbose=0, includeattr=False, gettimes=False):
    # make sure input is a list
    ncfilelist = getFileList(ncfileinput)
    ncfile=ncfilelist[0]
    suf='_avg'

    # Create a fresh db dictionary
    db = {}
    eps = 1.0E-10
    t1 = timerange[0]-eps
    t2 = timerange[1]
    if gettimes: db['times'] = []

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
        localNcount = 0
        with xr.open_dataset(ncfile, group=group) as ds:
            if 'x' not in ds:
                xm = ds['coordinates'].data[:,0]
                ym = ds['coordinates'].data[:,1]
                zm = ds['coordinates'].data[:,2]
                dtime=xr.open_dataset(ncfile)
                dtime.close()
                db['x'] = xm
                db['y'] = ym
                db['z'] = zm
            # Set up the initial mean fields
            zeroarray = np.zeros(len(ds.num_points))
            for v in varnames:
                if v+suf not in db:
                    db[v+suf] = np.full_like(zeroarray, 0.0)
            if len(extrafuncs)>0:
                for f in extrafuncs:
                    if f['name']+suf not in db:
                        db[f['name']+suf] = np.full_like(zeroarray, 0.0)
            # Loop through and accumulate
            for itime, t in enumerate(timevec):
                    t1 = t
                    if verbose: progress(localNcount+1, Ntotal)
                    if gettimes: db['times'].append(float(t))
                    vdat = {}
                    for v in varnames:
                        vdat[v] = ds[v][itime,:]
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
    if verbose:
        print("Ncount = %i"%Ncount)
        print()
    # include attributes
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
