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

import postproamrwindsample as ppsample
import postproamrwindsample_xarray as ppsamplexr

# Load the libraries
import numpy as np
import matplotlib.pyplot as plt
import copy
import xarray as xr

# === Stuff for the netcdf file handling ====
try:
    from netCDF4 import Dataset
    netCDF4loaded = True
except:
    netCDF4loaded = False

def reshapedict(ds, iplane, vvars):
    (Ny, Nx)=ds['x'][iplane,:,:].shape
    ix = np.array([range(0,Nx) for i in range(0,Ny)]).reshape((Nx*Ny,1))
    iy = np.array([[i]*Nx for i in range(0,Ny)]).reshape((Nx*Ny,1))
    iz = np.array([[0]*Nx for i in range(0,Ny)]).reshape((Nx*Ny,1))
    xm = ds['x'][iplane,:,:].reshape((Nx*Ny,1))
    ym = ds['y'][iplane,:,:].reshape((Nx*Ny,1))
    zm = ds['z'][iplane,:,:].reshape((Nx*Ny,1))
    outdat = np.hstack((iz, iy, ix, xm, ym, zm))
    for v in vvars:
        vdat = ds[v][iplane,:,:].reshape((Nx*Ny,1))
        outdat = np.hstack((outdat, vdat))
    return outdat

def avgNCplaneXR(ncfilename, tavg, group, iplane, verbose=False, replacenan=False):
    vvars = ['velocityx','velocityy','velocityz']
    vavg  = ['velocityx_avg','velocityy_avg','velocityz_avg']
    ds = ppsamplexr.avgPlaneXR(ncfilename, tavg,
                               extrafuncs=[],
                               varnames=vvars,
                               groupname=group, verbose=verbose, includeattr=False, replacenan=replacenan)
    avgdat = reshapedict(ds, iplane, vavg)
    headers="Plane_Number Index_j Index_i coordinates[0] coordinates[1] coordinates[2] velocity_probe[0] velocity_probe[1] velocity_probe[2]"
    return avgdat, headers.split()

def avgNCPlane(ncfilename, tindices, filename, group='p_f', verbose=False):
    """    
    Average the netcdf data over tindices and optionally save to file
    """
    if (not netCDF4loaded): Exception("netCDF4 not loaded") 
    ncdat  = Dataset(ncfilename, 'r')

    allvx  = ncdat[group].variables['velocityx']
    allvy  = ncdat[group].variables['velocityy']
    allvz  = ncdat[group].variables['velocityz']
    allT   = ncdat[group].variables['temperature']
        
    # Get coordinates
    # -- Zindex of all planes matching z --
    #zind = (ncdat[group].variables['coordinates'][:,2]==zplane)
    allx = ncdat[group].variables['coordinates'][:,0]
    ally = ncdat[group].variables['coordinates'][:,1]
    allz = ncdat[group].variables['coordinates'][:,2]
    # -- Get the i,j indices ---
    #ji=np.array([[j, i] for j in range(ncdat[group].ijk_dims[1]) for i in range(ncdat[group].ijk_dims[0])])
    kji=np.array([[k, j, i] for k in range(ncdat[group].ijk_dims[2]) for j in range(ncdat[group].ijk_dims[1]) for i in range(ncdat[group].ijk_dims[0])])
    
    # Get the means 
    avgvx = np.mean(allvx[tindices, :], axis=0)
    avgvy = np.mean(allvy[tindices, :], axis=0)
    avgvz = np.mean(allvz[tindices, :], axis=0)
    avgT  = np.mean(allT[tindices, :], axis=0)
    # Assemble the return matrix
    avgdat = np.vstack((kji.transpose(), allx,ally,allz, avgvx, avgvy, avgvz, avgT)).transpose()
    if len(filename)>0:
        header="Plane_Number Index_j Index_i coordinates[0] coordinates[1] coordinates[2] velocity_probe[0] velocity_probe[1] velocity_probe[2] temperature_probe[0]"
        np.savetxt(filename, avgdat, header=header)
    return avgdat, header.split()

def extractNCplane(ncdat, tindex, group='p_f'):
    """
    Extracts a plane at time tindex out of the ncdat netcdf data
    """
    # Get variables and coordinates
    t      = ncdat['time'][:]
    allvx  = ncdat[group].variables['velocityx']
    allvy  = ncdat[group].variables['velocityy']
    allvz  = ncdat[group].variables['velocityz']
    allT   = ncdat[group].variables['temperature']
    allx   = ncdat[group].variables['coordinates'][:,0]
    ally   = ncdat[group].variables['coordinates'][:,1]
    allz   = ncdat[group].variables['coordinates'][:,2]
    # -- Get the i,j indices ---
    kji      = np.array([[k, j, i] for k in range(ncdat[group].ijk_dims[2]) for j in range(ncdat[group].ijk_dims[1]) for i in range(ncdat[group].ijk_dims[0])])
    slicevx  = allvx[tindex,:]
    slicevy  = allvy[tindex,:]
    slicevz  = allvz[tindex,:]
    sliceT   = allT[tindex,:]
    planedat = np.vstack((kji.transpose(), allx,ally,allz, slicevx, slicevy, slicevz, sliceT)).transpose()
    return planedat

# ===============================
# Load all of the information needed from the file
def loadplanefile(filename):
    dat=np.loadtxt(filename, skiprows=2)
    # Get the maximum indices
    numplanes = int(max(dat[:,0]))
    Numj      = int(max(dat[:,1]))
    Numi      = int(max(dat[:,2]))
    #print numplanes, Numi, Numj
    fname, fext = os.path.splitext(filename)
    if ((fext == '.gz') or (fext == '.GZ')):
        with gzip.open(filename) as fp:
            timestring = fp.readline().strip().split()[1]
            headers    = fp.readline().strip('#').split()[:]
    else:
        with open(filename) as fp:
            timestring = fp.readline().strip().split()[1]
            headers    = fp.readline().strip('#').split()[:]
    time=float(timestring)
    #print time, headers
    fp.close()
    return dat, time, headers

def groupvars(allvarlist):
    """
    group the list of variables
    """
    justvarnames = [x.split("[")[0] for x in allvarlist]
    uniquevars   = []
    [uniquevars.append(x) for x in justvarnames if x not in uniquevars]
    varsizes = [[x, justvarnames.count(x)] for x in uniquevars]
    return varsizes

def getvelocityindices(header):
    varstart    = 6
    allvars     = header[varstart:]
    groupedvars = groupvars(allvars)
    varnames    = [x[0] for x in groupedvars]
    istart      = varnames.index('velocity_probe')
    return [int(varstart+istart), 
            int(varstart+istart+1), 
            int(varstart+istart+2)]


def avgplanefiles(filelist, verbose=False):
    """
    Average the plane files given in filelist
    """
    N = len(filelist)
    for ifile, filename in enumerate(filelist):
        if (verbose): 
            print('Loading [%i/%i]: %s'%(ifile+1, N,os.path.basename(filename)))
        dat, time, headers=loadplanefile(filename)
        if (ifile==0): 
            avgdat=dat
        else:
            avgdat=avgdat+dat
    return avgdat/(float(N)), headers


def loadavg(filelist, loadfromplanes, avgsavefile, 
            nctindices=[], verbose=False):
    """
    Load the average of all planes
    """
    # TODO: fix this for XR-array loading
    if (loadfromplanes):
        if len(nctindices)>0:
            # load from NC file
            avgdat, headers=avgNCPlane(filelist, nctindices, avgsavefile, 
                                       verbose=verbose)
        else:
            # load text file planes and average them
            avgdat, headers=avgplanefiles(filelist, verbose=verbose)
            # Save it
            if len(avgsavefile)>0:
                np.savetxt(avgsavefile, avgdat, header=' '.join(headers))
    else:
        avgdat=np.loadtxt(avgsavefile)
        with open(avgsavefile) as fp:
            headers    = fp.readline().strip('#').split()[:]
    return avgdat, headers

def getavgwind(avgdat, headers, iplane):
    Ni, Nj, Nplanes, iuvw = getsizesindices(avgdat, headers)
    dat  = avgdat[avgdat[:,0]==iplane,:]
    iu   = iuvw[0]
    iv   = iuvw[1]
    iw   = iuvw[2]
    avgu = np.mean(dat[:,iu])
    avgv = np.mean(dat[:,iv])
    avgw = np.mean(dat[:,iw])
    theta = np.arctan(avgv/avgu)*180.0/np.pi
    return [avgu, avgv, avgw], 270-theta


def getsizesindices(dat, headers):
    # Get the indices for u, v, w velocities
    if len(headers)>0: iuvw = getvelocityindices(headers)
    else:              iuvw = []
    Nplanes = max(dat[:,0])+1
    Nj      = max(dat[:,1])+1
    Ni      = max(dat[:,2])+1
    return int(Ni), int(Nj), int(Nplanes), iuvw
    
def getplaneindex(i, j, iplane, Ni, Nj):
    return i + Ni*j + Nj*Ni*iplane

def sanitizepoint(pt, Ni, Nj, Nplanes, lastpointperiodic=True):
    ipt    = pt[0]
    jpt    = pt[1]
    iplane = pt[2]
    if ((iplane<0)or(iplane>Nplanes)):
        print("ERROR in iplane=%i, NOT 0< %i < %i"%(iplane, iplane,Nplanes))
        sys.exit(1)
    pad = 0
    if (lastpointperiodic): pad = 1
    if (ipt >= Ni): ipt = ipt - (Ni - pad)
    if (jpt >= Nj): jpt = jpt - (Nj - pad)
    if (ipt <   0): ipt = ipt + (Ni - pad)
    if (jpt <   0): jpt = jpt + (Nj - pad)
    return [ipt, jpt, iplane]

def convertUVWtoLongLat(uvw, avguvw):
    """Converts u, v, w velocities to longitudinal, lateral, and vertical
    velocities
    """
    magU = np.sqrt(avguvw[0]**2 + avguvw[1]**2)
    nx   = avguvw[0]/magU
    ny   = avguvw[1]/magU
    nz   = 0
    longdir = np.array([nx, ny, nz])
    zdir    = np.array([0, 0, 1])
    latdir  = np.cross(zdir, longdir)
    
    ulong   = np.dot(uvw, longdir)
    ulat    = np.dot(uvw, latdir)
    uvert   = np.dot(uvw, zdir)
    return [ulong, ulat, uvert]

def makeRij(ij, allplist, filelist, loadfromplanes, avgsavefile, iplane, group,
            avgdat = None, headers=None, timerange=None,
            ncfilename='', verbose=False, norm=1, skip=1,
            replacenan=False):
    # Get the average data
    if (avgdat is None) and (headers is None): 
        avgdat, headers       = loadavg(filelist, loadfromplanes, avgsavefile, 
                                        verbose=verbose)
    Ni, Nj, Nplanes, iuvw = getsizesindices(avgdat, headers)

    Nplist   = len(allplist)
    Npt      = len(allplist[0])
        
    allxdist = []

    for plist in allplist:
        #Rij = np.zeros(Npt)

        # -- Construct the distance vector --
        # get the first coordinate
        p0   = plist[0]
        i0   = getplaneindex(p0[0], p0[1], p0[2], Ni, Nj)
        xyz0 = avgdat[i0, 3:6]
        xdist  = []
        #print(xyz0)
        # Loop through all points
        for ip, ptx in enumerate(plist):
            if (ip==0): 
                xdist.append(0)
            else:
                pt    = sanitizepoint(ptx, Ni, Nj, Nplanes)
                i    = getplaneindex(pt[0], pt[1], pt[2], Ni, Nj)
                xyzi = avgdat[i, 3:6]
                #print(xyzi)
                delta = np.linalg.norm(xyzi-xyz0)
                xdist.append(delta)
        allxdist.append(xdist)

    allRij   = np.zeros((Nplist, Npt))
    u0prime2 = np.zeros((Nplist, Npt))
    u1prime2 = np.zeros((Nplist, Npt))

    #if len(ncfilename)>0:
    #    if (not netCDF4loaded): Exception("netCDF4 not loaded") 
    #    ncdata   = Dataset(ncfilename, 'r')

    if len(ncfilename)>0:
        t1 = timerange[0]
        t2 = timerange[1]
        timevec = ppsample.getVar(ppsample.loadDataset(ncfilename), 'time')
        # Extract the relevant times
        filelist=[]
        for ti, t in enumerate(timevec):
            if (t1 <= t) and (t <= t2):
                filelist.append(ti)
        # Load the dataset
        ds=xr.open_dataset(ncfilename, group=group)
        reshapeijk = ds.attrs['ijk_dims'][::-1]
        xm = ds['coordinates'].data[:,0].reshape(tuple(reshapeijk))
        ym = ds['coordinates'].data[:,1].reshape(tuple(reshapeijk))
        zm = ds['coordinates'].data[:,2].reshape(tuple(reshapeijk))
        db = {}
        db['x'] = xm
        db['y'] = ym
        db['z'] = zm        
        
    # -- Construct the Rij --
    # Loop through all files
    for ifile, filename in enumerate(filelist[::skip]):
        if (verbose): 
            if len(ncfilename)>0:
                #statusstring='Computing [%i/%i]'%(ifile+1, len(filelist))
                ppsamplexr.progress(ifile+1, len(filelist[::skip]), digits=2)
            else:
                shortfname=os.path.basename(filename)            
                statusstring='Computing [%i/%i]: %s'%(ifile+1, len(filelist[::skip]), shortfname)
                print(statusstring)
        if len(ncfilename)>0:
            tindex = filename
            #dat = extractNCplane(ncdata, tindex)  # Old way of extracting data
            vvars = ['velocityx', 'velocityy', 'velocityz']
            for v in vvars:
                db[v] = ppsamplexr.nonan(ppsamplexr.extractvar(ds, v, tindex), replacenan)
            dat = reshapedict(db, iplane, vvars)
        else:
            dat, time, headers=loadplanefile(filename)
        for ilist, plist in enumerate(allplist):
            p0   = sanitizepoint(plist[0], Ni, Nj, Nplanes) #plist[0]
            i0   = getplaneindex(p0[0], p0[1], p0[2], Ni, Nj)
            avguvw0 = [avgdat[i0,iuvw[0]], 
                       avgdat[i0,iuvw[1]], 
                       avgdat[i0,iuvw[2]]]
            u0   = dat[i0, iuvw[0]] - avguvw0[0]
            v0   = dat[i0, iuvw[1]] - avguvw0[1]
            w0   = dat[i0, iuvw[2]] - avguvw0[2]
            ulonglat0 = convertUVWtoLongLat([u0, v0, w0], avguvw0)
            for ip, ptx in enumerate(plist):
                pt    = sanitizepoint(ptx, Ni, Nj, Nplanes)
                i1    = getplaneindex(pt[0], pt[1], pt[2], Ni, Nj)
                avguvw1 = [avgdat[i1, iuvw[0]], 
                           avgdat[i1, iuvw[1]], 
                           avgdat[i1, iuvw[2]]]
                u1    = dat[i1, iuvw[0]] - avguvw1[0]
                v1    = dat[i1, iuvw[1]] - avguvw1[1]
                w1    = dat[i1, iuvw[2]] - avguvw1[2]
                ulonglat1 = convertUVWtoLongLat([u1, v1, w1], avguvw1)
                dir1 = ij[0]
                dir2 = ij[1]
                upup = ulonglat0[dir1]*ulonglat1[dir2]
                u0prime2[ilist][ip] = u0prime2[ilist][ip] + ulonglat0[dir1]**2
                u1prime2[ilist][ip] = u1prime2[ilist][ip] + ulonglat1[dir1]**2

                #if (verbose): print('upup = '+repr(upup))
                #Rij[ip] = Rij[ip] + upup
                allRij[ilist][ip] = allRij[ilist][ip]  + upup 
    allRij = allRij/float(Npt)
    u0prime2 = u0prime2/float(Npt)
    u1prime2 = u1prime2/float(Npt)
    if norm==1:
        for i in range(Nplist):
            for j in range(Npt):
                allRij[i][j] = allRij[i][j]/(np.sqrt(u0prime2[i][j])*np.sqrt(u1prime2[i][j]))
    elif norm==2:
        for i in range(np.shape(allRij)[0]):
            allRij[i,:]=allRij[i,:]/allRij[i,0]
    return allxdist, allRij
        

def plotprobes(allplist, dat, headers, verbose=False, skip=1):
    """
    """
    Ni, Nj, Nplanes, iuvw = getsizesindices(dat, headers)

    # Get the max dimensions
    xmax      = max(dat[:,3])
    ymax      = max(dat[:,4])
    xmin      = min(dat[:,3])
    ymin      = min(dat[:,4])    
    # build the domain outline
    xlines = [xmin, xmin, xmax, xmax, xmin]
    ylines = [ymin, ymax, ymax, ymin, ymin]
    plt.plot(xlines, ylines, 'b-')
    # plot the probe points
    for ilist, plist in enumerate(allplist):
        #if (verbose): print("On %i of %i"%(ilist, len(allplist)))
        if (verbose):
            sys.stdout.write("\r%d%%" % int((ilist+1)*100.0/len(allplist)))
            sys.stdout.flush()
        for ip, pt in enumerate(plist[::skip]):
            pts   = sanitizepoint(pt, Ni, Nj, Nplanes)
            i1    = getplaneindex(pts[0], pts[1], pts[2], Ni, Nj)
            xyz1  = dat[i1, 3:6]
            color='gray'
            ms = 2
            if (ip==0): 
                ms = 10
                color='g'
            if (i1>=Ni*Nj*(pts[2]+1)): 
                ms = 10
                color='r'
            plt.plot(xyz1[0], xyz1[1], '.', color=color, markersize=ms)
    L=xmax-xmin
    H=ymax-ymin
    plt.xlim([xmin-0.1*L, xmax+0.1*L])
    plt.ylim([ymin-0.1*H, ymax+0.1*H])
    return

def calclengthscale(xi, Rij, max=0):
    #if (max==0): max=len(xi)+1
    if (max==0):
        max = next(x for x, val in enumerate(Rij) if val < 0.0) 
    integral = np.trapz(Rij[:max], x=xi[:max])
    return integral

def makeprobeline(startpts, winddir, Lmin, dat):
    """
    """
    # Get the dimensions
    Ni, Nj, Nplanes, iuvw = getsizesindices(dat, '')
    i1    = getplaneindex(0, 0, 0, Ni, Nj)
    i2    = getplaneindex(1, 0, 0, Ni, Nj)
    dx    = np.linalg.norm(dat[i1, 3:6]-dat[i2, 3:6])
    i1    = getplaneindex(0, 0, 0, Ni, Nj)
    i2    = getplaneindex(0, 1, 0, Ni, Nj)
    dy    = np.linalg.norm(dat[i1, 3:6]-dat[i2, 3:6])
    # Convert the wind direction
    theta = (270.0-winddir)*np.pi/180        # In x-y coordinate system
    nhat  = np.array([np.cos(theta), np.sin(theta), 0])
    nhatsign = np.array([np.sign(nhat[0]), np.sign(nhat[1]), 0])
    # Find the right march length
    eps   = 1.0E-8
    if np.abs(np.dot(nhat, [1, 0, 0])) < eps:
        lmarch = dy
    elif np.abs(np.dot(nhat, [0, 1, 0])) < eps:
        lmarch = dx
    else:
        ly = dy/np.sin(theta)
        lx = dx/np.cos(theta)
        lmarch = abs(min(ly, lx))
    #print('nhat   = '+repr(nhat))
    #print('dx     = '+repr(dx))
    #print('dy     = '+repr(dy))
    #print('lmarch = '+repr(lmarch))

    alllines=[]
    for startp in startpts:
        # Get the starting position
        p0   = sanitizepoint(startp, Ni, Nj, Nplanes) 
        i0   = getplaneindex(p0[0], p0[1], p0[2], Ni, Nj)
        startxy = np.array(dat[i0, 3:6])
        Ltotal  = 0.0
        plist   = [copy.deepcopy(p0)]
        currpt  = startxy       # In physical space
        currptraster = startxy  # The current line tip, discretized
        currptidx    = p0
        while (Ltotal < Lmin):
            nextpt = currpt + lmarch*nhat
            Ltotal = Ltotal + lmarch

            # Test to see if advance to next point
            deltapt = nextpt - currptraster
            if (abs(deltapt[0]) > 0.5*dx): 
                currptidx[0] = currptidx[0] + int(nhatsign[0])
                currptraster[0] = currptraster[0] + dx
            if (abs(deltapt[1]) > 0.5*dy): 
                currptidx[1] = currptidx[1] + int(nhatsign[1])
                currptraster[1] = currptraster[1] + dy
            #print('currptidx = '+repr(currptidx))
            #print('plist = '+repr(plist))
            plist.append(copy.deepcopy(currptidx))
            # Move to next point
            currpt = nextpt
        alllines.append(plist)
    return alllines

def makeprobegrid(startpts, Lx, Ly, dat):
    """
    """
    # Get the dimensions
    Ni, Nj, Nplanes, iuvw = getsizesindices(dat, '')
    i1    = getplaneindex(0, 0, 0, Ni, Nj)
    i2    = getplaneindex(1, 0, 0, Ni, Nj)
    dx    = np.linalg.norm(dat[i1, 3:6]-dat[i2, 3:6])
    i1    = getplaneindex(0, 0, 0, Ni, Nj)
    i2    = getplaneindex(0, 1, 0, Ni, Nj)
    dy    = np.linalg.norm(dat[i1, 3:6]-dat[i2, 3:6])

    for startp in startpts:
        # Get the starting position
        p0   = sanitizepoint(startp, Ni, Nj, Nplanes) 
