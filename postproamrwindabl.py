#!/usr/bin/env python
#
# Copyright (c) 2022, Alliance for Sustainable Energy
#/
# This software is released under the BSD 3-clause license. See LICENSE file
# for more details.
#
#
#

import numpy as np
import math
import sys
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import os.path as path
from collections            import OrderedDict 
from scipy import interpolate
from scipy.optimize import curve_fit
import mmap

scalarvars=[u'time', u'Q', u'Tsurf', u'ustar', u'wstar', u'L', u'zi', u'abl_forcing_x', u'abl_forcing_y']

stdvars = ['u',         'v',      'w',        'theta', 
           u"u'u'_r",  u"u'v'_r", u"u'w'_r", 
           u"v'v'_r",  u"v'w'_r", u"w'w'_r",
           u"u'theta'_r", u"v'theta'_r", u"w'theta'_r", 
           'k_sgs', 'k_rans', 'sdr', 'eps', 'mueff',
           'abl_meso_forcing_mom_x', 'abl_meso_forcing_mom_y',
           'abl_meso_forcing_mom_theta']

exprvars = { "u":'[u]',
             "v":'[v]',
             "w":'[w]',
             "theta":'[T]',
             u"u'u'_r": '[uu]',
             u"u'v'_r":'[uv]',
             u"u'w'_r":'[uw]', 
             u"v'v'_r":'[vv]',
             u"v'w'_r":'[vw]',
             u"w'w'_r":'[ww]',
             u"u'theta'_r":'[uT]',
             u"v'theta'_r":'[vT]',
             u"w'theta'_r":'[wT]',
             'k_sgs':'[k_sgs]',
             'k_rans':'[k_rans]',
             'mueff':'[mueff]',
             'sdr':'[sdr]',
             'eps':'[eps]',
             'abl_meso_forcing_mom_x':'[abl_meso_forcing_mom_x]',
             'abl_meso_forcing_mom_y':'[abl_meso_forcing_mom_y]',
             'abl_meso_forcing_mom_theta':'[abl_meso_forcing_mom_theta]',
           }

def timeaverage(t, dat, t1, t2):
    Ndim   = len(np.shape(dat))
    tfiltered   = t[(t>=t1)&(t<=t2)]
    if Ndim==1:
        datfiltered = dat[(t>=t1)&(t<=t2)]
    else:
        datfiltered = dat[(t>=t1)&(t<=t2),:]
        Nvars  = len(dat[0,:])
    tstart = tfiltered[0]
    tend   = tfiltered[-1]
    avgdat = 0.0 if Ndim==1 else np.zeros(Nvars)
    for i in range(len(tfiltered)-1):
        dt     = tfiltered[i+1] - tfiltered[i]
        if Ndim==1:
            avgdat = avgdat + 0.5*dt*(datfiltered[i+1] + datfiltered[i])
        else:
            avgdat = avgdat + 0.5*dt*(datfiltered[i+1,:] + datfiltered[i,:])
    return avgdat/(tend-tstart)

def loadnetcdffile(filename, usemmap=False):
    if path.exists(filename):
        if usemmap:
            print("Loading entire file into memory...")
            with open(filename, 'rb') as f:
                mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ,
                               flags=mmap.MAP_PRIVATE)
                ncread = mm.read()
                return Dataset('inmemory.nc', memory=ncread)
        else:
            return Dataset(filename)
    else:
        print("%s DOES NOT EXIST."%filename)
        return None

def loadProfileData(d, varslist=stdvars, group='mean_profiles', avgt=[], 
                    usemapped=True):
    alldat={}
    t = d.variables['time'][:]
    alldat['time'] = t
    alldat['avgt'] = avgt
    if usemapped and ('hmapped' in d['mean_profiles'].variables.keys()):
        alldat['z'] = d['mean_profiles'].variables['hmapped'][:]
    else:
        alldat['z'] = d['mean_profiles'].variables['h'][:]
    for var in varslist:
        print('Loading '+var)
        x = d[group].variables[var][:,:]
        if len(avgt)>=2:
            t1 = avgt[0]
            t2 = avgt[1]
            alldat[var] = timeaverage(t, x, t1, t2)
        else:
            alldat[var] = x
    return alldat

def matchvarstimes(dic, varnames, avgt):
    # get all keys in dic
    allkeys = [key for key, x in dic.items()]
    for var in varnames:
        if var not in allkeys: return False
    if dic['avgt'] != avgt:    return False
    return True

def calculateShearAlpha(allvars, ncdat=None, avgt=None,span=None):

    # compute Umag
    u_mag = np.sqrt(allvars['u']**2 + allvars['v']**2)
    z     = allvars['z']
    dudz = (u_mag[1:]-u_mag[0:-1])/(z[1:]-z[0:-1])
    dudz=np.append(dudz, dudz[-1])
    alpha=z/u_mag*dudz

    return z, alpha

def calculateShearAlpha_Fit(allvars, ncdat=None, avgt=None,span=None):

    #define functional form for wind speed profile
    def func(x,a,b):
        return b*x**a

    u_mag = np.sqrt(allvars['u']**2 + allvars['v']**2)
    z     = allvars['z']
    if span == None:
        print("Rotor span not specified. Fitting alpha over entire vertical domain")
        popt, pcov = curve_fit(func,z,u_mag)
    else:
        #only perform fit over rotor span
        z_span = (z >= span[0]) & (z <= span[1])
        popt, pcov = curve_fit(func,z[z_span],u_mag[z_span])
    return z, np.full_like(z,popt[0])

def calculateVeer(allvars, ncdat=None, avgt=None,span=None):
    #approximate the veer as d\Theta/dz with centered difference
    wind_dir = 270-np.arctan2(allvars['v'], allvars['u'])*180.0/math.pi
    z     = allvars['z']
    dwindDirdz = (wind_dir[1:]-wind_dir[0:-1])/(z[1:]-z[0:-1])
    dwindDirdz=np.append(dwindDirdz, dwindDirdz[-1])
    return z, dwindDirdz

def calculateVeer_Fit(allvars, ncdat=None, avgt=None,span=None):
    wind_dir = 270-np.arctan2(allvars['v'], allvars['u'])*180.0/math.pi
    z     = allvars['z']

    #calculate hub-height wind direction accounting for the discontinuity from 0/360 deg 
    ydata = wind_dir - wind_dir[0]
    for j in range(len(ydata))[1:]:
        temp1 = ydata[j]
        temp2 = ydata[j] + 360
        temp3 = ydata[j] - 360
        tempvector = [temp1,temp2,temp3]
        tempvector_absolute = np.absolute(tempvector)
        min_value = min(tempvector_absolute)
        min_index = np.nonzero(tempvector_absolute == min_value)
        temp = np.asarray(min_index[0])
        ydata[j] = tempvector[temp[0]]

    def func(x, a, b):
        return a*x+b

    if span == None:
        print("Rotor span not specified. Fitting veer over entire vertical domain")
        popt, pcov = curve_fit(func,z,ydata)
    else:
        #only perform fit over rotor span
        z_span = (z >= span[0]) & (z <= span[1])
        popt, pcov = curve_fit(func,z[z_span],ydata[z_span])

    return z, np.full_like(z,popt[0])

def calculateObukhovL(allvars, ncdat=None, avgt=None,span=None):
    k = 0.40
    g = 9.81
    z     = allvars['z']
    ustar = timeAvgScalar(ncdat, 'ustar', avgt)
    Oblength = -ustar**3/(k*g/allvars['theta']*allvars[u"w'theta'_r"])
    return z, Oblength

def calculateExpr(expr, allvars, avgt, ncdat, usemapped=True):
    requiredvars = ['u', 'v']
    if not matchvarstimes(allvars, requiredvars, avgt):
        # Load the data from the ncdat file
        var = loadProfileData(ncdat, varslist=requiredvars, avgt=avgt)
    else:
        var = allvars
    # Calculate the expression
    Nz  = len(var['z'])
    vec = []

    for i in range(Nz):
        answer=expr
        for v in requiredvars:
            exprv = exprvars[v]
            answer=answer.replace(exprv.encode().decode('utf-8'), '('+repr(var[v][i])+')')
        vec.append(eval(answer))
    # compute U horizontal
    return var['z'], np.array(vec)

statsprofiles_ = OrderedDict()
def registerstatsprofile(f):
    defdict = {'requiredvars':f.requiredvars,
                'header':f.header,
                'expr':f.expr, 
                'funcstring':f.funcstring}
    statsprofiles_[f.key] = defdict
    #print("Added "+f.key+" profile")
    return f

@registerstatsprofile
class velocityprof():
    key          = 'velocity'
    requiredvars = ['u', 'v', 'w']
    header       = 'u v w'
    expr         = '[[u], [v], [w]]'
    funcstring   = False
    
# A dictionary with all of the variables you can plot
statsprofiles=OrderedDict([
    ('velocity', {'requiredvars':['u', 'v', 'w'],     
                  'header':'u v w',
                  'expr':'[[u], [v], [w]]', 
                  'funcstring':False}),
    ('Uhoriz',   {'requiredvars':['u', 'v'],          
                  'header':'Uhoriz',
                  'expr':'np.sqrt([u]**2 + [v]**2)', 
                  'funcstring':False}),
    ('WindDir',  {'requiredvars':['u', 'v'],          
                  'header':'WindDir',
                  'expr':'270-np.arctan2([v], [u])*180.0/math.pi', 
                  'funcstring':False}),
    ('Temperature', {'requiredvars':['theta'],     
                     'header':'T',
                     'expr':'[T]', 
                     'funcstring':False}),
    ('TI_TKE',   {'requiredvars':['u', 'v', u"u'u'_r", u"v'v'_r", u"w'w'_r",], 
                  'header':'TI_TKE',
                  'expr':'np.sqrt(([uu]+[vv]+[ww])/3.0)/np.sqrt([u]**2 + [v]**2)', 
                  'funcstring':False}),
    ('TI_horiz', {'requiredvars':['u', 'v', u"u'u'_r", u"v'v'_r"], 
                  'header':'TI_horiz',
                  'expr':'np.sqrt([uu]+[vv])/np.sqrt([u]**2 + [v]**2)', 
                  'funcstring':False}),
    ('TKE',      {'requiredvars':[u"u'u'_r", u"v'v'_r", u"w'w'_r",], 
                  'header':'TKE',
                  'expr':'0.5*([uu]+[vv]+[ww])', 
                  'funcstring':False}),
    ('KSGS',      {'requiredvars':['k_sgs'], 
                   'header':'k_sgs',
                   'expr':'[k_sgs]', 
                  'funcstring':False}),
    ('KRANS',     {'requiredvars':['k_rans'], 
                   'header':'k_rans',
                   'expr':'[k_rans]', 
                   'funcstring':False}),
    ('SDR_OMEGA',     {'requiredvars':['sdr'], 
                   'header':'sdr',
                   'expr':'[sdr]', 
                   'funcstring':False}),
    ('ReStresses',{'requiredvars':[u"u'u'_r",  u"u'v'_r", u"u'w'_r", 
                                   u"v'v'_r",  u"v'w'_r", u"w'w'_r",], 
                   'header':'uu uv uw vv vw ww',
                   'expr':'[[uu], [uv], [uw], [vv], [vw], [ww]]', 
                   'funcstring':False}),
    ('Tfluxes',{'requiredvars':[u"u'theta'_r", u"v'theta'_r", u"w'theta'_r",], 
                   'header':'uT vT wT',
                   'expr':'[[uT], [vT], [wT]]', 
                   'funcstring':False}),
    ('MUEFF', {'requiredvars':['mueff'], 
                   'header':'mueff',
                   'expr':'[mueff]', 
                   'funcstring':False}),
    ('Alpha',    {'requiredvars':['u', 'v'],          
                  'header':'alpha',
                  'expr':'calculateShearAlpha', 
                  'funcstring':True}),
    ('Alpha-Fit', {'requiredvars':['u', 'v'],
                  'header':'alpha',
                  'expr':'calculateShearAlpha_Fit', 
                  'funcstring':True}),
    ('Veer',  {'requiredvars':['u', 'v'], 
                  'header':'veer',
                  'expr':'calculateVeer',
                  'funcstring':True}),
    ('Veer-Fit',  {'requiredvars':['u', 'v'],
                  'header':'veer',
                  'expr':'calculateVeer_Fit',
                  'funcstring':True}),
    ('ObukhovL', {'requiredvars':['theta', u"w'theta'_r"],
                  'header':'ObukhovL',
                  'expr':'calculateObukhovL', 
                  'funcstring':True}),
    ('MMC-forcing', {'requiredvars':['abl_meso_forcing_mom_x',
                                     'abl_meso_forcing_mom_y',
                                    ],
                     'header':'abl_meso_forcing_mom_x abl_meso_forcing_mom_y',
                     'expr':'[[abl_meso_forcing_mom_x], [abl_meso_forcing_mom_y]]',
                     'funcstring':False}),
])
    
class CalculatedProfile:
    def __init__(self, requiredvars, expr, ncdat, allvardata, avgt, span=None,header='',
                 funcstring=False, usemapped=True):
        self.requiredvars = requiredvars
        self.expr         = expr
        self.ncdat        = ncdat
        self.allvardata   = allvardata
        self.avgt         = avgt
        self.vec          = None
        self.funcstring   = funcstring
        self.header       = header
        self.usemapped    = usemapped
        self.span         = span

    @classmethod
    def fromdict(cls, d, ncdat, allvardata, avgt, span=None,usemapped=True):
        return cls(d['requiredvars'], d['expr'], ncdat, allvardata, avgt,span,
                   header=d['header'], funcstring=d['funcstring'],
                   usemapped=usemapped)

    def calculate(self, allvars=None, avgt=None,span=None):
        if allvars is None: allvars = self.allvardata
        if avgt    is None: avgt    = self.avgt
        if span    is None: span    = self.span
        if not matchvarstimes(allvars, self.requiredvars, avgt):
            # Load the data from the ncdat file
            var = loadProfileData(self.ncdat, 
                                  varslist=self.requiredvars, 
                                  avgt=avgt, usemapped=self.usemapped)
            self.allvardata = var
        else:
            var = allvars        
        # Now evalulate the function
        if self.funcstring:
            z, vec = eval(self.expr+"(var, ncdat=self.ncdat, avgt=avgt,span=span)")
        else:
            # Calculate the expression
            Nz  = len(var['z'])
            vec = []
            for i in range(Nz):
                answer=self.expr
                for v in self.requiredvars:
                    exprv = exprvars[v]
                    answer= answer.replace(exprv.encode().decode('utf-8'), 
                                           '('+repr(var[v][i])+')')
                evalans = eval(answer)
                vec.append(evalans)
        vec = np.array(vec)
        self.z = var['z']
        self.vec = vec
        return var['z'], vec

    def save(self, filename, allvars=None, avgt=None, extraheader=''):
        # Calculate the quantity
        z, vec = self.calculate(allvars=allvars, avgt=avgt)
        # Save it to the filename
        savedat = np.vstack((z, vec.transpose())).transpose()
        if len(extraheader)>0:
            header = extraheader + "\nz "+self.header
        else:
            header = "z "+self.header
        np.savetxt(filename, savedat, header=header)
        return

def extractScalarTimeHistory(ncdat, var):
    # Pull out the time
    t = np.array(ncdat.variables['time'])
    v = np.array(ncdat.variables[var])
    return t, v

def timeAvgScalar(ncdat, var, avgt):
    # Pull out the time history first
    t, v = extractScalarTimeHistory(ncdat, var)
    # Average it 
    avgv = timeaverage(t, v, avgt[0], avgt[1])
    return avgv
        
def printReport(ncdat, heights, avgt, span,verbose=True):
    """
    Print out a report of the ABL statistics at given heights
    """
    # Dict which holds all of the output variables
    reportvars={}

    # Get the scalar quantities
    avgustar = timeAvgScalar(ncdat, 'ustar', avgt)
    reportvars['ustar'] = avgustar

    avgzi= timeAvgScalar(ncdat, 'zi', avgt)
    reportvars['zi'] = avgzi 

    # Get the profile quantities
    profvars = ['Uhoriz', 'WindDir', 'TI_TKE', 'TI_horiz', 'Alpha', 'Alpha-Fit','ObukhovL','Veer','Veer-Fit']
    # Build the list of all required variables
    requiredvars = []
    for var in profvars:
        neededvars = statsprofiles[var]['requiredvars']
        requiredvars = list(set(requiredvars) | set(neededvars))
    # Load the variables
    alldata = loadProfileData(ncdat, varslist=requiredvars, avgt=avgt)

    # Get the profile quantities
    for var in profvars:
        # Get the quantity at every height
        prof=CalculatedProfile.fromdict(statsprofiles[var],ncdat, alldata, avgt,span)
        z, qdat = prof.calculate()
        interpf = interpolate.interp1d(z, qdat)
        reportvars[var]=[interpf(z) for z in heights]

    if verbose:
        # Print the header
        sys.stdout.write('%9s '%'z')
        for var in profvars:  sys.stdout.write('%12s '%var)
        sys.stdout.write('\n')
        sys.stdout.write('%9s '%'===')
        for var in profvars:  sys.stdout.write('%12s '%'====')        
        sys.stdout.write('\n')
        # write the results
        for iz, z in enumerate(heights):
            sys.stdout.write('%9.2f '%z)
            for var in profvars :
                sys.stdout.write('%12e '%reportvars[var][iz])
            sys.stdout.write('\n') 
        sys.stdout.write('\n')
        sys.stdout.write('ustar: %f'%reportvars['ustar'])
        sys.stdout.write('\n')
        sys.stdout.write('zi: %f'%reportvars['zi'])
        sys.stdout.write('\n')
    return reportvars

if __name__ == "__main__":
    ncdat=loadnetcdffile('abl_statistics00000.nc')
    avgt=[15000, 20000]
    heights=[60.0, 91.0]
    #printReport(ncdat, heights, avgt, verbose=True)
    CalculatedProfile.fromdict(statsprofiles['velocity'], ncdat, {}, avgt).save('velocity.dat')
