#!/usr/bin/env python
#
#

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import os.path as path

stdvars = ['u',         'v',      'w',        'theta', 
           u"u'u'_r",  u"u'v'_r", u"u'w'_r", 
           u"v'v'_r",  u"v'w'_r", u"w'w'_r",
           u"u'theta'_r", u"v'theta'_r", u"w'theta'_r",]

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
        avgdat = avgdat + 0.5*dt*(datfiltered[i+1,:] + datfiltered[i,:])
    return avgdat/(tend-tstart)

def loadnetcdffile(filename):
    if path.exists(filename):
        return Dataset(filename)
    else:
        print("%s DOES NOT EXIST.")
        return None

def loadProfileData(d, varslist=stdvars, group='mean_profiles', avgt=[]):
    alldat={}
    #print(d['mean_profiles'].variables)
    t = d.variables['time'][:]
    alldat['avgt'] = avgt
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

def calculateUhoriz(allvars, avgt, ncdat):
    requiredvars = ['u', 'v']
    if not matchvarstimes(allvars, requiredvars, avgt):
        # Load the data from the ncdat file
        var = loadProfileData(ncdat, varslist=requiredvars, avgt=avgt)
    else:
        var = allvars
    # compute U horizontal
    return var['z'], np.sqrt(var['u']**2 + var['v']**2)


def calculateShearAlpha(allvars):
    # compute Umag
    u_mag = np.sqrt(allvars['u']**2 + allvars['v']**2)
    z     = allvars['z']
    dudz = (u_mag[1:]-u_mag[0:-1])/(z[1:]-z[0:-1])
    dudz=np.append(dudz, dudz[-1])
    alpha=z/u_mag*dudz
    return z, alpha

def calculateExpr(expr, allvars, avgt, ncdat):
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
            answer=answer.replace(exprv.decode('utf-8'), '('+repr(var[v][i])+')')
        vec.append(eval(answer))
    # compute U horizontal
    return var['z'], np.array(vec)

# A dictionary with all of the variables you can plot
profiles={'velocity': [['u', 'v', 'w'],     'u v w',
                       '[[u], [v], [w]]'],
          'Uhoriz':   [['u', 'v'],          'Uhoriz',
                       'np.sqrt([u]**2 + [v]**2)'],
          'tke':      [[u"u'u'_r", u"v'v'_r", u"v'v'_r",], 'tke'
                       '0.5*([uu]**2+[vv]**2+[ww]**2)'],
}
    
class CalculatedProfile:
    def __init__(self, requiredvars, expr, ncdat, allvardata, avgt, header='',
                 funcstring=False):
        self.requiredvars = requiredvars
        self.expr         = expr
        self.ncdat        = ncdat
        self.allvardata   = allvardata
        self.avgt         = avgt
        self.vec          = None
        self.funcstring   = funcstring

    def calculate(self, allvars=None, avgt=None):
        if allvars is None: allvars = self.allvardata
        if avgt    is None: avgt    = self.avgt
        if not matchvarstimes(allvars, self.requiredvars, avgt):
            # Load the data from the ncdat file
            var = loadProfileData(self.ncdat, 
                                  varslist=self.requiredvars, 
                                  avgt=avgt)
            self.allvardata = var
        else:
            var = allvars        
        # Now evalulate the function
        if self.funcstring:
            z, vec = eval(self.expr+"(var)")
        else:
            # Calculate the expression
            Nz  = len(var['z'])
            vec = []
            for i in range(Nz):
                answer=self.expr
                for v in self.requiredvars:
                    exprv = exprvars[v]
                    answer= answer.replace(exprv.decode('utf-8'), 
                                           '('+repr(var[v][i])+')')
                evalans = eval(answer)
                vec.append(evalans)
        vec = np.array(vec)
        self.z = var['z']
        self.vec = vec
        return var['z'], vec
        
        
