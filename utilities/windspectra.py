#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Defaults
defaultwindow = {'choice':'tukey','alpha':0.1}
uKaimalconst  = {'a':105.0, 'b':33.0, 'alphaExp':1.0,'betaExp':5.0/3.0}
vKaimalconst  = {'a':17.0,  'b':9.5,  'alphaExp':1.0,'betaExp':5.0/3.0}
wKaimalconst  = {'a':2.1,   'b':5.3,  'alphaExp':5.0/3.0,'betaExp':1.0}

# The Tukey window
# see https://en.wikipedia.org/wiki/Window_function#Tukey_window
def tukeyWindow(N, params={'alpha':0.1}):
    alpha = params['alpha']
    w = np.zeros(N)
    L = N+1
    for n in np.arange(0, int(N//2) + 1):
        if ((0 <= n) and (n < 0.5*alpha*L)):
            w[n] = 0.5*(1.0 - np.cos(2*np.pi*n/(alpha*L)))
        elif ((0.5*alpha*L <= n) and (n <= N/2)):
            w[n] = 1.0
        else:
            print("Something wrong happened at n = ",n)
        if (n != 0): w[N-n] = w[n]
    return w

# FFT's a signal, returns 1-sided frequency and spectra (non-normalized)
def getFFT(t, y):
    n    = len(y)
    k    = np.arange(n)
    dt   = np.mean(np.diff(t))
    frq  = k/(n*dt)
    FFTy = np.fft.fft(y) # /(len(y))
    
    # Take the one sided version of it
    freq = frq[range(int(n//2))]
    FFTy = FFTy[range(int(n//2))]
    return freq, FFTy

# Takes a wind speed signal ws and returns the spectra
#  Be sure to look at 
#  https://www.nwpsaf.eu/publications/tech_reports/nwpsaf-kn-tr-008.pdf
#  for normalization!
def getWindSpectra(t, ws, window=defaultwindow):
    N      = len(ws)
    dt     = np.mean(np.diff(t))
    # Get the window
    if window['choice']=='tukey':    w = tukeyWindow(N, params=window)
    else:                            w = 1 
    # FFT to get the coefficients
    f, fWS = getFFT(t, w*(ws-np.mean(ws)))
    Su     = dt/N*abs(fWS)**2
    return f, Su

# average wind spectra across buckets
def avgWindSpectra(t, u, avgbins=[], window=defaultwindow):
    numbins=len(avgbins)
    if (numbins<1):  
        binbuckets = [[np.min(t), np.max(t)]]
        numbins = 1
    else: 
        binbuckets = avgbins
    for it, tbin in enumerate(binbuckets):
        t1  = tbin[0]
        t2  = tbin[1]
        #print(t1,t2)
        tfilter   = ((t[:] >= t1) & (t[:] <= t2))
        tbin = t[tfilter]
        ubin = u[tfilter]
        f, Su = getWindSpectra(tbin, ubin, window=window)
        if (it == 0):
            favg = f
            Savg = Su
        else:
            # Double check freqencies match
            if (np.sum(np.abs(np.array(f)-np.array(favg))) > 1.0E-9):
                print("frequencies in bins don't match!")
                sys.exit(1)
            # Add to average
            Savg = Savg + Su
    Savg = Savg/numbins
    return favg, Savg

# IEC Kaimal spectra
def getIECKaimal(f, z, WS, sigma, direction='u'):
    if type(WS) is float:    avgU = WS  
    else:                    avgU = abs(np.mean(WS))
    if type(sigma) is float: stdU = sigma
    else:                    stdU = np.std(WS)
    # Turbulence scale parameter
    if z<=60: lambda1 = 0.7*z
    else:     lambda1 = 42
    # Integral scale parameter
    if direction=='u': Lu   = 8.10*lambda1
    if direction=='v': Lu   = 2.70*lambda1
    if direction=='w': Lu   = 0.66*lambda1
    #stdU = np.std(WS)
    #print("avgU = ",avgU)
    #print("stdU = ",stdU)
    Su   = 4.0*Lu/avgU/(1.0 + 6.0*f*Lu/avgU)**(5.0/3.0)
    #Su   = Su*f
    Su   = Su*stdU*stdU
    return Su

# Get the Kaimal spectral model, normalized
# Returns f*Su/sigma^2
def getKaimal(f, z, WS, params=uKaimalconst):
    if type(WS) is float:  avgU = WS  
    else:                  avgU = abs(np.mean(WS))
    stdU = np.std(WS)
    a     = params['a']
    b     = params['b']
    alpha = params['alphaExp']
    beta  = params['betaExp']
    n     = f*z/avgU
    Su    = a*n/(1.0 + b*n**alpha)**(beta)
    #Su   = a*n/(1.0 + b*n)**(5.0/3.0)
    return Su

# Converts the u,v velocities in x,y to longitudinal and lateral
# velocities
def convertUxytoLongLat(u, v):
    avgU = np.mean(u)
    avgV = np.mean(v)
    magU = np.sqrt(avgU**2 + avgV**2)
    nx   = avgU/magU
    ny   = avgV/magU
    ulong = u*nx + v*ny
    ulat  = u*ny - v*nx
    return ulong, ulat
 
# Average the spectra over multiple locations
# given by locations in planefilelist
def avgSpectraFiles(planefilelist, zheight, datadir='.',verbose=True, avgbins=[]):
    # set initial values
    iplane = 0
    all_ulongavgs = []
    lent = 0
    # Loop through and get spectra
    for planefile in planefilelist:
        if not os.path.isfile(datadir+'/'+planefile): continue
        dat       = np.loadtxt(datadir+'/'+planefile, skiprows=1)
        filterdat=dat[dat[:,3]==zheight,:]
        t=filterdat[:,0]
        u=filterdat[:,4]
        v=filterdat[:,5]
        w=filterdat[:,6]
        #if ((iplane != 0) and (lent != len(t))): continue
        if ((iplane != 0) and (len(t)<lent)): continue
        if iplane == 0:    lent    = len(t)
        if verbose: print("Loaded plane "+planefile)
        all_ulongavgs.append(np.mean(u))
        f, Suu      = avgWindSpectra(t[:lent], u[:lent], avgbins)
        f, Svv      = avgWindSpectra(t[:lent], v[:lent], avgbins)
        f, Sww      = avgWindSpectra(t[:lent], w[:lent], avgbins)
        if iplane == 0:
            lent    = len(t)
            favg    = f
            Suu_avg = Suu
            Svv_avg = Svv
            Sww_avg = Sww
        else:
            Suu_avg = Suu_avg + Suu
            Svv_avg = Svv_avg + Svv
            Sww_avg = Sww_avg + Sww
        iplane = iplane+1
    # Average the spectra
    Suu_avg = Suu_avg/iplane
    Svv_avg = Svv_avg/iplane
    Sww_avg = Sww_avg/iplane
    if verbose: print("Averaged over %i planes"%iplane)
    return favg, Suu_avg, Svv_avg, Sww_avg, np.mean(all_ulongavgs)

def NarrowToOctaveBand(freq, dat, n):
    """
    Convert dat from narrowband freq to 1/n octave band
    """
    df = np.mean(np.diff(freq))
    lowf  = 1000 # Start at 1000 Hz
    prevf = lowf*(2**(1.0/n))
    while (prevf-lowf)>df:
        prevf = lowf
        lowf  = lowf/(2**(1.0/n))

    # Calculate center frequencies
    fcenter = [lowf]
    while (fcenter[-1] < max(freq)):
        fcenter.append(fcenter[-1]*(2**(1.0/n)))
    # Calculate low, high, and bandwidth from center frequencies
    flow  = [f/np.sqrt(2**(1.0/n)) for f in fcenter]
    fhigh = [f*np.sqrt(2**(1.0/n)) for f in fcenter]
    bandw = np.array(fhigh)-np.array(flow)

    # bin the data
    bindat   = np.zeros(len(fcenter))
    bincount = np.zeros(len(fcenter))
    for fi, f in enumerate(freq):
        for j in range(len(fcenter)):
            if ((flow[j]<= f) and (f<fhigh[j])): 
                bincount[j] += 1
                bindat[j]   += dat[fi]
                break

    avgdat  = []
    avgflow = []
    avgfc   = []
    avgfhigh= []
    for i in range(len(fcenter)): 
        if bincount[i]>0:
            avgdat.append(bindat[i]/bincount[i])
            avgfc.append(fcenter[i])
            avgflow.append(flow[i])
            avgfhigh.append(fhigh[i])

    return np.array(avgfc), np.array(avgdat)
