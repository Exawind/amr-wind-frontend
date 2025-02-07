# Get the location where this script is being run
import sys, os
scriptpath = os.path.dirname(os.path.realpath(__file__))
basepath   = os.path.dirname(scriptpath)
utilpath   = os.path.join(basepath, "utilities")
# Add any possible locations of amr-wind-frontend here
amrwindfedirs = ['../',
                 utilpath,
                 basepath]
for x in amrwindfedirs: sys.path.insert(1, x)

from postproengine import registerplugin, mergedicts, registeraction
import windspectra
import postproamrwindsample_xarray as ppsamplexr
import numpy as np
import pandas as pd
import postproamrwindabl as ppabl

def get_angular_wavenumbers(N,L):
    k = np.zeros(N)
    counter = 0
    for i in range(0,int(N/2)+1):
        k[counter] = 2*np.pi/L * i 
        counter += 1
    for i in range(1,int(N/2)):
        k[counter] = -2*np.pi/L*(N/2-i) 
        counter += 1
    return k

def extract_1d_from_meshgrid(Z):
    # Check along axis 0
    unique_rows = np.unique(Z, axis=0)
    if unique_rows.shape[0] == 1:
        return unique_rows[0],1

    # Check along axis 1
    unique_cols = np.unique(Z, axis=1)
    if unique_cols.shape[1] == 1:
        return unique_cols[:, 0],0

def read_cart_data(ncfile,varnames,group,trange,iplanes,xaxis,yaxis):
    db = ppsamplexr.getPlaneXR(ncfile,[0,1],varnames,groupname=group,verbose=0,includeattr=True,gettimes=True,timerange=trange)
    if iplanes == None: 
        if isinstance(db['offsets'], np.ndarray):
            iplanes = list(range(len(db['offsets'])))
        else:
            iplanes = [0,]
    if not isinstance(iplanes, list): iplanes = [iplanes,]
    if ('a1' in [xaxis, yaxis]) or ('a2' in [xaxis, yaxis]) or ('a3' in [xaxis, yaxis]):
        compute_axis1axis2_coords(db,rot=0)
        R = get_mapping_xyz_to_axis1axis2(db['axis1'],db['axis2'],db['axis3'],rot=0)
        origin = db['origin']
        origina1a2a3 = R@db['origin']
        offsets = db['offsets']
        offsets = [offsets] if (not isinstance(offsets, list)) and (not isinstance(offsets,np.ndarray)) else offsets

    heights = np.zeros(len(iplanes))
    XX  = np.array(db[xaxis])
    YY = np.array(db[yaxis])

    t = np.asarray(np.array(db['times']).data)

    udata = {}
    for iplaneiter, iplane in enumerate(iplanes):
        if ('a1' in [xaxis, yaxis]) or ('a2' in [xaxis, yaxis]) or ('a3' in [xaxis, yaxis]):
            heights[iplaneiter] = origina1a2a3[-1] + offsets[iplane]
        else:
            heights[iplaneiter] = db['z'][iplane,0,0]
        x,axisx = extract_1d_from_meshgrid(XX[iplane,:,:])
        y,axisy = extract_1d_from_meshgrid(YY[iplane,:,:])

        #set data to time,x,y
        permutation = [0,axisx+1,axisy+1]
        udata[iplane] = np.zeros((len(t),len(x),len(y),3))
        for i,tstep in enumerate(db['timesteps']):
            if ('velocitya' in varnames[0]) or ('velocitya' in varnames[1]) or ('velocitya' in varnames[2]):
                ordered_data = np.transpose(np.array(db['velocitya3'][tstep]),permutation)
                udata[iplane][i,:,:,0] = ordered_data[iplane,:,:]

                ordered_data = np.transpose(np.array(db['velocity'+xaxis][tstep]),permutation)
                udata[iplane][i,:,:,1] = ordered_data[iplane,:,:]

                ordered_data = np.transpose(np.array(db['velocity'+yaxis][tstep]),permutation)
                udata[iplane][i,:,:,2] = ordered_data[iplane,:,:]
            else:
                ordered_data = np.transpose(np.array(db['velocityx'][tstep]),permutation)
                udata[iplane][i,:,:,0] = ordered_data[iplane,:,:]

                ordered_data = np.transpose(np.array(db['velocityy'][tstep]),permutation)
                udata[iplane][i,:,:,1] = ordered_data[iplane,:,:]

                ordered_data = np.transpose(np.array(db['velocityz'][tstep]),permutation)
                udata[iplane][i,:,:,2] = ordered_data[iplane,:,:]
    return udata , heights, x , y , t, iplanes 
@registerplugin
class wavenumberspectra_executor():
    """
    Computes 2D wavenumber spectra 
    """
    name      = "wavenumber_spectra"                # Name of task (this is same as the name in the yaml)
    blurb     = "Calculates 2D wavenumber spectra in x and y"  # Description of task
    inputdefs = [
        # --- Required inputs ---
        {'key':'name',    'required':True,  'help':'An arbitrary name',  'default':''},
        {'key':'ncfile',  'required':True,  'help':'NetCDF file of horizontal planes',        'default':''},
        {'key':'group',   'required':True,  'help':'Group name in netcdf file',  'default':''},
        {'key':'trange',  'required':True,  'default':[], 'help':'Which times to average over', }, 

        # --- Option inputs ---
        {'key':'num_bins',  'required':False,  'default':50, 'help':'How many wavenumber bins to use for spectra', }, 
        {'key':'type',    'required':False,  'default':['energy','horiz','vertical','kol'], 'help':"What type of spectra to compute: 'energy', 'horiz', 'vertical', 'kol'. Default is all.", }, 
        {'key':'varnames',  'required':False,  'default':['velocityx', 'velocityy', 'velocityz'],
         'help':'Variables to extract from the netcdf file',},        
        {'key':'iplanes',       'required':False,  'default':None,'help':'i-index of planes to postprocess', },
        {'key':'csvfile',  'required':False,  'help':'Filename to save spectra to',  'default':''},

        {'key':'xaxis',    'required':False,  'default':'x','help':'Which axis to use on the abscissa', },
        {'key':'yaxis',    'required':False,  'default':'y','help':'Which axis to use on the ordinate', },

        {'key':'C_kol',    'required':False,  'default':1.5,'help':'Kolmogorov constant to use for theoretical spectra', },
        {'key':'diss_rate',    'required':False,  'default':1.0,'help':'Dissipation rate for theoretical spectra', },
        {'key':'remove_endpoint_x','required':False,  'default':False,'help':'Remove one endpoint in x before FFT if periodic signal is sampled twice at endpoints.', },
        {'key':'remove_endpoint_y','required':False,  'default':False,'help':'Remove one endpoint in y before FFT if periodic signal is sampled twice at endpoints.', },
    ]
    actionlist = {}                    # Dictionary for holding sub-actions
    example = """
    wavenumber_spectra:
        name: Spectra_027
        ncfile: XYdomain_027_30000.nc
        group: Farm_XYdomain027
        csvfile: E_spectra_Z027.csv
        trange:  [15000, 20000]
        iplanes: 0 
    """
    # --- Stuff required for main task ---
    def __init__(self, inputs, verbose=False):
        self.yamldictlist = []
        inputlist = inputs if isinstance(inputs, list) else [inputs]
        for indict in inputlist:
            self.yamldictlist.append(mergedicts(indict, self.inputdefs))
        print('Initialized '+self.name)
        return

    def compute_2D_spectra(self,E,kx_vec,ky_vec,num_bins):

        #generate list of 2D waveumber magnitudes
        k_mags = np.sqrt(kx_vec ** 2 + ky_vec ** 2)
        k_min = 0
        k_max = np.max(k_mags)

        #divide into annular rings based on num_bins parameter
        bins = np.linspace(k_min, k_max, num_bins + 1)  

        E_spec     = np.zeros(num_bins) #hold 2D wavenumber spectra E(|k|)
        bin_counts = np.zeros(num_bins) #number of wavenumbers per bin

        for kxiter , kx in enumerate(kx_vec):
            for kyiter , ky in enumerate(ky_vec):
                kmag = np.sqrt(kx**2 + ky**2)
                if kmag > 0: 
                    bin_index = np.where((bins[:-1] <= kmag) & (kmag < bins[1:]))[0]
                    E_spec[bin_index] += E[kxiter,kyiter] #sum energy in each wavenumber bin 
                    bin_counts[bin_index] += 1.0 

        kmag_centers = 0.5 * (bins[:-1] + bins[1:]) #determine wavenumber centers of each bin
        bin_area =  np.pi * bins[1:]**2 - np.pi * bins[:-1]**2 #total 2D area of annular region

        #E(|k|) = \circ_int 0.5 * <u_i(k) u*_i(k)>dS \approx \sum_{|k| \in bin} 0.5 * <u_i(k) u*_i(k)> * A/count
        E_spec = E_spec * bin_area / bin_counts 
        return kmag_centers,E_spec
    
    def execute(self, verbose=False):
        # Do any task-related here
        if verbose: print('Running '+self.name)
        for iplane, plane in enumerate(self.yamldictlist):
            # Run any necessary stuff for this task
            ncfile   = plane['ncfile']
            group    = plane['group']
            csvfile  = plane['csvfile']
            trange   = plane['trange']
            varnames = plane['varnames']
            iplanes  = plane['iplanes']
            xaxis    = plane['xaxis']
            yaxis    = plane['yaxis']
            type_spec = plane['type']
            num_bins = plane['num_bins']
            C_kol = plane['C_kol']
            diss_rate = plane['diss_rate']
            remove_endpoint_x = plane['remove_endpoint_x']
            remove_endpoint_y = plane['remove_endpoint_y']
            if not isinstance(type_spec, list): type_spec = [type_spec,]

            # Read in the cartesian data
            udata_cart,heights,x,y,times,iplanes = read_cart_data(ncfile,varnames,group,trange,iplanes,xaxis,yaxis)
            #udata[iplane] = np.zeros((len(t),len(x),len(y),3))

            E_spec = {}
            for iplane in iplanes:
                #subtract temporal mean of velocity components
                udata_mean = np.mean(udata_cart[iplane],axis=0,keepdims=True)
                udata_fluc = udata_cart[iplane] - udata_mean

                if remove_endpoint_x:
                    udata_fluc = udata_fluc[:,:-1,:,:]
                    x = x[:-1]

                if remove_endpoint_y:
                    udata_fluc = udata_fluc[:,:,:-1,:]
                    y = y[:-1]


                #Compute wavenumbers
                Nx = len(x)
                Ny = len(y)
                dx = x[1] - x[0] 
                dy = y[1] - y[0] 

                kx = np.fft.fftfreq(Nx,dx) * 2 * np.pi
                ky = np.fft.fftfreq(Ny,dy) * 2 * np.pi

                #Fourier transform in space
                uhat = np.fft.fft(np.fft.fft(udata_fluc[:,:,:,0],axis=1),axis=2)
                vhat = np.fft.fft(np.fft.fft(udata_fluc[:,:,:,1],axis=1),axis=2)
                what = np.fft.fft(np.fft.fft(udata_fluc[:,:,:,2],axis=1),axis=2)

                # Compute fourier transform of two-point correlation tensor 
                Phi_11 = np.mean((np.abs(uhat)**2),axis=0)
                Phi_22 = np.mean((np.abs(vhat)**2),axis=0)
                Phi_33 = np.mean((np.abs(what)**2),axis=0)

                # For 2D wavenumbers, E(|k|) ~ L^3/T^2, \Phi(k) ~ L^4/T^2 
                # \Phi is energy density in 2D wavesapce. 
                # Divide by cell area in wavespace to get L^2 factor
                dkA = (kx[1]-kx[0]) * (ky[1]-ky[0]) 
                Phi_11 /= dkA
                Phi_22 /= dkA
                Phi_33 /= dkA


                for spec_type in type_spec:
                    if spec_type == 'energy':
                        E = 0.5*(Phi_11 + Phi_22 + Phi_33)
                    elif spec_type == 'horiz':
                        E = 0.5*(Phi_11 + Phi_22)
                    elif spec_type == 'vertical':
                        E = 0.5*(Phi_33)
                    elif spec_type == 'kol':
                        E = np.zeros_like(Phi_11) #Kolmogorov Spectrum
                        for kxiter , kx_val in enumerate(kx):
                            for kyiter , ky_val in enumerate(ky):
                                kmag = np.sqrt(kx_val ** 2 + ky_val **2)
                                if kmag > 0:
                                    E[kxiter,kyiter]  = 1.0/(2 * np.pi * kmag) * C_kol * diss_rate ** (2/3) * kmag ** (-5/3) # See Pope 6.193 but derive for 2D
                    else:
                        print("Error: Unknown spectra type. Exiting")
                        sys.exit()
                    kmag_centers, E_spec[spec_type] = self.compute_2D_spectra(E,kx,ky,num_bins)

            # Save data to csv file
            dfcsv = pd.DataFrame()
            dfcsv['kmag']  = kmag_centers
            for spec_type in type_spec:
                dfcsv[spec_type] = E_spec[spec_type]
            dfcsv.to_csv(csvfile, index=False, sep=',')
            
            # Do any sub-actions required for this task
            for a in self.actionlist:
                action = self.actionlist[a]
                # Check to make sure required actions are there
                if action.required and (action.actionname not in yamldict.keys()):
                    # This is a problem, stop things
                    raise ValueError('Required action %s not present'%action.actionname)
                if action.actionname in self.yamldictlist[iplane].keys():
                    actionitem = action(self, self.yamldictlist[iplane][action.actionname])
                    actionitem.execute()
        return
