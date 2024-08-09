# Get the location where this script is being run
import sys, os
scriptpath = os.path.dirname(os.path.realpath(__file__))
basepath   = os.path.dirname(scriptpath)
# Add any possible locations of amr-wind-frontend here
amrwindfedirs = ['../',
                 basepath]
for x in amrwindfedirs: sys.path.insert(1, x)

from postproengine import registerplugin, mergedicts, registeraction
from postproengine import compute_axis1axis2_coords, get_mapping_xyz_to_axis1axis2
import postproamrwindsample_xarray as ppsamplexr
import postproamrwindsample as ppsample
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import importlib.util
import sys

# Add check for samwich package
name = 'samwich'
spec = importlib.util.find_spec(name)
if spec is not None:
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    from samwich.dataloaders import PlanarData
    from samwich.waketrackers import track,Plotter
    usesamwich=True
else:
    usesamwich=False

"""
Plugin for computing wake meandering statistics

See README.md for details on the structure of classes here
"""

def get_wake_centers(u,YY,ZZ,method='ConstantArea',weighted_center=True,args=None):
    datadict = {}
    datadict['y'] = np.copy(YY[:,:].T)
    y_grid_center = (datadict['y'][-1,0] + datadict['y'][0,0])/2.0 
    datadict['z'] = np.copy(ZZ[:,:].T)
    datadict['u'] = u[:,:,:,0]

    wakedata = PlanarData(datadict)
    wake = track(wakedata.sliceI(),method=method,verbose=True)
    wake.remove_shear(method='fringe',Navg=u.shape[0])
    wake.wake_tracked = False
    if method=='Gaussian':
        yc,zc = wake.find_centers(umin=None,sigma=args)
    if method=='ConstantArea':
        yc,zc = wake.find_centers(args,weighted_center=weighted_center)
    if method=='ConstantFlux':
        print('ref thrust (momentum deficit) is',args,'N')
        flux = lambda u,u_w: -u * u_w  # function arguments correspond to field_names
        yc,zc = wake.find_centers(args,flux_function=flux,field_names=('u','u_tot'),weighted_center=weighted_center)
    return wake, yc + y_grid_center ,zc


@registerplugin
class postpro_wakemeander():
    """
    Wake meandering postprocess 
    """
    # Name of task (this is same as the name in the yaml)
    name      = "wake_meander"
    # Description of task
    blurb     = "Compute wake meandering statistics"
    inputdefs = [
        # -- Execute parameters ----
        {'key':'iplane',       'required':False,  'default':[0,],
         'help':'i-index of planes to postprocess', },
        {'key':'name',     'required':True,  'default':'',
         'help':'An arbitrary name',},
        {'key':'ncfile',   'required':True,  'default':'',
        'help':'NetCDF sampling files of cross-flow planes', },
        {'key':'group',   'required':False,  'default':None,
         'help':'Which group to pull from netcdf file', },
        {'key':'varnames',  'required':False,  'default':['velocityx', 'velocityy', 'velocityz'],
         'help':'Variables to extract from the netcdf file',},        
        {'key':'trange',    'required':True,  'default':[0,1],
            'help':'Which times to postprocess', }, 
        {'key':'yhub',    'required':False,  'default':None,
            'help':'Lateral hub-height center', }, 
        {'key':'zhub',    'required':False,  'default':None,
            'help':'Vertical hub-height', }, 
        {'key':'method',    'required':True,  'default':'ConstantArea',
            'help':'Method for computing wake center. Options include: ConstantArea, ConstantFlux, Gaussian', }, 
        {'key':'diam',    'required':False,  'default':0,
            'help':'Rotor diameter', }, 
        {'key':'Uinf',    'required':False,  'default':None,
            'help':'U velocity for approximating rotor thrust for ConstantFlux method', }, 
        {'key':'Ct',    'required':False,  'default':0,
            'help':'Thrust coefficient for approximating rotor thrust for ConstantFlux method', }, 
        {'key':'rotthrust',    'required':False,  'default':None,
            'help':'Target rotor thrust for ConstantFlux method.', }, 
        {'key':'savefile', 'required':False,  'default':"",
            'help':'File to save timeseries of wake centers, per iplane', }, 
        {'key':'output_dir',  'required':False,  'default':'./','help':'Directory to save results'},
        {'key':'weighted_center',  'required':False,  'default':True,'help':'If True, calculate the velocity-deficit-weighted "center of mass"; if False, calculate the geometric center of the wake.'},
        {'key':'axis_rotation',  'required':False,  'default':0,
         'help':'Degrees to rotate axis for velocitya1,a2,a3 transformation',},                
        {'key':'xaxis',    'required':False,  'default':'y',
        'help':'Which axis to use on the abscissa', },
        {'key':'yaxis',    'required':False,  'default':'z',
        'help':'Which axis to use on the ordinate', },
    ]
    example = """
    wake_meander:
        iplane:
            - 5
            - 6
        name: Wake YZ plane
        ncfile: YZcoarse_103125.nc
        trange: [27950,28450]
        group: T0_YZdomain
        yhub: 1000
        zhub: 150
        method: ConstantArea
        #method: ConstantFlux
        #method: Gaussian
        diam: 240
        savefile: wake_center_{iplane}.csv
        output_dir: ./wake_meandering/
        Uinf: 9.0
        Ct: 1.00

        plot:
            xlabel: 'Y [m]'
            ylabel: 'Z [m]'
            iter: 0
            savefile: wake_center_{iplane}.png

        statistics:
            savefile: wake_stats_{iplane}.csv
            mean: True
            std: True
            anisotropy: True
            compute_uv: True
            pklfile: pcs.pkl
    """
    actionlist = {}                    # Dictionary for holding sub-actions

    # --- Stuff required for main task ---
    def __init__(self, inputs, verbose=False):
        self.yamldictlist = []
        inputlist = inputs if isinstance(inputs, list) else [inputs]
        for indict in inputlist:
            self.yamldictlist.append(mergedicts(indict, self.inputdefs))
        if verbose: print('Initialized '+self.name)
        return

    def extract_1d_from_meshgrid(self,Z):
        # Check along axis 0
        unique_rows = np.unique(Z, axis=0)
        if unique_rows.shape[0] == 1:
            return unique_rows[0],1

        # Check along axis 1
        unique_cols = np.unique(Z, axis=1)
        if unique_cols.shape[1] == 1:
            return unique_cols[:, 0],0
    
    def execute(self, verbose=False):
        if verbose: print('Running '+self.name)
        # Loop through and create plots
        for entryiter , entry in enumerate(self.yamldictlist):
            iplanes  = entry['iplane']
            trange   = entry['trange']
            ncfile   = entry['ncfile']
            group    = entry['group']
            self.yhub     = entry['yhub']
            self.zhub     = entry['zhub']
            diam     = entry['diam']
            weighted_center = entry['weighted_center']
            Ct       = entry['Ct']
            Uinf     = entry['Uinf']
            RotThrust= entry['rotthrust']
            Uinf     = entry['Uinf']
            method   = entry['method']
            self.xaxis    = entry['xaxis']
            self.yaxis    = entry['yaxis']
            savefile = entry['savefile']
            self.varnames = entry['varnames']
            self.output_dir =  entry['output_dir']
            self.axis_rotation = entry['axis_rotation']

            if not isinstance(iplanes, list): iplanes = [iplanes,]
            #Get all times if not specified 
            filelist = []
            for fileiter in range(0,len(ncfile)):
                filelist.append(ncfile[fileiter])

            udata = {}
            xc = {}
            self.db = ppsamplexr.getPlaneXR(ncfile,[0,1],self.varnames,groupname=group,verbose=0,includeattr=True,gettimes=True,timerange=trange)

            # Convert to native axis1/axis2 coordinates if necessary
            self.natural_axes = False
            # can we move this to __init__ so that we maintain plane locations in a3 direction?
            if ('a1' in [self.xaxis, self.yaxis]) or ('a2' in [self.xaxis, self.yaxis]) or ('a3' in [self.xaxis, self.yaxis]):
                self.natural_axes = True
                compute_axis1axis2_coords(self.db,rot=0)
                R = get_mapping_xyz_to_axis1axis2(self.db['axis1'],self.db['axis2'],self.db['axis3'],rot=0)
                origin = self.db['origin']
                origina1a2a3 = R@self.db['origin']
                offsets = self.db['offsets']
                offsets = [offsets] if (not isinstance(offsets, list)) and (not isinstance(offsets,np.ndarray)) else offsets

            natural_velocities=False
            if ('velocitya1' in self.varnames) or ('velocitya2' in self.varnames) or ('velocitya3' in self.varnames):
                natural_velocities=True

            if natural_velocities and not self.natural_axes:
                print("Error: Must xaxis and yaxis in terms of a1,a2,a3 for natural velocity components")
                sys.exit()

            t = np.asarray(np.array(self.db['times']).data)
            for iplane in iplanes:
                YY = np.array(self.db[self.xaxis])
                ZZ = np.array(self.db[self.yaxis])
                y,axisy = self.extract_1d_from_meshgrid(YY[iplane,:,:])
                z,axisz = self.extract_1d_from_meshgrid(ZZ[iplane,:,:])

                if not self.natural_axes:
                    xc[iplane] = self.db['x'][iplane,0,0]
                else:
                    xc[iplane] = origina1a2a3[-1] + offsets[iplane]

                udata[iplane] = np.zeros((len(t),len(y),len(z),3))

                permutation = [0,axisy+1,axisz+1]
                for i,tstep in enumerate(self.db['timesteps']):
                    if natural_velocities:
                        ordered_data = np.transpose(np.array(self.db['velocitya3'][tstep]),permutation)
                        udata[iplane][i,:,:,0] = ordered_data[iplane,:,:]

                        ordered_data = np.transpose(np.array(self.db['velocity'+self.xaxis][tstep]),permutation)
                        udata[iplane][i,:,:,1] = ordered_data[iplane,:,:]

                        ordered_data = np.transpose(np.array(self.db['velocity'+self.yaxis][tstep]),permutation)
                        udata[iplane][i,:,:,2] = ordered_data[iplane,:,:]
                    else:
                        ordered_data = np.transpose(np.array(self.db['velocityx'][tstep]),permutation)
                        udata[iplane][i,:,:,0] = ordered_data[iplane,:,:]

                        ordered_data = np.transpose(np.array(self.db['velocityy'][tstep]),permutation)
                        udata[iplane][i,:,:,1] = ordered_data[iplane,:,:]

                        ordered_data = np.transpose(np.array(self.db['velocityz'][tstep]),permutation)
                        udata[iplane][i,:,:,2] = ordered_data[iplane,:,:]

            YY , ZZ = np.meshgrid(y,z)
            arg=None
            if method == 'ConstantArea':
                arg = np.pi * (diam/2.0)**2
            elif method == 'ConstantFlux':
                if RotThrust != None:
                    arg = RotThrust
                else:
                    arg = 0.5 * np.pi * (diam/2.0)**2 * Ct * Uinf ** 2
            elif method == 'Gaussian':
                arg = diam/2.0
            else: 
                print("Error: Method " + str(method) + " uknown. Exiting.")
                sys.exit()

            for iplane in iplanes:
                self.dfcenters = pd.DataFrame()
                self.iplane = iplane
                self.dfcenters['t'] = t
                if self.natural_axes:
                    self.dfcenters['a3c'] = xc[iplane]
                else:
                    self.dfcenters['xc'] = xc[iplane]

                if not usesamwich:
                    print("Error: Samwich package required to compute wake centers")
                    sys.exit()

                self.wake, self.dfcenters[self.xaxis+'c'], self.dfcenters[self.yaxis+'c'] = get_wake_centers(udata[iplane],YY,ZZ,method=method,weighted_center=weighted_center,args=arg)

                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)

                if len(savefile)>0:
                    savefname = savefile.format(iplane=self.iplane)
                    savefname = os.path.join(self.output_dir, savefname)
                    self.dfcenters.to_csv(savefname,index=False,sep=',')

                # Do any sub-actions required for this task
                for a in self.actionlist:
                    action = self.actionlist[a]
                    # Check to make sure required actions are there
                    if action.required and (action.actionname not in self.yamldictlist[entryiter].keys()):
                        # This is a problem, stop things
                        raise ValueError('Required action %s not present'%action.actionname)
                    if action.actionname in self.yamldictlist[entryiter].keys():
                        actionitem = action(self, self.yamldictlist[entryiter][action.actionname])
                        actionitem.execute()
        return 

    @registeraction(actionlist)
    class plot():
        actionname = 'plot'
        blurb      = 'Plot contour with wake boundary and center '
        required   = False
        actiondefs = [
        {'key':'dpi',       'required':False,  'default':125,
         'help':'Figure resolution', },
        {'key':'figsize',   'required':False,  'default':[12,8],
         'help':'Figure size (inches)', },
        {'key':'savefile',  'required':False,  'default':'',
         'help':'Filename to save the picture', },
        {'key':'xlabel',    'required':False,  'default':'X [m]',
         'help':'Label on the X-axis', },
        {'key':'ylabel',    'required':False,  'default':'Y [m]',
         'help':'Label on the Y-axis', },
        {'key':'title',     'required':False,  'default':'',
         'help':'Title of the plot',},
        {'key':'cmin',   'required':False,  'default':None,
         'help':'Minimum contour level',},
        {'key':'cmax',   'required':False,  'default':None,
         'help':'Maximum contour level',},
        {'key':'iter',   'required':False,  'default':0,
         'help':'Iteration in time to plot',},
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent
            print('Initialized '+self.actionname+' inside '+parent.name)
            return

        def execute(self):
            print('Executing '+self.actionname)
            figsize  = self.actiondict['figsize']
            dpi      = self.actiondict['dpi']
            savefile = self.actiondict['savefile']
            xlabel   = self.actiondict['xlabel']
            ylabel   = self.actiondict['ylabel']
            cmin     = self.actiondict['cmin']
            cmax     = self.actiondict['cmax']
            title    = self.actiondict['title']
            itime    = self.actiondict['iter']

            fig, ax = plt.subplots(1,1,figsize=(figsize[0],figsize[1]), dpi=dpi)
            self.parent.wake.clear_plot()

            if not cmin == None and not cmax == None:
                self.parent.wake.plot_contour(vmin=cmin,vmax=cmax,cmap='coolwarm',itime=itime,outline=True,markercolor='k')
            else:
                self.parent.wake.plot_contour(cmap='coolwarm',itime=itime,outline=True,markercolor='k')
                #plotter = Plotter(y=self.parent.wake.y,z=self.parent.wake.z,u=self.parent.wake.u,figsize=figsize,dpi=dpi,cmap='coolwarm')
                #plotter.add('test',self.parent.wake)
                #plotter.plot()
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(eval("f'{}'".format(title)))

            if len(savefile)>0:
                savefname = savefile.format(iplane=self.parent.iplane)
                savefname = os.path.join(self.parent.output_dir, savefname)
                plt.savefig(savefname)
            return

    @registeraction(actionlist)
    class statistics():
        actionname = 'statistics'
        blurb      = 'Compute wake meandering statistics'
        required   = False
        actiondefs = [
        {'key':'savefile',  'required':False,  'default':'',
         'help':'Filename to save statistics', },
        {'key':'mean',  'required':False,  'default':True,
         'help':'Boolean to compute mean wake center', },
        {'key':'std',  'required':False,  'default':True,
         'help':'Boolean to compute std wake center', },
        {'key':'anisotropy',  'required':False,  'default':False,
         'help':'Boolean to compute wake anisotropy metric', },
        {'key':'compute_uv',  'required':False,  'default':False,
         'help':'Boolean to compute eigenvectors of PCA', },
        {'key':'pklfile',  'required':False,  'default':"",
         'help':'File to save eigenvectors of PCA', },
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent
            print('Initialized '+self.actionname+' inside '+parent.name)
            return

        def execute(self):
            print('Executing '+self.actionname)
            savefile = self.actiondict['savefile']
            computeMean = self.actiondict['mean']
            computeStd = self.actiondict['std']
            computeAniso = self.actiondict['anisotropy']
            compute_uv = self.actiondict['compute_uv']
            pklfile  = self.actiondict['pklfile']

            wake_meandering_stats = {}
            if self.parent.yhub != None and self.parent.zhub !=None:
                calc_hub_dist = True
                dist_from_hub = np.sqrt( (self.parent.dfcenters['zc']-self.parent.zhub)**2 + (self.parent.dfcenters['yc']-self.parent.yhub)**2 )
            else:
                calc_hub_dist = False


            if computeMean:
                if self.parent.natural_axes:
                    wake_meandering_stats['a3c_mean'] = np.mean(self.parent.dfcenters['a3c'])
                else:
                    wake_meandering_stats['xc_mean'] = np.mean(self.parent.dfcenters['xc'])

                wake_meandering_stats[self.parent.xaxis+'c_mean'] = np.mean(self.parent.dfcenters[self.parent.xaxis+'c'])
                wake_meandering_stats[self.parent.yaxis+'c_mean'] = np.mean(self.parent.dfcenters[self.parent.yaxis+'c'])
                if calc_hub_dist:
                    wake_meandering_stats['hub_distance_mean'] = np.mean(dist_from_hub)

            if computeStd:
                if self.parent.natural_axes:
                    wake_meandering_stats['a3c_std'] = np.std(self.parent.dfcenters['a3c'])
                else:
                    wake_meandering_stats['xc_std'] = np.std(self.parent.dfcenters['xc'])
                wake_meandering_stats[self.parent.xaxis+'c_std'] = np.std(self.parent.dfcenters[self.parent.xaxis+'c'])
                wake_meandering_stats[self.parent.yaxis+'c_std'] = np.std(self.parent.dfcenters[self.parent.yaxis+'c'])
                if calc_hub_dist:
                    wake_meandering_stats['hub_distance_std'] = np.std(dist_from_hub)

            if computeAniso:
                try:
                    numSamples = self.parent.wake.u.shape[0]
                    eig_ratio = np.zeros(numSamples)
                    if compute_uv:
                        pcs = np.zeros((numSamples,2,2))
                    for sample in range(numSamples):
                        wakeu = self.parent.wake.u[sample,:,:]
                        wakey = self.parent.wake.y
                        wakez = self.parent.wake.z
                        mask = wakeu < self.parent.wake.Clevels[sample]
                        masku = wakeu * mask
                        masky = wakey * mask
                        maskz = wakez * mask
                        pca_data = np.zeros((len(maskz[maskz != 0]),2))
                        pca_data[:,0] = masky[maskz != 0]
                        pca_data[:,1] = maskz[maskz != 0] - np.mean(maskz[maskz != 0])
                        if compute_uv:
                            U, S, Vt = np.linalg.svd(pca_data,full_matrices=False)
                            pc = U
                            pcs[sample,:,:] = np.dot(pc.T,pca_data)
                        else:
                            S = np.linalg.svd(pca_data,full_matrices=False,compute_uv=False)
                        eig_ratio[sample]= S[1]/S[0]
                    wake_meandering_stats['aniso_mean'] = np.mean(eig_ratio)
                    wake_meandering_stats['aniso_std']  = np.std(eig_ratio)
                    if compute_uv:
                        with open(pklfile, 'wb') as file:
                            pickle.dump(pcs, file)
                except:
                    print("Error computing PCA for ansitropy metric")

            wake_meandering_stats = pd.DataFrame(wake_meandering_stats,index=[0])
            print(wake_meandering_stats)
            if len(savefile)>0:
                savefname = savefile.format(iplane=self.parent.iplane)
                savefname = os.path.join(self.parent.output_dir, savefname)
                wake_meandering_stats.to_csv(savefname,index=False,sep=',')
            return
