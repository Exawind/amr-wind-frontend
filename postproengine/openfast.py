# Get the location where this script is being run
import sys, os
scriptpath = os.path.dirname(os.path.realpath(__file__))
basepath   = os.path.dirname(scriptpath)
# Add any possible locations of amr-wind-frontend here
amrwindfedirs = ['../',
                 basepath]
for x in amrwindfedirs: sys.path.insert(1, x)

from postproengine import registerplugin, mergedicts, registeraction
import postproamrwindsample_xarray as ppsamplexr
import postproamrwindsample as ppsample
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import fatpack
import scipy
import re
import linecache

"""
Plugin for postprocessing openfast data

See README.md for details on the structure of classes here
"""

def approximate_awc_time_interval(AWC,rotspeed,time,windspeed,diam,st,t1,t2):
    omega_e = 2*np.pi * float(st) * float(windspeed) / float(diam)
    time_interval = t2-t1
    t2_history = []
    t2_temp = t2
    AWC = AWC.lower()
    for i in range(10): # iterate 10 times to converge
        mask = (time <= t2_temp) & (time  >= t1)
        RotSpeed_Window = rotspeed[mask]
        meanRPM = np.mean(RotSpeed_Window)
        Omega = meanRPM*2*np.pi/60

        if 'baseline' in AWC:
            return t2_temp
        elif 'side' in AWC or 'up' in AWC or 'cl' in AWC or 'n1p1m' in AWC:
            period = (2*omega_e / (2*np.pi))**(-1)
        elif 'pulse' in AWC or 'n0' in AWC:
            period = (omega_e / (2*np.pi))**(-1)
        elif 'helix' in AWC or 'n1m' in AWC:
            period = ( (Omega + omega_e) / (2*np.pi))**(-1)
        elif 'cc_helix' in AWC or 'n1p' in AWC:
            period = ( (Omega - omega_e) / (2*np.pi))**(-1)
        else:
            return t2_temp

        dt = time[1]-time[0]
        t2_history = np.append(t2_history, t2_temp)
        t2_temp = t1 + np.floor(time_interval/ period) * period
        t2_temp = np.floor(t2_temp/dt)*dt

    return t2_temp

def makeSecBladeDF(csvfile, rpts, bladekeysdict):
    """
    Make a dictionary with blade sectional loading quantities
    """
    df=pd.read_csv(csvfile, comment='#',)
    bladedf = {}
    bladedf['rpts'] = rpts
    for k, bladekeys in bladekeysdict.items():
        alphadat = [float(df[k][0]) for k in bladekeys]
        bladedf[k] = alphadat
    return bladedf

@registerplugin
class postpro_openfast():
    """
    Postprocessing for openfast data
    """
    # Name of task (this is same as the name in the yaml)
    name      = "openfast"
    # Description of task
    blurb     = "Postprocessing of openfast variables"
    inputdefs = [
        # -- Execute parameters ----
        {'key':'name',     'required':True,  'default':'',
         'help':'An arbitrary name',},
        {'key':'filename',   'required':True,  'default':'',
        'help':'Openfast output file', },
        {'key':'vars',  'required':True,  'default':['Time',],
         'help':'Variables to extract from the openfast file',},        
        {'key':'extension',  'required':False,  'default':'.csv','help':'The extension to use for the csv files'},
        {'key':'output_dir',  'required':False,  'default':'./','help':'Directory to save results'},
        {'key':'useregex',  'required':False,  'default':False,
         'help':'Use regex expansion in vars list'},

    ]
    actionlist = {}                    # Dictionary for holding sub-actions
    notes = """

Currently the `openfast` executor is only capable of acting on text
output from OpenFAST, corresponding to OutFileFmt=1 in the fst input
file.
    
The `useregex` option allows multiple variables in the `vars` list to
be specified through a regex [regular
expression](https://en.wikipedia.org/wiki/Regular_expression).  For
instance, `^Rot` will match any variable that starts with `Rot`, such
as `RotSpeed` or `RotTorq`.
    """
    example = """
```yaml
openfast:
- name: NREL5MW_SECLOADS
  filename: RUNDIR/T0_NREL5MW_v402_ROSCO/openfast-cpp/5MW_Land_DLL_WTurb_cpp/5MW_Land_DLL_WTurb_cpp.out
  vars: 
  - Time
  - BldPitch1
  - BldPitch2
  - BldPitch3
  - GenPwr
  - '^Rot'
  - 'AB1N...Alpha'
  - 'AB1N...Phi'
  - 'AB1N...Cl'
  - 'AB1N...Cd'
  - 'AB1N...Fx'
  - 'AB1N...Fy'
  - RotSpeed
  output_dir: RESULTSDIR
  useregex: True
  csv:  # Store information to CSV files
    individual_files: False
  operate:
    operations: 
    - mean
    trange: [300, 900]
    pitch_travel: True
  spanwiseloading:
    bladefile: RUNDIR/T0_NREL5MW_v402_ROSCO/openfast/5MW_Baseline/NRELOffshrBsline5MW_AeroDyn_blade.dat
    bladevars: [Alpha, Phi, Cl, Cd, Fx, Fy]
    meancsvfile: RESULTSDIR/NREL5MW_SECLOADS_mean.csv
    savecsvfile: RESULTSDIR/NREL5MW_SECLOADS_mean_rpts.csv
    radialstations: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    prefix: AB1N
```
"""
    # --- Stuff required for main task ---
    def __init__(self, inputs, verbose=False):
        self.yamldictlist = []
        inputlist = inputs if isinstance(inputs, list) else [inputs]
        for indict in inputlist:
            self.yamldictlist.append(mergedicts(indict, self.inputdefs))
        if verbose: print('Initialized '+self.name)
        return
    
    def execute(self, verbose=False):
        if verbose: print('Running '+self.name)
        # Loop through and create plots
        for entryiter , entry in enumerate(self.yamldictlist):
            names = entry['name']
            if type(entry['name']) is str:
                names = []
                names.append(entry['name'])

            filenames = entry['filename']
            if type(entry['filename']) is str:
                filenames = []
                filenames.append(entry['filename'])
            if entry['useregex']:
                allvars=linecache.getline(filenames[0],7).split()
                searchstr = entry['vars']
                validvars = []
                for s in searchstr:
                    validvars.append([x for x in allvars if bool(re.search(s,x))])
                # Flatten and make list only have unique entries
                varnames = list(dict.fromkeys(sum(validvars, [])))
            else:
                varnames   = list(entry['vars'])
            self.extension  = entry['extension']
            self.output_dir =  entry['output_dir']

            for fileiter , file  in enumerate(filenames):
                self.name  = names[fileiter]
                print(self.name, file)
                self.df = pd.read_csv(file,sep='\s+',skiprows=(0,1,2,3,4,5,7), comment='#', usecols=lambda col: any(keyword in col for keyword in varnames))

                #drop any duplicate times due to restarts. 
                self.df = self.df.drop_duplicates(subset='Time', keep='last')
                #sort values in time 
                self.df = self.df.sort_values(by='Time')
                #drop any times withmissing or NaN columns
                self.df = self.df.dropna(axis=0)

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


    # --- Inner classes for action list ---
    @registeraction(actionlist)
    class write_to_csv():
        actionname = 'csv'
        blurb      = 'Writes the openfast variables to a csv file'
        required   = False
        actiondefs = [
            {'key':'individual_files', 'required':False,  'help':'Write each variable to a separate csv file',  'default':True},
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent

            print('Initialized '+self.actionname+' inside '+parent.name)
            return

        def execute(self):
            print('Executing '+self.actionname)
            individual_files = self.actiondict['individual_files']
            extension = self.parent.extension
            output_dir=  self.parent.output_dir
            prefix = self.parent.name

            # Go to the run directory
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            if not extension == None:
                csvfile = os.path.join(output_dir, prefix+extension)
            else:
                csvfile = os.path.join(output_dir, prefix)

            #print("Writing global csv file: ",csvfile)
            self.parent.df.to_csv(csvfile, index=False,float_format='%.15f')

            if individual_files: 

                # Iterate over the columns excluding the first one
                for column in self.parent.df.columns:
                    #print("Writing data entry: ",column)

                    # Create a new DataFrame with the first column and the current column
                    subset_df = self.parent.df[['Time', column]]

                    # Define the filename
                    filename = prefix + f'_{column}'
                    if not extension == None:
                        filename += extension

                    # Write the subset DataFrame to a CSV file
                    subset_df.to_csv(os.path.join(output_dir, filename), index=False)

            return 

    @registeraction(actionlist)
    class operate_and_write():
        actionname = 'operate'
        blurb      = 'Operates on the openfast data and saves to a csv file'
        required   = False
        actiondefs = [
            {'key':'operations',  'required':True,  'default':['mean','std','DEL','pwelch','running_avg'],'help':'List of operations to perform (mean,std,DEL,pwelch,running_avg)'},
            {'key':'trange',    'required':False,  'default':[],'help':'Times to apply operation over'}, 
            {'key':'awc_period', 'required':False,  'default':False,'help':'Average over equal periods for AWC forcing'},
            {'key':'awc',  'required':False,  'default':'baseline','help':'AWC case name [baseline,n0,n1p,n1m,n1p1m_cl00,n1p1m_cl90]'},
            {'key':'St',  'required':False,  'default':0.3,'help':'Forcing Strouhal number'},
            {'key':'diam',  'required':False,  'default':0,'help':'Turbine diameter'},
            {'key':'U_st',  'required':False,  'default':0,'help':'Wind speed to define Strouhal number'},
            {'key':'nperseg',  'required':False,  'default':4096, 'help':'Number of samples per segment used in pwelch'},
            {'key':'pitch_travel',    'required':False,  'default':False,'help':'Option to compute the pitch travel'}, 
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent

            print('Initialized '+self.actionname+' inside '+parent.name)
            return

        def execute(self):
            print('Executing '+self.actionname)
            operations =  self.actiondict['operations']
            extension = self.parent.extension
            trange     =  self.actiondict['trange']
            awc_period =  self.actiondict['awc_period']
            pitch_travel = self.actiondict['pitch_travel']
            output_dir=  self.parent.output_dir
            prefix = self.parent.name

            if trange==[]:
                trange.append(self.parent.df['Time'].iloc[0])
                trange.append(self.parent.df['Time'].iloc[-1])

            
            if awc_period:
                St   =  self.actiondict['St']
                diam =  self.actiondict['diam']
                awc  =  self.actiondict['awc']
                windspeed =  self.actiondict['U_st']
                rotspeed = self.parent.df['RotSpeed']
                time = self.parent.df['Time']
                trange[-1] = approximate_awc_time_interval(awc,rotspeed,time,windspeed,diam,St,trange[0],trange[-1])
                print('--> awc trange: ',trange)

            total_time = trange[-1]-trange[0]
            mask = (self.parent.df['Time'] >= trange[0]) & (self.parent.df['Time'] <= trange[-1])
            filtered_df = self.parent.df[mask]

            if 'mean' in operations:
                csvfile = os.path.join(output_dir, prefix + "_mean" + extension)
                mean_df = pd.DataFrame(columns=self.parent.df.columns)
                mean_df.loc[0] = filtered_df.mean()
                mean_df.to_csv(csvfile, index=False,float_format='%.15f')

            if 'std' in operations:
                csvfile = os.path.join(output_dir, prefix + "_std" + extension)
                std_df = pd.DataFrame(columns=self.parent.df.columns)
                std_df.loc[0] = filtered_df.std()
                std_df.to_csv(csvfile, index=False,float_format='%.15f')

            if 'DEL' in operations:
                DEL_df = pd.DataFrame(columns=self.parent.df.columns)
                DEL_df.loc[0]=0
                for column in filtered_df:
                    if not column  == 'Time':
                        try:
                            binNum = 100
                            m=10
                            ranges = fatpack.find_rainflow_ranges(np.asarray(filtered_df[column][mask].values))
                            Nrf, Srf = fatpack.find_range_count(ranges,binNum)
                            DELs = Srf**m * Nrf / total_time
                            DEL = DELs.sum() ** (1/m)
                            DEL_df[column]=DEL
                        except:
                            print("---> Warning, cannot compute DEL of: ",column, ". Setting to 0")
                csvfile = os.path.join(output_dir, prefix + "_DEL" + extension)
                DEL_df.to_csv(csvfile, index=False,float_format='%.15f')

            if 'pwelch' in operations:
                nperseg = self.actiondict['nperseg']
                pwelch_df = pd.DataFrame(index=range(int(nperseg/2+1)),columns=self.parent.df.columns)
                pwelch_df = pwelch_df.drop('Time', axis=1)
                for column in filtered_df:
                    if not column  == 'Time':
                        try:
                            x = np.asarray(filtered_df[column][mask].values)
                            fs= 1/(filtered_df['Time'][1]-filtered_df['Time'][0])
                            f, Pxx_den = scipy.signal.welch(x,fs=fs,nperseg=nperseg)
                            pwelch_df.index=f
                            pwelch_df[column]=Pxx_den
                        except:
                            print("---> Warning, cannot compute pwelch of: ",column, ". Setting to 0")
                csvfile = os.path.join(output_dir, prefix + "_pwelch" + extension)
                pwelch_df.to_csv(csvfile, index=True, index_label='Freq',float_format='%.15f')

            if 'running_avg' in operations:
                running_avg_df = pd.DataFrame(columns=self.parent.df.columns)
                def running_average(data):
                    running_avg = 0.0
                    count = 0
                    running_avgs = []

                    for value in data:
                        count += 1
                        running_avg = running_avg + (value - running_avg) / count
                        running_avgs.append(running_avg)
                    return running_avgs

                for column in filtered_df:
                    if not column  == 'Time':
                        running_avg_df[column]= running_average(filtered_df[column][mask].values)
                    else:
                        running_avg_df[column]= filtered_df[column][mask].values

                csvfile = os.path.join(output_dir, prefix + "_running_avg" + extension)
                running_avg_df.to_csv(csvfile, index=False,float_format='%.15f')

            if 'pitch_travel':
                pitch_travel_columns  = filtered_df.filter(regex='Pitch',axis=1).columns
                if pitch_travel_columns.empty:
                    print("No variables found containing 'Pitch'. Add to array of vars.")
                else:
                    pitch_travel_df = pd.DataFrame(0,index=range(1),columns=pitch_travel_columns)
                    for column in pitch_travel_columns:
                        pitch_travel = np.sum(np.abs(np.asarray(filtered_df[column][mask].values)))
                        pitch_travel_df[column] = pitch_travel
                    csvfile = os.path.join(output_dir, prefix + "_pitch_travel" + extension)
                    pitch_travel_df.to_csv(csvfile, index=False,float_format='%.15f')

            return 

    @registeraction(actionlist)
    class spanwiseloading():
        actionname = 'spanwiseloading'
        blurb      = 'Reformats time history csv data to spanwise loading profiles'
        required   = False
        actiondefs = [
            {'key':'bladefile', 'required':True,
             'help':'AeroDyn blade file',  'default':''},
            {'key':'bladevars', 'required':True,
             'help':'List of blade variables to extract, such as Alpha, Cl, Cd, etc.',  'default':[]},
            {'key':'meancsvfile', 'required':True,
             'help':'mean csv file (output from above)',  'default':''},
            {'key':'savecsvfile', 'required':True,
             'help':'output csv file',  'default':''},
            {'key':'radialstations', 'required':True,
             'help':'list of radial blade stations',  'default':[1,2,3]},
            {'key':'prefix', 'required':False,
             'help':'Prefix in front of each openfast var', 'default':'AB1N'},
        ]
        
        def __init__(self, parent, inputs):
            self.actiondict = mergedicts(inputs, self.actiondefs)
            self.parent = parent
            print('Initialized '+self.actionname+' inside '+parent.name)
            return

        def execute(self):
            print('Executing '+self.actionname)
            bladefile   = self.actiondict['bladefile']
            bladevars   = self.actiondict['bladevars']
            meancsvfile = self.actiondict['meancsvfile']
            savecsvfile = self.actiondict['savecsvfile']
            rstations   = self.actiondict['radialstations']
            prefix      = self.actiondict['prefix']

            # Load the blade stations
            bladedat  = np.genfromtxt(bladefile, skip_header=6, comments='!')
            rpts = np.array([bladedat[i-1,0] for i in rstations])

            # Get the list of keys that need to be pulled out
            blistdict  = {}
            for suffix in bladevars:
                blistdict[suffix] = [prefix+('%03i'%i)+suffix for i in rstations]

            d = makeSecBladeDF(meancsvfile, rpts, blistdict)
            pddf = pd.DataFrame(d).to_csv(savecsvfile, index=False,
                                          float_format='%.15f')
            return
