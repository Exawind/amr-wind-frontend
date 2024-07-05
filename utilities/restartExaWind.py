#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
#
# Copyright (c) 2022, Alliance for Sustainable Energy
#
# This software is released under the BSD 3-clause license. See LICENSE file
# for more details.
#

# Get the location where this script is being run
import sys, os
scriptpath = os.path.dirname(os.path.realpath(__file__))
basepath   = os.path.dirname(scriptpath)

sys.path.insert(1, scriptpath)
sys.path.insert(1, basepath)

# Load the libraries
import amrwind_frontend  as amrwind
import argparse
import glob
import re
import numpy as np

import restartAMRWind
from netCDF4 import Dataset

from functools import reduce  # forward compatibility for Python 3
import operator

try:
    import argcomplete
    has_argcomplete = True
except:
    has_argcomplete = False

# Load ruamel or pyyaml as needed
try:
    import ruamel.yaml
    #yaml = ruamel.yaml.YAML(typ='unsafe', pure=True)
    yaml = ruamel.yaml.YAML(typ='rt')
    #print("# Loaded ruamel.yaml")
    useruamel=True
    loaderkwargs = {}
    dumperkwargs = {}
    
    #loaderkwargs = {'Loader':yaml.RoundTripLoader}
    #dumperkwargs = {'Dumper':yaml.RoundTripDumper, 'indent':4, 'default_flow_style':False} # 'block_seq_indent':2, 'line_break':0, 'explicit_start':True,
    Loader=yaml.load
    #print("Done ruamel.yaml")
except:
    import yaml as yaml
    print("# Loaded yaml")
    useruamel=False
    loaderkwargs = {}
    dumperkwargs = {'default_flow_style':False }
    Loader=yaml.safe_load

if useruamel:
    from ruamel.yaml.comments import CommentedMap 
    def comseq(d):
        """
        Convert OrderedDict to CommentedMap
        """
        if isinstance(d, OrderedDict):
            cs = CommentedMap()
            for k, v in d.items():
                cs[k] = comseq(v)
            return cs
        return d

# Get the ending string
getsuffix = lambda x: re.search(r"(\d+)$", x).group()

def getFromDict(dataDict, mapList):
    """
    gets the entry from dataDict which corresponds to the list of keywords in mapList
    """
    return reduce(operator.getitem, mapList, dataDict)

def setInDict(dataDict, mapList, value, verbose=False):
    if verbose:
        keyloc = '.'.join([str(x) for x in mapList])
        print('SET  %-40s = '%keyloc+repr(value))
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

def getNewFilename(filename):
    basename, ext =os.path.splitext(os.path.basename(filename))
    trailingnums = re.search(r"(\d+)$", basename) #.group()
    if trailingnums is None:
        newnum='1'
        newbase = basename
    else:
        oldnum = trailingnums.group()
        Nlength= len(oldnum)
        newnum = int(oldnum)+1
        newnum = str(newnum).zfill(Nlength)
        newbase= basename[:-Nlength]

    newfilename=newbase+newnum+ext
    #print(basename, ext, newnum, newfilename)
    return newfilename

def getrsttime(rstfile, index=-1):
    rstdat = Dataset(rstfile)
    time = np.array(rstdat['time_whole'])
    return float(time[index])

def getlatestfile(filebase):
    list_of_files = glob.glob(filebase)
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def updateAMRWind(inputfile, chkdirs, outfile, verbose=False):
    # Load the input file
    case = amrwind.MyApp.init_nogui()
    case.loadAMRWindInput(inputfile, printunused=False)

    # Get the latest checkpoint
    if len(chkdirs)==0:
        chkprefix  = case.getAMRWindInput('check_file')
        chkdirlist = glob.glob(chkprefix+"*/")        
    else:
        chkdirlist = chkdirs

    latestdir = restartAMRWind.getLatestCHKDir(chkdirlist)

    # Set the latest check point for restart
    case.setAMRWindInput('restart_file', latestdir)    
    
    newinput=case.writeAMRWindInput(outfile)
    if verbose:
        print(newinput)

def updateNaluWind(inputfile, outputfile, restart_time, OF_iter, verbose=False):
    # Load the NaluWind inputfile
    with open(inputfile, 'r') as fp:
        naluyaml = Loader(fp, **loaderkwargs)
        realm0 = naluyaml['realms'][0]
        # Change the mesh related items
        #realm0['automatic_decomposition_type'] = 'None'
        setInDict(naluyaml, ['realms', 0, 'automatic_decomposition_type'], 'None', verbose=verbose)
        if 'rebalance_mesh' in realm0:
            realm0.pop('rebalance_mesh')
        if 'stk_rebalance_method' in realm0:
            realm0.pop('stk_rebalance_method')
        if 'mesh_transformation' in realm0:
            realm0.pop('mesh_transformation')

            
        # Change the realm restart time
        #realm0['restart']['restart_time'] = restart_time
        setInDict(naluyaml, ['realms', 0, 'restart', 'restart_time'], restart_time, verbose=verbose)

        # Incremement the restart data_base_names
        oldrstname = realm0['restart']['restart_data_base_name']
        rstname    = os.path.basename(oldrstname)
        oldpath    = os.path.dirname(oldrstname)
        newpath    = getNewFilename(oldpath)
        newrstname = os.path.join(newpath, rstname)
        #realm0['restart']['restart_data_base_name'] = newrstname
        setInDict(naluyaml, ['realms', 0, 'restart', 'restart_data_base_name'], newrstname, verbose=verbose)

        # Incremement the output data_base_names        
        oldoutname = realm0['output']['output_data_base_name']
        outname    = os.path.basename(oldoutname)
        oldpath    = os.path.dirname(oldoutname)
        newpath    = getNewFilename(oldpath)
        newoutname = os.path.join(newpath, outname)
        #realm0['output']['output_data_base_name'] = newoutname
        setInDict(naluyaml, ['realms', 0, 'output', 'output_data_base_name'], newoutname, verbose=verbose)

        # Change the openfast time
        dt_FAST = realm0['openfast_fsi']['dt_FAST']
        #realm0['openfast_fsi']['t_start'] = dt_FAST*OF_iter
        setInDict(naluyaml, ['realms', 0, 'openfast_fsi', 't_start'], dt_FAST*OF_iter, verbose=verbose)

        # Change Time_Integrators
        time_integ = naluyaml['Time_Integrators']
        time_integ[0]['StandardTimeIntegrator']['start_time'] = restart_time
        setInDict(naluyaml, ['Time_Integrators', 0, 'StandardTimeIntegrator', 'start_time'], restart_time, verbose=verbose)
        
        outfile=sys.stdout if outputfile.strip()=='sys.stdout' else open(outputfile, 'w')
        print('Writing new nalu yaml file to: '+repr(outputfile))
        yaml.dump(naluyaml, outfile, **dumperkwargs)
        if outfile != sys.stdout:
            outfile.close()

    return
        
# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":
    helpstring = """
Restart a hybrid Exawind simulation
    """
    
    # Handle arguments
    parser     = argparse.ArgumentParser(description=helpstring,
                                         formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.add_argument(
        "inputfile",
        help="ExaWind input file",
        type=str,
    )
    parser.add_argument('-v', '--verbose', 
                        action='count', 
                        help="Verbosity level (multiple levels allowed)",
                        default=0)
    parser.add_argument(
        "-o",
        "--outputfile",
        help="new input file",
        default='',
        type=str,
        required=True,
    )
    parser.add_argument(
        "--chkfile",
        help="Use this amr-wind restart file",
        default='',
        type=str,
        #required=True,
    )    
    parser.add_argument(
        "--chkpiter",
        help="iteration number for chkp files",
        default='',
        type=int,
        required=True,
    )
    parser.add_argument(
        "--additer",
        help="Number of iterations to add to exawind run",
        default=200,
        type=int,
        required=True,
    )

    # Load the options
    if has_argcomplete: argcomplete.autocomplete(parser)    
    args      = parser.parse_args()
    inputfile = args.inputfile
    outputfile= args.outputfile
    chkpiter  = args.chkpiter
    verbose   = args.verbose
    additer   = args.additer
    print('Reading '+inputfile)
    
    # --- Load the inputfile ---
    # Check if file exists
    if not os.path.isfile(inputfile):
        print("ERROR: %s does not exist"%inputfile)
        raise Exception('File does not exist')
    # Load the file
    with open(inputfile, 'r') as fp:
        yamldict = Loader(fp, **loaderkwargs)

    # Get a list of nalu input files
    nalu_wind_inputs = yamldict['exawind']['nalu_wind_inp']
    nalufilelist     = [x['base_input_file'] for x in nalu_wind_inputs] 
    uniquelist = list(set(nalufilelist))
    # Check if the uniquelist has more than 1 file
    if len(uniquelist)>1:
        raise RuntimeError('More than 1 nalu_wind_inp provided')
    
    # Get a list of the nalu restart files
    nalurestarts     = [x['replace']['realms'][0]['restart'] for x in nalu_wind_inputs]
    lastrstfile      = getlatestfile(nalurestarts[0]['restart_data_base_name']+'.*')
    nalu_restarttime = getrsttime(lastrstfile)
    
    # ==== Set up the nalu-wind restart file ==== 
    old_nalu_inp = uniquelist[0]
    new_nalu_inp = getNewFilename(old_nalu_inp)
    updateNaluWind(old_nalu_inp, new_nalu_inp, nalu_restarttime, chkpiter, verbose=verbose)
    print()
    
    # ==== Set up the amr-wind restart file ==== 
    # Load certain inputs
    amr_wind_inp = yamldict['exawind']['amr_wind_inp']
    new_amr_wind_inp = getNewFilename(amr_wind_inp)
    # Update the latest checkpoint file
    updateAMRWind(amr_wind_inp, [], new_amr_wind_inp, verbose=verbose)

    # ==== Set up the exawind restart file ==== 
    #yamldict['exawind']['amr_wind_inp'] = new_amr_wind_inp
    setInDict(yamldict, ['exawind', 'amr_wind_inp'], new_amr_wind_inp, verbose=verbose)    

    # Load the turbine info
    turbine_info = yamldict['turbine_info']
    Nturbines = len(turbine_info)
    # Change every turbine
    for iturb in range(Nturbines):
        turbname = 'turb'+repr(iturb)
        yamlturbname = yamldict['turbine_info'][turbname]
        naluturb     = yamldict['exawind']['nalu_wind_inp'][iturb]
        # -- Set openfast_restart
        yamlpath = ['exawind', 'nalu_wind_inp', iturb, 'replace', 'realms', 0, 'openfast_fsi','Turbine0','restart_filename']
        if 'openfast_restart' in yamlturbname:
            OFrestart = yamlturbname['openfast_restart']
            yamlturbname.pop('openfast_restart')
        else:
            OFrestart = getFromDict(yamldict, yamlpath)
        basename, ext =os.path.splitext(OFrestart)
        newchkp = basename+'.'+repr(chkpiter)
        setInDict(yamldict, yamlpath,
                  newchkp, verbose=verbose)
        
        # -- Set force log file
        yamlpath = ['exawind', 'nalu_wind_inp', iturb, 'replace', 'realms', 0, 'post_processing',0,'output_file_name']
        if 'force' in yamlturbname:
            oldforce = yamlturbname['force']
            yamlturbname.pop('force')
        else:
            oldforce = getFromDict(yamldict, yamlpath)
        newforce = getNewFilename(oldforce)
        setInDict(yamldict,
                  yamlpath,
                  newforce, verbose=verbose)

        # -- Set output ---
        yamlpath = ['exawind', 'nalu_wind_inp', iturb, 'replace', 'realms', 0, 'output','output_data_base_name']
        if 'output' in yamlturbname:
            oldoutname = yamlturbname['output']
            yamlturbname.pop('output')
        else:
            oldoutname = getFromDict(yamldict, yamlpath)
        outname    = os.path.basename(oldoutname)
        oldpath    = os.path.dirname(oldoutname)
        newpath    = getNewFilename(oldpath)
        newoutname = os.path.join(newpath, outname)
        setInDict(yamldict, yamlpath,
                  newoutname, verbose=verbose)        

        # -- Set restart --
        yamlpath = ['exawind', 'nalu_wind_inp', iturb, 'replace', 'realms', 0, 'restart','restart_data_base_name']
        if 'restart' in yamlturbname:
            oldrstname = yamlturbname['restart']
            yamlturbname.pop('restart')
        else:
            oldrstname = getFromDict(yamldict, yamlpath)
        rstname    = os.path.basename(oldrstname)
        oldpath    = os.path.dirname(oldrstname)
        newpath    = getNewFilename(oldpath)
        newrstname = os.path.join(newpath, rstname)
        setInDict(yamldict, yamlpath, 
                  newrstname, verbose=verbose)
        setInDict(yamldict, 
                  ['exawind', 'nalu_wind_inp', iturb, 'replace', 'realms', 0, 'mesh',],
                  oldrstname, verbose=verbose)

        # -- Set the log file
        yamlpath = ['exawind', 'nalu_wind_inp', iturb, 'logfile']
        if 'logfile' in yamlturbname:
            oldlogname = yamlturbname['logfile']
            yamlturbname.pop('logfile')
        else:
            oldlogname = getFromDict(yamldict, yamlpath)
        newlogname = getNewFilename(oldlogname)
        setInDict(yamldict, yamlpath, 
                  newlogname, verbose=verbose)
        
        
        # -- Remove mesh transformation
        if 'mesh_transformation' in naluturb['replace']['realms'][0]:
            naluturb['replace']['realms'][0].pop('mesh_transformation')

        # -- change the base_input_file
        setInDict(yamldict, 
                  ['exawind', 'nalu_wind_inp', iturb, 'base_input_file'],
                  new_nalu_inp, verbose=verbose)
           
        #print(yamldict['turbine_info']['turb'+repr(iturb)])
    yamlpath = ['exawind', 'num_timesteps']
    curriter = getFromDict(yamldict, yamlpath)
    setInDict(yamldict, yamlpath, curriter+additer, verbose=verbose)
    
    # === Dump the new yamlfile ====
    outfile=sys.stdout if outputfile.strip()=='sys.stdout' else open(outputfile, 'w')
    print('Writing new exawind yaml file to: '+repr(outputfile))    
    yaml.dump(yamldict, outfile, **dumperkwargs)
    if outfile != sys.stdout:
        outfile.close()
