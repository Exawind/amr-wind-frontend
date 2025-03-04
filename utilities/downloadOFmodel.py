#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

# Get the location where this script is being run
import sys, os
scriptpath = os.path.dirname(os.path.realpath(__file__))
basepath   = os.path.dirname(scriptpath)

# Add any possible locations of amr-wind-frontend here
amrwindfedirs = ['../',
                 basepath]
import shutil
for x in amrwindfedirs: sys.path.insert(1, x)
import argparse
import subprocess

import OpenFASTutil as OpenFAST
import ast

import urllib.request
import ssl

try:
    import argcomplete
    has_argcomplete = True
except:
    has_argcomplete = False

# Get YAML modules
import ruamel.yaml    
yaml = ruamel.yaml.YAML(typ='rt')
Loader= yaml.load
loaderkwargs = {}
dumperkwargs = {}
#dumperkwargs = {'indent':4, 'default_flow_style':False} # 'block_seq_indent':2, 'line_break':0, 'explicit_start':True, 

dictdefault = lambda d, k, default: d[k] if k in d else default

def downloadmodel(modelsource):
    """
    Download the OpenFAST model from a git repo
    """
    gitrepo     = modelsource['gitrepo']
    gitdirs     = modelsource['gitdirs']
    downloaddir = dictdefault(modelsource, 'downloaddir', '')
    branch      = dictdefault(modelsource, 'branch', None)
    compilecmd  = dictdefault(modelsource, 'compilecmd', None)

    # Clone the repo
    gitdlcmds   = ['git', 'clone']
    if branch is not None:
        gitdlcmds += ['-b', branch ]
    gitdlcmds  += [ '-n', '--depth=1', '--filter=tree:0' ]
    gitdlcmds  += [ gitrepo]
    if len(downloaddir)>0:
        gitdlcmds  += [ downloaddir ]
    print('EXECUTING '+' '.join(gitdlcmds))
    subprocess.run(gitdlcmds)

    # Switch to the directory
    if len(downloaddir)>0:
        workingdir = downloaddir
    else:
        workingdir = gitrepo.rsplit('/', 1)[-1].replace('.git','')
    curdir = os.getcwd()
    os.chdir(workingdir)

    # Check out the directories
    gitdircmds  = ['git', 'sparse-checkout', 'set', '--no-cone', ]
    gitdircmds += gitdirs
    subprocess.run(gitdircmds)
    print('EXECUTING '+' '.join(gitdircmds))
    subprocess.run(['git','checkout'])

    # Capture the git hash
    githash = subprocess.getoutput("git rev-parse --short HEAD")
    print('GITHASH = %s'%githash)
    with open('downloadmodel_githash.txt', 'w') as f:
        f.write(githash)
    
    # Go back to the original directory
    os.chdir(curdir)

    # Copy the directories (if necessary)
    if 'copyaction' in modelsource:
        source = modelsource['copyaction']['source']
        dest   = modelsource['copyaction']['dest']
        if not isinstance(source, list):
            source = [source]
        for sdir in source:
            shutil.copytree(sdir, dest, dirs_exist_ok=True)
        with open(os.path.join(dest, 'downloadmodel_githash.txt'), 'w') as f:
            f.write(githash)


    # Download any files from url sources
    if 'urlfiles' in modelsource:
        context = ssl._create_unverified_context()   # Avoid SSL certificate errors
        for urlf in modelsource['urlfiles']:
            url      = urlf[0]
            filedest = urlf[1]
            with urllib.request.urlopen(url, context=context) as f:
                print('Downloading '+url)
                html = f.read().decode('utf-8')
                with open(filedest, 'w') as outf:
                    outf.write(html)

    # Compile the DISCON library (if necessary)
    if compilecmd is not None:
        os.system(compilecmd)

    if ('deleteafterdownload' in modelsource) and modelsource['deleteafterdownload']:
        shutil.rmtree(workingdir)

    return


def editmodel(modelparams, tagedits=True):
    """
    Edit OpenFAST model parameters
    """
    fstfilename = modelparams['fstfilename']
    postconfigcmd = dictdefault(modelparams, 'postconfigcmd', None)

    # Edit fst parameters
    if 'FSTFile' in modelparams:
        fstparams = modelparams['FSTFile']
        print('Editing '+fstfilename)
        OpenFAST.editFASTfile(fstfilename, fstparams, tagedits=tagedits)

    # Edit parameters in each of these files
    filelist = ['EDFile', 'AeroFile', 'ServoFile', 'HydroFile', 'MooringFile', 'SubFile']
    for editfile in filelist:
        if editfile in modelparams:
            params = modelparams[editfile]
            # Search for unallowed values:
            for k, g in params.items():
                if g=='PLEASEEDITTHIS':
                    print('ERROR: PLEASEEDITTHIS not allowed for %s in %s'%(k, editfile))
                    raise ValueError('Abort')
            OFfile = OpenFAST.getFileFromFST(fstfilename, editfile)
            print('Editing '+OFfile)
            OpenFAST.editFASTfile(OFfile, params, tagedits=tagedits)

    # Edit DISCON parameters
    if 'DISCONFile' in modelparams:
        params = modelparams['DISCONFile']
        SDfile = OpenFAST.getFileFromFST(fstfilename, 'ServoFile')
        DISCONfile   = OpenFAST.getFileFromFST(SDfile, 'DLL_InFile')
        OpenFAST.editDISCONfile(DISCONfile, params)

    # run any post configuration commands
    if postconfigcmd is not None:
        os.system(postconfigcmd)

    return

def writeTurbineYaml(inputdict, filename):
    # Open the file and write it
    outfile = sys.stdout if filename == sys.stdout else open(filename, 'w')
    # Write out the header comment
    outfile.write("# ----- BEGIN turbine type setup input file ----\n")
    yaml.dump(inputdict, outfile, 
              **dumperkwargs)
    outfile.write("# ----- END turbine type input file ------\n")
    if filename != sys.stdout: 
        outfile.close()
        print("Saved turbine setup to %s"%filename)
    return

def processModelDict(yamldict, tagedits=True):
    if not yamldict:
        # Empty dictionary, do nothing
        raise ValueError('Empty dictionary')
    
    # Download the model
    if 'modelsource' in yamldict:
        modelsource = yamldict['modelsource']
        downloadmodel(modelsource)

    # Edit the openfast parameters in the files
    if 'modelparams' in yamldict:
        modelparams = yamldict['modelparams']
        editmodel(modelparams, tagedits=tagedits)

    # Write the turbine model yaml file
    writeturbyaml = dictdefault(yamldict, 'writeturbineyaml', False)
    if writeturbyaml and 'turbines' in yamldict:
        inputdict = {}
        inputdict['turbines'] = yamldict['turbines']
        outfile = dictdefault(yamldict, 'turbineyamlfile', sys.stdout)
        print(outfile)
        writeTurbineYaml(inputdict, outfile)
    return

# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":
    helpstring = """
    Edit an openFAST model
    """

    exampleyaml = """
# Download the openfast model from this repo
modelsource:
  gitrepo: git@github.com:IEAWindTask37/IEA-15-240-RWT.git
  gitdirs:
    - OpenFAST/IEA-15-240-RWT-UMaineSemi
    - OpenFAST/IEA-15-240-RWT
  branch: main                      # branch to clone (optional)
  #downloaddir: IEA-15-240-RWT-GIT   # destination for clone (optional)
  copyaction:                       # copy files out from git repo (optional)
    source: IEA-15-240-RWT/OpenFAST
    dest: Floating-IEA-15-240-RWT
  #compilecmd: SHELL COMMAND HERE   # Shell command/script to compile DISCON (optional)
  deleteafterdownload: True         # Delete the git repo after d/l (optional)

# Edit the model parameters in this section
modelparams:
  # Name of the FST file 
  fstfilename: Floating-IEA-15-240-RWT/IEA-15-240-RWT-UMaineSemi/IEA-15-240-RWT-UMaineSemi.fst
  # Shell script comamand to run after any OpenFAST parameter changes (optional)
  #postconfigcmd: SHELL COMMAND HERE

  # Specify any changes to OpenFAST parameters below
  # Possible files to edit are: FSTFile, EDFile, AeroFile, ServoFile, HydroFile, MooringFile, SubFile, DISCONFile
  FSTFile:
    DT: 0.005
    CompInflow: 2
  AeroFile:
    WakeMod: 0
  DISCONFile:
    Fl_Mode: 0

# Write the turbine yamlfile
writeturbineyaml: True
turbineyamlfile: floatingIEA15MW.yaml

# This will be copied over to turbineyamlfile
turbines:
  IEA15MW_ALM:     # This is an arbitrary, unique name
    # OpenFAST files from the repo git@github.com:IEAWindTask37/IEA-15-240-RWT.git
    turbinetype_name:             "IEA15MW_ALM"
    turbinetype_comment:          "OpenFAST 3.5 model"
    Actuator_type:                TurbineFastLine
    Actuator_openfast_input_file: OpenFAST3p5_Floating_IEA15MW/IEA-15-240-RWT-UMaineSemi/IEA-15-240-RWT-UMaineSemi.fst
    Actuator_rotor_diameter:      240
    Actuator_hub_height:          150
    Actuator_num_points_blade:    50
    Actuator_num_points_tower:    12
    Actuator_epsilon:             [2.0, 2.0, 2.0]
    Actuator_epsilon_tower:       [2.0, 2.0, 2.0]
    Actuator_openfast_start_time: 0.0
    Actuator_openfast_stop_time:  10000.0
    Actuator_nacelle_drag_coeff:  0.5
    Actuator_nacelle_area:        49.5
    Actuator_output_frequency:    10
    turbinetype_filedir:          OpenFAST3p5_Floating_IEA15MW/
"""
    
    # Handle arguments
    parser     = argparse.ArgumentParser(description=helpstring,
                                         formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.add_argument(
        "inputfile",
        help="input YAML file",
        nargs='*',
        type=str,
    )
    parser.add_argument('--example', 
                        help="Provide an example of an yaml file",
                        default=False,
                        action='store_true',
                        required=False)    
    parser.add_argument('--notagedits', 
                        help="Do not tag edits in the openfast files",
                        default=False,
                        action='store_true')
    
    # Load the options
    if has_argcomplete: argcomplete.autocomplete(parser)
    args      = parser.parse_args()
    inputfile = args.inputfile
    notagedits= args.notagedits
    example   = args.example
    
    if example:
        print(exampleyaml)
        sys.exit(0)

    if len(args.inputfile)<1:
        parser.print_help()
        sys.exit(0)

    for yamlfile in inputfile:
        # Load the input file
        yamldict = {}
        with open(yamlfile, 'r') as f:
            yamldict = Loader(f, **loaderkwargs)
            #print(yamldict)

        processModelDict(yamldict, tagedits=(not notagedits))
