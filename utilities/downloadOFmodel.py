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

dictdefault = lambda d, k, default: d[k] if k in d else default

def downloadmodel(modelsource):
    """
    Download the OpenFAST model from a git repo
    """
    gitrepo     = modelsource['gitrepo']
    gitdirs     = modelsource['gitdirs']
    downloaddir = dictdefault(modelsource, 'downloaddir', '')
    branch      = dictdefault(modelsource, 'branch', None)

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

    if 'deleteafterdownload' in modelsource:
        shutil.rmtree(workingdir)

    return


def editmodel(modelparams):
    """
    Edit OpenFAST model parameters
    """
    fstfilename = modelparams['fstfilename']

    # Edit fst parameters
    if 'FSTFile' in modelparams:
        fstparams = modelparams['FSTFile']
        print('Editing '+fstfilename)
        OpenFAST.editFASTfile(fstfilename, fstparams, tagedits=(not notagedits))

    # Edit parameters in each of these files
    filelist = ['EDFile', 'AeroFile', 'ServoFile', 'HydroFile', 'MooringFile', 'SubFile']
    for editfile in filelist:
        if editfile in modelparams:
            params = modelparams[editfile]
            OFfile = OpenFAST.getFileFromFST(fstfilename, editfile)
            print('Editing '+OFfile)
            OpenFAST.editFASTfile(OFfile, params, tagedits=(not notagedits))
        
    if 'DISCONFile' in modelparams:
        params = modelparams['DISCONFile']
        SDfile = OpenFAST.getFileFromFST(fstfilename, 'ServoFile')
        DISCONfile   = OpenFAST.getFileFromFST(SDfile, 'DLL_InFile')
        OpenFAST.editDISCONfile(DISCONfile, params)
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
  #downloaddir: IEA-15-240-RWT-GIT   # destination for clone (optional)
  copyaction:                       # copy files out from git repo (optional)
    source: IEA-15-240-RWT/OpenFAST
    dest: Floating-IEA-15-240-RWT
  deleteafterdownload: True         # Delete the git repo after d/l (optional)

# Edit the model parameters in this section
modelparams:
  fstfilename: Floating-IEA-15-240-RWT/IEA-15-240-RWT-UMaineSemi/IEA-15-240-RWT-UMaineSemi.fst
  # Specify any changes to OpenFAST parameters below
  # Possible files to edit are: FSTFile, EDFile, AeroFile, ServoFile, HydroFile, MooringFile, SubFile, DISCONFile
  FSTFile:
    DT: 0.005
    CompInflow: 2
  AeroFile:
    WakeMod: 0
  DISCONFile:
    Fl_Mode: 0
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
        with open(inputfile[0], 'r') as f:
            yamldict = Loader(f, **loaderkwargs)
            print(yamldict)

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
            editmodel(modelparams)
