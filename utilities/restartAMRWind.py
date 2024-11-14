#!/usr/bin/env python
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

def getRestartInput(inputfile, outputfile='', verbose=False):
    # Start the AMR-Wind case
    case = amrwind.MyApp.init_nogui()
    case.loadAMRWindInput(inputfile, printunused=False)
    # Get the chk restart prefix
    chkprefix = case.getAMRWindInput('check_file')
    globresult = glob.glob(chkprefix+"*/")
    latestdir = max(globresult, key=os.path.getctime)
    if verbose: 
        print("CHK PREFIX: "+chkprefix)
        print(globresult)
        print(latestdir)
    # Set the restart time
    case.setAMRWindInput('restart_file', latestdir)
    # Write the new outputfile
    newinput=case.writeAMRWindInput(outputfile)
    if verbose:
        print(newinput)
    return

def getLatestCHKDir(dirlist, criteria='lastmodified'):
    if criteria == 'lastmodified':
        latestdir = max(dirlist, key=os.path.getctime)
    return latestdir
        
# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":
    helpstring = """
Restart an AMR-Wind simulation

Example usage:
    ./restartAMRWind.py INPUT.inp -o OUTPUT.inp
    """

    # Handle arguments
    parser     = argparse.ArgumentParser(description=helpstring,
                                         formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.add_argument(
        "inputfile",
        help="AMR-Wind input file",
        type=str,
    )
    parser.add_argument('-v', '--verbose', 
                        action='count', 
                        help="Verbosity level (multiple levels allowed)",
                        default=0)
    parser.add_argument(
        "-o",
        "--outfile",
        help="new input file (default: dump to screen if verbose)",
        default='',
        type=str,
        #required=True,
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        nargs='+',
        default='',
        help="Choose the latest directories from these checkpoints",
        dest='checkpoint',
    )
    parser.add_argument(
        "--noturbines",
        dest='noturbines',
        help="Don't restart any openfast turbines (Default: False)",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "--stop-time",
        dest='stoptime',
        help="Set the new stop time (Default: None)",
        default=None,
    )
    parser.add_argument(
        "--max-step",
        dest='maxstep',
        help="Set the new max step (Default: None)",
        default=None,
    )
    parser.add_argument(
        "--amr-wind-version",
        dest='amrwindver',
        help="Set which version of amr-wind to use [latest, legacy] (Default: latest)",
        default='latest',
    )

    # Load the options
    args      = parser.parse_args()
    inputfile = args.inputfile
    outfile   = args.outfile
    verbose   = args.verbose
    chkdirs   = args.checkpoint
    noturbs   = args.noturbines
    stoptime  = args.stoptime
    maxstep   = args.maxstep
    amrwindver= args.amrwindver

    # Load the input file
    case = amrwind.MyApp.init_nogui()
    case.loadAMRWindInput(inputfile, printunused=False)

    # Get the latest checkpoint
    if len(chkdirs)==0:
        chkprefix  = case.getAMRWindInput('check_file')
        chkdirlist = glob.glob(chkprefix+"*/")        
    else:
        chkdirlist = chkdirs
        
    #latestdir = max(chkdirlist, key=os.path.getctime)
    #print(latestdir)
    latestdir = getLatestCHKDir(chkdirlist)

    # Set the latest check point for restart
    case.setAMRWindInput('restart_file', latestdir)    

    # Set the new stop time
    if (stoptime is not None):
        case.setAMRWindInput('stop_time', float(stoptime))

    # Set the new stop time
    if (maxstep is not None):
        case.setAMRWindInput('max_step', int(maxstep))

    # Set the restarts for openfast turbine
    physics  = case.getAMRWindInput('physics')
    actuator = case.getAMRWindInput('ActuatorForcing')
    if (not noturbs) and actuator and ('Actuator' in physics):
        case.restartOpenFASTinput(latestdir)
        
    newinput=case.writeAMRWindInput(outfile, amr_wind_version=amrwindver)
    if verbose:
        print(newinput)
