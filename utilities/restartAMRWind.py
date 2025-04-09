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

try:
    import argcomplete
    has_argcomplete = True
except:
    has_argcomplete = False

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

def getLatestCHKDir(dirlist, criteria='lastiter'):
    latestdir = None
    if criteria == 'lastcreated':
        # Choose the directory with the last creation time
        latestdir = max(dirlist, key=os.path.getctime)
    elif criteria == 'lastiter':
        # Choose the directory with the highest iteration number
        simiter = []
        for d in dirlist:
            simiter.append(int(case.readCheckpointHeader(d, linenum=3)))
        maxindex = simiter.index(max(simiter))
        latestdir=dirlist[maxindex]
        #print(dirlist[maxindex], simiter[maxindex])
    elif criteria == 'lastsimtime':
        # Choose the directory with the largest simulation time
        simtimes = []
        for d in dirlist:
            simtimes.append(float(case.readCheckpointHeader(d, linenum=4)))
        maxindex = simtimes.index(max(simtimes))
        latestdir=dirlist[maxindex]
        #print(dirlist[maxindex], simtimes[maxindex])
    # Warn if latestdir contains ".old."
    if ".old." in latestdir:
        print("WARNING: %s is a backed-up checkpoint directory.  Make sure this is correct."%latestdir)
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
    parser.add_argument(
        "--sort-method",
        dest='sortmethod',
        help="Choose which method to use when selecting the restart checkpoint",
        default='lastsimtime',
        choices=['lastsimtime', 'lastiter','lastcreated']
    )

    
    # Load the options
    if has_argcomplete: argcomplete.autocomplete(parser)
    args      = parser.parse_args()
    inputfile = args.inputfile
    outfile   = args.outfile
    verbose   = args.verbose
    chkdirs   = args.checkpoint
    noturbs   = args.noturbines
    stoptime  = args.stoptime
    maxstep   = args.maxstep
    amrwindver= args.amrwindver
    sortmethod= args.sortmethod

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
    latestdir = getLatestCHKDir(chkdirlist, criteria=sortmethod)

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
