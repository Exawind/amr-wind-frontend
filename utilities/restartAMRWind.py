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

# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":
    helpstring = """Restart an AMR-Wind simulation
    """
    # Handle arguments
    parser     = argparse.ArgumentParser(description=helpstring)
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
        help="new input file",
        default='',
        type=str,
        required=True,
    )

    # Load the options
    args      = parser.parse_args()
    inputfile = args.inputfile
    outfile   = args.outfile
    verbose   = args.verbose

    # Get the new restart input file
    getRestartInput(inputfile, outputfile=outfile, verbose=verbose)
