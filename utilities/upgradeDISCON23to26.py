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

import argparse
from enum import Enum
import fileinput

def replacelines(txtfile, replacelines, replacetxt):
    """
    Replace some lines in a file
    """
    skiplines = range(replacelines[0], replacelines[1]+1)
    addedtext = False
    for line in fileinput.input(txtfile, inplace=True, backup='.bak'):
        if fileinput.filelineno() in skiplines:
            if not addedtext:
                sys.stdout.write(replacetxt)
                addedtext = True
        else:
            sys.stdout.write(str(line))
    return

def insertlines(txtfile, insertafter, inserttxt):
    """
    Insert some lines after a position in the file
    
    """
    for line in fileinput.input(txtfile, inplace=True, backup='.bak'):
        sys.stdout.write(str(line))
        if fileinput.filelineno() == insertafter:
            sys.stdout.write(inserttxt)            
    return

def deletelines(txtfile, dellines):
    """
    Delete some lines in a file
    """
    for line in fileinput.input(txtfile, inplace=True, backup='.bak'):
        if fileinput.filelineno() in dellines:
            pass
        else:
            sys.stdout.write(str(line))            
    return

def getlines(txtfile, linenums):
    """
    Get specific lines from txtfile
    """
    with open(txtfile) as f:
        alllines = f.readlines()
    subset = []
    for i in linenums: subset.append(alllines[i-1])
    return subset

def getfirstword(txtfile, linenum):
    alllines = getlines(txtfile, [linenum])
    word     = alllines[0].split('!')[0]
    return word

def testgetlines():
    origtxt="""1\n2\n3\n4\n5\n6\n"""
    text_file = open("sample.txt", "w")
    text_file.write(origtxt)
    text_file.close()
    subset=getlines("sample.txt", [2,4,5])
    print(subset)
    return

defaultdict = {
    'TD_Mode':'0',
    'OL_Mode':'0',
    'PA_Mode':'2',
    'Ext_Mode':'0',
    'ZMQ_Mode':'0',
}

def upgradeDISCON26(disconfile, verbosity=0, vardict=defaultdict):
    extraEndLines = """\

!------- Open Loop Control -----------------------------------------------------
"unused"            ! OL_Filename       - Input file with open loop timeseries (absolute path or relative to this file)
0                   ! Ind_Breakpoint    - The column in OL_Filename that contains the breakpoint (time if OL_Mode = 1)
0                   ! Ind_BldPitch      - The column in OL_Filename that contains the blade pitch input in rad
0                   ! Ind_GenTq         - The column in OL_Filename that contains the generator torque in Nm
0                   ! Ind_YawRate       - The column in OL_Filename that contains the generator torque in Nm

!------- Pitch Actuator Model -----------------------------------------------------
1.570800000000        ! PA_CornerFreq     - Pitch actuator bandwidth/cut-off frequency [rad/s]
0.707000000000        ! PA_Damping        - Pitch actuator damping ratio [-, unused if PA_Mode = 1]

!------- External Controller Interface -----------------------------------------------------
"unused"            ! DLL_FileName        - Name/location of the dynamic library in the Bladed-DLL format
"unused"            ! DLL_InFile          - Name of input file sent to the DLL (-)
"DISCON"            ! DLL_ProcName        - Name of procedure in DLL to be called (-) 

!------- ZeroMQ Interface ---------------------------------------------------------
"tcp://localhost:5555"            ! ZMQ_CommAddress     - Communication address for ZMQ server, (e.g. "tcp://localhost:5555") 
2                   ! ZMQ_UpdatePeriod    - Call ZeroMQ every [x] seconds, [s]"""
    # Add lines at the end
    if verbosity>0: 
        print("=== Adding lines at the end: ===")
        print(extraEndLines)
    insertlines(disconfile, 119, extraEndLines)

    # ==== Set up the yaw control ====
    # Get the original yaw inputs
    YCtrlDict = {
        'Y_Rate':        getfirstword(disconfile, 96),
        'Y_MErrSet':     getfirstword(disconfile, 93),
        'Y_IPC_IntSat':  getfirstword(disconfile, 87),
        'Y_IPC_KP':      getfirstword(disconfile, 89),
        'Y_IPC_KI':      getfirstword(disconfile, 90),
    }

    YawCtrlTemplate = """\
!------- YAW CONTROL ------------------------------------------------------
0.00000             ! Y_uSwitch		- Wind speed to switch between Y_ErrThresh. If zero, only the first value of Y_ErrThresh is used [m/s]
4.000000  8.000000  ! Y_ErrThresh		- Yaw error threshold. Turbine begins to yaw when it passes this. [rad^2 s]
{Y_Rate:20}! Y_Rate			- Yaw rate [rad/s]
{Y_MErrSet:20}! Y_MErrSet			- Yaw alignment error, set point [rad]
{Y_IPC_IntSat:20}! Y_IPC_IntSat		- Integrator saturation (maximum signal amplitude contribution to pitch from yaw-by-IPC), [rad]
{Y_IPC_KP:20}! Y_IPC_KP			- Yaw-by-IPC proportional controller gain Kp
{Y_IPC_KI:20}! Y_IPC_KI			- Yaw-by-IPC integral controller gain Ki
"""
    YawCtrlLines = YawCtrlTemplate.format(**YCtrlDict)
    # Add Yaw Control lines
    if verbosity>0: 
        print("=== Replacing with yaw control section: ===")
        print(YawCtrlLines)
    cutlines = [85, 96]
    replacelines(disconfile, cutlines, YawCtrlLines)

    # ==== Set up IPC variables ====
    IPC_VrampLine ="""\
8.592000  10.740000  ! IPC_Vramp		- Start and end wind speeds for cut-in ramp function. First entry: IPC inactive, second entry: IPC fully active. [m/s]
"""
    IPC_KPLine   = """\
0.000e+00 0.000e+00 ! IPC_KP			- Proportional gain for the individual pitch controller: first parameter for 1P reductions, second for 2P reductions, [-]
"""
    # Add Yaw Control lines
    if verbosity>0: 
        print("=== Adding IPC lines: ===")
        print(IPC_VrampLine)
        print(IPC_KPLine)
    insertlines(disconfile, 46, IPC_KPLine)
    insertlines(disconfile, 45, IPC_VrampLine)


    # ==== Set up filter variables ====
    filterlines1 = """\
0.20944             ! F_WECornerFreq    - Corner frequency (-3dB point) in the first order low pass filter for the wind speed estimate [rad/s].
0.17952             ! F_YawErr          - Low pass filter corner frequency for yaw controller [rad/s].
"""
    filterlines2 = """\
0.01042             ! F_FlHighPassFreq    - Natural frequency of first-order high-pass filter for nacelle fore-aft motion [rad/s].
"""
    # Add filter lines
    if verbosity>0: 
        print("=== Adding filter lines: ===")
        print(filterlines1)
        print(filterlines2)
    insertlines(disconfile, 27, filterlines2)
    insertlines(disconfile, 26, filterlines1)

    # ==== Set up controller flags ====
    TDModeLine = """\
{TD_Mode:20}! TD_Mode           - Tower damper mode {{0: no tower damper, 1: feed back translational nacelle accelleration to pitch angle}}
""".format(**vardict)
    ModeLines = """\
{OL_Mode:20}! OL_Mode           - Open loop control mode {{0: no open loop control, 1: open loop control vs. time}}
{PA_Mode:20}! PA_Mode           - Pitch actuator mode {{0 - not used, 1 - first order filter, 2 - second order filter}}
{Ext_Mode:20}! Ext_Mode          - External control mode {{0 - not used, 1 - call external dynamic library}}
{ZMQ_Mode:20}! ZMQ_Mode          - Fuse ZeroMQ interaface {{0: unused, 1: Yaw Control}}
""".format(**vardict)
    # Add controller lines
    if verbosity>0: 
        print("=== Adding controller lines: ===")
        print(TDModeLine)
        print(ModeLines)
    insertlines(disconfile, 19, ModeLines)
    insertlines(disconfile, 18, TDModeLine)

    return
    

# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":
    helpstring = """Upgrade the version of an OpenFAST model
    """

    # Handle arguments
    parser     = argparse.ArgumentParser(description=helpstring)
    parser.add_argument(
        "disconfile",
        help="DISCON input file",
        type=str,
    )
    parser.add_argument('-v', '--verbose', 
                        action='count', 
                        help="Verbosity level (multiple levels allowed)",
                        default=0)

    # Load the options
    args      = parser.parse_args()
    filename  = args.disconfile
    verbose   = args.verbose

    #testgetlines()
    upgradeDISCON26(filename, verbose)
