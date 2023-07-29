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
import OpenFASTutil  as OpenFAST
import findOFversion as findOFversion
import argparse
from enum import Enum
import fileinput

def verinfo2tuple(verinfo):
    return (verinfo['major'], verinfo['minor'])

def checkallowedversions(verinfo):
    """
    Check to make sure that the version in verinfo is allowed
    """
    verlist = verinfo2tuple(verinfo) 
    if verlist in findOFversion.allowedversions:
        return True
    else:
        return False


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

def extractlines(filename, linenums):
    with open(filename) as file:
        lines = [line.rstrip() for line in file]
    n1=linenums[0]-1
    n2=linenums[1]
    return lines[n1:n2]

def findlinewith(filename, substring):
    with open(filename) as file:
        lines = [line.rstrip() for line in file]
    for ix, x in enumerate(lines):
        if substring in x:
            return ix+1
    return -1

def testreplacelines():
    origtxt="""1\n2\n3\n4\n5\n6\n"""
    newlines="""A
B
C
"""
    text_file = open("sample.txt", "w")
    text_file.write(origtxt)
    text_file.close()
    replacelines("sample.txt", [2,6], newlines)
    return


def testinsertlines():
    origtxt="""1\n2\n3\n4\n5\n6\n"""
    newlines="""A
B
C
"""
    text_file = open("sample.txt", "w")
    text_file.write(origtxt)
    text_file.close()
    insertlines("sample.txt", 1, newlines)
    return

def testdeletelines():
    origtxt="""1\n2\n3\n4\n5\n6\n"""
    text_file = open("sample.txt", "w")
    text_file.write(origtxt)
    text_file.close()
    deletelines("sample.txt", [2,5])
    return

# =========================================================================
# Function decorator for upgrade functions
upgradefunctionlist = {}

# See https://stackoverflow.com/questions/5929107/decorators-with-parameters
def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer

@parametrized
def register_upgradefunction(f, curver, targetver):
    upgradefunctionlist[curver] = {'func':f, 'newver':targetver}
    return f

# See this document for all changes between versions
# https://openfast.readthedocs.io/en/main/source/user/api_change.html

@register_upgradefunction((2,5), (2,6))
def upgrade_2_5(fstfile, verbosity, **kwargs):
    """
    Upgrade OpenFAST model from 2.5 to 2.6
    """
    # Get the input files
    AeroDynFile   = OpenFAST.getFileFromFST(fstfile, 'AeroFile')
    TwrShadowTF   = OpenFAST.getVarFromFST(AeroDynFile, 'TwrShadow').upper()
    TwrShadowVal  = 1 if TwrShadowTF == 'TRUE' else 0
    OpenFAST.editFASTfile(AeroDynFile, {'TwrShadow':TwrShadowVal})

    # Add the TwrTI column
    TwrTI = " 1.0E-1"
    with open(AeroDynFile) as file:
        ADlines = [line.rstrip() for line in file]
    ADfirstword = [x.strip().split()[0].lower() for x in ADlines if len(x.strip().split())>0]
    linestart   = ADfirstword.index('twrelev')+3
    NumTwrNds   = int(OpenFAST.getVarFromFST(AeroDynFile, 'NumTwrNds'))
    linenums    = [linestart, linestart+NumTwrNds-1]
    nodelines   = extractlines(AeroDynFile, linenums)
    newnodelines= ''
    for x in nodelines:
        newnodelines += x+TwrTI+'\n'
    replacelines(AeroDynFile, linenums, newnodelines)
    if verbosity>0:
        print("Adding TwrTI")
        print(newnodelines)

    return

@register_upgradefunction((2,6), (3,0))
def upgrade_2_6(fstfile, verbosity, **kwargs):
    """
    Upgrade OpenFAST model from 2.6 to 3.0
    """
    # Get the ServoDyn File
    ServoDynFile = OpenFAST.getFileFromFST(fstfile, 'ServoFile')
    cutlines = [60, 64]
    structinputs = \
"""---------------------- STRUCTURAL CONTROL --------------------------------------
0             NumBStC      - Number of blade structural controllers (integer)
"unused"      BStCfiles    - Name of the files for blade structural controllers (quoted strings) [unused when NumBStC==0]
0             NumNStC      - Number of nacelle structural controllers (integer)
"unused"      NStCfiles    - Name of the files for nacelle structural controllers (quoted strings) [unused when NumNStC==0]
0             NumTStC      - Number of tower structural controllers (integer)
"unused"      TStCfiles    - Name of the files for tower structural controllers (quoted strings) [unused when NumTStC==0]
0             NumSStC      - Number of substructure structural controllers (integer)
"unused"      SStCfiles    - Name of the files for substructure structural controllers (quoted strings) [unused when NumSStC==0]
"""
    if verbosity>0: 
        print("Editing "+ServoDynFile+"\nInserting")
        print(structinputs)
    replacelines(ServoDynFile, cutlines, structinputs)
    return

@register_upgradefunction((3,0), (3,1))
def upgrade_3_0(fstfile, verbosity, **kwargs):
    """
    Upgrade OpenFAST model from 3.0 to 3.1
    """

    # Get the input files
    AeroDynFile   = OpenFAST.getFileFromFST(fstfile, 'AeroFile')
    ServoDynFile  = OpenFAST.getFileFromFST(fstfile, 'ServoFile')
    ElastoDynFile = OpenFAST.getFileFromFST(fstfile, 'EDFile')

    # --- Edit the FST file ---
    environmentfst = """\
0   		       MHK         - MHK turbine type (switch) {0=Not an MHK turbine; 1=Fixed MHK turbine; 2=Floating MHK turbine}
---------------------- ENVIRONMENTAL CONDITIONS --------------------------------
    9.80665   Gravity         - Gravitational acceleration (m/s^2)
      1.225   AirDens         - Air density (kg/m^3)
          0   WtrDens         - Water density (kg/m^3)
  1.464E-05   KinVisc         - Kinematic viscosity of working fluid (m^2/s)
        335   SpdSound        - Speed of sound in working fluid (m/s)
     103500   Patm            - Atmospheric pressure (Pa) [used only for an MHK turbine cavitation check]
       1700   Pvap            - Vapour pressure of working fluid (Pa) [used only for an MHK turbine cavitation check]
          0   WtrDpth         - Water depth (m)
          0   MSL2SWL         - Offset between still-water level and mean sea level (m) [positive upward]
"""
    # Add lines after CompIce (line 20)
    if verbosity>0: 
        print("Editing "+fstfile+"\nInserting")
        print(environmentfst)
    insertlines(fstfile, 20, environmentfst)
    
    # --- Edit the Aerodyn file --- 
    environmentAD = """\
======  Environmental Conditions  ===================================================================
"default"     AirDens            - Air density (kg/m^3)
"default"     KinVisc            - Kinematic air viscosity (m^2/s)
"default"     SpdSound           - Speed of sound (m/s)
"default"     Patm               - Atmospheric pressure (Pa) [used only when CavitCheck=True]
"default"     Pvap               - Vapour pressure of fluid (Pa) [used only when CavitCheck=True]
"""
    # Replace the current environmental condition lines
    if verbosity>0: 
        print("Editing "+AeroDynFile+"\nReplacing lines 15-21 with")
        print(environmentAD)
    cutlines = [15,21]
    replacelines(AeroDynFile, cutlines, environmentAD)

    # --- Edit the ElastoDyn file --- 
    if verbosity>0: 
        print("Editing "+ElastoDynFile+"\nDeleting lines 7-8 with")
    # Remove the environmental lines from ElastoDyn
    deletelines(ElastoDynFile, [7, 8])

    # --- Edit the ServoDyn file ---  
    flowcontroltxt = """\
---------------------- AERODYNAMIC FLOW CONTROL --------------------------------
          0   AfCmode      - Airfoil control mode {0: none, 1: cosine wave cycle, 4: user-defined from Simulink/Labview, 5: user-defined from Bladed-style DLL} (switch)
          0   AfC_Mean     - Mean level for cosine cycling or steady value (-) [used only with AfCmode==1]
          0   AfC_Amp      - Amplitude for for cosine cycling of flap signal (-) [used only with AfCmode==1]
          0   AfC_Phase    - Phase relative to the blade azimuth (0 is vertical) for for cosine cycling of flap signal (deg) [used only with AfCmode==1]
"""
    cablecontroltxt = """\
---------------------- CABLE CONTROL -------------------------------------------
          0   CCmode       - Cable control mode {0: none, 4: user-defined from Simulink/Labview, 5: user-defined from Bladed-style DLL} (switch)
"""
    if verbosity>0: 
        print("Editing "+ServoDynFile+"\nInserting")
        print(flowcontroltxt)
        print("AND")
        print(cablecontroltxt)
    # Add cable control lines after SStCfiles (line 68 of old v3.0 file)
    insertlines(ServoDynFile, 68, cablecontroltxt)
    # Add flow control lines after NacYaw (line 59 of old v3.0 file)
    insertlines(ServoDynFile, 59, flowcontroltxt)

    return

@register_upgradefunction((3,1), (3,2))
def upgrade_3_1(fstfile, verbosity, **kwargs):
    """
    Upgrade OpenFAST model from 3.1 to 3.2
    """
    # Nothing needed here
    # Note that there are TurbSim changes in 
    # https://openfast.readthedocs.io/en/main/source/user/api_change.html#openfast-v3-1-0-to-openfast-v3-2-0
    return

@register_upgradefunction((3,2), (3,3))
def upgrade_3_2(fstfile, verbosity, **kwargs):
    """
    Upgrade OpenFAST model from 3.2 to 3.3
    """
    # Nothing needed here
    # Note that there are FAST.Farm and AeroDyn Driver changes
    # https://openfast.readthedocs.io/en/main/source/user/api_change.html#openfast-v3-1-0-to-openfast-v3-2-0
    return

@register_upgradefunction((3,3), (3,4))
def upgrade_3_3(fstfile, verbosity, **kwargs):
    """
    Upgrade OpenFAST model from 3.3 to 3.4
    """
    # See https://openfast.readthedocs.io/en/main/source/user/api_change.html#openfast-v3-3-0-to-openfast-v3-4-0
    # Get the input files
    AeroDynFile   = OpenFAST.getFileFromFST(fstfile, 'AeroFile')

    BouyancyAD="""\
False         Buoyancy           - True Buoyancy - Include buoyancy effects? (flag)
"""

    ExtraAD="""\
====== Hub Properties ============================================================================== [used only when Buoyancy=True]
        1.0   VolHub - Hub volume (m^3)
        0.0   HubCenBx - Hub center of buoyancy x direction offset (m)
====== Nacelle Properties ========================================================================== [used only when Buoyancy=True]
        1.0   VolNac - Nacelle volume (m^3)
0.0 0.0 0.0   NacCenB - Position of nacelle center of buoyancy from yaw bearing in nacelle coordinates (m)
====== Tail fin Aerodynamics ========================================================================
False         TFinAero - Calculate tail fin aerodynamics model (flag)
"None"        TFinFile - Input file for tail fin aerodynamics [used only when TFinAero=True]
"""
    TwrCb = " 0.0E+00"

    # Get NumTwrNds
    NumTwrNds = int(OpenFAST.getVarFromFST(AeroDynFile, 'NumTwrNds'))

    with open(AeroDynFile) as file:
        ADlines = [line.rstrip() for line in file]
    ADfirstword = [x.strip().split()[0].lower() for x in ADlines if len(x.strip().split())>0]

    # --- Remove RtAeroCp and RtAeroCt from OutList ---
    OutList = OpenFAST.getVarFromFST(AeroDynFile, 'OutList')
    # Make new OutList
    newoutlist = ' '.join(OutList)
    newoutlist = newoutlist.replace('"','').replace(',','\n')
    newoutlist = newoutlist.replace('RtAeroCp','')
    newoutlist = newoutlist.replace('RtAeroCt','')
    newoutlist = newoutlist.replace('RtAeroFyh','')
    newoutlist = newoutlist.replace('RtAeroFzh','')
    newoutlist = newoutlist.replace('RtAeroPwr','')
    newoutlist = '\n'.join(newoutlist.split())
    newoutlist += '\n'

    l1 = ADfirstword.index('outlist')+2
    l2 = ADfirstword.index('end')
    replacelines(AeroDynFile, [l1, l2], newoutlist)
    if verbosity>0: 
        print("Changing AeroDyn OutList to")
        print(newoutlist)

    # --- Add TwrCb to Tower nodes ---
    #print("NumTwrNds = "+repr(NumTwrNds))
    linestart = ADfirstword.index('twrelev')+3
    linenums = [linestart, linestart+NumTwrNds-1]
    nodelines= extractlines(AeroDynFile, linenums)
    newnodelines = ''
    for x in nodelines: 
        newnodelines += x+TwrCb+'\n'
    replacelines(AeroDynFile, linenums, newnodelines)
    if verbosity>0: 
        print("Changing tower nodes to")
        print(newnodelines)

    # --- Add ExtraAD ---
    line_insert = findlinewith(AeroDynFile, 'ADBlFile(3)')
    insertlines(AeroDynFile, line_insert, ExtraAD)
    if verbosity>0:
        print("ADDED")
        print(ExtraAD)

    # --- Add Bouyancy ---
    insertlines(AeroDynFile, 12, BouyancyAD)
    if verbosity>0:
        print("ADDED")
        print(BouyancyAD)

    return


def upgrade_AD_to_dev(fstfile, verbosity, **kwargs):
    """
    Upgrade OpenFAST AD model to include dev inputs
    """
    # Get the input files
    AeroDynFile   = OpenFAST.getFileFromFST(fstfile, 'AeroFile')

    BouyancyAD="""\
False         Buoyancy           - AD DEV
"""

    ExtraAD="""\
======  TEMP LINE BREAK ==============================================================================
        1.0   VolHub
        0.0   HubCenBx
======  TEMP LINE BREAK ==============================================================================
        1.0   VolNac
0.0 0.0 0.0   NacCenB
======  TEMP LINE BREAK ==============================================================================
False         TFinAero
"None"        TFinFile
"""

    TwrCb = " 0.0E+00"

    # Get NumTwrNds
    NumTwrNds = int(OpenFAST.getVarFromFST(AeroDynFile, 'NumTwrNds'))

    with open(AeroDynFile) as file:
        ADlines = [line.rstrip() for line in file]
    ADfirstword = [x.strip().split()[0].lower() for x in ADlines if len(x.strip().split())>0]

    # --- Remove RtAeroCp and RtAeroCt from OutList ---
    OutList = OpenFAST.getVarFromFST(AeroDynFile, 'OutList')
    # Make new OutList
    newoutlist = ' '.join(OutList)
    newoutlist = newoutlist.replace('"','').replace(',','\n')
    newoutlist = newoutlist.replace('RtAeroCp','')
    newoutlist = newoutlist.replace('RtAeroCt','')
    newoutlist = newoutlist.replace('RtAeroFyh','')
    newoutlist = newoutlist.replace('RtAeroFzh','')
    newoutlist = newoutlist.replace('RtAeroPwr','')
    newoutlist = '\n'.join(newoutlist.split())
    newoutlist += '\n'

    l1 = ADfirstword.index('outlist')+2
    l2 = ADfirstword.index('end')
    replacelines(AeroDynFile, [l1, l2], newoutlist)
    if verbosity>0: 
        print("Changing AeroDyn OutList to")
        print(newoutlist)

    # --- Add TwrCb to Tower nodes ---
    #print("NumTwrNds = "+repr(NumTwrNds))
    linestart = ADfirstword.index('twrelev')+3
    linenums = [linestart, linestart+NumTwrNds-1]
    nodelines= extractlines(AeroDynFile, linenums)
    newnodelines = ''
    for x in nodelines: 
        newnodelines += x+TwrCb+'\n'
    replacelines(AeroDynFile, linenums, newnodelines)
    if verbosity>0: 
        print("Changing tower nodes to")
        print(newnodelines)

    # --- Add ExtraAD ---
    line_insert = findlinewith(AeroDynFile, 'ADBlFile(3)')
    insertlines(AeroDynFile, line_insert, ExtraAD)
    if verbosity>0:
        print("ADDED")
        print(ExtraAD)

    # --- Add Bouyancy ---
    insertlines(AeroDynFile, 12, BouyancyAD)
    if verbosity>0:
        print("ADDED")
        print(BouyancyAD)

    return

def repeatedupgrade(fstfile, initver, targetver, verbosity, maxupgrades=10):
    """
    Perform multiple upgrades on an OpenFAST input file
    """
    numupgrades=0
    curver = initver
    while (numupgrades<maxupgrades) and (curver != targetver):
        # Get the upgrade function for the current version
        upgradefunc = upgradefunctionlist[curver]['func']
        newver      = upgradefunctionlist[curver]['newver']
        print("**** UPGRADING "+fstfile+" TO v%i.%i"%(newver[0], newver[1]))
        upgradefunc(fstfile, verbosity)        
        numupgrades += 1
        curver       = newver
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
        "fstfile",
        help="OpenFAST fst file",
        type=str,
    )
    parser.add_argument('--major', 
                        help="Major version",
                        required=True,
                        default=3)
    parser.add_argument('--minor', 
                        help="Minor version",
                        required=True,
                        default=1)
    parser.add_argument('--ADdev', 
                        help="Upgrade Aerodyn to OpenFAST dev version",
                        default=False,
                        action='store_true')
    parser.add_argument('-v', '--verbose', 
                        action='count', 
                        help="Verbosity level (multiple levels allowed)",
                        default=0)

    # Load the options
    args      = parser.parse_args()
    filename  = args.fstfile
    verbose   = args.verbose
    major     = int(args.major)
    minor     = int(args.minor)
    ADdev     = args.ADdev

    # Get the current version of the input file
    initverinfo, match = findOFversion.findversion(filename, verbosity=verbose)
    #print(initverinfo, match.name)
    print("Current model version: v%i.%i"%(initverinfo['major'], initverinfo['minor']))

    # make sure the model matches a version
    if match != findOFversion.versionmatch.MATCH:
        print("Cannot find OpenFAST version for "+filename)
        sys.exit()

    # make sure the target version is acceptible
    targetversion = {'major':major, 'minor':minor}
    if checkallowedversions(targetversion):
        print("Target version v%i.%i: OK"%(targetversion['major'], targetversion['minor']))
    else:
        print("Target version v%i.%i: NOT OK"%(targetversion['major'], targetversion['minor']))

    #print(upgradefunctionlist)
    #upgrade_2_6(filename, verbose)
    #upgrade_3_0(filename, verbose)
    #testreplacelines()
    #testinsertlines()
    #testdeletelines()
    repeatedupgrade(filename, 
                    verinfo2tuple(initverinfo), verinfo2tuple(targetversion), 
                    verbose)
    
    if ADdev:
        upgrade_AD_to_dev(filename, verbose)

