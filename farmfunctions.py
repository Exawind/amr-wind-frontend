#!/usr/bin/env python
#
# Copyright (c) 2022, Alliance for Sustainable Energy
#
# This software is released under the BSD 3-clause license. See LICENSE file
# for more details.
#
"""
Wind farm functions
"""
import numpy as np
import pandas as pd
import sys, os, csv
import re
import shlex
from collections            import OrderedDict 
try:
    from  tkyamlgui import moretypes
except:
    pass

# Load the right version of StringIO
if sys.version_info[0] < 3: 
    from StringIO import StringIO
    import Tkinter as Tk
    import tkFileDialog as filedialog
else:
    from io import StringIO
    import tkinter as Tk
    from tkinter import filedialog as filedialog

from plotfunctions import plotRectangle
import OpenFASTutil as OpenFAST


# Load UTM library
try:
    import utm
    useutm = True
except:
    useutm = False

# Load ruamel or pyyaml as needed
try:
    import ruamel.yaml as YAML
    yaml = YAML(typ='unsafe', pure=True)
    #print("# Loaded ruamel.yaml")
    useruamel=True
    loaderkwargs = {'Loader':yaml.RoundTripLoader}
    dumperkwargs = {'Dumper':yaml.RoundTripDumper, 'indent':4, 'default_flow_style':False} # 'block_seq_indent':2, 'line_break':0, 'explicit_start':True, 
except:
    import yaml as yaml
    #print("# Loaded yaml")
    useruamel=False
    loaderkwargs = {}
    dumperkwargs = {'default_flow_style':False }

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

# Helper functions to get defaults from a dictionary
getdictval = lambda d, k, default: default[k] if k not in d else d[k]

def loadcsv(f, stringinput=False, reqheaders=None, optheaders=None,
            **kwargs):
    """
    Load the csv input, whether from string or filename
    """
    # Get input f into a string
    if stringinput:
        s = f.lstrip()
    else:
        s = open(f, 'r').read().lstrip()

    # Remove comments 
    cleanstr = re.sub(r'(?m)^ *#.*\n?', '', s)

    # Check for header
    firstline = cleanstr.splitlines()[0]
    firstlinewords = [x.strip().lower() for x in firstline.split(',')]
    if reqheaders is not None:
        hasheader =  all(elem in firstlinewords  for elem in reqheaders)
    else:
        hasheader = False

    ## Check for header
    #sniffer   = csv.Sniffer()
    #hasheader = sniffer.has_header(cleanstr)

    #print("hasheader = %s"%repr(hasheader))
    
    header = None
    colheaders=reqheaders+optheaders
    if hasheader:
        header     = 0
        colheaders = None

    # Convert string to io stream
    finput = StringIO(cleanstr)
    df = pd.read_csv(finput, header=header, names=colheaders, **kwargs)

    # Double check headers to make sure it has the required headers
    if hasheader:
        # first rename the headers
        renamemap = {}
        for x in list(df.columns): renamemap[x] = str(x.strip().lower())
        df.rename(columns=renamemap, inplace=True)
        csvheaders=list(df.columns)
        #csvheaders=[x.strip().lower() for x in list(df.columns)]
        #print(csvheaders)
        for header in reqheaders:
            if header.lower() not in csvheaders:
                print('ERROR: required data column %s not present in data'%(header))
    return df

def dataframe2dict(df, reqheaders, optheaders, dictkeys=[]):
    """
    Convert dataframe to a list of dictionaries
    dictkeys are keys in optheaders which should be parsed as dictionaries
    """
    listdict = []
    for index, row in df.iterrows():
        rowdict = OrderedDict()
        for key in reqheaders:
            rowdict[key] = row[key]
        # Add the optional headers
        for key in optheaders:
            if key in list(df.columns):
                rowdict[key] = parseoptions(row[key]) if key in dictkeys else row[key]  
            else:
                rowdict[key] = None
        listdict.append(rowdict)
    return listdict

def parseoptions(optionstr):
    """
    Splits an option string into a dictionary.  optionstr should be of
    the form "key1:val1 key2:val2"
    """
    sanitizestr = optionstr if isinstance(optionstr, str) else str(optionstr)
    allopts = shlex.split(sanitizestr.replace(';',','))
    optdict = OrderedDict()
    for opt in allopts:
        optsplit = opt.split(':')
        key = optsplit[0]
        if len(optsplit)>1:
            val = optsplit[1]
        else: 
            val = None
        optdict[key] = val
    return optdict

def button_loadcsv(self, filenameinput, csvtextbox):
    """
    Button to load a CSV file
    """
    # Get the filename to load
    csvfile  = self.inputvars[filenameinput].getval()

    # Check if file exists
    if not os.path.isfile(csvfile):
        print("ERROR: %s does not exist"%csvfile)
        return

    # Load the filename and display it in the text box    
    csvstr = open(csvfile, 'r').read().lstrip()
    self.inputvars[csvtextbox].setval(csvstr)
    return

def loadcsv2textbox(self, csvtextbox, csvfile):
    """
    Loads CSV file to a textbox
    """
    # Check if file exists
    if not os.path.isfile(csvfile):
        print("ERROR: %s does not exist"%csvfile)
        return

    # Load the filename and display it in the text box    
    csvstr = open(csvfile, 'r').read().lstrip()
    self.inputvars[csvtextbox].setval(csvstr)
    return

def resetFarmSetup(self):
    """
    Resets all variables with 'farmsetup' in outputdef to their defaults
    """
    for key, var in self.inputvars.items():
        outputkey = 'farmsetup'
        if outputkey in var.outputdef:
            var.setdefault()
    return

# ----------- Functions for refinement zones ---------------
def get_turbProp(self, tdict):
    # Get the default type
    default_type   = self.inputvars['Actuator_default_type'].getval()
    default_type   = None if len(default_type)==0 else default_type
    default_type   = default_type[0] if isinstance(default_type, list) else default_type

    turbtype = default_type if 'Actuator_type' not in tdict else tdict['Actuator_type']
    turbtype = turbtype[0] if isinstance(turbtype, list) else turbtype

    # Get default diameter and HH
    if 'Actuator_%s_rotor_diameter'%turbtype in self.inputvars:
        default_turbD  = self.inputvars['Actuator_%s_rotor_diameter'%turbtype].getval()
    else:
        default_turbD  = 100.0
    if 'Actuator_%s_hub_height'%turbtype in self.inputvars:
        default_hh     = self.inputvars['Actuator_%s_hub_height'%turbtype].getval()
    else:
        default_hh     = 100.0
        
    # Get the real values
    turbhh   = default_hh  if tdict['Actuator_hub_height'] is None else tdict['Actuator_hub_height']
    turbD    = default_turbD if tdict['Actuator_rotor_diameter'] is None else tdict['Actuator_rotor_diameter']
    return turbD, turbhh

def calc_FarmAvgProp(self):
    # Go through all turbines
    allturbines  = self.listboxpopupwindict['listboxactuator']
    alltags      = allturbines.getitemlist()
    keystr       = lambda n, d1, d2: d2.name
    #print(alltags)
    acceptableTurbTypes = ['TurbineFastLine', 'TurbineFastDisk',
                           'UniformCtDisk' , 'JoukowskyDisk']
    Nturbs       = 0
    AvgHH        = 0.0
    AvgTurbD     = 0.0
    AvgCenter    = np.array([0.0, 0.0, 0.0])
    for turb in alltags:
        tdict = allturbines.dumpdict('AMR-Wind', subset=[turb], keyfunc=keystr)
        if tdict['Actuator_type'] in acceptableTurbTypes:
            Nturbs    += 1
            AvgCenter += tdict['Actuator_base_position']
            turbD, turbhh = self.get_turbProp(tdict)
            AvgHH     += turbhh
            AvgTurbD  += turbD
            #AvgHH     += tdict['Actuator_hub_height']
            #AvgTurbD  += tdict['Actuator_rotor_diameter']
            #print(tdict['Actuator_base_position'])
            #print(tdict['Actuator_hub_height'])
        else:
            print('ERROR: '+tdict['Actuator_type']+' is not a recognized disk type')
    if Nturbs>0:
        AvgHH     /= Nturbs
        AvgTurbD  /= Nturbs
        AvgCenter /= Nturbs
    else:
        print("ERROR: calc_FarmAvgProp(): No turbines found to average over.")
    return AvgCenter, AvgTurbD, AvgHH

def refine_calcZone(zonename, zonedict, zonecenter, sx, cx, vx, scale):
    refinedict = {}

    # Get the distances
    upstream   = scale*zonedict['upstream']
    downstream = scale*zonedict['downstream']
    lateral    = scale*zonedict['lateral']
    below      = scale*zonedict['below']
    above      = scale*zonedict['above']

    # Calculate the corner
    corner     = zonecenter - below*vx - upstream*sx - lateral*cx
    axis1      = (upstream+downstream)*sx
    axis2      = 2.0*lateral*cx
    axis3      = (below+above)*vx
    
    # Edit the parameters of the refinement window
    refinedict['tagging_name']         = zonename
    refinedict['tagging_shapes']       = zonename
    refinedict['tagging_type']         = 'GeometryRefinement'
    refinedict['tagging_level']        = zonedict['level']
    refinedict['tagging_geom_type']    = 'box'
    refinedict['tagging_geom_origin']  = list(corner)
    refinedict['tagging_geom_xaxis']   = list(axis1)
    refinedict['tagging_geom_yaxis']   = list(axis2)
    refinedict['tagging_geom_zaxis']   = list(axis3)
    return refinedict

def refine_createZoneForTurbine(self, turbname, turbinedict, zonedict,
                                defaultopt):
    # Get the wind direction
    winddir = self.inputvars['ABL_winddir'].getval()

    # Get the turbine properties
    base_position = np.array(turbinedict['Actuator_base_position'])
    turbD, turbhh = self.get_turbProp(turbinedict)
    turbyaw       = winddir if turbinedict['Actuator_yaw'] is None else turbinedict['Actuator_yaw']

    # Get the zone options
    units   = getdictval(zonedict['options'], 'units', defaultopt).lower()
    orient  = getdictval(zonedict['options'], 'orientation', defaultopt).lower()
    # Set scale and orientation axes
    scale = turbD if units=='diameter' else 1.0
    if orient == 'x':
        streamwise  = np.array([1.0, 0.0, 0.0])
        crossstream = np.array([0.0, 1.0, 0.0])
        vert        = np.array([0.0, 0.0, 1.0])
    elif orient == 'y':
        streamwise  = np.array([0.0, 1.0, 0.0])
        crossstream = np.array([1.0, 0.0, 0.0])
        vert        = np.array([0.0, 0.0, 1.0])        
    elif orient == 'nacdir':
        streamwise, crossstream, vert = self.convert_winddir_to_xy(turbyaw)
    elif isFloat(orient):
        streamwise, crossstream, vert = self.convert_winddir_to_xy(float(orient))
    else:  # Use the wind direction
        streamwise, crossstream, vert = self.convert_winddir_to_xy(winddir)
        
    # Get the name
    zonename = '%s_level_%i_zone'%(turbname, zonedict['level'])

    zonecenter = base_position + turbhh*vert
    refinedict = refine_calcZone(zonename, zonedict, zonecenter, 
                                 streamwise, crossstream, vert, scale)

    return refinedict

def refine_createZoneForFarm(self, zonedict, autofarmcenter, AvgTurbD, AvgHH,
                             defaultopt):
    # Get the wind direction
    winddir = self.inputvars['ABL_winddir'].getval()

    # Get the zone options
    units   = getdictval(zonedict['options'], 'units', defaultopt).lower()
    orient  = getdictval(zonedict['options'], 'orientation', defaultopt).lower()
    # Set scale and orientation axes
    scale   = AvgTurbD if units=='diameter' else 1.0
    if orient == 'x':
        streamwise  = np.array([1.0, 0.0, 0.0])
        crossstream = np.array([0.0, 1.0, 0.0])
        vert        = np.array([0.0, 0.0, 1.0])
    elif orient == 'y':
        streamwise  = np.array([0.0, 1.0, 0.0])
        crossstream = np.array([1.0, 0.0, 0.0])
        vert        = np.array([0.0, 0.0, 1.0])        
    elif orient == 'nacdir':
        print("Zone orientation nacdir not possible for farm zone.")
        print("Using wind direction instead")
        streamwise, crossstream, vert = self.convert_winddir_to_xy(winddir)
    elif isFloat(orient):
        streamwise, crossstream, vert = self.convert_winddir_to_xy(float(orient))
    else:  # Use the wind direction
        streamwise, crossstream, vert = self.convert_winddir_to_xy(winddir)

    # Set the farm center
    center=getdictval(zonedict['options'], 'center', defaultopt).lower()
    if center == 'specified':
        defaultctr = {'centerx':0.0, 'centery':0.0, 'centerz':0.0}
        # Use a specified center location
        centerx = float(getdictval(zonedict['options'], 'centerx', defaultctr))
        centery = float(getdictval(zonedict['options'], 'centery', defaultctr))
        centerz = float(getdictval(zonedict['options'], 'centerz', defaultctr))
        zonecenter = np.array([centerx, centery, centerz])
        if 'name' in zonedict['options']:
            zonename = zonedict['options']['name']
        else:
            zonename = 'Farm_level_%i_center_%.0f_%.0f_%.0f'%(zonedict['level'], centerx, centery, centerz)
    else:
        # Use the farm center
        if self.inputvars['turbines_autocalccenter'].getval() == True:
            usecenter = autofarmcenter
            centerz   = usecenter[2] + AvgHH
        else:
            usecenter = self.inputvars['turbines_farmcenter'].getval()
            centerz   = AvgHH
        zonecenter = np.array([usecenter[0], usecenter[1], centerz])
        # Get the name
        zonename = 'Farm_level_%i_zone'%(zonedict['level'])

    refinedict = refine_calcZone(zonename, zonedict, zonecenter, 
                                 streamwise, crossstream, vert, scale)

    return refinedict

def refine_createAllZones(self):
    """
    Create all of the refinement zones
    """
    # Default dictionary for optional inputs
    defaultopt = {'orientation':'winddir',   # winddir/nacdir/x/y/float
                  'units':'diameter',        # diameter/meter
                  'center':'turbine',        # turbine/farm
                  'applyto':'',              # Act only on specific turbs
              }

    # Get the csv input
    csvstring  = self.inputvars['refine_csvtextbox'].getval()
    reqheaders = ['level', 'upstream', 'downstream', 
                  'lateral', 'below', 'above']
    optheaders = ['options']
    #print(csvstring)
    df         = loadcsv(csvstring, stringinput=True, 
                         reqheaders=reqheaders, optheaders=optheaders)
    alldf = dataframe2dict(df, reqheaders, optheaders, dictkeys=optheaders)
    #for zone in alldf: print(zone['options'])

    # See if any zones are farm-centered
    allcenters = [getdictval(z['options'], 'center', defaultopt).lower() for z in alldf]
    #print(allcenters)
    if 'farm' in allcenters:
        #print("Calculating farm center")
        AvgCenter, AvgTurbD, AvgHH = calc_FarmAvgProp(self)
        #print("AvgCenter = "+repr(AvgCenter))
        #print("AvgTurbD  = "+repr(AvgTurbD))
        #print("AvgHH     = "+repr(AvgHH))
    else:
        AvgHH        = 0.0
        AvgTurbD     = 0.0
        AvgCenter    = np.array([0.0, 0.0, 0.0])

    # Get all turbine properties
    allturbines  = self.listboxpopupwindict['listboxactuator']
    alltags      = allturbines.getitemlist()
    keystr       = lambda n, d1, d2: d2.name

    # Get the wind direction
    self.ABL_calculateWDirWS()

    # Delete all old zones (if necessary)
    if self.inputvars['refine_deleteprev'].getval():
        alltagging  = self.listboxpopupwindict['listboxtagging']
        alltagging.deleteall()

    # Go through all rows and create zones
    for zone in alldf:
        center=getdictval(zone['options'], 'center', defaultopt).lower()
        filterstr=getdictval(zone['options'], 'applyto', defaultopt)
        if center=='turbine':
            # Apply to specific turbines
            if len(filterstr)>0:
                applyturbs = [x for x in alltags if bool(re.search(filterstr, x))]
            else:
                applyturbs = alltags
            for turb in applyturbs:
                tdict = allturbines.dumpdict('AMR-Wind', 
                                             subset=[turb], keyfunc=keystr)
                refinedict = refine_createZoneForTurbine(self, turb, tdict, 
                                                         zone, defaultopt)
                #print(refinedict)
                if refinedict is not None:
                    self.add_tagging(refinedict)
        else:
            # Apply to the farm center
            refinedict = refine_createZoneForFarm(self, zone, AvgCenter, 
                                                  AvgTurbD, AvgHH, defaultopt)
            #print(refinedict)
            if refinedict is not None:
                self.add_tagging(refinedict)

    # Automatically set the max_level value
    self.autoMaxLevel()
    return

# ----------- Functions for wind farm turbines -------------
def convertLatLong(x, y, useutm, coordsys, stoponerror=True):
    """
    Convert lat/long to utm x/y if necessary
    """
    turbx = x
    turby = y
    if coordsys=='latlong':
        if useutm:
            utmxy     = utm.from_latlon(x, y)
            turbx     = utmxy[0]
            turby     = utmxy[1]
        else:
            print("ERROR: UTM conversion not available ")
            if stoponerror: sys.exit(1)
    return turbx, turby


def getTurbAvgCenter(self, turbdf, updatewidget=False, convertlatlong=True):
    """
    Calculate the farm center based on turbine locations
    """
    AvgCenter    = np.array([0.0, 0.0])
    latcol       = 0 # which coordinate is latitude
    longcol      = 1 # which coordinate is longitude

    # Get the average center
    if self.inputvars['turbines_autocalccenter'].getval():
        for turb in turbdf: 
            AvgCenter += np.array([turb['x'], turb['y']])
        AvgCenter    = AvgCenter/len(turbdf)
    else:
        AvgCenter    = self.inputvars['turbines_farmcenter'].getval()

    # Convert from lat/long if necessary
    coordsys = self.inputvars['turbines_coordsys'].getval()
    if coordsys=='latlong' and convertlatlong:
        # Convert AvgCenter to lat/long
        if useutm:
            utmxy     = utm.from_latlon(AvgCenter[latcol], 
                                        AvgCenter[longcol])
            AvgCenter = [utmxy[0], utmxy[1]]
            #utm.from_latlon(row['ylat'], row['xlong'])
        else:
            print("ERROR: UTM conversion not available ")
    if updatewidget:
        self.inputvars['turbines_farmcenter'].setval(AvgCenter, 
                                                     forcechange=True)

    return AvgCenter

def turbines_getAllTurbineTypes(self):
    """
    Get a list of all turbine types from csv input
    """
    reqheaders = ['name', 'x', 'y', 'type', 'yaw', 'hubheight']
    optheaders = ['options']

    # Get the csv input
    csvstring  = self.inputvars['turbines_csvtextbox'].getval()    
    df         = loadcsv(csvstring, stringinput=True, 
                         reqheaders=reqheaders, optheaders=optheaders)
    alldf = dataframe2dict(df, reqheaders, optheaders, dictkeys=optheaders)

    # Get the turbine list
    allturbtypes = self.listboxpopupwindict['listboxturbinetype'].getitemlist()

    # build the list of turbine types
    includedturbtypes = []
    for turb in alldf:
        turbtype = turb['type'].strip()
        if turbtype not in allturbtypes:
            print("ERROR: %s is not in list all turbine types:%s"%(turbtype, 
                                                                   allturbtypes))
            continue
        if (turbtype not in includedturbtypes):
            includedturbtypes.append(turbtype)
    return includedturbtypes

def isInt(s):
    try:
        int(s)
        return True
    except:
        return False

def isFloat(s):
    try:
        float(s)
        return True
    except:
        return False

def convertString(s):
    if isInt(s):   return int(s)
    if isFloat(s): return float(s)
    else:          return s

def extractkeystartingwith(d, key, removeprefix=False):
    """
    Extract all keys from dict that start with key
    """
    returndict = OrderedDict()
    for k, v in d.items():
        if k.startswith(key):
            newkey = k.replace(key, '', 1) if removeprefix else k
            returndict[newkey] = convertString(v)
    return returndict


def turbines_createAllTurbines(self):
    """
    Create all of the turbines from csv input
    """
    # Default dictionary for optional inputs
    defaultopt = {'copyfast':False,          # True/False
              }

    reqheaders = ['name', 'x', 'y', 'type', 'yaw', 'hubheight']
    optheaders = ['options']

    # Get the csv input
    csvstring  = self.inputvars['turbines_csvtextbox'].getval()    
    #print(csvstring)
    df         = loadcsv(csvstring, stringinput=True, 
                         reqheaders=reqheaders, optheaders=optheaders)
    alldf = dataframe2dict(df, reqheaders, optheaders, dictkeys=optheaders)
    #for zone in alldf: print(zone['options'])

    # Get all turbine properties
    allturbines  = self.listboxpopupwindict['listboxactuator']
    alltags      = allturbines.getitemlist()
    keystr       = lambda n, d1, d2: d2.name

    # Calculate the farm center
    AvgCenter = getTurbAvgCenter(self, alldf)    
    #print("AvgCenter = "+repr(AvgCenter))

    createnewdomain = self.inputvars['turbines_createnewdomain'].getval()

    # Set prob_lo/prob_hi/n_cell if necessary
    if createnewdomain:
        # Get the farm domain size
        domainsize   = self.inputvars['turbines_domainsize'].getval()    
        if domainsize is None:
            # WARNING
            print("ERROR: Farm domain size is not valid!")
            return
        if self.inputvars['turbines_freespace'].getval():
            groundoffset = -0.5*domainsize[2]
        else:
            groundoffset = 0.0
        corner1 = [AvgCenter[0] - 0.5*domainsize[0],
                   AvgCenter[1] - 0.5*domainsize[1],
                   0.0+groundoffset]
        corner2 = [AvgCenter[0] + 0.5*domainsize[0],
                   AvgCenter[1] + 0.5*domainsize[1],
                   domainsize[2]+groundoffset]
        self.inputvars['prob_lo'].setval(corner1)
        self.inputvars['prob_hi'].setval(corner2)

        # Set the mesh size (if necessary)
        backgrounddx = self.inputvars['turbines_backgroundmeshsize'].getval()
        if backgrounddx is not None:
            Nx = int(round(domainsize[0]/backgrounddx))
            Ny = int(round(domainsize[1]/backgrounddx))
            Nz = int(round(domainsize[2]/backgrounddx))
            self.inputvars['n_cell'].setval([Nx, Ny, Nz])

    # Make sure to add turbines to simulation
    source_terms = self.inputvars['ICNS_source_terms'].getval()
    if source_terms is None: source_terms = []
    if 'ActuatorForcing' not in source_terms:
        source_terms.append('ActuatorForcing')
        self.inputvars['ICNS_source_terms'].setval(source_terms)
    #print(source_terms)

    # Make sure to add Actuator to icns.physics
    physicsterms = self.inputvars['physics'].getval()
    if physicsterms is None: physicsterms = []
    if 'Actuator' not in physicsterms:
        physicsterms.append('Actuator')
        self.inputvars['physics'].setval(physicsterms)

    # Delete all old turbines (if necessary)
    if self.inputvars['turbines_deleteprev'].getval():
        allturbines.deleteall()

    # Add all turbines
    # Get the turbine list
    allturbinemodels = self.listboxpopupwindict['listboxturbinetype']
    allturbtypes     = allturbinemodels.getitemlist()

    # Get the wind direction
    self.ABL_calculateWDirWS()
    winddir = self.inputvars['ABL_winddir'].getval()

    coordsys = self.inputvars['turbines_coordsys'].getval()
    for turb in alldf:
        turbtype = turb['type'].strip()
        # Check the turbine type
        if turbtype not in allturbtypes:
            print("ERROR: turbine type %s not found for turbine %s"%(turbtype,turb['name']))
            continue

        modelparams = allturbinemodels.dumpdict('AMR-Wind',
                                                subset=[turbtype],
                                                keyfunc=lambda n, d1, d2: d2.name)
        model_zHH   = modelparams['Actuator_hub_height']

        # Set the turbine xy
        turbx, turby = convertLatLong(turb['x'], turb['y'], useutm, coordsys)

        # ==== Set the turbine dictionary ====
        turbdict = self.get_default_actuatordict()
        turbdict['Actuator_name']          = turb['name']

        # Set the hub-height
        try:      # Hub-height specified in CSV
            hubheight = float(turb['hubheight'])
            turbdict['Actuator_hub_height'] = hubheight
        except:   # Hub-height left alone
            hubheight = model_zHH

        turbdict['Actuator_base_position'] = [turbx, turby, hubheight - model_zHH]

        # Set the yaw
        try:
            turbyaw = float(turb['yaw'])
        except:
            turbyaw = winddir
        turbdict['Actuator_yaw'] = turbyaw

        # Get all of the turbine model defaults
        turbdict = self.turbinemodels_applyturbinemodel(turbdict,
                                                        turbtype,
                                                        docopy=True, 
                                                        updatefast=True)

        # Set any AMR-Wind options
        AMRoptions = extractkeystartingwith(turb['options'], 'AMRparam_', removeprefix=True)

        def tryeval(s) :
            try:
                x = eval(s)
            except:
                x = s
            return x

        if bool(AMRoptions):
            for key, val in AMRoptions.items():
                if isinstance(val, str): val = val.replace(';',',')
                turbdict[key] =  tryeval(val)
                print("Setting "+key+" to "+repr(turbdict[key]))

        # Process options if using OpenFAST model
        if turbdict['Actuator_type'] in ['TurbineFastLine', 'TurbineFastDisk']:
            fstfile = turbdict['Actuator_openfast_input_file']
            options = turb['options']
            #print("turbine fst file: "+fstfile)
            #print("turbine options: "+repr(options))
            # Make any edits to the FST file
            FSToptions = extractkeystartingwith(options, 'FSTparam_', removeprefix=True)
            if bool(FSToptions):
                print(FSToptions)
                OpenFAST.editFASTfile(fstfile, FSToptions)
            # Make any edits to AeroDyn
            ADoptions = extractkeystartingwith(options, 'ADparam_', removeprefix=True)
            if bool(ADoptions):
                print(ADoptions)
                ADfile   = OpenFAST.getFileFromFST(fstfile,'AeroFile')
                OpenFAST.editFASTfile(ADfile, ADoptions)
            # Make any edits to ServoDyn
            SDoptions = extractkeystartingwith(options, 'SDparam_', removeprefix=True)
            if bool(SDoptions):
                print(SDoptions)
                SDfile   = OpenFAST.getFileFromFST(fstfile,'ServoFile')
                OpenFAST.editFASTfile(SDfile, SDoptions)
            # Make any edits to ElastoDyn
            EDoptions = extractkeystartingwith(options, 'EDparam_', removeprefix=True)
            if bool(EDoptions):
                print(EDoptions)
                EDfile   = OpenFAST.getFileFromFST(fstfile,'EDFile')
                OpenFAST.editFASTfile(EDfile, EDoptions)

            # Make any edits to HydroDyn
            HDoptions = extractkeystartingwith(options, 'HDparam_', removeprefix=True)
            if bool(HDoptions):
                print(HDoptions)
                HDfile   = OpenFAST.getFileFromFST(fstfile,'HydroFile')
                OpenFAST.editFASTfile(HDfile, HDoptions)

            # Make any edits to MoorDyn
            MDoptions = extractkeystartingwith(options, 'MDparam_', removeprefix=True)
            if bool(MDoptions):
                print(MDoptions)
                MDfile   = OpenFAST.getFileFromFST(fstfile,'MooringFile')
                OpenFAST.editFASTfile(MDfile, MDoptions)

            # Make any edits to DISCON
            DISCONoptions = extractkeystartingwith(options, 'DISCONparam_', removeprefix=True)
            if bool(DISCONoptions):
                print(DISCONoptions)
                SDfile       = OpenFAST.getFileFromFST(fstfile,'ServoFile')
                DISCONfile   = OpenFAST.getFileFromFST(SDfile, 'DLL_InFile')
                OpenFAST.editDISCONfile(DISCONfile, DISCONoptions)


        # Add the turbine to the list
        self.add_turbine(turbdict, verbose=False)

    return

def turbines_previewAllTurbines(self, ax=None):
    """
    Plot all of the turbines from the csv input
    """
    reqheaders = ['name', 'x', 'y', 'type', 'yaw', 'hubheight']
    optheaders = ['options']

    # Get the csv input
    csvstring  = self.inputvars['turbines_csvtextbox'].getval()    
    #print(csvstring)
    df         = loadcsv(csvstring, stringinput=True, 
                         reqheaders=reqheaders, optheaders=optheaders)
    alldf = dataframe2dict(df, reqheaders, optheaders, dictkeys=optheaders)
    #for turb in alldf: print("%10s %f %f %10s"%(turb['name'], turb['x'], turb['y'], turb['type']))

    createnewdomain = self.inputvars['turbines_createnewdomain'].getval()

    # Set prob_lo/prob_hi/n_cell if necessary
    if createnewdomain:
        # Calculate the farm center
        AvgCenter = getTurbAvgCenter(self, alldf)    
        #print("AvgCenter = "+repr(AvgCenter))
            
        # Get the farm domain size
        domainsize   = self.inputvars['turbines_domainsize'].getval()    
        if domainsize is None:
            # WARNING
            print("ERROR: Farm domain size is not valid!")
            return
    else:
        corner1 = self.inputvars['prob_lo'].getval()
        corner2 = self.inputvars['prob_hi'].getval()
        domainsize = np.array(corner2) - np.array(corner1)
        AvgCenter  = (np.array(corner2) + np.array(corner1))*0.5

    # Set the corner points
    corner1 = [AvgCenter[0] - 0.5*domainsize[0],
               AvgCenter[1] - 0.5*domainsize[1],
               0.0]

    corner2 = [AvgCenter[0] + 0.5*domainsize[0],
               AvgCenter[1] + 0.5*domainsize[1],
               domainsize[2]]

    # Clear and resize figure
    if ax is None: ax=self.setupfigax()

    # Do the domain plot first
    ix = 0; xstr='X'
    iy = 1; ystr='Y'
    x1, y1, x2, y2  = plotRectangle(ax, corner1, corner2, ix, iy,
                                    color='gray', ec='k', alpha=0.25)
    # Plot the turbines
    coordsys = self.inputvars['turbines_coordsys'].getval()
    addturbinename = self.inputvars['turbines_plotnames'].getval()
    for turb in alldf: 
        turbx, turby = convertLatLong(turb['x'], turb['y'], useutm, coordsys)

        # plot the point
        ax.plot(turbx, turby, marker='s', color='k', markersize=8)
        if addturbinename:
            ax.text(turbx+50, turby+50, turb['name'],
                    color='r', ha='right', va='top', fontdict={'fontsize':8})

    # --------------------------------
    # Set some plot formatting parameters

    if coordsys=='latlong' or coordsys=='utm':
        # this this for different hemispheres
        xstr, ystr = 'EASTING', 'NORTHING'

    ax.set_xlim([AvgCenter[0]-domainsize[0]*0.55, 
                 AvgCenter[0]+domainsize[0]*0.55])
    ax.set_ylim([AvgCenter[1]-domainsize[1]*0.55, 
                 AvgCenter[1]+domainsize[1]*0.55])

    ax.set_aspect('equal')
    ax.set_xlabel('%s [m]'%xstr)
    ax.set_ylabel('%s [m]'%ystr)
    ax.set_title(r'Wind Farm Preview')
    self.figcanvas.draw()

    return

# ----------- Functions for sample plane creation ----------
def sampling_createDictForTurbine(self, turbname, tdict, pdict, defaultopt):
    """
    Creates a sampling dictionary for turbine-oriented probes
    """
    # Get the wind direction
    winddir = self.inputvars['ABL_winddir'].getval()

    # Get the turbine properties
    base_position = np.array(tdict['Actuator_base_position'])
    turbD, turbhh = self.get_turbProp(tdict)
    turbyaw       = winddir if tdict['Actuator_yaw'] is None else tdict['Actuator_yaw']

    # Get the zone options
    units   = getdictval(pdict['options'], 'units', defaultopt).lower()
    orient  = getdictval(pdict['options'], 'orientation', defaultopt).lower()
    usedx   = getdictval(pdict['options'], 'usedx', defaultopt)
    outputto= getdictval(pdict['options'], 'outputto', defaultopt)
    outputfreq = getdictval(pdict['options'], 'outputfreq', defaultopt)
    outputvars = getdictval(pdict['options'], 'outputvars', defaultopt)
    outputderived = getdictval(pdict['options'], 'outputderived', defaultopt)
    if outputvars is not None:
        outputvars = outputvars.split(',')
        #print('outputvars = '+repr(outputvars))

    # Set scale and orientation axes
    scale = turbD if units=='diameter' else 1.0
    if orient == 'x':
        streamwise  = np.array([1.0, 0.0, 0.0])
        crossstream = np.array([0.0, 1.0, 0.0])
        vert        = np.array([0.0, 0.0, 1.0])
    elif orient == 'y':
        streamwise  = np.array([0.0, 1.0, 0.0])
        crossstream = np.array([1.0, 0.0, 0.0])
        vert        = np.array([0.0, 0.0, 1.0])        
    elif orient == 'nacdir':
        streamwise, crossstream, vert = self.convert_winddir_to_xy(turbyaw)
    elif isFloat(orient):
        streamwise, crossstream, vert = self.convert_winddir_to_xy(float(orient))
    else:  # Use the wind direction
        streamwise, crossstream, vert = self.convert_winddir_to_xy(winddir)

    # Turbine hub center
    hubcenter = base_position + turbhh*vert 
    # # Get the distances
    # upstream   = scale*pdict['upstream']
    # downstream = scale*pdict['downstream']
    # lateral    = scale*pdict['lateral']
    # below      = scale*pdict['below']
    # above      = scale*pdict['above']

    # Get the name and probe type
    probename = '%s_%s'%(turbname, pdict['name'])
    probetype = pdict['type'].lower().strip()

    sampledict = {}
    # Set the output postprocessing object
    if outputto is None:
        sampledict['sampling_outputto'] = self.getPostProSamplingDefault()
    else:
        self.addPostProSamplingObject(outputto,
                                      output_freq=outputfreq,
                                      fields=outputvars,
                                      derived_fields=outputderived)
        sampledict['sampling_outputto'] = [outputto]
    # --- Create centerline sampling probes --- 
    if probetype == 'centerline':
        # Calculate the start, end, and number of points
        upstream   = scale*float(pdict['upstream'])
        downstream = scale*float(pdict['downstream'])
        clstart = hubcenter - upstream*streamwise
        clend   = hubcenter + downstream*streamwise

        # Calculate the grid points
        if usedx is None:
            N1 = int(pdict['n1'])
        else:
            N1 = int(round((upstream+downstream)/(scale*float(usedx))))+1

        # Set up the sampling dict
        sampledict['sampling_name']         = probename
        sampledict['sampling_type']         = 'LineSampler'
        sampledict['sampling_l_num_points'] = N1
        sampledict['sampling_l_start']      = clstart
        sampledict['sampling_l_end']        = clend
    # --- Create rotorplane sampling plane --- 
    elif probetype == 'rotorplane':
        # Calculate the geometry
        upstream   = scale*float(pdict['upstream'])
        #print('below = ['+repr(pdict['below'].strip())+']')
        pdictbelow = repr(pdict['below']).replace("'",'').strip()
        pdictabove = repr(pdict['above']).replace("'",'').strip()
        pdictlateral = repr(pdict['lateral']).replace("'",'').strip()
        below      = 0.5*turbD if len(pdictbelow)<1 else scale*float(pdictbelow)
        above      = 0.5*turbD if len(pdictabove)<1 else scale*float(pdictabove)
        lateral    = 0.5*turbD if len(pdictlateral)<1 else scale*float(pdictlateral)
        origin     = hubcenter - upstream*streamwise
        origin     = origin - lateral*crossstream - below*vert

        # Calculate dimensions
        L1         = 2.0*lateral
        L2         = above + below

        # Calculate the grid points
        if usedx is None:
            N1 = int(pdict['n1'])
            N2 = int(pdict['n2'])
        else:
            N1 = int(round((L1)/(scale*float(usedx))))+1
            N2 = int(round((L2)/(scale*float(usedx))))+1

        # Set up the sampling dict
        sampledict['sampling_name']         = probename
        sampledict['sampling_type']         = 'PlaneSampler'
        sampledict['sampling_p_num_points'] = [N1, N2]
        sampledict['sampling_p_origin']     = origin
        sampledict['sampling_p_axis1']      = crossstream*L1
        sampledict['sampling_p_axis2']      = vert*L2

        # Calculate offsets
        noffsets   = int(getdictval(pdict['options'], 'noffsets', defaultopt))
        if noffsets>0:
            downstream = scale*float(pdict['downstream'])        
            offsetvec  = np.linspace(0, upstream+downstream, noffsets+1)
            offsetstr  = ' '.join([repr(x) for x in offsetvec])
            sampledict['sampling_p_normal']  = streamwise
            sampledict['sampling_p_offset_vector'] = streamwise
            sampledict['sampling_p_offsets'] = offsetstr
    # --- Create hub-height sampling planes --- 
    elif probetype == 'hubheight':
        # Calculate the geometry
        upstream   = scale*float(pdict['upstream'])
        downstream = scale*float(pdict['downstream'])
        lateral    = scale*float(pdict['lateral'])
        below      = scale*float(pdict['below'])

        origin     = hubcenter - upstream*streamwise - below*vert
        origin     = origin - lateral*crossstream

        # Calculate dimensions
        L1         = upstream + downstream
        L2         = 2.0*lateral

        # Calculate the grid points
        if usedx is None:
            N1 = int(pdict['n1'])
            N2 = int(pdict['n2'])
        else:
            N1 = int(round((L1)/(scale*float(usedx))))+1
            N2 = int(round((L2)/(scale*float(usedx))))+1

        # Set up the sampling dict
        sampledict['sampling_name']         = probename
        sampledict['sampling_type']         = 'PlaneSampler'
        sampledict['sampling_p_num_points'] = [N1, N2]
        sampledict['sampling_p_origin']     = origin
        sampledict['sampling_p_axis1']      = L1*streamwise
        sampledict['sampling_p_axis2']      = L2*crossstream

        # Calculate offsets
        noffsets   = int(getdictval(pdict['options'], 'noffsets', defaultopt))
        if noffsets>0:
            above      = scale*float(pdict['above'])
            offsetvec  = np.linspace(0, above+below, noffsets+1)
            offsetstr  = ' '.join([repr(x) for x in offsetvec])
            sampledict['sampling_p_normal']  = vert
            sampledict['sampling_p_offset_vector']  = vert
            sampledict['sampling_p_offsets'] = offsetstr
    # --- Create streamwise sampling planes --- 
    elif probetype == 'streamwise':
        # Calculate the geometry
        upstream   = scale*float(pdict['upstream'])
        downstream = scale*float(pdict['downstream'])
        below      = scale*float(pdict['below'])
        above      = scale*float(pdict['above'])

        origin     = hubcenter - upstream*streamwise - below*vert

        # Calculate dimensions
        L1         = upstream + downstream
        L2         = (above + below)

        # Calculate the grid points
        if usedx is None:
            N1 = int(pdict['n1'])
            N2 = int(pdict['n2'])
        else:
            N1 = int(round((L1)/(scale*float(usedx))))+1
            N2 = int(round((L2)/(scale*float(usedx))))+1

        # Set up the sampling dict
        sampledict['sampling_name']         = probename
        sampledict['sampling_type']         = 'PlaneSampler'
        sampledict['sampling_p_num_points'] = [N1, N2]
        sampledict['sampling_p_origin']     = origin
        sampledict['sampling_p_axis1']      = L1*streamwise
        sampledict['sampling_p_axis2']      = L2*vert

        # Calculate offsets
        noffsets   = int(getdictval(pdict['options'], 'noffsets', defaultopt))
        if noffsets>0:
            lateral    = scale*float(pdict['lateral'])
            offsetvec  = np.linspace(0, lateral, noffsets+1)
            offsetstr  = ' '.join([repr(x) for x in offsetvec])
            sampledict['sampling_p_normal']  = crossstream
            sampledict['sampling_p_offset_vector']  = crossstream
            sampledict['sampling_p_offsets'] = offsetstr        
    else:
        print("ERROR: probetype %s not recognized"%probetype)

    return sampledict

def intersectLinePlane(l0, l, p0, n):
    """
    Returns the intersection of a line and a plane
    https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
    """
    d = np.dot(p0 - l0, n)/np.dot(l, n)
    return d

def intersectLineDomain(l0, l, problo, probhi):
    """
    Returns the forward and backward intersection points
    """
    allplanes = [{'name':'xlo', 'p0':problo, 'n':[1, 0, 0]},
                 {'name':'ylo', 'p0':problo, 'n':[0, 1, 0]},
                 {'name':'zlo', 'p0':problo, 'n':[0, 0, 1]},
                 {'name':'xhi', 'p0':probhi, 'n':[1, 0, 0]},
                 {'name':'yhi', 'p0':probhi, 'n':[0, 1, 0]},
                 {'name':'zhi', 'p0':probhi, 'n':[0, 0, 1]},
                 ]
    lnorm = l/np.linalg.norm(l)
    # intersect line with all planes
    for plane in allplanes:
        plane['d'] = intersectLinePlane(l0, lnorm,
                                        np.array(plane['p0']), np.array(plane['n']))
    # get the closest intersection planes
    alldist = np.array([p['d'] for p in allplanes])
    forwardind  = np.where(alldist > 0.0, alldist, np.inf).argmin()   # smallest positive number
    backwardind = np.where(alldist < 0.0, alldist, -np.inf).argmax()  # smallest negative number
    return alldist[forwardind], alldist[backwardind]

def sampling_createDictForFarm(self, pdict, AvgCenter,
                               AvgTurbD, AvgHH, defaultopt):
    """
    Creates a sampling dictionary for farm-oriented probes
    """
    # Get the wind direction
    winddir = self.inputvars['ABL_winddir'].getval()

    # Get the zone options
    units   = getdictval(pdict['options'], 'units', defaultopt).lower()
    orient  = getdictval(pdict['options'], 'orientation', defaultopt).lower()
    usedx   = getdictval(pdict['options'], 'usedx', defaultopt)
    center  = getdictval(pdict['options'], 'center', defaultopt).lower()
    outputto= getdictval(pdict['options'], 'outputto', defaultopt)
    outputfreq = getdictval(pdict['options'], 'outputfreq', defaultopt)
    outputvars = getdictval(pdict['options'], 'outputvars', defaultopt)
    outputderived = getdictval(pdict['options'], 'outputderived', defaultopt)
    wholedomain= bool(getdictval(pdict['options'], 'wholedomain', defaultopt))
    if outputvars is not None:
        outputvars = outputvars.split(',')
        #print('outputvars = '+repr(outputvars))

    wholedomaineps = 1.0E-4

    # Set scale and orientation axes
    scale   = AvgTurbD if units=='diameter' else 1.0
    if orient == 'x':
        streamwise  = np.array([1.0, 0.0, 0.0])
        crossstream = np.array([0.0, 1.0, 0.0])
        vert        = np.array([0.0, 0.0, 1.0])
    elif orient == 'y':
        streamwise  = np.array([0.0, 1.0, 0.0])
        crossstream = np.array([1.0, 0.0, 0.0])
        vert        = np.array([0.0, 0.0, 1.0])
    elif isFloat(orient):
        streamwise, crossstream, vert = self.convert_winddir_to_xy(float(orient))
    else:  # Use the wind direction
        streamwise, crossstream, vert = self.convert_winddir_to_xy(winddir)

    # Set the farm center
    if center == 'specified':
        # Use the specified center
        defaultctr = {'centerx':0.0, 'centery':0.0, 'centerz':0.0}
        # Use a specified center location
        centerx = float(getdictval(pdict['options'], 'centerx', defaultctr))
        centery = float(getdictval(pdict['options'], 'centery', defaultctr))
        centerz = float(getdictval(pdict['options'], 'centerz', defaultctr))
        probecenter = np.array([centerx, centery, centerz])
    else:
        #print('AvgCenter = '+repr(AvgCenter))
        # Use the farm center
        if self.inputvars['turbines_autocalccenter'].getval() == True:
            usecenter = AvgCenter
            centerz   = usecenter[2] + AvgHH
        else:
            usecenter = self.inputvars['turbines_farmcenter'].getval()
            usecenter.append(0.0)
            centerz   = AvgHH
        probecenter = np.array([usecenter[0], usecenter[1], usecenter[2]+AvgHH])

    # Get the name and probe type
    probename = '%s_%s'%("Farm", pdict['name'])
    probetype = pdict['type'].lower().strip()

    sampledict = {}
    # Set the output postprocessing object
    if outputto is None:
        sampledict['sampling_outputto'] = self.getPostProSamplingDefault()
    else:
        self.addPostProSamplingObject(outputto,
                                      output_freq=outputfreq,
                                      fields=outputvars,
                                      derived_fields=outputderived)
        sampledict['sampling_outputto'] = [outputto]

    # --- Create centerline sampling probes --- 
    if probetype == 'centerline':
        # Calculate the start, end, and number of points
        upstream   = scale*float(pdict['upstream'])
        downstream = scale*float(pdict['downstream'])
        clstart = probecenter - upstream*streamwise
        clend   = probecenter + downstream*streamwise

        # Calculate the grid points
        if usedx is None:
            N1 = int(pdict['n1'])
        else:
            N1 = int(round((upstream+downstream)/(scale*float(usedx))))+1

        # Set up the sampling dict
        sampledict['sampling_name']         = probename
        sampledict['sampling_type']         = 'LineSampler'
        sampledict['sampling_l_num_points'] = N1
        sampledict['sampling_l_start']      = clstart
        sampledict['sampling_l_end']        = clend
    # --- Create hub-height sampling planes --- 
    elif probetype == 'hubheight':
        # Calculate the geometry
        upstream   = scale*float(pdict['upstream'])
        downstream = scale*float(pdict['downstream'])
        lateral    = scale*float(pdict['lateral'])
        below      = scale*float(pdict['below'])

        if wholedomain:
            prob_lo     = self.inputvars['prob_lo'].getval()
            prob_hi     = self.inputvars['prob_hi'].getval()
            streamwise  = np.array([1.0, 0.0, 0.0])
            crossstream = np.array([0.0, 1.0, 0.0])
            vert        = np.array([0.0, 0.0, 1.0])
            origin      = np.array([prob_lo[0]+wholedomaineps, prob_lo[1]+wholedomaineps, probecenter[2]])
            L1          = prob_hi[0] - prob_lo[0] - 2.0*wholedomaineps
            L2          = prob_hi[1] - prob_lo[1] - 2.0*wholedomaineps
        else:
            origin     = probecenter - upstream*streamwise - below*vert
            origin     = origin - lateral*crossstream

            # Calculate dimensions
            L1         = upstream + downstream
            L2         = 2.0*lateral

        # Calculate the grid points
        if usedx is None:
            N1 = int(pdict['n1'])
            N2 = int(pdict['n2'])
        else:
            N1 = int(round((L1)/(scale*float(usedx))))+1
            N2 = int(round((L2)/(scale*float(usedx))))+1

        # Set up the sampling dict
        sampledict['sampling_name']         = probename
        sampledict['sampling_type']         = 'PlaneSampler'
        sampledict['sampling_p_num_points'] = [N1, N2]
        sampledict['sampling_p_origin']     = origin
        sampledict['sampling_p_axis1']      = L1*streamwise
        sampledict['sampling_p_axis2']      = L2*crossstream

        # Calculate offsets
        noffsets   = int(getdictval(pdict['options'], 'noffsets', defaultopt))
        if noffsets>0:
            above      = scale*float(pdict['above'])
            offsetvec  = np.linspace(0, above+below, noffsets+1)
            offsetstr  = ' '.join([repr(x) for x in offsetvec])
            sampledict['sampling_p_normal']  = vert
            sampledict['sampling_p_offset_vector']  = vert
            sampledict['sampling_p_offsets'] = offsetstr
    # --- Create rotorplane sampling plane --- 
    elif probetype == 'rotorplane':
        # Calculate the geometry
        upstream   = scale*float(pdict['upstream'])
        below      = scale*float(pdict['below'])
        above      = scale*float(pdict['above'])
        lateral    = scale*float(pdict['lateral'])

        origin     = probecenter - upstream*streamwise
        origin     = origin - lateral*crossstream - below*vert

        # Calculate dimensions
        L1         = 2.0*lateral
        L2         = (above + below)

        # Calculate the grid points
        if usedx is None:
            N1 = int(pdict['n1'])
            N2 = int(pdict['n2'])
        else:
            N1 = int(round((L1)/(scale*float(usedx))))+1
            N2 = int(round((L2)/(scale*float(usedx))))+1

        # Set up the sampling dict
        sampledict['sampling_name']         = probename
        sampledict['sampling_type']         = 'PlaneSampler'
        sampledict['sampling_p_num_points'] = [N1, N2]
        sampledict['sampling_p_origin']     = origin
        sampledict['sampling_p_axis1']      = L1*crossstream
        sampledict['sampling_p_axis2']      = L2*vert
        # Calculate offsets
        noffsets   = int(getdictval(pdict['options'], 'noffsets', defaultopt))
        if noffsets>0:
            downstream = scale*float(pdict['downstream'])        
            offsetvec  = np.linspace(0, upstream+downstream, noffsets+1)
            offsetstr  = ' '.join([repr(x) for x in offsetvec])
            sampledict['sampling_p_normal']  = streamwise
            sampledict['sampling_p_offset_vector']  = streamwise
            sampledict['sampling_p_offsets'] = offsetstr

    # --- Create streamwise sampling planes --- 
    elif probetype == 'streamwise':
        if wholedomain:
            problo     = self.inputvars['prob_lo'].getval()
            probhi     = self.inputvars['prob_hi'].getval()
            upstream, downstream = intersectLineDomain(probecenter, -streamwise, problo, probhi)
            # Calculate the geometry
            upstream   = np.abs(upstream) - wholedomaineps
            downstream = np.abs(downstream) - wholedomaineps
            below      = probecenter[2] - problo[2] - wholedomaineps
            above      = probhi[2] - probecenter[2] - wholedomaineps
        else:
            # Calculate the geometry
            upstream   = scale*float(pdict['upstream'])
            downstream = scale*float(pdict['downstream'])
            below      = scale*float(pdict['below'])
            above      = scale*float(pdict['above'])

        origin     = probecenter - upstream*streamwise - below*vert

        # Calculate dimensions
        L1         = upstream + downstream
        L2         = (above + below)

        # Calculate the grid points
        if usedx is None:
            N1 = int(pdict['n1'])
            N2 = int(pdict['n2'])
        else:
            N1 = int(round((L1)/(scale*float(usedx))))+1
            N2 = int(round((L2)/(scale*float(usedx))))+1

        # Set up the sampling dict
        sampledict['sampling_name']         = probename
        sampledict['sampling_type']         = 'PlaneSampler'
        sampledict['sampling_p_num_points'] = [N1, N2]
        sampledict['sampling_p_origin']     = origin
        sampledict['sampling_p_axis1']      = L1*streamwise
        sampledict['sampling_p_axis2']      = L2*vert

        # Calculate offsets
        noffsets   = int(getdictval(pdict['options'], 'noffsets', defaultopt))
        if noffsets>0:
            lateral    = scale*float(pdict['lateral'])
            offsetvec  = np.linspace(0, lateral, noffsets+1)
            offsetstr  = ' '.join([repr(x) for x in offsetvec])
            sampledict['sampling_p_normal']  = crossstream
            sampledict['sampling_p_offset_vector']  = crossstream
            sampledict['sampling_p_offsets'] = offsetstr        
    else:
        print("ERROR: probetype %s not recognized for farm centers"%probetype)

    return sampledict

def sampling_createAllProbes(self, verbose=False):
    """
    Create all of the sample probes from csv input
    """
    # Default dictionary for optional inputs
    defaultopt = {'orientation':'winddir',   # winddir/nacdir/x/y/float
                  'units':'diameter',        # diameter/meter
                  'center':'turbine',        # turbine/farm/specified
                  'usedx':None,              # use this mesh size
                  'noffsets':0,              # number of offsets
                  'outputto':None,           # Output to this sampler object
                  'outputfreq':None,         # Output at this frequency
                  'outputvars':None,         # Output these variables
                  'outputderived':None,      # Output these derived field variables
                  'applyto':'',              # Act only on specific turbs
                  'wholedomain':False        # Create this probe across whole domain
                 }

    reqheaders = ['name', 'type', 'upstream', 'downstream', 'lateral', 
                  'below', 'above', 'n1', 'n2']
    optheaders = ['options']

    # Get the csv input
    csvstring  = self.inputvars['sampling_csvtextbox'].getval()    
    df         = loadcsv(csvstring, stringinput=True, 
                         reqheaders=reqheaders, optheaders=optheaders)
    alldf = dataframe2dict(df, reqheaders, optheaders, dictkeys=optheaders)

    # DEPRECATED, REMOVE IN FUTURE
    # # Make sure to add sampling to the outputs
    # ppro_items = self.inputvars['post_processing'].getval()
    # if 'sampling' not in ppro_items:
    #     ppro_items.append('sampling')
    #     self.inputvars['post_processing'].setval(ppro_items)
    #     self.inputvars['post_processing'].onoffctrlelem(None)
    
    # Get all turbine properties
    allturbines  = self.listboxpopupwindict['listboxactuator']
    alltags      = allturbines.getitemlist()
    keystr       = lambda n, d1, d2: d2.name

    # See if any zones are farm-centered
    allcenters = [getdictval(z['options'], 'center', defaultopt).lower() for z in alldf]
    #print(allcenters)
    if 'farm' in allcenters:
        AvgCenter, AvgTurbD, AvgHH = calc_FarmAvgProp(self)
    else:
        AvgHH        = 100.0
        AvgTurbD     = 100.0
        AvgCenter    = np.array([0.0, 0.0, 0.0])

    # Get the wind direction
    self.ABL_calculateWDirWS()

    # Delete all old zones (if necessary)
    if self.inputvars['sampling_deleteprev'].getval():
        allsampling  = self.listboxpopupwindict['listboxsampling']
        allsampling.deleteall()

    # Go through all rows and create sampling probes
    for probe in alldf:
        center = getdictval(probe['options'], 'center', defaultopt).lower()
        filterstr = getdictval(probe['options'], 'applyto', defaultopt)
        if center=='turbine':
            # Apply to specific turbines
            if len(filterstr)>0:
                applyturbs = [x for x in alltags if bool(re.search(filterstr, x))]
            else:
                applyturbs = alltags
            for turb in applyturbs:
                tdict = allturbines.dumpdict('AMR-Wind', 
                                             subset=[turb], keyfunc=keystr)
                sampledict = sampling_createDictForTurbine(self, turb, tdict,
                                                           probe, defaultopt)
                if verbose: print(sampledict)
                if sampledict is not None:
                    self.add_sampling(sampledict, verbose=verbose)
        elif (center=='farm') or (center=='specified'):
            sampledict = sampling_createDictForFarm(self, probe, 
                                                    AvgCenter, AvgTurbD, AvgHH,
                                                    defaultopt)
            if verbose: print(sampledict)
            if sampledict is not None:
                self.add_sampling(sampledict, verbose=verbose)
        else:
            print("ERROR: option center=%s not recognized"%center)
    return

# ----------- Functions related to parameter sweeps ----
def sweep_setBCTable(self, inflow='mass_inflow', outflow='pressure_outflow'):
    """
    Sets the boundary condition table for all wind direction angles
    Figures out which sides should be periodic and which are inflow/outflow
    """
    eps      = 1.0E-5
    BCtable  = [ 
        # mindir, maxdir,  xper,  yper,    xlo,     xhi,     ylo,     yhi
        [  0-eps,   0+eps, True,   False,  None,    None,    outflow, inflow ],
        [  0+eps,  90-eps, False,  False,  outflow, inflow,  outflow, inflow ],
        [ 90-eps,  90+eps, False,  True,   outflow, inflow,  None,    None ],
        [ 90+eps, 180-eps, False,  False,  outflow, inflow,  inflow,  outflow ],
        [180-eps, 180+eps, True,   False,  None,    None,    inflow,  outflow ],
        [180+eps, 270-eps, False,  False,  inflow,  outflow, inflow,  outflow ],
        [270-eps, 270+eps, False,  True,   inflow,  outflow, None,    None ],
        [270+eps, 360-eps, False,  False,  inflow,  outflow, outflow, inflow ],
    ]
    return BCtable

def sweep_findBCinTable(self, WDir, table):
    for entry in table:
        if (entry[0]<=WDir) and (WDir <= entry[1]): return entry
    print('Cannot find BC entry corresponding to WDir: '+repr(WDir))
    sys.exit(1)  # Problem if it gets to here
    return

def setBC(self, bcsurf, bctype, velocity, density):
    if bctype is None: return
    if bctype == 'pressure_outflow':
        self.setAMRWindInput(bcsurf+'.type',     'pressure_outflow')
        self.setAMRWindInput(bcsurf+'.density',  None)
        self.setAMRWindInput(bcsurf+'.velocity', [None, None, None])
    if bctype == 'mass_inflow':
        self.setAMRWindInput(bcsurf+'.type',     'mass_inflow')
        self.setAMRWindInput(bcsurf+'.density',  density)
        self.setAMRWindInput(bcsurf+'.velocity', velocity)

def sweep_SetupRunParamSweep(self, preSetupFunc=None, postSetupFunc=None, verbose=False):
    """
    Set up and run a parameter sweep 
    """
    str2list = lambda strinput: [float(x) for x in strinput.replace(',',' ').replace(';', ' ').split()]

    # Get the wind speed and direction lists    
    try: 
        WSlist   = str2list(self.inputvars['sweep_windspeeds'].getval())
    except:
        print("ERROR: error in sweep_windspeeds")
        return
    try:
        WDirlist = str2list(self.inputvars['sweep_winddirs'].getval())
    except:
        print("ERROR: error in sweep_winddirs")
        return
        
    # Get the case name 
    casename_template = self.inputvars['sweep_caseprefix'].getval()
    # Add an index to the prefix if necessary
    if ('{' not in casename_template) and ('}' not in casename_template):
        casename_template += '_{CASENUM}'

    # Get the directory 
    dirname_template = self.inputvars['sweep_dirprefix'].getval()
    # Add an index to the prefix if necessary
    if ('{' not in dirname_template) and ('}' not in dirname_template):
        dirname_template += '_{CASENUM}'

    # Add any formatting necessary to string
    fmtstr = lambda x, y: x.format(**y)

    # Construct the list of runs
    runlist = []
    # Add in however many loops are necessary here
    for WS in WSlist:
        for WDir in WDirlist:
            runlist.append({'WS':WS, 'WDir':WDir})

    
    # Get the current directory
    cwd = os.getcwd()
    
    # Print the header
    if verbose:
        print("%10s %12s %12s %20s"%("NUM", "WS", "WDir", "Case name"))
        print("%10s %12s %12s %20s"%("---", "--", "----", "---------"))

    # Loop over all wind speeds and directions
    for icase, case in enumerate(runlist):
        WS   = case['WS']
        WDir = case['WDir']

        # Run the user-defined pre setup function
        namevars={'WS':WS, 'WDir':WDir, 'CASENUM':icase}
        casevars={}
        if preSetupFunc is not None:
            casevars=preSetupFunc(self, icase)
        namevars.update(casevars)
        if 'WDir' in casevars: WDir = casevars['WDir']
        if 'WS'   in casevars: WS   = casevars['WS']

        casename = fmtstr(casename_template, namevars)
        if verbose:
            print("%10i %12.5f %12.5f %20s"%(icase, WS, WDir, casename))
        
        # Create the directories if necessary
        if self.inputvars['sweep_usenewdirs'].getval():
            dirname = fmtstr(dirname_template, namevars)
            if not os.path.exists(dirname): os.makedirs(dirname)
            os.chdir(dirname)
            case['dir'] = dirname
        else:
            case['dir'] = cwd

        # Set the WS/WDir
        self.inputvars['ABL_windspeed'].setval(WS, forcechange=True)
        self.inputvars['ABL_winddir'].setval(WDir, forcechange=True)
        self.ABL_calculateWindVector()

        velocity = self.inputvars['ABL_velocity'].getval()
        density  = self.inputvars['density'].getval()

        inflowmode=self.inputvars['sweep_inflowmode'].getval()
        if inflowmode=='uniform':
            self.inputvars['ConstValue_velocity'].setval(velocity, forcechange=True)

        # Set up the boundary conditions
        if inflowmode=='uniform':
            BCtable = sweep_setBCTable(self)
            BCentry = sweep_findBCinTable(self, WDir, BCtable)
            self.setAMRWindInput('is_periodicx', BCentry[2])
            self.setAMRWindInput('is_periodicy', BCentry[3])
            setBC(self, 'xlo', BCentry[4], velocity, density)
            setBC(self, 'xhi', BCentry[5], velocity, density)
            setBC(self, 'ylo', BCentry[6], velocity, density)
            setBC(self, 'yhi', BCentry[7], velocity, density)
        else:
            print("ERROR: boundary conditions for %s not yet implemented"%inflowmode)
        # Set up the turbines
        turbines_createAllTurbines(self)
        
        # Set up the refinement regions
        refine_createAllZones(self)

        # Set up the sampling probes
        sampling_createAllProbes(self)

        # Run the user-defined post setup function
        if postSetupFunc is not None:
            postSetupFunc(self, icase)

        # Write the AMR-Wind input file
        AMRinputfile = casename+".inp"
        self.writeAMRWindInput(AMRinputfile)

        # Write the submission file
        writesubmit = self.inputvars['sweep_createsubmitscript'].getval()
        dosubmit    = self.inputvars['sweep_submitjob'].getval()
        if writesubmit:
            AMRsubmitfile = casename+".sh"
            self.submitscript_savescript(scriptfilename=AMRsubmitfile, 
                                         submit=dosubmit)

        case['submitted'] = dosubmit
        

        # Update the run list
        case['casename']  = casename
        case['inputfile'] = AMRinputfile

        # Done creating the case, get back to cwd
        if self.inputvars['sweep_usenewdirs'].getval():
            os.chdir(cwd)

    # Write out the log file list of runs
    logfilename = self.inputvars['sweep_logfile'].getval()
    if logfilename:        
        logfile=sys.stdout if logfilename.strip()=='sys.stdout' else open(logfilename, 'w')
        for run in runlist:
            dumpdict = {run['casename']:run}
            yaml.dump(dumpdict, logfile, **dumperkwargs)
        if logfile != sys.stdout:
            logfile.close()
        
    return

# ----------- Functions related to I/O  ----------------
def writeFarmSetupYAML(self, filename, verbose=True):
    """
    Write out the farm setup parameters into a YAML file
    """

    # Embed the turbine types if needed
    embedturbinetype = self.inputvars['farm_embedturbinetype'].getval()
    embedturbinedict = {}
    if embedturbinetype:
        turbtypelist = turbines_getAllTurbineTypes(self)
        #print(turbtypelist)
        for turbtype in turbtypelist:
            allturbinemodels = self.listboxpopupwindict['listboxturbinetype']
            for k, g in allturbinemodels.alldataentries.items():
                if k==turbtype:
                    embedturbinedict[k] = g

    # Embed the AMR-Wind input file if needed
    amrwindinput = ''
    embedamrwindinput = self.inputvars['farm_embedamrwindinput'].getval()
    if embedamrwindinput:
        amrwindinput = self.writeAMRWindInput('')
    formattedinput = ''
    for line in amrwindinput.splitlines(): formattedinput += line.rstrip()+'\n'
    self.inputvars['wfarm_embedamrwindinput'].setval(formattedinput)

    # Get all farmsetup inputs into a dict
    inputdict = dict(self.getDictFromInputs('farmsetup', onlyactive=False))

    # Get the help dict
    helpdict = self.getHelpFromInputs('farmsetup', 'help', onlyactive=False)

    if embedturbinetype:
        inputdict['wfarm_embedturbinetype'] = embedturbinedict

    if useruamel: 
        inputdict = comseq(self.getDictFromInputs('farmsetup', onlyactive=False))
        if embedturbinetype:
            inputdict['wfarm_embedturbinetype'] = embedturbinedict

        yaml.scalarstring.walk_tree(inputdict)
        for k,v in inputdict.items():
            if k in helpdict:
                if self.inputvars[k].inputtype == moretypes.textbox:
                    if hasattr(yaml.comments.CommentedMap, "yaml_set_comment_before_after_key"):
                        inputdict.yaml_set_comment_before_after_key(k, before=helpdict[k])
                elif helpdict[k].startswith('##'):
                    if hasattr(yaml.comments.CommentedMap, "yaml_set_comment_before_after_key"):
                        inputdict.yaml_set_comment_before_after_key(k, before="\n ")
                        inputdict.yaml_add_eol_comment(helpdict[k], k, column=40)
                else:
                    inputdict.yaml_add_eol_comment(helpdict[k], k, column=40)

    # Open the file and write it
    outfile = sys.stdout if filename == sys.stdout else open(filename, 'w')
    # Write out the header comment
    outfile.write("# ----- BEGIN Farm setup input file ----\n")
    yaml.dump(inputdict, outfile, 
              **dumperkwargs)
    outfile.write("# ----- END Farm setup input file ------\n")
    if filename != sys.stdout: 
        outfile.close()
        print("Saved farm setup to %s"%filename)
    return

def loadFarmSetupYAML(self, loadfile, stringinput=False):
    """
    Load the farm setup from a YAML file
    """
    if useruamel: Loader=yaml.load
    else:         Loader=yaml.safe_load
    if stringinput:
        yamldict = Loader(loadfile, **loaderkwargs)
    else:
        # Check if file exists
        if not os.path.isfile(loadfile):
            print("ERROR: %s does not exist"%loadfile)
            return
        # Load the file
        with open(loadfile, 'r') as fp:
            yamldict = Loader(fp, **loaderkwargs)
        print("Loaded farm setup from %s"%loadfile)

    if 'wfarm_embedturbinetype' in yamldict:
        embedturbdict = yamldict['wfarm_embedturbinetype']
        #print(embedturbdict)
        if self.inputvars['farm_embedturbinetype'].getval():
            allturbinemodels = self.listboxpopupwindict['listboxturbinetype']
            turbmodellist = allturbinemodels.getitemlist()
            # #print(turbmodellist)
            # # Delete any existing turbine models with the same name
            # for k, g in embedturbdict.items():
            #     if k in turbmodellist:
            #         print("pop "+k)
            #         allturbinemodels.alldataentries.pop(k)
            # Load the turbine type
            allturbinemodels.populatefromdict(embedturbdict,
                                              deleteprevious=True, 
                                              verbose=False,
                                              forcechange=True)
            #print(allturbinemodels.getitemlist())
        # Delete the key from the yamldict
        yamldict.pop('wfarm_embedturbinetype', None)


    # Set the values of each variable
    for key, val in yamldict.items():
        self.inputvars[key].setval(val, forcechange=True)

    # Load the embedded AMR-Wind inputs
    if self.inputvars['farm_loadembedamrwindinput'].getval():
        embeddedinput=self.inputvars['wfarm_embedamrwindinput'].getval()
        self.loadAMRWindInput(embeddedinput, string=True)
    return

def button_saveFarmSetupYAML(self):
    """
    Button to save the farm setup
    """
    farmfile  = self.inputvars['farm_setupfile'].getval()
    kwargs = {'filetypes':[("YAML files","*.yaml *.yml"), 
                           ("all files","*.*")]}
    if len(farmfile)==0:
        initialdir = './'
        initialfile= None
        #print('Blank farm setup file provided.  Cannot save.')
        #return
    else:
        initialdir  = os.path.dirname(farmfile)
        initialfile = os.path.basename(farmfile)
    if farmfile=='sys.stdout': 
        farmfile=sys.stdout
    else:
        farmfile  = filedialog.asksaveasfilename(initialdir  = initialdir,
                                                 initialfile = initialfile,
                                                 title = "Save as farm YAML file",
                                                 **kwargs)
    if isinstance(farmfile, str) and len(farmfile)==0:
        return
    if isinstance(farmfile, str): 
        self.inputvars['farm_setupfile'].setval(farmfile)

    self.writeFarmSetupYAML(farmfile)
    return

def button_loadFarmSetupYAML(self):
    """
    Button to load the farm setup
    """
    farmfile  = self.inputvars['farm_setupfile'].getval()
    # Check if file exists
    if not os.path.isfile(farmfile):
        kwargs = {'filetypes':[("YAML files","*.yaml *.yml"), 
                               ("all files","*.*")]}
        farmfile  = filedialog.askopenfilename(initialdir = "./",
                                               title = "Select farm YAML file",
                                               **kwargs)
        #print("ERROR: %s does not exist"%farmfile)
        #return
    if len(farmfile)==0:
        return
    self.inputvars['farm_setupfile'].setval(farmfile)
    self.loadFarmSetupYAML(farmfile, stringinput=False)

    return

def button_clearSetupYAMLfile(self):
    """
    Button to load the farm setup
    """
    self.inputvars['farm_setupfile'].setval('')

# ------------------------------------------------------

def runtest1():
    testdata="""
#
# This is comment 1
 A, b, C, d, e
 1, 2, 3, 4, x:1 b:2
5, 6, 7, 8, xy:2;4;5
# This is comment 2
 1, 2, 3, 4, 
  5, 6, 7, 8, ad
"""
    testdatanohead="""
1, 2, 3, 4
5, 6, 7, 8
1, 2, 3, 4
5, 6, 7, 8
"""

    reqheaders = ['a', 'b', 'c', 'd']
    optheaders = ['e']
    #df = loadcsv(testdata, stringinput=True)
    df = loadcsv(testdatanohead, stringinput=True, 
                 reqheaders=reqheaders, optheaders=optheaders)
    #df = loadcsv('csvtest.csv', stringinput=False)
    print(df)
    print("Test parseoptions:")
    print(parseoptions("key1:val1 key2:3 key3:0.01"))
    print("test dataframe2dict")
    for k in dataframe2dict(df, reqheaders, optheaders, dictkeys=['e']):
        print(k)

    #loadFarmSetupYAML(testinp, stringinput=True)    

if __name__ == "__main__":
    runtest1()
    #runtest2()
    #inputdict = {'a':'apple', 'b':'boy'}
    #yaml.dump(dict(inputdict), sys.stdout, default_flow_style=False)
