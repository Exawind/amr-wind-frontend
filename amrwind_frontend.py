#!/usr/bin/env python
#
# Copyright (c) 2022, Alliance for Sustainable Energy
#
# This software is released under the BSD 3-clause license. See LICENSE file
# for more details.
#

import sys, os, re, shutil
# import the tkyamlgui library
scriptpath=os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, scriptpath+'/tkyamlgui')
sys.path.insert(1, scriptpath)

import numpy as np
from functools import partial
import tkyamlgui as tkyg
import postproamrwindabl    as postpro
import postproamrwindsample as ppsample

if sys.version_info[0] < 3:
    import Tkinter as Tk
    import tkFileDialog as filedialog
else:
    import tkinter as Tk
    from tkinter import filedialog as filedialog

import traceback

# Try the loading the Xvfb package
import platform
hasxvfb=False
if platform.system()=='Linux':
    try:
        from xvfbwrapper import Xvfb
        hasxvfb=True
    except:
        hasxvfb=False
        
    
import numpy as np
from collections            import OrderedDict 
from matplotlib.collections import PatchCollection
from matplotlib.patches     import Rectangle
import matplotlib.pyplot    as plt
from matplotlib.lines       import Line2D
import argparse
import subprocess
import signal
import copy

# amrwind-frontend libraries
import validateinputs
import OpenFASTutil as OpenFAST
#from plotfunctions import readCartBoxFile, plotRectangle, plot3DBox, rotatepoint, plotTurbine


# -------------------------------------------------------------
class MyApp(tkyg.App, object):
    def __init__(self, *args, **kwargs):
        self.abl_stats = None
        self.sample_ncdat = None
        super(MyApp, self).__init__(*args, **kwargs)
        self.fig.clf()
        self.fig.text(0.35,0.5,'Welcome to\nAMR-Wind')
        self.formatgridrows()
        self.extradictparams = OrderedDict()
        self.abl_profiledata = {}
        self.savefile        = ''

        # for fast output plotting
        self.fast_outdata    = None
        self.fast_outfiles   = None
        self.fast_headers    = None
        self.fast_units      = None

        # variables for local run
        self.localrun_process= None

        # build the map going from AMR-Wind var --> amrwind_frontend var
        self.amrkeydict      = OrderedDict()
        for key, var in self.inputvars.items():
            outputkey = 'AMR-Wind'
            if outputkey in var.outputdef:
                self.amrkeydict[var.outputdef[outputkey]] = key

        # Load any turbines
        self.turbinemodels_populate()

        # Define alias functions for get_default_* dicts
        self.get_default_samplingdict = \
            self.listboxpopupwindict['listboxsampling'].getdefaultdict
        self.get_default_taggingdict = \
            self.listboxpopupwindict['listboxtagging'].getdefaultdict
        self.get_default_turbinetypedict = \
            self.listboxpopupwindict['listboxturbinetype'].getdefaultdict
        self.get_default_actuatordict = \
            self.listboxpopupwindict['listboxactuator'].getdefaultdict

        # Shorthand aliases to add things
        self.add_turbine  = partial(self.add_populatefromdict,'listboxactuator')
        self.add_sampling = partial(self.add_populatefromdict,'listboxsampling')
        self.add_tagging  = partial(self.add_populatefromdict,'listboxtagging')

        # Shorthand aliases to edit things
        self.edit_turbine  = partial(self.edit_entryval, 'listboxactuator')
        self.edit_sampling = partial(self.edit_entryval, 'listboxsampling')
        self.edit_tagging  = partial(self.edit_entryval, 'listboxtagging')

        # Shorthand aliases to load CSV files
        from farmfunctions import loadcsv2textbox
        self.loadTurbineCSVFile = partial(loadcsv2textbox, self,
                                          'turbines_csvtextbox')
        self.loadRefineCSVFile  = partial(loadcsv2textbox, self,
                                          'refine_csvtextbox')

        self.report_callback_exception = self.showerror
        return

    def showerror(self, *args):
        err = traceback.format_exception(*args)
        
    # Used to define alias for populatefromdict()
    def add_populatefromdict(self, key, d, **kwargs):
        deleteprevious = False
        if 'deleteprevious' in kwargs:
            deleteprevious = kwargs['deleteprevious']
            del kwargs['deleteprevious']
        self.listboxpopupwindict[key].populatefromdict({'x':d}, 
                                                       deleteprevious=deleteprevious,
                                                       **kwargs)
        return

    # Used to define alias for populatefromdict()
    def edit_entryval(self, listboxkey, entry, key, val):
        self.listboxpopupwindict[listboxkey].setentryval(entry, key, val,
                                                         outputtag='AMR-Wind')
        return    

    @classmethod
    def init_nogui(cls, *args, **kwargs):
        localconfigdir=os.path.join(scriptpath,'local')
        if 'localconfigdir' in kwargs:
            localconfigdir=kwargs['localconfigdir']
            del kwargs['localconfigdir']
        if hasxvfb:
            try:
                vdisplay = Xvfb()
                vdisplay.start()
            except:
                pass
        return cls(configyaml=os.path.join(scriptpath,'config.yaml'), 
                   localconfigdir=localconfigdir, scriptpath=scriptpath,
                   withdraw=True, **kwargs)

    def reloadconfig(self):
        with open('config.yaml') as fp:
            if tkyg.useruemel: Loader=tkyg.yaml.load
            else:              Loader=tkyg.yaml.safe_load
            self.yamldict = Loader(fp)
        #print('Reloaded config')
        for listboxdict in self.yamldict['listboxpopupwindows']:
            frame  = self.tabframeselector(listboxdict)
            name   = listboxdict['name']
            popupdict = self.yamldict['popupwindow'][listboxdict['popupinput']]
            self.listboxpopupwindict[name] = tkyg.listboxpopupwindows(self, frame, listboxdict, popupdict)
        return

    @classmethod
    def ifbool(cls, x):
        if not isinstance(x, bool): return x
        return 'true' if x else 'false'

    def tellMeAbout(self, name):
        """
        Query an AMR-Wind input or AMR-Wind frontend input
        """
        inputkey = None
        if name in self.amrkeydict:
            inputkey = self.amrkeydict[name]
        if name in self.inputvars:
            inputkey = name
        # Check if it can't find anything
        if inputkey is None:
            print("Unknown input "+name)
            return
        # Set inputs
        inputvar = self.inputvars[inputkey]
        infodict = OrderedDict()
        infodict['Internal name'] = inputvar.name
        infodict['AMR-Wind name'] = str(None) if 'AMR-Wind' not in inputvar.outputdef else inputvar.outputdef['AMR-Wind']
        infodict['Help'] = str(None) if 'help' not in inputvar.outputdef else inputvar.outputdef['help']
        infodict['Variable type'] = str(inputvar.inputtype)
        infodict['GUI Label']     = inputvar.label
        infodict['Default value'] = str(inputvar.defaultval)
        infodict['Option list']   = str(inputvar.optionlist) if inputvar.optionlist else None

        for k, g in infodict.items(): print('%-20s: %s'%(k,g))
        return

    def setAMRWindInput(self, name, val, updatectrlelem=True, **kwargs):
        """
        Use this function to set the AMR-Wind keyword name to value.
        """
        try:
            if name in self.amrkeydict:
                inputkey = self.amrkeydict[name]
            if name in self.inputvars:
                inputkey = name
            self.inputvars[inputkey].setval(val, **kwargs)

            if updatectrlelem and (self.inputvars[inputkey].ctrlelem is not None):
                self.inputvars[inputkey].linkctrlelem(self.subframes, 
                                                      self.inputvars)
                self.inputvars[inputkey].onoffctrlelem(None)

        except:
            print("Cannot set "+name)
        return

    def getAMRWindInput(self, name, **kwargs):
        """
        Use this function to get the value of an AMR-Wind keyword.
        """
        returnval = None
        try:
            if name in self.amrkeydict:
                inputkey = self.amrkeydict[name]
            if name in self.inputvars:
                inputkey = name
            returnval = self.inputvars[inputkey].getval(**kwargs)
        except:
            print("Cannot get value of "+name)
        return returnval

    def getAMRWindInputType(self, name):
        """
        Use this function to get the input type of an AMR-Wind keyword.
        """
        returnval = None
        try:
            if name in self.amrkeydict:
                inputkey = self.amrkeydict[name]
            if name in self.inputvars:
                inputkey = name
            returnval = self.inputvars[inputkey].inputtype
        except:
            print("Cannot get inputtype of "+name)
        return returnval


    def getTaggingKey(self, keyname, listboxdict, datadict):
        keyheader = 'tagging.'
        intername = ''
        #getinputtype = lambda l,n: [x['inputtype'] for x in l if x['name']==n]
        if datadict.name.startswith('tagging_geom'):
            #print(keyname+datadict.outputdef['AMR-Wind']+" needs fixing!")
            keynamedict = self.listboxpopupwindict['listboxtagging'].dumpdict('AMR-Wind', subset=[keyname])
            if keyname+".shapes" in keynamedict:
                intername=keynamedict[keyname+".shapes"].strip()+"."
            #print(listboxdict)
        return keyheader+keyname+"."+intername+datadict.outputdef['AMR-Wind']

    def writeAMRWindInput(self, filename, verbose=False, 
                          outputextraparams=True, comments=True):
        """
        Write out the input file for AMR-Wind
        TODO: Do more sophisticated output control later
        """
        self.postLoad_SetOnOffCtrlElem()

        inputdict = self.getDictFromInputs('AMR-Wind')

        # Get the sampling outputs
        samplingkey = lambda n, d1, d2: d1['outputprefix']['AMR-Wind']+'.'+n+'.'+d2.outputdef['AMR-Wind']
        sampledict  = self.listboxpopupwindict['listboxsampling'].dumpdict('AMR-Wind', keyfunc=samplingkey)
        
        taggingdict = self.listboxpopupwindict['listboxtagging'].dumpdict('AMR-Wind', keyfunc=self.getTaggingKey)

        actuatordict= self.listboxpopupwindict['listboxactuator'].dumpdict('AMR-Wind', keyfunc=samplingkey)

        # Construct the output dict
        outputdict=inputdict.copy()
        if len(sampledict)>0:
            commentdict = {'#comment_sampledict':'\n#---- sample defs ----'}
            if comments: outputdict.update(commentdict)
            outputdict.update(sampledict)
        if len(taggingdict)>0:
            commentdict = {'#comment_taggingdict':'\n#---- tagging defs ----'}
            if comments:  outputdict.update(commentdict)
            outputdict.update(taggingdict)
        if len(actuatordict)>0:
            commentdict = {'#comment_actuatordict':'\n#---- actuator defs ----'}
            if comments:  outputdict.update(commentdict)
            outputdict.update(actuatordict)

        # Add any extra parameters
        if outputextraparams:
            commentdict = {'#comment_extradict':'\n#---- extra params ----'}
            if comments:  outputdict.update(commentdict)
            outputdict.update(self.extradictparams)

        # Add the end comment
        if comments: 
            outputdict.update({'#comment_end':'#== END AMR-WIND INPUT =='})

        # Get the help dict
        helpdict = self.getHelpFromInputs('AMR-Wind', 'help')

        # Convert the dictionary to string output
        returnstr = ''
        if len(filename)>0:  f=open(filename, "w")
        for key, val in outputdict.items():
            outputkey = key
            # Write out a comment
            if (key[0] == '#') and comments:
                try:
                    writestr = bytes(val, "utf-8").decode("unicode_escape")
                except:
                    writestr = val.decode('string_escape')
                if verbose: print(writestr)
                if len(filename)>0: f.write(writestr+"\n")
                returnstr += writestr+"\n"
                continue
            elif (key[0] == '#'):
                continue
            # convert val to string
            if val is None:
                continue
            elif isinstance(val, list):
                outputstr=' '.join([str(self.ifbool(x)) for x in val])
            else:
                outputstr=str(self.ifbool(val))
            if len(outputstr)>0 and (outputstr != 'None'):
                writestr = "%-40s = %-20s"%(outputkey, outputstr)
                # Add any help to this
                if comments and (outputkey in helpdict):
                    writestr += "# "+helpdict[outputkey]
                if verbose: print(writestr)
                if len(filename)>0: f.write(writestr+"\n")
                returnstr += writestr+"\n"
        if len(filename)>0:     
            f.close()
            self.savefile = filename
        
        return returnstr

    def writeAMRWindInputGUI(self):
        filename  = filedialog.asksaveasfilename(initialdir = "./",
                                                 title = "Save AMR-Wind file",
                                                 filetypes=[("input files","*.inp"),
                                                            ("all files","*.*")])
        if len(filename)>0:
            self.writeAMRWindInput(filename)
            self.savefile = filename
        return

    def saveAMRWindInputGUI(self):
        if len(self.savefile)>0:
            self.writeAMRWindInput(self.savefile)
            print("Saved "+self.savefile)
        else:
            self.writeAMRWindInputGUI()
        return

    def dumpAMRWindInputGUI(self):
        return tkyg.messagewindow(self, self.writeAMRWindInput(''), 
                                  height=40, title='Preview Input File')

    def showyamlmesg(self, helpkey, category='helpwindows'):
        """
        Displays a help message in yamldict[category][helpkey]
        """
        mesg = self.yamldict[category][helpkey]['mesg']
        opts = tkyg.getdictval(self.yamldict['helpwindows'][helpkey], 
                               'options', {})
        return tkyg.messagewindow(self, mesg, **opts)

    def getInputHelp(self, search=''):
        # Print the header
        print("%-40s %-40s %-10s %s"%("INTERNAL NAME", "AMRWIND PARAMETER",
                                      "DEFAULT VAL", "/ DESCRIPTION"))
        # Print each widget
        for widget in self.yamldict['inputwidgets']:
            appname  = widget['name']
            amrname  = '' 
            default  = '' if 'defaultval' not in widget else repr(widget['defaultval'])
            helpstr  = '' if 'help' not in widget else repr(widget['help'])
            if 'outputdef' in widget:
                if 'AMR-Wind' in widget['outputdef']:
                    amrname = widget['outputdef']['AMR-Wind']
            # Combine and search if necessary
            allstrings = appname+' '+default+' '+helpstr+' '+amrname
            hassearchterm = False
            if ((len(search)==0) or (search.upper() in allstrings.upper())):
                hassearchterm = True
            #printentry = True if len(search)==0 else False
            if hassearchterm:
                print("%-40s %-40s %-10s %s"%(appname, amrname, default, helpstr))
        return

    @classmethod
    def processline(cls, inputline):
        line = inputline.partition('#')[0]
        line = line.rstrip()
        if len(line)>0:
            line = line.split('=')
            key  = line[0].strip()
            data = line[1].strip()
            return key, data
        return None, None

    @classmethod
    def AMRWindStringToDict(cls, string):
        returndict = OrderedDict()
        for line in string.split('\n'):
            key, data = cls.processline(line)
            if key is not None: returndict[key] = data
        return returndict
            
    @classmethod
    def AMRWindInputToDict(cls, filename):
        returndict = OrderedDict()
        with open(filename) as f:
            for line in f:
                key, data = cls.processline(line)
                if key is not None: returndict[key] = data
        return returndict

    @classmethod
    def AMRWindExtractSampleDict(cls, inputdict, template, sep=[".","."]):
        """
        From input dict, extract all of the sampling probe parameters
        """
        pre='sampling'
        dictkeypre= 'sampling_'
        samplingdict = OrderedDict()
        if pre+'.labels' not in inputdict: return samplingdict, inputdict
        extradict = inputdict.copy()

        # Create the markers for probe/plane/line sampler
        lmap = {}
        lmap['probesampler'] = 'pf_'
        lmap['linesampler']  = 'l_'
        lmap['planesampler'] = 'p_'

        # Get the sampling labels
        allkeys = [key for key, item in inputdict.items()]
        samplingkeys = [key for key in allkeys if key.lower().startswith(pre)]
        samplingnames = inputdict[pre+'.labels'].strip().split()
        extradict.pop(pre+'.labels')

        getinputtype = lambda l,n: [x['inputtype'] for x in l if x['name']==n]
        matchlisttype = lambda x, l: x.split() if isinstance(l, list) else x
        #print(getinputtype(template['inputwidgets'], 'sampling_p_offsets')[0])

        for name in samplingnames:
            probedict=OrderedDict()
            # Process all keys for name
            prefix    = pre+sep[0]+name+sep[1]
            probekeys = [k for k in samplingkeys if k.startswith(prefix) ]
            # First process the type
            probetype = tkyg.getdictval(inputdict, prefix+'type', None)
            l = lmap[probetype.lower()]
            if probetype is None: 
                print("ERROR: %s is not found!"%prefix+'type')
                continue
            probedict[dictkeypre+'name']  = name
            probedict[dictkeypre+'type']  = probetype
            # Remove them from the list & dict
            probekeys.remove(prefix+'type')
            extradict.pop(prefix+'type')

            # Go through the rest of the keys
            for key in probekeys:
                suffix = key[len(prefix):]
                probedictkey = dictkeypre+l+suffix
                # Check what kind of data it's supposed to provide
                inputtype=getinputtype(template['inputwidgets'], probedictkey)[0]
                data = matchlisttype(inputdict[key], inputtype)
                probedict[probedictkey] = data
                extradict.pop(key)
            samplingdict[name] = probedict.copy()

        #print(samplingdict)
        return samplingdict, extradict

    @classmethod
    def AMRWindExtractTaggingDict(cls, inputdict, template, sep=['.','.','.']):
        """
        From input dict, extract all of the tagging parameters
        """
        pre='tagging'
        dictkeypre= 'tagging_'
        taggingdict = OrderedDict()
        if pre+'.labels' not in inputdict: return taggingdict, inputdict
        extradict = inputdict.copy()
        
        # Get the tagging labels
        allkeys = [key for key, item in inputdict.items()]
        taggingkeys = [key for key in allkeys if key.lower().startswith(pre)]
        taggingnames = inputdict[pre+'.labels'].strip().split()
        extradict.pop(pre+'.labels')

        # This lambda returns the inputtype corresponding to entry in
        # dictionary l with name n
        getinputtype = lambda l,n: [x['inputtype'] for x in l if x['name']==n]
        matchlisttype = lambda x, l: x.split() if isinstance(l, list) else x
        for name in taggingnames:
            itemdict = OrderedDict()
            # Process all keys for name
            prefix      = pre+sep[0]+name+sep[1]

            tagkeys = [k for k in taggingkeys if k.startswith(prefix) ]
            # First process the type
            tagtype = tkyg.getdictval(inputdict, prefix+'type', None)
            if tagtype is None: 
                print("ERROR: %s is not found!"%prefix+'type')
                continue
            itemdict[dictkeypre+'name']  = name
            itemdict[dictkeypre+'type']  = tagtype
            #print("Found tagging: %s"%name)
            #print("      type: %s"%tagtype)
            # Remove them from the list & dict
            tagkeys.remove(prefix+'type')
            extradict.pop(prefix+'type')

            readtaggingkeys = True
            taginsert       = ''
            if tagtype=='GeometryRefinement':
                # Get the level
                suffix    = 'level'
                tagdictkey= dictkeypre+suffix     
                key       = prefix+suffix
                if key in inputdict:
                    leveldata = int(inputdict[key])
                    itemdict[tagdictkey] = leveldata
                    extradict.pop(key)
                
                # Get the shapes
                suffix = 'shapes'
                tagdictkey = dictkeypre+suffix     
                key       = prefix+suffix
                shapedata = inputdict[key]
                #print("shapedata = %s"%shapedata)
                extradict.pop(key)
                itemdict[tagdictkey] = shapedata
                if (len(shapedata.strip().split())>1):
                    print(" ERROR: More than one geometric refinement shape ")
                    print(" ERROR: Can't handle that at the moment")
                    readtaggingkeys = False
                else:
                    prefix  = pre+sep[0]+name+sep[1]+shapedata.strip()+sep[2]
                    tagkeys = [k for k in taggingkeys if k.startswith(prefix) ]
                    taginsert = 'geom_'

            if not readtaggingkeys: continue
            #print(taggingkeys)
            #print(tagkeys)
            # Go through the rest of the keys
            for key in tagkeys:
                suffix = key[len(prefix):]
                tagdictkey = dictkeypre+taginsert+suffix
                #print(tagdictkey)
                # Check what kind of data it's supposed to provide
                inputtype=getinputtype(template['inputwidgets'], tagdictkey)[0]
                data = matchlisttype(inputdict[key], inputtype)
                itemdict[tagdictkey] = data
                extradict.pop(key)
            taggingdict[name] = itemdict.copy()
        return taggingdict, extradict

    def AMRWindExtractActuatorDict(self, inputdict, template, sep=['.','.','.']):
        """
        From input dict, extract all of the sampling probe parameters
        """
        pre='Actuator'
        dictkeypre= 'Actuator_'
        actuatordict = OrderedDict()
        if pre+'.labels' not in inputdict: return actuatordict, inputdict
        extradict = inputdict.copy()

        # Get the Actuator labels
        allkeys = [key for key, item in inputdict.items()]
        actuatorkeys = [key for key in allkeys if key.lower().startswith(pre.lower())]
        actuatornames = inputdict[pre+'.labels'].strip().split()
        extradict.pop(pre+'.labels')
        #print(actuatorkeys)
        getinputtype = lambda l,n: [x['inputtype'] for x in l if x['name']==n]
        matchlisttype = lambda x, l: x.split() if isinstance(l, list) else x
        #print(getinputtype(template['inputwidgets'], 'sampling_p_offsets')[0])

        for name in actuatornames:
            probedict=OrderedDict()
            # Process all keys for name
            prefix    = pre+sep[0]+name+sep[1]
            probekeys = [k for k in actuatorkeys if k.startswith(prefix) ]
            # First process the type
            probetype = tkyg.getdictval(inputdict, prefix+'type', None)
            l = '' #lmap[probetype.lower()]
            if probetype is None:
                probetype = self.inputvars['Actuator_default_type'].getval()
                probetype = str(probetype[0])
            else:
                probekeys.remove(prefix+'type')
                extradict.pop(prefix+'type')
            probedict[dictkeypre+'name']  = name
            probedict[dictkeypre+'type']  = probetype
            # Remove them from the list & dict
            # Go through the rest of the keys
            for key in probekeys:
                suffix = key[len(prefix):]
                probedictkey = dictkeypre+l+suffix
                #print(probedictkey)
                # Check what kind of data it's supposed to provide
                inputtype=getinputtype(template['inputwidgets'], probedictkey)[0]
                data = matchlisttype(inputdict[key], inputtype)
                probedict[probedictkey] = data
                extradict.pop(key)
            #print(probedict)
            actuatordict[name] = probedict.copy()

        #print(samplingdict)
        return actuatordict, extradict

    def setInternalVars(self):
        """
        Set any internal variables necessary after loading AMR-Wind inputs
        """
        # Set the time control variables
        if self.inputvars['fixed_dt'].getval() > 0.0:
            self.inputvars['time_control'].setval(['const dt'])
        else:
            self.inputvars['time_control'].setval(['max cfl'])            
        # Set the ABL boundary plane variables
        if self.inputvars['ABL_bndry_io_mode'].getval() != "-1":
            self.inputvars['ABL_useboundaryplane'].setval(True)
        return

    def postLoad_SetOnOffCtrlElem(self):
        self.inputvars['zlo_type'].onoffctrlelem(None)
        self.inputvars['zlo_temperature_type'].onoffctrlelem(None)
        return

    def loadAMRWindInput(self, filename, string=False, printunused=False):
        if string:
            amrdict=self.AMRWindStringToDict(filename)
        else:
            amrdict=self.AMRWindInputToDict(filename)
        extradict=self.setinputfromdict('AMR-Wind', amrdict)

        # Input the sampling probes
        samplingdict, extradict = \
            self.AMRWindExtractSampleDict(extradict, 
            self.yamldict['popupwindow']['sampling'])
        self.listboxpopupwindict['listboxsampling'].populatefromdict(samplingdict, forcechange=True)
        # Input the tagging/refinement zones
        taggingdict, extradict = \
            self.AMRWindExtractTaggingDict(extradict, 
            self.yamldict['popupwindow']['tagging'])
        self.listboxpopupwindict['listboxtagging'].populatefromdict(taggingdict, forcechange=True)

        # Input the turbine actuators
        actuatordict, extradict = \
            self.AMRWindExtractActuatorDict(extradict, 
            self.yamldict['popupwindow']['turbine'])
        self.listboxpopupwindict['listboxactuator'].populatefromdict(actuatordict, forcechange=True)
        
        if printunused and len(extradict)>0:
            print("# -- Unused variables: -- ")
            for key, data in extradict.items():
                print("%-40s= %s"%(key, data))

        # Set the internal variables
        self.setInternalVars()

        # link any widgets necessary
        for key,  inputvar in self.inputvars.items():
            if self.inputvars[key].ctrlelem is not None:
                self.inputvars[key].onoffctrlelem(None)

        self.postLoad_SetOnOffCtrlElem()

        self.extradictparams = extradict
        return extradict

    def loadAMRWindInputGUI(self):
        kwargs = {'filetypes':[("Input files","*.inp *.i"), ("all files","*.*")]}
        filename  = filedialog.askopenfilename(initialdir = "./",
                                               title = "Select AMR-Wind file",
                                               **kwargs)
        if len(filename)>0:
            self.loadAMRWindInput(filename, printunused=True)
            self.savefile = filename
        return

    def donothing_button(self):
        print("Does nothing")
        return

    def menubar(self, root):
        """ 
        Adds a menu bar to root
        See https://www.tutorialspoint.com/python/tk_menu.htm
        """
        menubar  = Tk.Menu(root)

        # File menu
        filemenu = Tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Save input file", 
                             command=self.saveAMRWindInputGUI)
        filemenu.add_command(label="Save input file As", 
                             command=self.writeAMRWindInputGUI)
        filemenu.add_command(label="Import AMR-Wind file", 
                             command=self.loadAMRWindInputGUI)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        # Plot menu
        plotmenu = Tk.Menu(menubar, tearoff=0)
        plotmenu.add_command(label="Plot domain", 
                             command=partial(self.launchpopupwin, 
                                             'plotdomain', savebutton=False))
        plotmenu.add_command(label="FAST outputs", 
                             command=partial(self.launchpopupwin, 
                                             'plotfastout', savebutton=False))
        menubar.add_cascade(label="Plot", menu=plotmenu)

        # run menu
        runmenu = Tk.Menu(menubar, tearoff=0)
        runmenu.add_command(label="Check Inputs", 
                            command=self.validateGUI)
        runmenu.add_command(label="Estimate mesh size", 
                            command=self.estimateMeshSize)
        runmenu.add_command(label="Preview Input File", 
                            command=self.dumpAMRWindInputGUI)
        runmenu.add_command(label="Local Run", 
                             command=partial(self.launchpopupwin, 
                                             'localrun', savebutton=False))
        runmenu.add_command(label="Job Submission", 
                             command=partial(self.launchpopupwin, 
                                             'submitscript', savebutton=False))

        menubar.add_cascade(label="Run", menu=runmenu)        

        # Help menu
        help_text = """This is AMR-Wind!"""
        helpmenu = Tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Help Index", 
                             command=partial(tkyg.donothing, root))
        helpmenu.add_command(label="About...", 
                             command=partial(tkyg.messagewindow, root,
                                             help_text))
        menubar.add_cascade(label="Help", menu=helpmenu)
        
        root.config(menu=menubar)
        return

    def validateGUI(self, **kwargs):
        result = self.validate(**kwargs)
        return tkyg.messagewindow(self, result, height=25)

    def validate(self, printeverything=True):
        # Load validateinputs plugins
        num_nonactive = 0
        num_active    = 0
        outputstr     = ""
        def printcat(x): print(x); return x+"\n"
        outputstr += printcat("-- Checking inputs --")
        resultclass = OrderedDict()
        for c in validateinputs.CheckStatus: resultclass[c.name] = []
        for p in validateinputs.pluginlist:
            active = True if "active" not in vars(p) else p.active
            if active:
                num_active = num_active+1
                results = p().check(self)
                for r in results:
                    resultclass[r['result'].name].append(r)
                    if printeverything:
                        outputstr += \
                            printcat("[%5s] %-20s %s"%(r['result'].name,
                                                       p.name+":"+r['subname'],
                                                       r['mesg']))
            else:
                num_nonactive = num_nonactive+1            
        outputstr += printcat('')
        outputstr += printcat("Results: ")
        for k, g in resultclass.items():
            outputstr += printcat(' %i %s'%(len(g), k))
        return outputstr
    
    def setupfigax(self, clear=True, subplot=111):
        # Clear and resize figure
        canvaswidget=self.figcanvas.get_tk_widget()
        w,h1 = self.winfo_width(), self.winfo_height()
        canvaswidget.configure(width=w-self.leftframew-10, height=h1-75)
        self.fig.clf()
        ax=self.fig.add_subplot(subplot)
        if clear: ax.clear()
        return ax

    from plotfunctions import plotDomain, readCartBoxFile, plotGenericProfile

    # import all wind farm functions
    from farmfunctions import button_loadcsv
    from farmfunctions import button_saveFarmSetupYAML, button_loadFarmSetupYAML
    from farmfunctions import button_clearSetupYAMLfile
    from farmfunctions import resetFarmSetup
    from farmfunctions import writeFarmSetupYAML, loadFarmSetupYAML
    from farmfunctions import refine_createAllZones, calc_FarmAvgProp, get_turbProp
    from farmfunctions import turbines_createAllTurbines, turbines_previewAllTurbines
    from farmfunctions import sampling_createAllProbes
    from farmfunctions import sweep_SetupRunParamSweep

    def getMaxLevel(self):
        max_level = 0
        # Search refinement levels
        allrefinements = self.listboxpopupwindict['listboxtagging']
        alltags        = allrefinements.getitemlist()

        #print(alltags)
        for tag in alltags:
            pdict = allrefinements.dumpdict('AMR-Wind',
                                            subset=[tag],
                                            keyfunc=lambda n, d1, d2: d2.name)
            # Handle the Cartesian Box Refinements
            if pdict['tagging_type'][0]=='CartBoxRefinement':
                filename = pdict['tagging_static_refinement_def']
                # Load the boxes
                allboxes = self.readCartBoxFile(filename)
                if len(allboxes)>max_level: max_level = len(allboxes)

            if pdict['tagging_type'][0]=='GeometryRefinement':
                if pdict['tagging_geom_type'][0]=='box':
                    ilevel = pdict['tagging_level']
                    if (ilevel is not None) and ilevel+1>max_level: 
                        max_level = ilevel+1
        return max_level

    def autoMaxLevel(self):
        """
        Automatically set the maximum refinement level
        """
        max_level = self.getMaxLevel()
        self.inputvars['max_level'].setval(max_level)
        return

    def estimateMeshSize(self, verbose=True, **kwargs):
        # Get the domain size
        prob_lo   = self.inputvars['prob_lo'].getval()
        prob_hi   = self.inputvars['prob_hi'].getval()

        # Get the level 0 mesh size/dx
        max_level = self.inputvars['max_level'].getval()
        n_cell    = self.inputvars['n_cell'].getval()
        
        # Get the level 0 volume and dx
        level0_Lx = np.array(prob_hi) - np.array(prob_lo)

        # Calculate the cell dimensions at every level
        level_dx = []
        level_dx.append(np.array([level0_Lx[0]/n_cell[0],
                                  level0_Lx[1]/n_cell[1], 
                                  level0_Lx[2]/n_cell[2],
                                  ]))
        for l in range(max_level):
            prevdx = level_dx[-1]
            level_dx.append(prevdx/2)

        #for dx in level_dx: print(dx)

        # Calculate the cell volumes at every level
        level_cellv=[dx[0]*dx[1]*dx[2] for dx in level_dx]            

        # -- Calculate the number of cells at every level --
        level_ncells = np.zeros(max_level+1)
        # At level 0
        level_ncells[0] = n_cell[0]*n_cell[1]*n_cell[2]

        # Search refinement levels
        allrefinements = self.listboxpopupwindict['listboxtagging']
        alltags        = allrefinements.getitemlist()

        #print(alltags)
        for tag in alltags:
            pdict = allrefinements.dumpdict('AMR-Wind',
                                            subset=[tag],
                                            keyfunc=lambda n, d1, d2: d2.name)
            #print(pdict)
            if pdict['tagging_type'][0]=='GeometryRefinement':
                if pdict['tagging_geom_type'][0]=='box':
                    origin = pdict['tagging_geom_origin']
                    xaxis  = pdict['tagging_geom_xaxis']
                    yaxis  = pdict['tagging_geom_yaxis']
                    zaxis  = pdict['tagging_geom_zaxis']
                    ilevel = pdict['tagging_level']
                    if ilevel+1 <= max_level:
                        # Calculate the volume
                        vol    = np.abs(np.dot(np.cross(xaxis, yaxis), zaxis))
                        ncells = int(vol/level_cellv[ilevel+1])
                        level_ncells[ilevel+1] += ncells
                        if verbose:
                            print("Refinement %s: level %i: %i cells"%(tag, 
                                                                       ilevel+1,
                                                                       ncells))
                    else:
                        # Refinement not applied
                        print("Refinement %s ignored. Level %i cells, max level %i"%(tag, ilevel+1, max_level)) 

                if pdict['tagging_geom_type'][0]=='cylinder':
                    cylstart  = pdict['tagging_geom_start']
                    cylend    = pdict['tagging_geom_end']
                    outerR    = pdict['tagging_geom_outer_radius']
                    innerR    = pdict['tagging_geom_inner_radius']
                    ilevel    = pdict['tagging_level']
                    if ilevel+1 <= max_level:
                        # Calculate the volume
                        cylL   = np.linalg.norm(np.array(cylend) - 
                                                np.array(cylstart))
                        vol    = np.pi*(outerR**2 - innerR**2)*cylL
                        ncells = int(vol/level_cellv[ilevel+1])
                        level_ncells[ilevel+1] += ncells
                        if verbose:
                            print("Refinement %s: level %i: %i cells"%(tag, 
                                                                       ilevel+1,
                                                                       ncells))
                    else:
                        # Refinement not applied
                        print("Refinement %s ignored. Level %i cells, max level %i"%(tag, ilevel+1, max_level)) 

                        
            # Handle the Cartesian Box Refinements
            if pdict['tagging_type'][0]=='CartBoxRefinement':
                filename = pdict['tagging_static_refinement_def']
                # Load the boxes
                allboxes = self.readCartBoxFile(filename)
                for ilevel, boxlevel in enumerate(allboxes):
                    for ibox, box in enumerate(boxlevel):
                        corner1 = np.array(box[0:3])
                        corner2 = np.array(box[3:6])
                        # Calculate the volume
                        boxdL   = corner2-corner1
                        boxV    = boxdL[0]*boxdL[1]*boxdL[2]
                        ncells  = int(boxV/level_cellv[ilevel+1])
                        level_ncells[ilevel+1] += ncells
                        if verbose:
                            print(" box %i level %i: %i cells"%(ibox, ilevel+1,
                                                                ncells))

        #for k, g in allrefinedicts.items(): print(k)
        # Print a summary of Ncells at each level
        print("ESTIMATED MESH SIZE")
        print("%8s %12s %30s"%("Level", "Ncells", "Cell Size"))
        for l, ncells in enumerate(level_ncells):
            print("%8i %12i %30s"%(l, ncells, 
                                   " x ".join([str(x) for x in level_dx[l]])))
        print("  TOTAL: %12i"%(sum(level_ncells)))

        return

    # ---- plot FAST outputs ---
    def FAST_addoutfilesGUI(self, window):
        files  = filedialog.askopenfilename(initialdir = "./",
                                            title = "Select FAST out file",
                                            multiple = True,
                                            filetypes = 
                                            [("FAST out files", "*.out"), 
                                            ("all files","*.*")])
        # Add files to the file listbox
        if window is not None:
            tkentry = window.temp_inputvars['plotfastout_files'].tkentry  
            for f in list(files): tkentry.insert(Tk.END, os.path.relpath(f))


    def FAST_loadallfiles(self, window, outfile=None):
        plotparams = self.popup_storteddata['plotfastout']
        if (outfile is not None) and isinstance(outfile, str):
            outfile = [outfile]
        if outfile is None: 
            outfile = window.temp_inputvars['plotfastout_files'].tkentry.get(0, Tk.END)
        self.fast_outdata = []
        self.fast_outfiles = []
        for f in list(outfile):
            print("Loading "+f)
            dat, headers, units = OpenFAST.loadoutfile(f)
            # To do: check to make sure headers line up
            self.fast_outdata.append(dat)
            self.fast_outfiles.append(f)
        self.fast_headers = headers
        self.fast_units   = units
        # Set the variables to plot
        if window is not None:
            tkentry = window.temp_inputvars['plotfastout_vars'].tkentry
            tkentry.delete(0, Tk.END)
            for h in self.fast_headers[1:]: tkentry.insert(Tk.END, h)
        # Set the file selectors
        if window is not None:
            N = len(window.temp_inputvars['plotfastout_files'].tkentry.get(0, Tk.END))
            for i in range(N):
                window.temp_inputvars['plotfastout_files'].tkentry.selection_set(i)
        return headers

    def FAST_plotoutputs(self, window=None, ax=None):
        # Clear and resize figure
        if ax is None: ax=self.setupfigax()

        plotparams = self.popup_storteddata['plotfastout']

        if window is None: 
            datindices = range(len(self.fast_outdata))
        else:
            allfiles  = window.temp_inputvars['plotfastout_files'].tkentry.get(0, Tk.END)
            plotfiles = window.temp_inputvars['plotfastout_files'].getval()
            datindices = [allfiles.index(x) for x in plotfiles]

        for i in datindices:
            dat = self.fast_outdata[i]
            for v in plotparams['plotfastout_vars']:
                indx = self.fast_headers.index(v)
                ax.plot(dat[:,0], dat[:,indx], 
                        label=self.fast_outfiles[i]+': '+v+' '+self.fast_units[indx])

        ax.legend(fontsize=8)
        ax.set_xlabel(self.fast_headers[0]+' '+self.fast_units[0])
        self.figcanvas.draw()
        return

    # ---- Check North/East orientations ---
    def check_NE_orthogonal(self, tol=1.0E-6):
        north   = self.inputvars['north_vector'].getval()
        east    = self.inputvars['east_vector'].getval()
        NdotE   = np.dot(np.array(north), np.array(east))
        result  = True if np.abs(NdotE)<tol else False
        return result

    def check_NE_onXYplane(self, tol=1.0E-6):
        north   = self.inputvars['north_vector'].getval()
        east    = self.inputvars['east_vector'].getval()
        Z       = np.array([0, 0, 1.0])
        NdotZ   = np.dot(np.array(north), Z)
        EdotZ   = np.dot(np.array(east), Z)
        if np.abs(NdotZ)<tol and np.abs(EdotZ)<tol:
            return True
        else:
            return False
    
    def get_N_angle_to_Y(self, tol=1.0E-10):
        north   = np.array(self.inputvars['north_vector'].getval())
        Y       = np.array([0, 1, 0])
        Z       = np.array([0, 0, 1])
        costheta = np.dot(north, Y)/(np.linalg.norm(north)*np.linalg.norm(Y))
        theta   = np.arccos(costheta)*180.0/np.pi
        if np.abs(np.dot(np.cross(north, Y), Z) < tol):
            sign = 1.0
        else:
            sign    = -np.sign(np.dot(np.cross(north, Y), Z))
        return sign*theta

    def convert_winddir_to_xy(self, winddir):
        # Check for North/East vector
        thetaoffset = self.get_N_angle_to_Y()
        # Calculate Wind Vector
        theta = (270.0+thetaoffset-winddir)*np.pi/180.0
        nx    = np.cos(theta)
        ny    = np.sin(theta)
        vertical    = np.array([0, 0, 1.0])
        streamwise  = np.array([nx, ny, 0.0])
        crossstream = -np.cross(streamwise, vertical)
        return streamwise, crossstream, vertical

    # ---- ABL wind calculation ----------
    def ABL_calculateWindVector(self):
        WS   = self.inputvars['ABL_windspeed'].getval()
        Wdir = self.inputvars['ABL_winddir'].getval()
        # Check for None
        if (WS is None) or (Wdir is None): 
            print("Error in WS = "+repr(WS)+" or Wdir = "+repr(Wdir))
            return
        # Check for North/East vector
        thetaoffset = self.get_N_angle_to_Y()
        # Calculate Wind Vector
        theta = (270.0+thetaoffset-Wdir)*np.pi/180.0
        Ux    = WS*np.cos(theta)
        Uy    = WS*np.sin(theta)
        Uz    = 0.0
        # Set ABL_velocity
        self.inputvars['ABL_velocity'].setval([Ux, Uy, Uz], forcechange=True)
        return

    def ABL_calculateWDirWS(self):
        Wvec   = self.inputvars['ABL_velocity'].getval()
        Uhoriz = np.sqrt(Wvec[0]**2 + Wvec[1]**2)
        # Check for North/East vector
        thetaoffset = self.get_N_angle_to_Y()
        theta  = 270.0+thetaoffset-np.arctan2(Wvec[1], Wvec[0])*180.0/np.pi
        self.inputvars['ABL_windspeed'].setval(Uhoriz, forcechange=True)
        self.inputvars['ABL_winddir'].setval(theta, forcechange=True)
        return

    # ---- ABL postprocessing options ----
    def ABLpostpro_getprofileslist(self):
        return [key for key, v in postpro.statsprofiles.items()]

    def ABLpostpro_getscalarslist(self):
        return postpro.scalarvars[1:]

    def ABLpostpro_loadnetcdffile(self, ablfile=None, updatetimes=False,
                                  usemmap=None):
        if ablfile is None:
            ablfile = self.inputvars['ablstats_file'].getval()
        if self.abl_stats is not None:
            self.abl_stats.close()
        if usemmap is None:
            loadinmemory = False
        else:
            loadinmemory = usemmap
        self.abl_stats = postpro.loadnetcdffile(ablfile, usemmap=loadinmemory)
        print("Loading %s"%ablfile)
        mint = min(self.abl_stats['time'])
        maxt = max(self.abl_stats['time']) 
        if updatetimes: self.inputvars['ablstats_avgt'].setval([mint, maxt])
        print("Time range: %f to %f"%(mint, maxt))
        print("Done.")
        return

    def ABLpostpro_plotprofiles(self, ax=None, plotvars=None, avgt=None, 
                                doplot=True, usemapped=None):
        # Get the list of selected quantities
        if plotvars is None:
            plotvars = self.inputvars['ablstats_profileplot'].getval()
        if avgt is None:
            avgt         = self.inputvars['ablstats_avgt'].getval()
        if ax is None: ax=self.setupfigax()
        if usemapped is None: 
            usemapped = self.inputvars['ablstats_usemapped'].getval()

        returndict = {}

        if len(plotvars)<1: return returndict
        for var in plotvars:
            prof=postpro.CalculatedProfile.fromdict(postpro.statsprofiles[var],
                                                    self.abl_stats,
                                                    self.abl_profiledata, avgt,
                                                    usemapped=usemapped)
            z, plotdat = prof.calculate()
            self.abl_profiledata = prof.allvardata.copy()
            N = np.shape(plotdat)
            if len(N)>1:
                # Break the header labels
                varlabels = postpro.statsprofiles[var]['header'].split()
                for i in range(N[1]):
                    if doplot: 
                        ax.plot(plotdat[:,i], z, label=var+': '+varlabels[i])
                    returndict[varlabels[i]] = {'z':z, 'data':plotdat[:,i]}
            else:
                if doplot:
                    ax.plot(plotdat, z, label=postpro.statsprofiles[var]['header'])
                returndict[postpro.statsprofiles[var]['header']] = {'z':z, 'data':plotdat}
        if doplot:
            # Format the plot
            ax.set_ylabel('z [m]')
            ax.legend()
            # Draw the figure
            self.figcanvas.draw()
        return returndict

    def ABLpostpro_plotscalars(self, ax=None, plotvars=None, avgt=None,
                               doplot=True):
        # Get the list of selected quantities
        if plotvars is None:
            plotvars     = self.inputvars['ablstats_scalarplot'].getval()
        if avgt is None:
            avgt         = self.inputvars['ablstats_avgt'].getval()
        if ax is None: ax=self.setupfigax()

        returndict = {}

        if len(plotvars)<1: return returndict
        for var in plotvars:
            print(var)
            t, v = postpro.extractScalarTimeHistory(self.abl_stats, var)
            if doplot: ax.plot(t, v, label=var)
            returndict[var] = {'t':t, 'data':v}

        if doplot:
            # Format the plot
            ax.set_xlabel('t [s]')
            ax.legend()
            # Draw the figure
            self.figcanvas.draw()
        return returndict

    def ABLpostpro_exportprofiles(self):
        # Get the list of selected quantities
        selectedvars = self.inputvars['ablstats_profileplot'].getval()
        avgt         = self.inputvars['ablstats_avgt'].getval()

        if len(selectedvars)<1: return
        filepre  = filedialog.asksaveasfilename(initialdir = "./",
                                                title="Specify filename prefix")

        for var in selectedvars:
            prof=postpro.CalculatedProfile.fromdict(postpro.statsprofiles[var],
                                                    self.abl_stats,
                                                    self.abl_profiledata, avgt)
            filename = filepre+"."+var+".dat"
            extraheader = "Avg from t=%e to t=%e"%(avgt[0], avgt[1])
            prof.save(filename,
                      allvars=self.abl_profiledata, 
                      avgt=avgt, extraheader=extraheader)
            print("saved "+filename)
            self.abl_profiledata = prof.allvardata.copy()
        return

    def ABLpostpro_printreport(self, avgt=None, avgz=None):
        if avgt is None:
            avgt         = self.inputvars['ablstats_avgt'].getval()
        if avgz is None:
            ablstats_avgz= self.inputvars['ablstats_avgz'].getval()
        else:
            ablstats_avgz= avgz
        if (ablstats_avgz is None) or (ablstats_avgz=='None') or (ablstats_avgz==''): 
            print('Error ablstats_avgz=%s is not valid.'%ablstats_avgz)
            return
        if isinstance(ablstats_avgz, str):
            avgz  = [float(z) for z in re.split(r'[,; ]+', ablstats_avgz)]
        report = postpro.printReport(self.abl_stats, avgz, avgt, verbose=True)
        return report

    # ---- Sample probe postprocessing options ----
    def Samplepostpro_loadnetcdffile(self, ncfile=None, usemmap=None):
        # Get the filename
        if ncfile is None:
            samplefile = self.inputvars['sampling_file'].getval()
        else:
            samplefile = ncfile
        if len(samplefile)==0:
            print("Empty filename, choose file first")
            return
        if usemmap is None:
            loadinmemory = self.inputvars['samplingprobe_usemmap'].getval()
        else:
            loadinmemory = usemmap
        # Close the previous file
        if self.sample_ncdat is not None:
            self.sample_ncdat.close()
        print("Loading %s"%samplefile)
        self.sample_ncdat = ppsample.loadDataset(samplefile, usemmap=loadinmemory)
        # Update the groups
        tkentry = self.inputvars['samplingprobe_groups'].tkentry
        groups  = self.Samplepostpro_getgroups()
        tkentry.delete(0, Tk.END)
        for g in groups: tkentry.insert(Tk.END, g)
        # Update the time labels
        self.Samplepostpro_updatetimes()

    def Samplepostpro_updatetimes(self):
        timevec = ppsample.getVar(self.sample_ncdat, 'time')
        curindex = self.inputvars['samplingprobe_plottimeindex'].getval()
        mintime = '0: %f'%timevec[0]
        curtime = '%i: %f'%(curindex, timevec[curindex])
        maxtime = '%i: %f'%(len(timevec)-1, timevec[-1])
        self.inputvars['samplingprobe_timeinfo'].setval([mintime, curtime, 
                                                         maxtime],
                                                        forcechange=True)

    def Samplepostpro_getprevtime(self):
        curindex = self.inputvars['samplingprobe_plottimeindex'].getval()
        timevec = ppsample.getVar(self.sample_ncdat, 'time')
        if curindex>1: newindex = curindex-1
        else:          newindex = len(timevec)-1
        self.inputvars['samplingprobe_plottimeindex'].setval(newindex)
        self.Samplepostpro_updatetimes()
    
    def Samplepostpro_getnexttime(self):
        curindex = self.inputvars['samplingprobe_plottimeindex'].getval()
        timevec = ppsample.getVar(self.sample_ncdat, 'time')
        if curindex<len(timevec)-1:   newindex = curindex + 1
        else:                         newindex = 0
        self.inputvars['samplingprobe_plottimeindex'].setval(newindex)
        self.Samplepostpro_updatetimes()

    def Samplepostpro_getgroups(self):
        if self.sample_ncdat is None:
            return []
        else:
            return ppsample.getGroups(self.sample_ncdat)

    def Samplepostpro_getvars(self):
        if self.sample_ncdat is None:
            return []
        else:
            group   = self.inputvars['samplingprobe_groups'].getval()
            if len(group)==0: 
                print("No group selected")
                return
            allvars = ppsample.getVarList(self.sample_ncdat, group=group[0])
            tkentry = self.inputvars['samplingprobe_variables'].tkentry
            tkentry.delete(0, Tk.END)
            for v in allvars: tkentry.insert(Tk.END, v)
            return allvars

    def Samplepostpro_getplot(self):
        # Get the group
        groupsel= self.inputvars['samplingprobe_groups'].getval()
        var     = self.inputvars['samplingprobe_variables'].getval()
        timeind = self.inputvars['samplingprobe_plottimeindex'].getval()

        axis1   = self.inputvars['samplingprobe_plotaxis1'].getval()
        axis2   = self.inputvars['samplingprobe_plotaxis2'].getval()
        kindex  = self.inputvars['samplingprobe_kindex'].getval()
        
        # Check to make sure some group/var/timeind is selected
        if len(groupsel)==0:
            print('No group selected')
            return
        if len(var)==0:
            print('No var selected')
            return

        # Get the probe type
        alltypes = set()
        for g in groupsel:
            alltypes.add(ppsample.getGroupSampleType(self.sample_ncdat, g))
        alltypeslist = list(alltypes)
        if len(alltypeslist)>1: 
            print("ERROR: Can't plot")
            print("Multiple sample types selected: "+repr(alltypeslist))
            return
        sampletype = alltypeslist[0]

        self.Samplepostpro_updatetimes()

        if sampletype == 'LineSampler':
            self.plotSampleLine(groupsel,var, timeind, axis1)
        elif sampletype == 'PlaneSampler':
            self.plotSamplePlane(groupsel, var, timeind, 
                                 kindex, axis1, axis2)
        else:
            print('sample type %s is not recognized'%sampletype)
        return

    def plotSampleLine(self, groups, var, tindex, plotaxis, ax=None,ncdat=None):
        if isinstance(groups, str): groups=[groups]
        # Clear and resize figure
        if ax is None: ax=self.setupfigax()
        if ncdat is None: ncdat = self.sample_ncdat
        
        # Dictionary to hold return stuff
        returndat = {}

        # Get the plot data
        for group in groups:
            xyz,linedat=ppsample.getLineSampleAtTime(ncdat, group, var, tindex)
            plotx = ppsample.getPlotAxis(xyz, plotaxis)
            returndat[group] = {'plotx': plotx, 'dat': linedat}
            for v in var:
                ax.plot(plotx, linedat[v], label=group+':'+v)            
        ax.set_xlabel(plotaxis)
        ax.legend(fontsize=10)

        timevec = ppsample.getVar(self.sample_ncdat, 'time')
        curindex = self.inputvars['samplingprobe_plottimeindex'].getval()
        ax.set_title('Time: %f'%(timevec[curindex]))        

        self.figcanvas.draw()

        return returndat

    def plotSamplePlane(self, groups, varselect, tindex, kindex, 
                        plotaxis1, plotaxis2, ax=None, ncdat=None,
                        colorbar=True, levels=41, **contourfargs):
        if isinstance(groups, str): groups=[groups]
        # Clear and resize figure
        if ax is None: ax=self.setupfigax()
        if ncdat is None: ncdat = self.sample_ncdat

        if isinstance(varselect, list):
            var = varselect[0]
            if len(varselect)>1:
                print("Multiple variables selected.  Using %s"%var)            
        else:
            var = varselect

        #nlevels = 40  # Number of contour levels
        # Get the plot data
        imvec = []
        for group in groups:
            x,y,z,s1,s2,v = ppsample.getPlaneSampleAtTime(self.sample_ncdat, 
                                                          group, var, tindex, 
                                                          kindex)
            if plotaxis1=='X': plotx = x
            if plotaxis1=='Y': plotx = y
            if plotaxis1=='Z': plotx = z
            if plotaxis1=='AUTO': plotx = s1
            if plotaxis2=='X': ploty = x
            if plotaxis2=='Y': ploty = y
            if plotaxis2=='Z': ploty = z
            if plotaxis2=='AUTO': ploty = s2

            # plot the mesh
            im = ax.contourf(plotx, ploty, v, levels, **contourfargs)
            imvec.append(im)
            #im.autoscale()
        if colorbar: self.fig.colorbar(im, ax=ax)

        xlabel = 'Axis1' if plotaxis1=='AUTO' else plotaxis1
        ylabel = 'Axis2' if plotaxis2=='AUTO' else plotaxis2
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Set the title
        timevec = ppsample.getVar(self.sample_ncdat, 'time')
        #curindex = self.inputvars['samplingprobe_plottimeindex'].getval()
        ax.set_title('Time: %f'%(timevec[tindex]))        
        ax.set_aspect('equal')

        self.figcanvas.draw()
        #self.figcanvas.show()
        return imvec

    # ---- Turbine model stuff ----
    def turbinemodels_loadfromfile(self, filename, deleteprevious=False):
        """
        Load all of the turbine model from the yaml file filename
        """
        with open(filename) as fp:
            filepath = os.path.dirname(os.path.realpath(filename))
            if tkyg.useruemel: Loader=tkyg.yaml.load
            else:              Loader=tkyg.yaml.safe_load
            turbmodeldict = Loader(fp)
            if 'turbines' in turbmodeldict:
                # Add in the file locations
                for turb in turbmodeldict['turbines']:
                    turbmodeldict['turbines'][turb]['turbinetype_filelocation']\
                        = filepath
                
                self.listboxpopupwindict['listboxturbinetype'].populatefromdict(turbmodeldict['turbines'], deleteprevious=deleteprevious, forcechange=True)
            else:  
                # No turbines in YAML file
                pass
        return 

    def turbinemodels_populate(self, deleteprevious=False, turbinedir=None):
        if turbinedir is None:
            turbinedir = self.inputvars['preferences_turbinedir'].getval()
        # Check if we need to prepend any paths
        if turbinedir.startswith('/'):
            loadpath = turbinedir
        else:
            loadpath = os.path.join(scriptpath, turbinedir)
        # Go through each file 
        for fname in os.listdir(loadpath):
            if not fname.startswith('.') and \
               (fname.endswith('.yaml') or fname.endswith('.YAML')):
                self.turbinemodels_loadfromfile(os.path.join(loadpath, fname), \
                                                deleteprevious=deleteprevious)
        return

    def turbinemodels_applyturbinemodel(self, inputdict, use_turbine_type, 
                                        windowinputs=None, docopy=False,
                                        updatefast=False, window=None):
        if len(use_turbine_type)==0: 
            return  # No turbine type selected, return
        # Get the turbine model
        allturbinemodels = self.listboxpopupwindict['listboxturbinetype']
        keystr = lambda n, d1, d2: d2.name
        modelparams = allturbinemodels.dumpdict('AMR-Wind', 
                                                subset=[use_turbine_type], 
                                                keyfunc=keystr)
        modelfiles  = allturbinemodels.dumpdict('amrwind_frontend', 
                                                subset=[use_turbine_type], 
                                                keyfunc=keystr)

        # -- Set all of the turbine parameters
        # Apply window inputs
        if windowinputs is not None:
            for key, item in modelparams.items():
                if (key in window.temp_inputvars) and (item is not None):
                    windowinputs[key].setval(item)

        # Apply dictionary inputs
        outdict = inputdict.copy()
        if len(inputdict)>0:
            for key, item in modelparams.items():
                if (key in inputdict) and (item is not None):
                    outdict[key] = item

        if docopy and (modelfiles['turbinetype_filedir'] != "None"):
            origdir = modelfiles['turbinetype_filelocation']
            copydir = modelfiles['turbinetype_filedir']
            if windowinputs is not None:
                newdir = windowinputs['Actuator_name'].getval()
            elif 'Actuator_name' in outdict:
                newdir = outdict['Actuator_name']
            
            # Set up the copy paths
            newdir  = newdir + '_'+copydir
            origdir = os.path.join(origdir, copydir)
            print("docopy = "+repr(docopy)+" from "+origdir+" to "+newdir)
            try:
                shutil.copytree(origdir, newdir)
            except:
                print("copy %s failed"%origdir)

            # -- Change any file references --
            # Change the window input stuff
            if windowinputs is not None:
                for key, inputvar in windowinputs.items():
                    if inputvar.inputtype is tkyg.moretypes.filename:
                        origval    = inputvar.getval()
                        path, base = os.path.split(origval)
                        pathsplit  = path.split(os.sep)
                        if len(pathsplit)==0: continue
                        # replace the filename
                        pathsplit[0] = pathsplit[0].replace(copydir, newdir)
                        # join it back together
                        newval     = os.path.join(*pathsplit)
                        newval     = os.path.join(newval, base)
                        print(newval)
                        inputvar.setval(newval)
            # Change the outdict stuff
            if len(inputdict)>0:
                for key, item in outdict.items():
                    if isinstance(item, str) and item.startswith(copydir):
                        origval    = item
                        path, base = os.path.split(origval)
                        pathsplit  = path.split(os.sep)
                        if len(pathsplit)==0: continue
                        pathsplit[0] = pathsplit[0].replace(copydir, newdir)
                        # join it back together
                        newval     = os.path.join(*pathsplit)
                        newval     = os.path.join(newval, base)
                        print(newval)
                        outdict[key] = newval

        if updatefast:
            self.turbinemodels_checkupdateFAST(window=windowinputs, inputdict=outdict)
        return outdict

    def turbinemodels_copytoturbine(self, window=None):
        """
        Copy a turbine model to a specific turbine instance
        """
        # Get the selected turbine model
        use_turbine_type = window.temp_inputvars['use_turbine_type'].getval()
        docopy           = window.temp_inputvars['copy_turb_files'].getval()
        updatefast       = window.temp_inputvars['edit_fast_files'].getval()
        
        if len(use_turbine_type)==0: 
            return  # No turbine type selected, return
        # Get the turbine model
        allturbinemodels = self.listboxpopupwindict['listboxturbinetype']
        keystr = lambda n, d1, d2: d2.name
        modelparams = allturbinemodels.dumpdict('AMR-Wind', 
                                                subset=[use_turbine_type], 
                                                keyfunc=keystr)
        modelfiles  = allturbinemodels.dumpdict('amrwind_frontend', 
                                                subset=[use_turbine_type], 
                                                keyfunc=keystr)
        # Set all of the turbine parameters
        for key, item in modelparams.items():
            if (key in window.temp_inputvars) and (item is not None):
                window.temp_inputvars[key].setval(item)
                #print('%s: %s'%(key, repr(item)))

        # Copy over the turbine files
        copydir = modelfiles['turbinetype_filedir']
        newdir  = window.temp_inputvars['Actuator_name'].getval()+'_'+copydir
        origdir = modelfiles['turbinetype_filelocation']
        if docopy and (copydir is not None) and (len(copydir)>0):
            origdir = os.path.join(origdir, copydir)
            #print("docopy = "+repr(docopy)+" from "+origdir+" to "+newdir)
            try:
                shutil.copytree(origdir, newdir)
            except:
                print("copy %s failed"%origdir)

            # Change any file references
            for key, inputvar in window.temp_inputvars.items():
                if inputvar.inputtype is tkyg.moretypes.filename:
                    origval    = inputvar.getval()
                    path, base = os.path.split(origval)
                    pathsplit  = path.split(os.sep)
                    if len(pathsplit)==0: continue
                    # replace the filename
                    pathsplit[0] = pathsplit[0].replace(copydir, newdir)
                    # join it back together
                    newval     = os.path.join(*pathsplit)
                    newval     = os.path.join(newval, base)
                    print(newval)
                    inputvar.setval(newval)
        if updatefast:
            self.turbinemodels_checkupdateFAST(window=window)
        return

    def turbinemodels_checkupdateFAST(self, window=None, inputdict={}):
        if (window is None) and (len(inputdict)==0):
            return

        # Get the actuator type
        if window is not None:
            Actuator_type = window.temp_inputvars['Actuator_type'].getval()
        else:
            Actuator_type = inputdict['Actuator_type']
        if Actuator_type not in ['TurbineFastLine', 'TurbineFastDisk']:
            # Not a FAST model, do nothing
            return

        # Get some FAST input values
        if window is not None:
            # Get the FAST file
            fstfile  = window.temp_inputvars['Actuator_openfast_input_file'].getval()
            yaw      = window.temp_inputvars['Actuator_yaw'].getval()
            density  = window.temp_inputvars['Actuator_density'].getval()
        else:
            fstfile  = inputdict['Actuator_openfast_input_file']
            yaw      = inputdict['Actuator_yaw']
            density  = inputdict['Actuator_density']

        # Check yaw
        EDfile   = OpenFAST.getFileFromFST(fstfile,'EDFile')
        EDdict   = OpenFAST.FASTfile2dict(EDfile)
        EDyaw    = float(EDdict['NacYaw'])

        TOL = 1.0E-6
        if (yaw is not None) and abs(yaw - (270.0-EDyaw))>TOL:
            # Correct the yaw in Elastodyn
            EDyaw = 270.0 - yaw
            print("Fixing yaw in %s"%EDfile)
            OpenFAST.editFASTfile(EDfile, {'NacYaw':EDyaw})

        # Check aerodyn
        CompAero = OpenFAST.getFileFromFST(fstfile,'CompAero')
        if (density is not None) and (CompAero != 0):
            # Check density
            AeroFile = OpenFAST.getFileFromFST(fstfile,'AeroFile')
            print(AeroFile)
            AeroDict = OpenFAST.FASTfile2dict(AeroFile)
            AirDens  = float(AeroDict['AirDens'])
            if abs(density - AirDens) > TOL:
                OpenFAST.editFASTfile(AeroFile, {'AirDens':density})        
        return

    # ---- Local run stuff ----
    def localrun_constructrunstring(self, window=None):
        localrunparams = self.popup_storteddata['localrun']
        modulecmd      = localrunparams['localrun_modules']
        mpicmd         = localrunparams['localrun_mpicmd']
        amrwindexe     = localrunparams['localrun_exe']
        logfile        = localrunparams['localrun_logfile']
        nproc          = localrunparams['localrun_nproc']

        # construct the execution command
        exestring = ""
        # Load the modules
        if len(modulecmd)>0: 
            exestring += modulecmd+"&& "
        # MPI command
        mpistring = "%s -np %i %s %s "%(mpicmd,nproc,amrwindexe,self.savefile)
        exestring += mpistring
        # Pipe to logfile
        if len(logfile)>0:
            exestring += "|tee %s"%logfile
        #print(localrunparams)
        return exestring

    def localrun_execute(self):
        # First save the file
        if len(self.savefile)>0:
            self.writeAMRWindInput(self.savefile)
            print("Saved "+self.savefile)
        else:
            self.writeAMRWindInputGUI()

        exestring = self.localrun_constructrunstring()
        print("Running")
        print(exestring)
        self.localrun_process = subprocess.Popen(exestring, 
                                                 preexec_fn=os.setsid,
                                                 shell=True)
        return

    def localrun_kill(self):
        if self.localrun_process is None:
            print("No running job")
            return
        if self.localrun_process.poll() is not None:
            print("Job completed with exit code %i"%self.localrun_process.poll())
        else:
            print("Terminating job")
            #self.localrun_process.kill()
            os.killpg(os.getpgid(self.localrun_process.pid), signal.SIGTERM)
        return 

    # ---- submit script stuff ----
    def submitscript_makescript(self, submitscript_inputfile, 
                                window=None):
        submitparams   = self.popup_storteddata['submitscript']
        scripttemplate = self.inputvars['submitscript_template'].getval()
        scripttemplate = scripttemplate[scripttemplate.find('#'):]

        # Replace the input file with the input filename
        replacevar     = submitparams['submitscript_replaceinputfilestring']
        submitscript   = scripttemplate.replace(replacevar, submitscript_inputfile.strip())

        if window is None:
            inputvars = self.yamldict['popupwindow']['submitscript']['inputwidgets']
            for inputitem in inputvars:
                if ('outputdef' in inputitem):
                    if 'replacevar' in inputitem['outputdef']:
                        replacevar = inputitem['outputdef']['replacevar']
                        name       = inputitem['name']
                        if (name in submitparams):
                            replacestr = str(submitparams[name]).strip()
                            if replacestr != 'None':
                                submitscript = submitscript.replace(replacevar, 
                                                                    replacestr)
        else:
            # get the list of variables to replace
            for key, item in window.temp_inputvars.items():
                if 'replacevar' in item.outputdef:
                    replacevar   = item.outputdef['replacevar']
                    replaceval   = item.getval()
                    if (replaceval is not None) and \
                       (str(replaceval) != 'None') and \
                       (len(str(replaceval))>0):
                        replacestr = str(replaceval).strip()
                        submitscript = submitscript.replace(replacevar, 
                                                            replacestr)
        if sys.version_info[0] < 3:
           formattedscript = submitscript.decode('string_escape')
        else:
           formattedscript = bytes(submitscript, "utf-8").decode("unicode_escape")
        #return submitscript
        return formattedscript


    def submitscript_previewscript(self, window=None):
        # Save the file first
        if len(self.savefile)==0: self.saveAMRWindInputGUI()
        submitscript = self.submitscript_makescript(self.savefile, 
                                                    window=window)
        if submitscript is None:
            print("Error in submit script")
            return
        formattedscript = copy.copy(submitscript)

        # Show the script in a message window
        tkyg.messagewindow(self,formattedscript, height=20, autowidth=True, 
                           title='Preview submit script')
        return

    def submitscript_savescript(self, window=None, submit=False, guimesg=False,
                                scriptfilename=None):
        if len(self.savefile)==0: self.saveAMRWindInputGUI()
        submitscript = self.submitscript_makescript(self.savefile, 
                                                    window=window)
        if submitscript is None:
            print("Error in submit script")
            return
        formattedscript = copy.copy(submitscript)

        # Save the script
        submitparams   = self.popup_storteddata['submitscript']
        if scriptfilename is None:
            filename       = submitparams['submitscript_filename']
        else:
            filename       = scriptfilename
        if len(filename)>0:
            f=open(filename, "w")
            f.write(formattedscript)
            f.close()
            print("Saved "+filename)
        else:
            errormesg="ERROR: Need to specify submission script filename. "
            print(errormesg)
            if guimesg:
                tkyg.messagewindow(self, errormesg, autowidth=True, 
                                   title='ERROR')    
            return

        # Submit the job if asked
        if submit:
            exestring = submitparams['submitscript_submitcmd']
            exestring += " "+filename
            print("Executing: "+exestring)
            try:
                joboutput = subprocess.check_output([exestring],
                                                    stderr=subprocess.STDOUT, 
                                                    shell=True)
                title='JOB SUBMITTED'
            except subprocess.CalledProcessError as e:
                joboutput = '%s'%(e.output)
                title='ERROR'
            print(joboutput)
            if guimesg:
                tkyg.messagewindow(self, str(joboutput.strip()), autowidth=True,
                                   title=title)    
        return

    # ---- Boundary plane restart stuff ----
    def boundaryplane_restartGUI(self):
        return

    def boundaryplane_restart(self, 
                              setIOmode=1, 
                              bndryfiles='',
                              inflowplanes=[],
                              autooutflow=True,
                              forcingdict={},
                              autoset_ABLForcing=True,
                              autoset_ABLMeanBoussinesq=True,
                              verbose=False):
        """Automatically sets the boundary conditions and parameters required
        to restart using boundary planes.
        """

        def printverbose(suffix, key):
            print(suffix+" "+key+" = "+repr(self.inputvars[key].getval()))
            return

        # Define the opposite face from a certain face
        oppositeface = {  
            'xlo':'xhi', 'xhi':'xlo',
            'ylo':'yhi', 'yhi':'ylo',
            'zlo':'zhi', 'zhi':'zlo',
        }

        # Set the boundary plane IO mode and files
        if setIOmode:
            self.inputvars['ABL_bndry_io_mode'].setval(str(setIOmode))
            if verbose: printverbose('SET','ABL_bndry_io_mode')

        if len(bndryfiles)>0:
            self.inputvars['ABL_bndry_file'].setval(bndryfiles)
            if verbose: printverbose('SET','ABL_bndry_file')
        
        # Set the boundary conditions
        ## First set the correct periodicity arguments
        if ('xlo' in inflowplanes) or ('xhi' in inflowplanes):
            self.inputvars['is_periodicx'].setval(False)
            if verbose: printverbose('SET', 'is_periodicx')
        if ('ylo' in inflowplanes) or ('yhi' in inflowplanes):
            self.inputvars['is_periodicy'].setval(False)
            if verbose: printverbose('SET', 'is_periodicy')
        ## Set the inflow mass flow BC
        for face in inflowplanes:
            density = self.inputvars['density'].getval()
            self.inputvars[face+'_type'].setval('mass_inflow')
            self.inputvars[face+'_density'].setval(density)
            self.inputvars[face+'_temperature'].setval(0.0)
            self.inputvars[face+'_tke'].setval(0.0)
            if verbose: 
                printverbose('SET', face+'_type')
                printverbose('SET', face+'_density')
                printverbose('SET', face+'_temperature')
                printverbose('SET', face+'_tke')
            
        ## Set the outflow boundary conditions
        if autooutflow:
            for face in inflowplanes:
                oface = oppositeface[face]
                self.inputvars[oface+'_type'].setval('pressure_outflow')
                self.inputvars[oface+'_density'].setval(None)   # UPDATE THIS!
                self.inputvars[oface+'_temperature'].setval(None)
                self.inputvars[oface+'_tke'].setval(None)
                if verbose: 
                    printverbose('SET', oface+'_type')
                    printverbose('SET', oface+'_density')
                    printverbose('SET', oface+'_temperature')
                    printverbose('SET', oface+'_tke')

        # Set the body force
        if len(forcingdict)>0:
            # Calculate the body force
            ncfile = forcingdict['ablstatfile']
            tavg   = forcingdict['tavg']
            abl_ncdat = postpro.loadnetcdffile(ncfile)
            ABL_X_FORCE = postpro.timeAvgScalar(abl_ncdat,'abl_forcing_x',tavg)
            ABL_Y_FORCE = postpro.timeAvgScalar(abl_ncdat,'abl_forcing_y',tavg)
            ABL_Z_FORCE = 0.0
            ABL_F_VEC   = [ABL_X_FORCE, ABL_Y_FORCE, ABL_Z_FORCE]
            # Set the body force
            self.inputvars['BodyForce'].setval(True)
            self.inputvars['BodyForce_magnitude'].setval(ABL_F_VEC)
            if verbose: 
                printverbose('SET','BodyForce')
                printverbose('SET','BodyForce_magnitude')

        # Set the ABL forcing off
        if autoset_ABLForcing:
            self.inputvars['ABLForcing'].setval(False)
            if verbose: printverbose('SET','ABLForcing')

        # Set the ABLMeanBoussinesq term
        if autoset_ABLMeanBoussinesq:
            self.inputvars['ABLMeanBoussinesq'].setval(True)
            if verbose: printverbose('SET','ABLMeanBoussinesq')

        # Set the ABL mode to local
        
        return

if __name__ == "__main__":
    title='AMR-Wind'
    localconfigdir=os.path.join(scriptpath,'local')

    # Check the command line arguments
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('inputfile', nargs='?')
    parser.add_argument('--ablstatsfile',   
                        default='',  
                        help="Load the ABL statistics file [default: None]")
    parser.add_argument('--farmfile',   
                        default='',  
                        help="Load the farm layout YAML file [default: None]")
    parser.add_argument('--samplefile',   
                        default='',  
                        help="Load the sample probe file [default: None]")
    parser.add_argument('--outputfile',   
                        default='',  
                        help="Write the output file [default: None]")
    parser.add_argument('--validate',   
                        action='store_true',  
                        help="Check input file for errors and quit [default: False]")
    parser.add_argument('--calcmeshsize',   
                        action='store_true',  
                        help="Estimate the meshsize [default: False]")
    parser.add_argument('--localconfigdir',   
                        default=localconfigdir,  
                        help="Local configuration directory [default: %s]"%localconfigdir)

    args         = parser.parse_args()
    inputfile    = args.inputfile
    ablstatsfile = args.ablstatsfile
    farmfile     = args.farmfile
    samplefile   = args.samplefile
    outputfile   = args.outputfile
    validate     = args.validate
    calcmesh     = args.calcmeshsize
    localconfigdir = args.localconfigdir

    # Validate the input file
    if validate:
        mainapp=MyApp.init_nogui(localconfigdir=localconfigdir)
        if inputfile is not None:
            mainapp.loadAMRWindInput(inputfile, printunused=True)
        mainapp.validate()
        sys.exit()

    if calcmesh:
        mainapp=MyApp.init_nogui(localconfigdir=localconfigdir)
        if inputfile is not None:
            mainapp.loadAMRWindInput(inputfile, printunused=True)
        mainapp.estimateMeshSize()
        sys.exit()

    # Instantiate the app
    mainapp=MyApp(configyaml=os.path.join(scriptpath,'config.yaml'), 
                  localconfigdir=localconfigdir, 
                  scriptpath=scriptpath,
                  #os.path.join(scriptpath,'local'),
                  title=title)
    mainapp.notebook.enable_traversal()

    # Load an inputfile
    if inputfile is not None:
        mainapp.loadAMRWindInput(inputfile, printunused=True)
        mainapp.savefile = inputfile
                
    if len(outputfile)>0:
        mainapp.writeAMRWindInput(outputfile, outputextraparams=True)


    # Load the farm file
    if len(farmfile)>0:
        mainapp.loadFarmSetupYAML(farmfile)

    # Load the abl statsfile
    if len(ablstatsfile)>0:
        mainapp.inputvars['ablstats_file'].setval(ablstatsfile)
        mainapp.ABLpostpro_loadnetcdffile()
        mainapp.notebook.select(6)

    # Load the samplefile
    if len(samplefile)>0:
        mainapp.inputvars['sampling_file'].setval(samplefile)
        mainapp.Samplepostpro_loadnetcdffile()
        mainapp.notebook.select(6)

    mainapp.mainloop()
