#!/usr/bin/env python

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

# -------------------------------------------------------------
def readCartBoxFile(filename):
    """
    Read the Cartesian box file
    """
    allboxes = []
    fname    = open(filename, 'r')
    # read the number of levels
    Nlevels = int(fname.readline().strip())
    for i in range(Nlevels):
        # Read the number of boxes at this level
        Nboxes = int(fname.readline().strip())
        levelboxes=[]
        for b in range(Nboxes):
            # Read each box
            boxline  = fname.readline().strip().split()
            box      = [float(x) for x in boxline]
            if len(box)!=6:
                print("Line does not contain 6 floats:")
                print(" %s"%boxline)
            levelboxes.append(box)
        allboxes.append(levelboxes)
    fname.close()
    return allboxes

def plotRectangle(figax, corner1, corner2, ix, iy, **kwargs):
    """
    Plot a rectangle onto figax
    """
    x1 = corner1[ix]
    y1 = corner1[iy]
    x2 = corner2[ix]
    y2 = corner2[iy]
    Lx = x2-x1
    Ly = y2-y1
    rect=Rectangle((x1, y1), Lx, Ly, **kwargs)
    figax.add_patch(rect)
    return x1, y1, x2, y2

def plot3DBox(figax, origin, xaxis, yaxis, zaxis, ix, iy, **kwargs):
    """
    Plots a 3D box on figax.  Corner is at origin 
    """
    # Define the faces to plot
    plotfaces = [[origin, xaxis, yaxis],
                 [origin, xaxis, zaxis],
                 [origin, yaxis, zaxis]]
    # Build list for each of three faces
    for face in plotfaces:
        p1     = np.array(face[0])
        p2     = p1+np.array(face[1])
        p3     = p2+np.array(face[2])
        p4     = p3-np.array(face[1])
        ptlist = [p1, p2, p3, p4]
        xlist  = [p[ix] for p in ptlist]
        ylist  = [p[iy] for p in ptlist]
        figax.fill(xlist, ylist, **kwargs)
    return

def rotatepoint(pt, orig, theta):
    """
    Rotates a point pt about origin orig
    Here theta is measured w.r.t. the x-axis
    """
    dx = pt[0]-orig[0]
    dy = pt[1]-orig[1]
    p2=[0.0, 0.0, 0.0]
    p2[0] = dx*np.cos(theta) - dy*np.sin(theta) + orig[0]
    p2[1] = dx*np.sin(theta) + dy*np.cos(theta) + orig[1]
    p2[2] = pt[2]
    return p2

def plotTurbine(figax, basexyz, hubheight, turbD, nacelledir, ix, iy, **kwargs):
    """
    Plot turbine on figax
    """
    turbR = 0.5*turbD
    Nsegs = 30  # Number of segments on rotor circumference
    # Construct the rotor diameter ring
    rotorpts = []
    for theta in np.linspace(0, 360, Nsegs+1):
        x = 0
        y = turbR*np.cos(theta*np.pi/180.0)
        z = turbR*np.sin(theta*np.pi/180.0)
        rotorpts.append([x,y,z])
    # Rotate the rotor ring to the right orientation
    rotatetheta = (270.0-nacelledir)*np.pi/180.0
    rotatedring = [rotatepoint(p, [0.0,0.0,0.0], rotatetheta) for p in rotorpts]
    # Translate the right to the right location
    hhpt     = np.array([0.0, 0.0, hubheight])
    rotorpts = [np.array(p)+np.array(basexyz)+hhpt for p in rotatedring]
    #print(rotorpts)
    # Get the list of x and y points and plot them
    xlist = [p[ix] for p in rotorpts]
    ylist = [p[iy] for p in rotorpts]
    figax.fill(xlist, ylist, **kwargs)
    #figax.plot(xlist, ylist, **kwargs)
    
    # Plot the turbine nacelle
    nacelleW = 0.1*turbD   # nacelle width (lateral)
    nacelleH = 0.1*turbD   # nacelle height
    nacelleL = 0.2*turbD   # nacelle length (streamwise)
    nacellecorner = [0, -nacelleW/2, -nacelleH/2]
    nacelleLaxis  = [nacelleL, 0, 0]
    nacelleWaxis  = [0, nacelleW, 0]
    nacelleHaxis  = [0, 0, nacelleH]
    # Rotate the points
    nacelleLaxis  = rotatepoint(nacelleLaxis, [0,0,0], rotatetheta)
    nacelleWaxis  = rotatepoint(nacelleWaxis, [0,0,0], rotatetheta)
    # rotate and translate the corner
    nacellecorner = rotatepoint(nacellecorner, [0,0,0], rotatetheta)
    nacellecorner = np.array(nacellecorner) + np.array(basexyz) + hhpt
    plot3DBox(figax, nacellecorner, nacelleLaxis, nacelleWaxis, nacelleHaxis,
              ix, iy, **kwargs)
    return

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

        # Define aliases
        self.get_default_samplingdict = \
            self.listboxpopupwindict['listboxsampling'].getdefaultdict
        self.get_default_taggingdict = \
            self.listboxpopupwindict['listboxtagging'].getdefaultdict
        self.get_default_turbinetypedict = \
            self.listboxpopupwindict['listboxturbinetype'].getdefaultdict
        self.get_default_actuatordict = \
            self.listboxpopupwindict['listboxactuator'].getdefaultdict

        return

    @classmethod
    def init_nogui(cls, *args, **kwargs):
        return cls(configyaml=os.path.join(scriptpath,'config.yaml'), 
                   localconfigdir=os.path.join(scriptpath,'local'), 
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

    def setAMRWindInput(self, name, val, **kwargs):
        """
        Use this function to set the AMR-Wind keyword name to value.
        """
        try:
            inputkey = self.amrkeydict[name]
            self.inputvars[inputkey].setval(val, **kwargs)
        except:
            print("Cannot set "+name)
        return

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
        if len(filename)>0:     f.close()
        
        return returnstr

    def writeAMRWindInputGUI(self):
        filename  = filedialog.asksaveasfilename(initialdir = "./",
                                                 title = "Save AMR-Wind file")
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
        tkyg.messagewindow(self, self.writeAMRWindInput(''), 
                           height=40)
        return

    def getInputHelp(self, search=''):
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

        # link any widgets necessary
        for key,  inputvar in self.inputvars.items():
            if self.inputvars[key].ctrlelem is not None:
                self.inputvars[key].onoffctrlelem(None)

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

        # Validate menu
        runmenu = Tk.Menu(menubar, tearoff=0)
        runmenu.add_command(label="Check Inputs", 
                            command=self.validate)
        runmenu.add_command(label="Local run", 
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

    def validate(self, printeverything=True):
        # Load validateinputs plugins
        num_nonactive = 0
        num_active    = 0
        print("-- Checking inputs --")
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
                        print("[%5s] %-20s %s"%(r['result'].name,
                                                p.name+":"+r['subname'],
                                                r['mesg']))
            else:
                num_nonactive = num_nonactive+1            
        print('')
        print("Results: ")
        for k, g in resultclass.items():
            print(' %i %s'%(len(g), k))
        return
    
    def setupfigax(self, clear=True, subplot=111):
        # Clear and resize figure
        canvaswidget=self.figcanvas.get_tk_widget()
        w,h1 = self.winfo_width(), self.winfo_height()
        canvaswidget.configure(width=w-self.leftframew-10, height=h1-75)
        self.fig.clf()
        ax=self.fig.add_subplot(subplot)
        if clear: ax.clear()
        return ax

    def plotDomain(self, ax=None):
        # Clear and resize figure
        if ax is None: ax=self.setupfigax()
        
        # Get the variables
        corner1  = self.inputvars['prob_lo'].getval()
        corner2  = self.inputvars['prob_hi'].getval()
        plotparams = self.popup_storteddata['plotdomain']
        xychoice = plotparams['plot_chooseview']
        if xychoice == 'XY':
            ix,iy = 0,1
            xstr, ystr='x','y'
        elif xychoice == 'XZ':
            ix,iy = 0,2
            xstr, ystr='x','z'
        elif xychoice == 'YZ':
            ix,iy = 1,2
            xstr, ystr='y','z'
        # Wind direction
        windvec  = self.inputvars['ABL_velocity'].getval()
        windh    = self.inputvars['forcing_height'].getval()
        # North direction
        northdir = self.inputvars['north_vector'].getval()

        # Do the domain plot here
        x1, y1, x2, y2  = plotRectangle(ax, corner1, corner2, ix, iy,
                                        color='gray', alpha=0.25)
        Lx = x2-x1
        Ly = y2-y1
        Cx = 0.5*(x1+x2)
        Cy = 0.5*(y1+y2)
        ax.set_xlim([Cx-Lx*0.55, Cx+Lx*0.55])
        ax.set_ylim([Cy-Ly*0.55, Cy+Ly*0.55])

        if plotparams['plot_windnortharrows']:
            # Plot the wind vector
            arrowlength = 0.1*np.linalg.norm([Lx, Ly])
            plotwindvec = np.array(windvec)
            plotwindvec = plotwindvec/np.linalg.norm(plotwindvec)*arrowlength
            windcenter = [Cx, Cy, windh]
            if np.linalg.norm([plotwindvec[ix], plotwindvec[iy]])>0.0:
                ax.arrow(windcenter[ix], windcenter[iy], 
                         plotwindvec[ix], plotwindvec[iy], 
                         width=0.05*arrowlength)
        
            # Plot the north arrow
            northlength = 0.1*np.linalg.norm([Lx, Ly])
            plotnorthvec  = np.array(northdir)
            plotnorthvec  = plotnorthvec/np.linalg.norm(plotnorthvec)*northlength
            compasscenter = [Cx-0.4*Lx, Cy+0.35*Ly, windh]
        
            if np.linalg.norm([plotnorthvec[ix], plotnorthvec[iy]])>0.0:
                ax.arrow(compasscenter[ix], compasscenter[iy], 
                         plotnorthvec[ix], plotnorthvec[iy], 
                         color='r', head_width=0.1*northlength, linewidth=0.5)
                ax.text(compasscenter[ix], 0.99*compasscenter[iy], 
                        'N', color='r', ha='right', va='top')

        # Plot the sample probes
        # ---------------------------
        if ((plotparams['plot_sampleprobes'] is not None) 
            and (len(plotparams['plot_sampleprobes'])>0)):
        #if plotparams['plot_sampleprobes']:
            allsamplingdata = self.listboxpopupwindict['listboxsampling']
            allprobes=allsamplingdata.getitemlist()
            keystr = lambda n, d1, d2: d2.name
            ms=2
            for p in plotparams['plot_sampleprobes']:
                pdict = allsamplingdata.dumpdict('AMR-Wind', subset=[p], keyfunc=keystr)
                if pdict['sampling_type'][0]=='LineSampler':
                    Npts  = pdict['sampling_l_num_points']
                    start = np.array(pdict['sampling_l_start'])
                    end   = np.array(pdict['sampling_l_end'])
                    dx    = (end-start)/(Npts-1.0)
                    pts   = []
                    for i in range(Npts):
                        pt = start + dx*i
                        pts.append(pt)
                    pts = np.array(pts)
                    ax.plot(pts[:,ix], pts[:,iy], '.', markersize=ms, label=p)
                if pdict['sampling_type'][0]=='PlaneSampler':
                    Npts   = pdict['sampling_p_num_points']
                    origin = np.array(pdict['sampling_p_origin'])
                    axis1  = np.array(pdict['sampling_p_axis1'])
                    axis2  = np.array(pdict['sampling_p_axis2'])
                    dx1    = axis1/(Npts[0]-1.0)
                    dx2    = axis2/(Npts[1]-1.0)
                    pts    = []
                    # TODO: add offset
                    for i in range(Npts[0]):
                        for j in range(Npts[1]):
                            pt = origin + i*dx1 + j*dx2
                            pts.append(pt)
                    pts = np.array(pts)
                    ax.plot(pts[:,ix], pts[:,iy], '.', markersize=ms, label=p)
            legendprobes=ax.legend(title="Sampling probes", fontsize=10,
                                   loc='upper right')
            for legend_handle in legendprobes.legendHandles:
                legend_handle._legmarker.set_markersize(9)
            plt.setp(legendprobes.get_title(),fontsize=10)
            ax.add_artist(legendprobes)

        # Plot the refinement boxes
        # ---------------------------
        if ((plotparams['plot_refineboxes'] is not None) and 
            (len(plotparams['plot_refineboxes'])>0)):
            #print(plotparams['plot_refineboxes'])
            allrefinements = self.listboxpopupwindict['listboxtagging']
            alltags        = allrefinements.getitemlist()
            keystr         = lambda n, d1, d2: d2.name
            
            # Need to validate maxlevel! Fix this!
            maxlevel       = self.inputvars['max_level'].getval()
            # Get the level colors
            try: 
                levelcolors=plt.rcParams['axes.color_cycle']
            except:
                levelcolors=plt.rcParams['axes.prop_cycle'].by_key()['color']

            for p in plotparams['plot_refineboxes']:
                pdict = allrefinements.dumpdict('AMR-Wind',
                                                subset=[p], keyfunc=keystr)
                # Plot the Cartesian Box Refinements
                if pdict['tagging_type'][0]=='CartBoxRefinement':
                    filename = pdict['tagging_static_refinement_def']
                    # Load the boxes
                    allboxes = readCartBoxFile(filename)
                    if len(allboxes)>maxlevel: maxlevel = len(allboxes)
                    for ilevel, boxlevel in enumerate(allboxes):
                        for box in boxlevel:
                            corner1 = box[0:3]
                            corner2 = box[3:6]
                            color   = levelcolors[ilevel]
                            plotRectangle(ax, corner1, corner2, ix, iy,
                                          facecolor=color, ec='k', lw=0.5, 
                                          alpha=0.90)
                # Plot the Geometry Refinements
                if pdict['tagging_type'][0]=='GeometryRefinement':
                    if pdict['tagging_geom_type']=='box':
                        origin = pdict['tagging_geom_origin']
                        xaxis  = pdict['tagging_geom_xaxis']
                        yaxis  = pdict['tagging_geom_yaxis']
                        zaxis  = pdict['tagging_geom_zaxis']
                        ilevel = pdict['tagging_level']
                        if ilevel is not None: 
                            color   = levelcolors[ilevel]
                        else:
                            color   = levelcolors[0]
                        #print("plotting box: ")
                        #print(" origin: "+repr(origin))
                        #print(" xaxis:  "+repr(xaxis))
                        #print(" yaxis:  "+repr(yaxis))
                        #print(" zaxis:  "+repr(zaxis))
                        plot3DBox(ax, origin, xaxis, yaxis, zaxis, ix, iy,
                                  lw=0.4, facecolor=color, alpha=0.90)
                    if pdict['tagging_geom_type']=='cylinder':
                        print("cylinder geometry refinement plotting not supported")

            # Add a legend with the level labels
            legend_el = []
            legend_label = []
            legend_el.append(Line2D([0],[0], 
                                    linewidth=0, marker='s', color='gray',
                                    alpha=0.25, label='Level 0'))
            legend_label.append('Level 0')
            for i in range(maxlevel):
                legend_el.append(Line2D([0],[0], 
                                        linewidth=0, marker='s',
                                        color=levelcolors[i+0], 
                                        alpha=0.75, 
                                        label='Level %i'%(i+0)))
                legend_label.append('Level %i'%(i+1))
            legendrefine = ax.legend(legend_el, legend_label, 
                                     frameon=True, numpoints=1, 
                                     fontsize=10, loc='lower right')
            ax.add_artist(legendrefine)

        # Plot the turbines
        # ---------------------------
        if ((plotparams['plot_turbines'] is not None) and 
            (len(plotparams['plot_turbines'])>0)):
            print("Plotting turbines")
            allturbines  = self.listboxpopupwindict['listboxactuator']
            alltags      = allturbines.getitemlist()
            keystr       = lambda n, d1, d2: d2.name

            # Get the defaults
            default_type   = self.inputvars['Actuator_default_type'].getval()
            default_type   = None if len(default_type)==0 else default_type
            default_type   = default_type[0] if isinstance(default_type, list) else default_type
            if 'Actuator_%s_rotor_diameter'%default_type in self.inputvars:
                default_turbD  = self.inputvars['Actuator_%s_rotor_diameter'%default_type].getval()
            else:
                default_turbD  = None
            if 'Actuator_%s_hub_height'%default_type in self.inputvars:
                default_hh     = self.inputvars['Actuator_%s_hub_height'%default_type].getval()
            else:
                default_hh     = None

            # Get the wind direction
            self.ABL_calculateWDirWS()
            winddir = self.inputvars['ABL_winddir'].getval()
            
            for turb in plotparams['plot_turbines']:
                tdict = allturbines.dumpdict('AMR-Wind',
                                             subset=[turb], keyfunc=keystr)
                turbtype = default_type if 'Actuator_type' not in tdict else tdict['Actuator_type']
                turbtype = turbtype[0] if isinstance(turbtype, list) else turbtype
                turbhh   = default_hh  if tdict['Actuator_hub_height'] is None else tdict['Actuator_hub_height']
                turbD    = default_turbD if tdict['Actuator_rotor_diameter'] is None else tdict['Actuator_rotor_diameter']
                
                basepos  = tdict['Actuator_base_position']
                yaw      = winddir #270.0

                if turbtype in ['TurbineFastLine', 'TurbineFastDisk']:
                    fstfile  = tdict['Actuator_openfast_input_file']
                    EDfile   = OpenFAST.getFileFromFST(fstfile,'EDFile')
                    EDdict   = OpenFAST.FASTfile2dict(EDfile)
                    EDyaw    = float(EDdict['NacYaw'])
                    yaw      = 270.0-EDyaw

                plotTurbine(ax, basepos, turbhh, turbD, yaw, ix, iy,
                            lw=1, color='k', alpha=0.75)
                
        # --------------------------------
        # Set some plot formatting parameters
        ax.set_aspect('equal')
        ax.set_xlabel('%s [m]'%xstr)
        ax.set_ylabel('%s [m]'%ystr)
        ax.set_title(r'Domain')
        self.figcanvas.draw()
        #self.figcanvas.show()

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
        return

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


    # ---- ABL wind calculation ----------
    def ABL_calculateWindVector(self):
        WS   = self.inputvars['ABL_windspeed'].getval()
        Wdir = self.inputvars['ABL_winddir'].getval()
        # Check for None
        if (WS is None) or (Wdir is None): 
            print("Error in WS = "+repr(WS)+" or Wdir = "+repr(Wdir))
            return
        # Check for North/East vector
        # TODO
        # Calculate Wind Vector
        theta = (270.0-Wdir)*np.pi/180.0
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
        # TODO
        theta  = 270.0-np.arctan2(Wvec[1], Wvec[0])*180.0/np.pi
        self.inputvars['ABL_windspeed'].setval(Uhoriz, forcechange=True)
        self.inputvars['ABL_winddir'].setval(theta, forcechange=True)
        return

    # ---- ABL postprocessing options ----
    def ABLpostpro_getprofileslist(self):
        return [key for key, v in postpro.statsprofiles.items()]

    def ABLpostpro_getscalarslist(self):
        return postpro.scalarvars[1:]

    def ABLpostpro_loadnetcdffile(self, ablfile=None, updatetimes=False):
        if ablfile is None:
            ablfile = self.inputvars['ablstats_file'].getval()
        if self.abl_stats is not None:
            self.abl_stats.close()
        self.abl_stats = postpro.loadnetcdffile(ablfile)
        print("Loading %s"%ablfile)
        mint = min(self.abl_stats['time'])
        maxt = max(self.abl_stats['time']) 
        if updatetimes: self.inputvars['ablstats_avgt'].setval([mint, maxt])
        print("Time range: %f to %f"%(mint, maxt))
        print("Done.")
        return

    def ABLpostpro_plotprofiles(self, ax=None, plotvars=None, avgt=None):
        # Get the list of selected quantities
        if plotvars is None:
            plotvars = self.inputvars['ablstats_profileplot'].getval()
        if avgt is None:
            avgt         = self.inputvars['ablstats_avgt'].getval()
        if ax is None: ax=self.setupfigax()

        if len(plotvars)<1: return
        for var in plotvars:
            #print(var)
            # initialize the profile
            prof=postpro.CalculatedProfile.fromdict(postpro.statsprofiles[var],
                                                    self.abl_stats,
                                                    self.abl_profiledata, avgt)
            z, plotdat = prof.calculate()
            self.abl_profiledata = prof.allvardata.copy()
            N = np.shape(plotdat)
            if len(N)>1:
                # Break the header labels
                varlabels = postpro.statsprofiles[var]['header'].split()
                for i in range(N[1]):
                    ax.plot(plotdat[:,i], z, label=var+': '+varlabels[i])
            else:
                ax.plot(plotdat, z, label=postpro.statsprofiles[var]['header'])
        # Format the plot
        ax.set_ylabel('z [m]')
        ax.legend()
        # Draw the figure
        self.figcanvas.draw()
        return

    def ABLpostpro_plotscalars(self, ax=None, plotvars=None, avgt=None):
        # Get the list of selected quantities
        if plotvars is None:
            plotvars     = self.inputvars['ablstats_scalarplot'].getval()
        if avgt is None:
            avgt         = self.inputvars['ablstats_avgt'].getval()
        if ax is None: ax=self.setupfigax()
        
        if len(plotvars)<1: return
        for var in plotvars:
            print(var)
            t, v = postpro.extractScalarTimeHistory(self.abl_stats, var)
            ax.plot(t, v, label=var)
        # Format the plot
        ax.set_xlabel('t [s]')
        ax.legend()
        # Draw the figure
        self.figcanvas.draw()
        return

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

    def ABLpostpro_printreport(self):
        avgt         = self.inputvars['ablstats_avgt'].getval()
        ablstats_avgz= self.inputvars['ablstats_avgz'].getval()
        if (ablstats_avgz is None) or (ablstats_avgz=='None') or (ablstats_avgz==''): 
            print('Error ablstats_avgz=%s is not valid.'%ablstats_avgz)
            return
        avgz         = [float(z) for z in re.split(r'[,; ]+', ablstats_avgz)]
        report = postpro.printReport(self.abl_stats, avgz, avgt, verbose=True)
        return

    # ---- Sample probe postprocessing options ----
    def Samplepostpro_loadnetcdffile(self, ncfile=None):
        if ncfile is None:
            samplefile = self.inputvars['sampling_file'].getval()
        else:
            samplefile = ncfile
        if len(samplefile)==0:
            print("Empty filename, choose file first")
            return
        if self.sample_ncdat is not None:
            self.sample_ncdat.close()
        print("Loading %s"%samplefile)
        self.sample_ncdat = ppsample.loadDataset(samplefile)
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

        # Get the plot data
        for group in groups:
            xyz,linedat=ppsample.getLineSampleAtTime(ncdat, group, var, tindex)
            plotx = ppsample.getPlotAxis(xyz, plotaxis)
            for v in var:
                ax.plot(plotx, linedat[v], label=group+':'+v)            
        ax.set_xlabel(plotaxis)
        ax.legend(fontsize=10)

        timevec = ppsample.getVar(self.sample_ncdat, 'time')
        curindex = self.inputvars['samplingprobe_plottimeindex'].getval()
        ax.set_title('Time: %f'%(timevec[curindex]))        

        self.figcanvas.draw()
        #self.figcanvas.show()
        return

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
            if plotaxis1=='S': plotx = s1
            if plotaxis2=='X': ploty = x
            if plotaxis2=='Y': ploty = y
            if plotaxis2=='Z': ploty = z
            if plotaxis2=='S': ploty = s2

            # plot the mesh
            im = ax.contourf(plotx, ploty, v, levels, **contourfargs)
            imvec.append(im)
            #im.autoscale()
        if colorbar: self.fig.colorbar(im, ax=ax)

        xlabel = 'Axis1' if plotaxis1=='S' else plotaxis1
        ylabel = 'Axis2' if plotaxis2=='S' else plotaxis2
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

    def turbinemodels_populate(self, deleteprevious=False):
        turbinedir = 'turbines'
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

    def turbinemodels_checkupdateFAST(self, window=None):
        Actuator_type = window.temp_inputvars['Actuator_type'].getval()
        if Actuator_type not in ['TurbineFastLine', 'TurbineFastDisk']:
            # Not a FAST model, do nothing
            return

        TOL = 1.0E-6
        # Get the FAST file
        fstfile  =window.temp_inputvars['Actuator_openfast_input_file'].getval()
        # Check yaw
        EDfile   = OpenFAST.getFileFromFST(fstfile,'EDFile')
        EDdict   = OpenFAST.FASTfile2dict(EDfile)
        EDyaw    = float(EDdict['NacYaw'])
        yaw      = window.temp_inputvars['Actuator_yaw'].getval()

        if (yaw is not None) and abs(yaw - (270.0-EDyaw))>TOL:
            # Correct the yaw in Elastodyn
            EDyaw = 270.0 - yaw
            print("Fixing yaw in %s"%EDfile)
            OpenFAST.editFASTfile(EDfile, {'NacYaw':EDyaw})

        # Check aerodyn
        density  = window.temp_inputvars['Actuator_density'].getval()
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
    def submitscript_makescript(self, submitscript_inputfile, window=None):
        submitparams   = self.popup_storteddata['submitscript']
        scripttemplate = self.inputvars['submitscript_template'].getval()
        scripttemplate = scripttemplate[scripttemplate.find('#'):]

        #submitscript_inputfile = self.savefile
        submitscript   = scripttemplate.replace('submitscript_inputfile', 
                                                submitscript_inputfile.strip())

        # get the list of variables to replace
        for key, item in window.temp_inputvars.items():
            if 'replacevar' in item.outputdef:
                replacevar   = item.outputdef['replacevar']
                replaceval   = item.getval()
                if replaceval is not None and (len(str(replaceval))>0):
                    replacestr = str(replaceval).strip()
                    submitscript = submitscript.replace(replacevar, replacestr)
        return submitscript

    def submitscript_previewscript(self, window=None):
        self.saveAMRWindInputGUI()
        submitscript = self.submitscript_makescript(self.savefile, 
                                                    window=window)
        if submitscript is None:
            print("Error in submit script")
            return
        #formattedscript = copy.copy(submitscript).decode('string_escape')
        if sys.version_info[0] < 3:
           formattedscript = submitscript.decode('string_escape')
        else:
           formattedscript = bytes(submitscript, "utf-8").decode("unicode_escape")
        # Show the script in a message window
        tkyg.messagewindow(self,formattedscript, height=20, autowidth=True)
        return

    def submitscript_savescript(self, window=None):
        self.saveAMRWindInputGUI()
        submitscript = self.submitscript_makescript(self.savefile, 
                                                    window=window)
        if submitscript is None:
            print("Error in submit script")
            return
        formattedscript = copy.copy(submitscript)
        if sys.version_info[0] < 3:
            formattedscript = submitscript.decode('string_escape')
        else:
            formattedscript = bytes(submitscript, "utf-8").decode("unicode_escape")
        # Save the script
        submitparams   = self.popup_storteddata['submitscript']
        filename       = submitparams['submitscript_filename']
        if len(filename)>0:
            f=open(filename, "w")
            f.write(formattedscript)
            f.close()
            print("Saved "+filename)
        return

if __name__ == "__main__":
    title='AMR-Wind'

    # Check the command line arguments
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('inputfile', nargs='?')
    parser.add_argument('--ablstatsfile',   
                        default='',  
                        help="Load the ABL statistics file [default: None]")
    parser.add_argument('--samplefile',   
                        default='',  
                        help="Load the sample probe file [default: None]")
    parser.add_argument('--outputfile',   
                        default='',  
                        help="Write the output file [default: None]")
    parser.add_argument('--validate',   
                        action='store_true',  
                        help="Check input file for errors and quit [default: False]")

    args         = parser.parse_args()
    inputfile    = args.inputfile
    ablstatsfile = args.ablstatsfile
    samplefile   = args.samplefile
    outputfile   = args.outputfile
    validate     = args.validate

    # Validate the input file
    if validate:
        mainapp=MyApp.init_nogui()
        mainapp.loadAMRWindInput(inputfile, printunused=True)
        mainapp.validate()
        sys.exit()

    # Instantiate the app
    mainapp=MyApp(configyaml=os.path.join(scriptpath,'config.yaml'), 
                  localconfigdir=os.path.join(scriptpath,'local'),
                  title=title)
    mainapp.notebook.enable_traversal()

    # Load an inputfile
    if inputfile is not None:
        mainapp.loadAMRWindInput(inputfile, printunused=True)
        mainapp.savefile = inputfile
                
    if len(outputfile)>0:
        mainapp.writeAMRWindInput(outputfile, outputextraparams=True)
    
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
