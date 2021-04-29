#!/usr/bin/env python

import sys, os, re
# import the tkyamlgui library
scriptpath=os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, scriptpath+'/tkyamlgui')
sys.path.insert(1, scriptpath)

import numpy as np
from functools import partial
import tkyamlgui as tkyg
import postproamrwindabl as postpro

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

# -------------------------------------------------------------
class MyApp(tkyg.App, object):
    def __init__(self, *args, **kwargs):
        super(MyApp, self).__init__(*args, **kwargs)
        self.fig.clf()
        self.fig.text(0.35,0.5,'Welcome to\nAMR-Wind')
        self.formatgridrows()
        self.extradictparams = OrderedDict()
        self.abl_stats = None
        self.abl_profiledata = {}
        return

    @classmethod
    def init_nogui(cls, *args, **kwargs):
        return cls(configyaml=scriptpath+'/config.yaml',withdraw=True,**kwargs)

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
            self.listboxpopupwindict[name] = tkyg.listboxpopupwindows(frame, listboxdict, popupdict)
        return

    @classmethod
    def ifbool(cls, x):
        if not isinstance(x, bool): return x
        return 'true' if x else 'false'

    def getTaggingKey(self, keyname, listboxdict, datadict):
        keyheader = 'tagging.'
        intername = ''
        #getinputtype = lambda l,n: [x['inputtype'] for x in l if x['name']==n]
        if datadict.name.startswith('tagging_geom'):
            #print(keyname+datadict.outputdef['AMR-Wind']+" needs fixing!")
            keynamedict = self.listboxpopupwindict['listboxtagging'].dumpdict('AMR-Wind', subset=[keyname])
            intername=keynamedict[keyname+".shapes"].strip()+"."
            #print(listboxdict)
        return keyheader+keyname+"."+intername+datadict.outputdef['AMR-Wind']

    def writeAMRWindInput(self, filename, verbose=False, 
                          outputextraparams=True):
        """
        Do more sophisticated output control later
        """
        inputdict = self.getDictFromInputs('AMR-Wind')

        # Get the sampling outputs
        samplingkey = lambda n, d1, d2: d1['outputprefix']['AMR-Wind']+'.'+n+'.'+d2.outputdef['AMR-Wind']
        sampledict= self.listboxpopupwindict['listboxsampling'].dumpdict('AMR-Wind', keyfunc=samplingkey)
        
        taggingdict = self.listboxpopupwindict['listboxtagging'].dumpdict('AMR-Wind', keyfunc=self.getTaggingKey)

        # Construct the output dict
        outputdict=inputdict.copy()
        outputdict.update(sampledict)
        outputdict.update(taggingdict)

        # Add any extra parameters
        if outputextraparams:
            outputdict.update(self.extradictparams)

        if len(filename)>0:  f=open(filename, "w")
        for key, val in outputdict.items():
            outputkey = key
            # convert val to string
            if val is None:
                continue
            elif isinstance(val, list):
                outputstr=' '.join([str(self.ifbool(x)) for x in val])
            else:
                outputstr=str(self.ifbool(val))
            if len(outputstr)>0 and (outputstr != 'None'):
                writestr = "%-40s = %s"%(outputkey, outputstr)
                if verbose: print(writestr)
                if len(filename)>0: f.write(writestr+"\n")
        if len(filename)>0:     f.close()
        #print(sampledict)
        return

    def writeAMRWindInputGUI(self):
        filename  = filedialog.asksaveasfilename(initialdir = "./",
                                                 title = "Save AMR-Wind file")
        if len(filename)>0:
            self.writeAMRWindInput(filename)
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
        From input dict, extract all of the sampling probe parameters
        """
        pre='tagging'
        dictkeypre= 'tagging_'
        taggingdict = OrderedDict()
        if pre+'.labels' not in inputdict: return taggingdict, inputdict
        extradict = inputdict.copy()
        
        # Get the sampling labels
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
                suffix = 'shapes'
                # Get the shapes
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
        self.listboxpopupwindict['listboxsampling'].populatefromdict(samplingdict)
        # Input the tagging/refinement zones
        taggingdict, extradict = \
            self.AMRWindExtractTaggingDict(extradict, 
            self.yamldict['popupwindow']['tagging'])
        self.listboxpopupwindict['listboxtagging'].populatefromdict(taggingdict)

        if printunused and len(extradict)>0:
            print("# -- Unused variables: -- ")
            for key, data in extradict.items():
                print("%-40s= %s"%(key, data))

        # link any widgets necessary
        for key,  inputvar in self.inputvars.items():
            if self.inputvars[key].ctrlelem is not None:
                self.inputvars[key].onoffctrlelem(None)
        return extradict

    def loadAMRWindInputGUI(self):
        filename  = filedialog.askopenfilename(initialdir = "./",
                                              title = "Select AMR-Wind file")
        if len(filename)>0:
            self.extradictparams = self.loadAMRWindInput(filename, printunused=True)
        return

    def menubar(self, root):
        """ 
        Adds a menu bar to root
        See https://www.tutorialspoint.com/python/tk_menu.htm
        """
        menubar  = Tk.Menu(root)

        # File menu
        filemenu = Tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Import AMR-Wind file", 
                             command=self.loadAMRWindInputGUI)
        filemenu.add_command(label="Save AMR-Wind file", 
                             command=self.writeAMRWindInputGUI)

        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        # Plot menu
        plotmenu = Tk.Menu(menubar, tearoff=0)
        plotmenu.add_command(label="Plot domain", 
                             command=partial(self.launchpopupwin, 
                                             'plotdomain', savebutton=False))
        menubar.add_cascade(label="Plot", menu=plotmenu)


        # Help menu
        helpmenu = Tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Help Index", 
                             command=partial(tkyg.donothing, root))
        helpmenu.add_command(label="About...", 
                             command=partial(tkyg.donothing, root))
        menubar.add_cascade(label="Help", menu=helpmenu)
        
        root.config(menu=menubar)
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
            maxlevel       = 0
            # Get the level colors
            try: 
                levelcolors=plt.rcParams['axes.color_cycle']
            except:
                levelcolors=plt.rcParams['axes.prop_cycle'].by_key()['color']

            for p in plotparams['plot_refineboxes']:
                pdict = allrefinements.dumpdict('AMR-Wind',
                                                subset=[p], keyfunc=keystr)
                #print(p)
                #print(pdict)
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
                                          alpha=0.75)
                            
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
                                        color=levelcolors[i], 
                                        alpha=0.75, 
                                        label='Level %i'%(i+1)))
                legend_label.append('Level %i'%(i+1))
            legendrefine = ax.legend(legend_el, legend_label, 
                                     frameon=True, numpoints=1, 
                                     fontsize=10, loc='lower right')
            ax.add_artist(legendrefine)

        # Set some plot formatting parameters
        ax.set_aspect('equal')
        ax.set_xlabel('%s [m]'%xstr)
        ax.set_ylabel('%s [m]'%ystr)
        ax.set_title(r'Domain')
        self.figcanvas.draw()
        #self.figcanvas.show()

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

    def ABLpostpro_loadnetcdffile(self, updatetimes=False):
        ablfile        = self.inputvars['ablstats_file'].getval()
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

    def ABLpostpro_plotprofiles(self):
        # Get the list of selected quantities
        selectedvars = self.inputvars['ablstats_profileplot'].getval()
        avgt         = self.inputvars['ablstats_avgt'].getval()
        ax=self.setupfigax()

        if len(selectedvars)<1: return
        for var in selectedvars:
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

    def ABLpostpro_plotscalars(self):
        # Get the list of selected quantities
        selectedvars = self.inputvars['ablstats_scalarplot'].getval()
        avgt         = self.inputvars['ablstats_avgt'].getval()
        ax=self.setupfigax()
        
        if len(selectedvars)<1: return
        for var in selectedvars:
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

if __name__ == "__main__":
    title='AMR-Wind'

    # Check the command line arguments
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('inputfile', nargs='?')
    parser.add_argument('--ablstatsfile',   
                        default='',  
                        help="Load the ABL statistics file [default: None]")
    parser.add_argument('--outputfile',   
                        default='',  
                        help="Write the output file [default: None]")
    args         = parser.parse_args()
    inputfile    = args.inputfile
    ablstatsfile = args.ablstatsfile
    outputfile   = args.outputfile

    mainapp=MyApp(configyaml=scriptpath+'/config.yaml', title=title)
    mainapp.notebook.enable_traversal()

    # Load an inputfile
    if inputfile is not None:
        mainapp.extradictparams = mainapp.loadAMRWindInput(inputfile, printunused=True)
    if len(outputfile)>0:
        mainapp.writeAMRWindInput(outputfile, outputextraparams=True)
        
    # Load the abl statsfile
    if len(ablstatsfile)>0:
        mainapp.inputvars['ablstats_file'].setval(ablstatsfile)
        mainapp.ABLpostpro_loadnetcdffile()
        mainapp.notebook.select(5)
    mainapp.mainloop()
