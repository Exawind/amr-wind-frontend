#!/usr/bin/env python

import sys
sys.path.insert(1, './tkyamlgui')
import numpy as np
from functools import partial
import tkyamlgui as tkyg
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
import argparse

class MyApp(tkyg.App, object):
    def __init__(self, *args, **kwargs):
        super(MyApp, self).__init__(*args, **kwargs)
        self.fig.clf()
        self.fig.text(0.35,0.5,'Welcome to\nAMR-Wind')
        self.formatgridrows()
        self.extradictparams = OrderedDict()
        return

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

    def writeAMRWindInput(self, filename, verbose=False, 
                          outputextraparams=True):
        """
        Do more sophisticated output control later
        """
        samplingkey = lambda n, d1, d2: d1['outputprefix']['AMR-Wind']+'/'+n+'.'+d2.outputdef['AMR-Wind']

        inputdict = self.getDictFromInputs('AMR-Wind')
        sampledict= self.listboxpopupwindict['listboxsampling'].dumpdict('AMR-Wind', keyfunc=samplingkey)
        # Construct the output dict
        outputdict=inputdict.copy()
        outputdict.update(sampledict)
        if outputextraparams:
            outputdict.update(self.extradictparams)

        if len(filename)>0:  f=open(filename, "w")
        for key, val in outputdict.items():
            outputkey = key
            # convert val to string
            if isinstance(val, list):
                outputstr=' '.join([str(self.ifbool(x)) for x in val])
            else:
                outputstr=str(self.ifbool(val))
            if len(outputstr)>0: writestr = "%-40s = %s"%(outputkey, outputstr)
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
    def AMRWindInputToDict(cls, filename):
        returndict = OrderedDict()
        with open(filename) as f:
            for line in f:
                line = line.partition('#')[0]
                line = line.rstrip()
                if len(line)>0:
                    line = line.split('=')
                    key  = line[0].strip()
                    data = line[1].strip()
                    returndict[key] = data
        return returndict

    @classmethod
    def AMRWindExtractSampleDict(cls, inputdict, template, sep=["/","."]):
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
                inputtype=getinputtype(template['inputwidgets'], probedictkey)[0]
                data = matchlisttype(inputdict[key], inputtype)
                probedict[probedictkey] = data
                extradict.pop(key)
            samplingdict[name] = probedict.copy()

        #print(samplingdict)
        return samplingdict, extradict

    def loadAMRWindInput(self, filename, printunused=False):
        amrdict=self.AMRWindInputToDict(filename)
        extradict=self.setinputfromdict('AMR-Wind', amrdict)

        # Input the sampling probes
        samplingdict, extradict = \
            self.AMRWindExtractSampleDict(extradict, 
            self.yamldict['popupwindow']['sampling'])
        self.listboxpopupwindict['listboxsampling'].populatefromdict(samplingdict)

        if printunused:
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

    def plotDomain(self):
        # Clear and resize figure
        canvaswidget=self.figcanvas.get_tk_widget()
        w,h1 = self.winfo_width(), self.winfo_height()
        canvaswidget.configure(width=w-self.leftframew-10, height=h1-75)
        self.fig.clf()
        ax=self.fig.add_subplot(111)
        ax.clear()
        
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

        # Do the actual plot here
        x1 = corner1[ix]
        y1 = corner1[iy]
        x2 = corner2[ix]
        y2 = corner2[iy]
        Lx = x2-x1
        Ly = y2-y1
        Cx = 0.5*(x1+x2)
        Cy = 0.5*(y1+y2)
        rect=Rectangle((x1, y1), Lx, Ly, color='gray', alpha=0.25)
        ax.add_patch(rect)
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
                ax.text(compasscenter[ix], compasscenter[iy], 
                        'N', color='r', ha='right', va='top')

        if plotparams['plot_sampleprobes']:
            allsamplingdata = self.listboxpopupwindict['listboxsampling']
            allprobes=allsamplingdata.getitemlist()
            keystr = lambda n, d1, d2: d2.name
            for p in allprobes:
                pdict = allsamplingdata.dumpdict('AMR-Wind', subset=[p], keyfunc=keystr)
                #print(pdict['sampling_type'])
        
        ax.set_aspect('equal')
        ax.set_xlabel('%s [m]'%xstr)
        ax.set_ylabel('%s [m]'%ystr)
        ax.set_title(r'Domain')
        self.figcanvas.draw()
        #self.figcanvas.show()

        return

if __name__ == "__main__":
    title='AMR-Wind input creator'

    # Check the command line arguments
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('inputfile', nargs='?')
    args   = parser.parse_args()
    inputfile = args.inputfile
    mainapp=MyApp(configyaml='config.yaml', title=title)

    if inputfile is not None:
        mainapp.extradictparams = mainapp.loadAMRWindInput(inputfile, printunused=True)
    mainapp.mainloop()
