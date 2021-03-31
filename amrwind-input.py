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

class MyApp(tkyg.App, object):
    def __init__(self, *args, **kwargs):
        super(MyApp, self).__init__(*args, **kwargs)
        self.fig.clf()
        self.fig.text(0.35,0.5,'Welcome to\nAMR-Wind')

        self.formatgridrows()
        return

    def getInputVal(self, inp):
        if inp.labelonly is True: return None
        val = inp.getval()
        return val
    
    def getDictFromInputs(self, tag):
        """
        Create a dict based on tag in outputdefs
        """
        output = OrderedDict()
        for key, var in self.inputvars.items():
            if tag in var.outputdef:
                outputkey = var.outputdef[tag]
                output[outputkey] = self.getInputVal(var)
                #print(outputkey+' = '+repr(output[outputkey]))
        return output

    def writeAMRWindInput(self):
        """
        Do more sophisticated output control later
        """
        inputdict = self.getDictFromInputs('AMR-Wind')
        for key, val in inputdict.items():
            outputkey = key
            # convert val to string
            if isinstance(val, list):
                outputstr=' '.join([str(x) for x in val])
            else:
                outputstr=str(val)
            print("%-40s = %s"%(outputkey, outputstr))
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

    def loadAMRWindInput(self, filename, printunused=False):
        amrdict=self.AMRWindInputToDict(filename)
        extradict=self.setinputfromdict('AMR-Wind', amrdict)
        if printunused:
            print("# -- Unused variables: -- ")
            for key, data in extradict.items():
                print("%-40s= %s"%(key, data))
        return extradict

    def loadAMRWindInputGUI(self):
        filename  = filedialog.askopenfilename(initialdir = "./",
                                              title = "Select AMR-Wind file")
        if len(filename)>0:
            extradict = self.loadAMRWindInput(filename, printunused=True)
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
        filemenu.add_command(label="Save", 
                             command=partial(tkyg.donothing, root))

        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=root.quit)
        menubar.add_cascade(label="File", menu=filemenu)

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
        xychoice = self.inputvars['plot_chooseview'].getval()
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
        
        ax.set_aspect('equal')
        ax.set_xlabel('%s [m]'%xstr)
        ax.set_ylabel('%s [m]'%ystr)
        ax.set_title(r'Domain')
        self.figcanvas.draw()
        #self.figcanvas.show()

        return

if __name__ == "__main__":
    title='AMR-Wind input creator'
    mainapp=MyApp(configyaml='config.yaml', title=title)
    #mainapp.loadAMRWindInput('abl.inp', printunused=True)
    mainapp.mainloop()
