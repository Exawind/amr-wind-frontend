import logging
import threading
import os, sys
import time
import matplotlib.pyplot    as plt

if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk
import warnings
warnings.filterwarnings('ignore')

# Load the screenshot library
scriptpath='../../'
sys.path.insert(1, scriptpath)
import genscreenshot as screenshot

# Set the tutorial properties
scrwidth  = 1280
scrheight = 800
imagedir  = 'images'

# Create the directory
if not os.path.exists(imagedir): os.makedirs(imagedir)

# create a virtual display
vdisplay = screenshot.start_Xvfb(width=scrwidth, height=scrheight)

# Set the logging functions
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, 
                    level=logging.INFO,
                    datefmt="%H:%M:%S")
logging.info("Main    : Starting script")


casedict={}
lock= threading.Lock()
t1 = threading.Thread(target=screenshot.start_instance, 
                     args=(1,casedict, lock))
t1.start()
logging.info("Main    : started case thread")
time.sleep(3)

case=casedict[1]
case.launchpopupwin('plotdomain', savebutton=False).okclose()

###########################
case.notebook.select(2)
case.setAMRWindInput('useWSDir',      True)
case.setAMRWindInput('ABL_windspeed', 10.0)
case.setAMRWindInput('ABL_winddir',   225.0)
case.ABL_calculateWindVector()
#logging.info("Main    : saving ABL_settings.png")
screenshot.Xvfb_screenshot(imagedir+'/ABL_settings.png', 
                           crop=(0, 0, 515, 400))
###########################

###########################
turbinescsv="""# CSV file should have columns with
# name, x, y, type, yaw, hubheight, options
T0, 500, 300, GE25test, , ,
T1, 500, 700, GE25test, , ,
"""
domainsize       = [1000,1000,1000]   # Domain size [x,y,z] in meters
backgrounddeltax = 10                 # Background mesh delta x in meters

# Set the parameters
case.setAMRWindInput('turbines_csvtextbox', turbinescsv)
case.setAMRWindInput('turbines_domainsize',         domainsize)
case.setAMRWindInput('turbines_backgroundmeshsize', backgrounddeltax)
case.setAMRWindInput('turbines_deleteprev', True)   # Delete any existing turbines from the system

#-------------------------------------------------------
#logging.info("Main    : saving farm_turbine_layout.png")
case.notebook.select(8)
case.toggledframes['frame_farmturbines'].setstate(True)

screenshot.scrollcanvas(case.notebook._tab['Farm'].canvas, 1.0)
screenshot.Xvfb_screenshot(imagedir+'/farm_turbine_layout.png', 
                           crop=(0,60,515,scrheight-175))
###########################

###########################
#logging.info("Main    : saving farm_turbine_layout_preview.png")
fig, ax = plt.subplots(figsize=(5,5), facecolor='w', dpi=150)
case.turbines_previewAllTurbines(ax=ax)
plt.tight_layout()
plt.savefig(imagedir+'/farm_turbine_layout_preview.png')
###########################

###########################
case.turbines_createAllTurbines()
case.notebook.select(4)
screenshot.Xvfb_screenshot(imagedir+'/farm_turbine_created.png', 
                           crop=(0,60,515,scrheight-350))
###########################



###########################
refinementcsv="""# CSV file should have columns with
# level, upstream, downstream, lateral, below, above, options
level, upstream, downstream, lateral, below, above, options
0,     1,    1,   1,   0.75, 1, 
1,     0.5,  0.5, 0.5, 0.75, 1, 
#center:farm
"""
case.setAMRWindInput('refine_csvtextbox', refinementcsv)
case.setAMRWindInput('refine_deleteprev', True)
#-------------------------------------------------------
#logging.info("Main    : saving farm_refinementspec.png")
time.sleep(0.25)
case.notebook.select(8)
case.toggledframes['frame_farmrefinement'].setstate(True)

screenshot.scrollcanvas(case.notebook._tab['Farm'].canvas, 1.0)
screenshot.Xvfb_screenshot(imagedir+'/farm_refinementspec.png', 
                           crop=(0,60+200,515,scrheight-120))
###########################


###########################
case.refine_createAllZones()
p = case.launchpopupwin('plotdomain', savebutton=False)
time.sleep(0.1)
p.temp_inputvars['plot_refineboxes'].tkentry.select_set(0, Tk.END)
p.temp_inputvars['plot_turbines'].tkentry.select_set(0, Tk.END)

screenshot.Xvfb_screenshot(imagedir+'/plotDomainWin_refinezone.png', 
                           crop=screenshot.getwinpos(p))
time.sleep(0.1)
p.okclose()
###########################

###########################
# Plot the domain
case.setAMRWindInput('max_level',2)
fig, ax = plt.subplots(figsize=(5,5), facecolor='w', dpi=150)
time.sleep(0.25)
case.plotDomain(ax=ax)
plt.tight_layout()
plt.savefig(imagedir+'/plotDomainFig_refineturbine.png')
###########################


###########################
samplingcsv="""# CSV file should have columns withturbinescsv=
# name, type, upstream, downstream, lateral, below, above, n1, n2, options
name, type, upstream, downstream, lateral, below, above, n1, n2, options
cl1, centerline, 1,  0, none, none,  none,  11, 11, none
rp1, rotorplane, 2,  0, none, none,  none,  11, 11, none
#hh1, hubheight,  2,  1, 1, 0,  none,  11, 11, usedx:0.5
sw1, streamwise, 2,  1, 1, 0.5,  0.5,  11, 11, usedx:0.25 noffsets:1
hh,  hubheight,  2,  1, 1, 0,  none,  11, 11, usedx:0.25 center:farm orientation:x
"""
case.setAMRWindInput('sampling_csvtextbox', samplingcsv)
case.setAMRWindInput('sampling_deleteprev', True)

time.sleep(0.25)
case.notebook.select(8)
case.toggledframes['frame_farmsampling'].setstate(True)

screenshot.scrollcanvas(case.notebook._tab['Farm'].canvas, 1.0)
screenshot.Xvfb_screenshot(imagedir+'/farm_samplingspec.png', 
                           crop=(0,60+260,515,scrheight-50))


case.sampling_createAllProbes(verbose=False)
###########################

###########################
case.refine_createAllZones()
p2 = case.launchpopupwin('plotdomain', savebutton=False)
time.sleep(0.1)
p2.temp_inputvars['plot_sampleprobes'].tkentry.select_set(0, Tk.END)
p2.temp_inputvars['plot_refineboxes'].tkentry.select_set(0, Tk.END)
p2.temp_inputvars['plot_turbines'].tkentry.select_set(0, Tk.END)

screenshot.Xvfb_screenshot(imagedir+'/plotDomainWin_samplingzone.png', 
                           crop=screenshot.getwinpos(p2))
time.sleep(0.2)
p2.okclose()
###########################

###########################
# Plot the domain
fig, ax = plt.subplots(figsize=(5,5), facecolor='w', dpi=150)
time.sleep(0.25)
case.plotDomain(ax=ax)
plt.tight_layout()
plt.savefig(imagedir+'/plotDomainFig_refineturbinesampling.png')
###########################

time.sleep(2)
case.quit()
t1.join()
