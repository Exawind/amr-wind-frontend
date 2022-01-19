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
import nbformat as nbf

getsubstring          = lambda x, s1, s2: x.split(s1)[-1].split(s2)[0]

setAMRWindInputString = screenshot.setAMRWindInputString
NBADDMARKDOWN = lambda nb, x: nb['cells'].append(nbf.v4.new_markdown_cell(x))
NBADDCELL     = lambda nb, x: nb['cells'].append(nbf.v4.new_code_cell(x.strip()))


# Get the yaml help file
farmyaml = os.path.join(scriptpath, 'farm.yaml')

# ========================================
# Set the tutorial properties
scrwidth  = 1280
scrheight = 800
imagedir  = 'images'
mdtemplate= 'tutorial3gui_template.md'
mdfile    = 'tutorial3gui.md'
nbfile    = 'tutorial3python.ipynb'
gennbfile = True
runjupyter= True
# ========================================

farmtab = 8
abltab  = 2

mdstr = ""
mdvar = {}

# Load the markdown template
with open (mdtemplate, "r") as myfile:
    mdstr=myfile.read()

# Create the directory
if not os.path.exists(imagedir): os.makedirs(imagedir)

# create a virtual display
vdisplay = screenshot.start_Xvfb(width=scrwidth, height=scrheight)

# Set the logging functions
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, 
                    level=logging.INFO,
                    datefmt="%H:%M:%S")

# Start the app
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

if gennbfile:
    nb = nbf.v4.new_notebook()
    nb['cells'] = []

###########################
if gennbfile:
    # Get the header
    txt = "# Tutorial 3: Setting up a farm calculation"
    NBADDMARKDOWN(nb, txt)

    # Load the modules
    txt = """\
# Load the modules
amrwindfedir = '../../'  # Location of amrwind-frontend directory
import sys, os
sys.path.insert(1, amrwindfedir)

# Load the libraries
import amrwind_frontend as amrwind
import matplotlib.pyplot    as plt

# Also ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Make all plots inline 
%matplotlib inline
"""
    NBADDCELL(nb, txt)
    txt="""\
# Start the AMR-Wind case
case = amrwind.MyApp.init_nogui()"""
    NBADDCELL(nb, txt)

    # Intro text
    txt = getsubstring(mdstr, '<!--INTROTEXT0-->', '<!--INTROTEXT1-->')
    NBADDMARKDOWN(nb, txt)

###########################
case.notebook.select(abltab)
useWSDir = True
WS   = 10     # Wind speed [m/s]
WDir = 225    # Wind direction

mdvar['useWSDir']         = useWSDir
mdvar['WS']               = WS   
mdvar['WDir']             = WDir 
mdvar['img_ABL_settings'] = imagedir+'/ABL_settings.png'

case.setAMRWindInput('useWSDir',      useWSDir)
case.setAMRWindInput('ABL_windspeed', WS,   forcechange=True)
case.setAMRWindInput('ABL_winddir',   WDir, forcechange=True)
case.ABL_calculateWindVector()

screenshot.Xvfb_screenshot(mdvar['img_ABL_settings'], crop=(0, 0, 515, 400))

if gennbfile:
    txt  = getsubstring(mdstr, '<!--WINDPROPSTART-->', '<!--WINDPROPEND-->')
    NBADDMARKDOWN(nb, txt.format(**mdvar))
    txt  = setAMRWindInputString(case, 'case', 'useWSDir')
    txt += setAMRWindInputString(case, 'case', 'ABL_windspeed',
                                 extra='forcechange=True')
    txt += setAMRWindInputString(case, 'case', 'ABL_winddir',  
                                 extra='forcechange=True')
    txt += "case.ABL_calculateWindVector()"
    NBADDCELL(nb, txt)
###########################

###########################
turbinescsv="""# CSV file should have columns with
# name, x, y, type, yaw, hubheight, options
T0, 500, 300, UnifCtTest, , ,
T1, 500, 700, UnifCtTest, , ,"""

domainsize       = [1000,1000,1000]   # Domain size [x,y,z] in meters
backgrounddeltax = 10                 # Background mesh delta x in meters

mdvar['turbinescsv']                 = turbinescsv
mdvar['img_farm_turbine_layout']     = imagedir+'/farm_turbine_layout.png'
mdvar['domainsize']                  = domainsize
mdvar['backgrounddeltax']            = backgrounddeltax
mdvar['turbinecsv_help']             = screenshot.gethelpmesg(farmyaml, 
                                                 'frame_farmturbinecsv')

# Set the parameters
case.setAMRWindInput('turbines_csvtextbox',         turbinescsv)
case.setAMRWindInput('turbines_domainsize',         domainsize)
case.setAMRWindInput('turbines_backgroundmeshsize', backgrounddeltax)
case.setAMRWindInput('turbines_deleteprev', True)   # Delete any existing turbines from the system

#-------------------------------------------------------
case.notebook.select(farmtab)
case.toggledframes['frame_farmturbines'].setstate(True)

screenshot.scrollcanvas(case.notebook._tab['Farm'].canvas, 1.0)
screenshot.Xvfb_screenshot(mdvar['img_farm_turbine_layout'], 
                           crop=(0,60,515,scrheight-175))
#--------------------------
if gennbfile:
    txt = """\
## Create wind farm layout and domain

We'll use a CSV format to provde the turbine layout information.  This
will generally take the form of something like 

```
{turbinescsv}
```

Here's what each of the columns mean:

{turbinecsv_help}
"""
    NBADDMARKDOWN(nb, txt.format(**mdvar))

    txt = """\
turbinescsv=\"\"\"
{turbinescsv}
\"\"\"
case.setAMRWindInput('turbines_csvtextbox',  turbinescsv)"""
    NBADDCELL(nb, txt.format(**mdvar))

    txt = """\
Note that if you already have a csv file saved somewhere, you can load that directly using the `loadTurbineCSVFile()` command, like
```python
case.loadTurbineCSVFile('turbines.csv')
```
instead of using `setAMRWindInput()` to set the variable directly.

We'll also need to set some thing things like the turbine domainsize
of `{domainsize}` and background mesh size of `{backgrounddeltax}`m."""
    NBADDMARKDOWN(nb, txt.format(**mdvar))

    txt  = setAMRWindInputString(case, 'case', 'turbines_domainsize')
    txt += setAMRWindInputString(case, 'case', 'turbines_backgroundmeshsize')
    txt += setAMRWindInputString(case, 'case', 'turbines_deleteprev')
    NBADDCELL(nb, txt)
###########################

###########################
fig, ax = plt.subplots(figsize=(5,5), facecolor='w', dpi=150)
case.turbines_previewAllTurbines(ax=ax)
plt.tight_layout()
mdvar['img_turbine_layout_preview']=imagedir+'/farm_turbine_layout_preview.png'
plt.savefig(mdvar['img_turbine_layout_preview'])
#--------------------------
if gennbfile:
    txt = """You can preview the locations and domain size using the `turbines_previewAllTurbines() command:`"""
    NBADDMARKDOWN(nb, txt.format(**mdvar))

    txt = """\
# Preview the turbine layout
fig, ax = plt.subplots(figsize=(5,5), facecolor='w', dpi=150)
case.turbines_previewAllTurbines(ax=ax)
"""
    NBADDCELL(nb, txt)

###########################

###########################
case.turbines_createAllTurbines()
case.notebook.select(4)
mdvar['img_farm_turbine_created'] = imagedir+'/farm_turbine_created.png'
screenshot.Xvfb_screenshot(mdvar['img_farm_turbine_created'], 
                           crop=(0,60,515,scrheight-350))
# ------------------
if gennbfile:
    txt="""In this next step, we tell it to actually create the turbines specified in the CSV input."""
    NBADDMARKDOWN(nb, txt.format(**mdvar))

    txt="""
case.turbines_createAllTurbines()

# Print out existing list of turbines, just to confirm that the turbines got made
print(case.listboxpopupwindict['listboxactuator'].getitemlist())
"""
    NBADDCELL(nb, txt)
###########################



###########################
refinementcsv="""# CSV file should have columns with
# level, upstream, downstream, lateral, below, above, options
level, upstream, downstream, lateral, below, above, options
0,     1,    1,   1,   0.75, 1,
1,     0.5,  0.5, 0.5, 0.75, 1,"""

mdvar['refinementcsv']           = refinementcsv
mdvar['img_farm_refinementspec'] = imagedir+'/farm_refinementspec.png'
mdvar['refinecsv_help']  = screenshot.gethelpmesg(farmyaml, 
                                                  'frame_farmrefinecsv')

case.setAMRWindInput('refine_csvtextbox', refinementcsv)
case.setAMRWindInput('refine_deleteprev', True)
#-------------------------------------------------------
#logging.info("Main    : saving farm_refinementspec.png")
time.sleep(0.25)
case.notebook.select(farmtab)
case.toggledframes['frame_farmrefinement'].setstate(True)

screenshot.scrollcanvas(case.notebook._tab['Farm'].canvas, 1.0)
screenshot.Xvfb_screenshot(mdvar['img_farm_refinementspec'], 
                           crop=(0,60+200,515,scrheight-120))
###########################


###########################
case.refine_createAllZones()
p = case.launchpopupwin('plotdomain', savebutton=False)
time.sleep(0.1)
p.temp_inputvars['plot_refineboxes'].tkentry.select_set(0, Tk.END)
p.temp_inputvars['plot_turbines'].tkentry.select_set(0, Tk.END)
mdvar['img_plotDomainWin_refinezone'] = imagedir+'/plotDomainWin_refinezone.png'
screenshot.Xvfb_screenshot(mdvar['img_plotDomainWin_refinezone'], 
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
mdvar['img_plotDomainFig_refineturbine'] = imagedir+'/plotDomainFig_refineturbine.png'
plt.savefig(mdvar['img_plotDomainFig_refineturbine'])
###########################


###########################
samplingcsv="""# CSV file should have columns withturbinescsv=
# name, type, upstream, downstream, lateral, below, above, n1, n2, options
name, type, upstream, downstream, lateral, below, above, n1, n2, options
cl1, centerline, 1,  0, none, none,  none,  11, 11, none
rp1, rotorplane, 2,  0, none, none,  none,  11, 11, none
sw1, streamwise, 2,  1, 1, 0.5,  0.5,  11, 11, usedx:0.25 noffsets:1
hh,  hubheight,  2,  1, 1, 0,  none,  11, 11, usedx:0.25 center:farm orientation:x"""

mdvar['samplingcsv']           = samplingcsv
mdvar['img_farm_samplingspec'] = imagedir+'/farm_samplingspec.png'
mdvar['samplingcsv_help']      = screenshot.gethelpmesg(farmyaml, 
                                                        'frame_farmrefinecsv')

case.setAMRWindInput('sampling_csvtextbox', samplingcsv)
case.setAMRWindInput('sampling_deleteprev', True)

time.sleep(0.25)
case.notebook.select(farmtab)
case.toggledframes['frame_farmsampling'].setstate(True)

screenshot.scrollcanvas(case.notebook._tab['Farm'].canvas, 1.0)
screenshot.Xvfb_screenshot(mdvar['img_farm_samplingspec'],
                           crop=(0,60+260,515,scrheight-50))


case.sampling_createAllProbes(verbose=False)
###########################

###########################
#case.refine_createAllZones()
p2 = case.launchpopupwin('plotdomain', savebutton=False)
time.sleep(0.1)
p2.temp_inputvars['plot_sampleprobes'].tkentry.select_set(0, Tk.END)
p2.temp_inputvars['plot_refineboxes'].tkentry.select_set(0, Tk.END)
p2.temp_inputvars['plot_turbines'].tkentry.select_set(0, Tk.END)

mdvar['img_plotDomainWin_samplingzone'] = imagedir+'/plotDomainWin_samplingzone.png'
screenshot.Xvfb_screenshot(mdvar['img_plotDomainWin_samplingzone'], 
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
mdvar['img_plotDomainFig_refineturbinesampling'] = imagedir+'/plotDomainFig_refineturbinesampling.png'
plt.savefig(mdvar['img_plotDomainFig_refineturbinesampling'])
###########################

###########################
# See the input file
inputfile = case.writeAMRWindInput('')
mdvar['amrwindinput1'] = "\n".join([s for s in inputfile.split("\n") if s])
###########################


###########################
# Set up a wind sweep
case.notebook.select(farmtab)
case.toggledframes['frame_runsweep'].setstate(True)

windspeeds   = [10, 20]        
winddirs     = [270, 225]      

caseprefix   = "Tutorial3_Case_{CASENUM}"     
usenewdirs   = False                          
logfile      = 'Tutorial3_logfile.yaml'       

mdvar['img_farm_runsweepspec'] = imagedir+'/farm_runsweepspec.png'
mdvar['sweep_windspeeds']      = ' '.join([repr(x) for x in windspeeds])
mdvar['sweep_winddirs']        = ' '.join([repr(x) for x in winddirs])
mdvar['caseprefix']            = caseprefix
mdvar['usenewdirs']            = usenewdirs
mdvar['logfile']               = logfile
                                                    
case.setAMRWindInput('sweep_windspeeds',  mdvar['sweep_windspeeds'] )
case.setAMRWindInput('sweep_winddirs',    mdvar['sweep_winddirs'])
case.setAMRWindInput('sweep_caseprefix',  caseprefix)
case.setAMRWindInput('sweep_usenewdirs',  usenewdirs)
case.setAMRWindInput('sweep_logfile',     logfile)

case.notebook.select(farmtab)
case.toggledframes['frame_farmrefinement'].setstate(True)
screenshot.scrollcanvas(case.notebook._tab['Farm'].canvas, 1.0)
screenshot.Xvfb_screenshot(mdvar['img_farm_runsweepspec'],
                           crop=(0,60+325,515,scrheight-10))
#--------------------
case.sweep_SetupRunParamSweep(verbose=True)
time.sleep(2)
# Load the logfile
with open (logfile, "r") as myfile:
    mdvar['logfileoutput']=myfile.read()
###########################

###########################
# Save the wind farm setup

mdvar['farm_setupfile']    = 'Tutorial3_WindFarmSetup.yaml'
mdvar['farm_usercomments'] = 'Tutorial3 wind farm setup parameters.'
mdvar['img_farm_savefarmsetup'] = imagedir+'/farm_savefarmsetup.png'

case.setAMRWindInput('farm_setupfile',    mdvar['farm_setupfile'])
case.setAMRWindInput('farm_usercomments', mdvar['farm_usercomments'])

case.notebook.select(farmtab)
case.toggledframes['frame_farmsetup1'].setstate(True)
screenshot.scrollcanvas(case.notebook._tab['Farm'].canvas, 0.0)
screenshot.Xvfb_screenshot(mdvar['img_farm_savefarmsetup'],
                           crop=(0,60,515,scrheight-300))
# write file
case.writeFarmSetupYAML(mdvar['farm_setupfile'])
with open (mdvar['farm_setupfile'], "r") as myfile:
    inputfile=myfile.read()
mdvar['farm_setupfile_output'] = "\n".join([s for s in inputfile.split("\n") if s])
###########################


###########################
# WRAP UP AND FINISH
# -------------------------

# Write the markdown file
logging.info("Main    : writing "+mdfile)
with open(mdfile, "w") as f:
    f.write(mdstr.format(**mdvar))

if gennbfile:
    logging.info("Main    : writing "+nbfile)
    nbf.write(nb, nbfile)
    if runjupyter:
        logging.info("Main    : running jupyter on "+nbfile)
        cmd='jupyter nbconvert --execute --inplace '+nbfile
        os.system(cmd)
    
# Quit and clean up
time.sleep(2)
case.quit()
t1.join()
