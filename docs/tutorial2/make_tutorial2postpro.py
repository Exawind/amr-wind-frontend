import logging
import threading
import os, sys
import time
import matplotlib.pyplot    as plt

if sys.version_info[0] < 3:
    import Tkinter as Tk
    from cStringIO import StringIO
else:
    import tkinter as Tk
    from io import StringIO

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

def SETINPUT(mddict, case, key, val,**kwargs):
    #mdkey = key.split('.')[-1] if len(key.split('.'))>1 else key
    mdkey = key.replace('.','_')
    mddict[mdkey] = val
    case.setAMRWindInput(key, val, **kwargs)
    return

# ========================================
# Set the tutorial properties
scrwidth  = 1280
scrheight = 800
imagedir  = 'images'
mdtemplate= 'tutorial2guipostpro_template.md'
mdfile    = 'tutorial2guipostpro.md'
nbfile    = 'tutorial2python.ipynb'
gennbfile = False
runjupyter= True
savefigs  = True
docheap   = True  # Use pre-saved output to speed up some steps
# ========================================

mdstr = ""
mdvar = {}
mdvar['makescript'] = __file__

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

###########################
# POSTPRO
#  Load and get the ABL report
case.notebook.select(6)

SETINPUT(mdvar, case, 'ablstats_file', 'post_processing/abl_statistics00000.nc')
SETINPUT(mdvar, case, 'ablstats_avgt', [15000, 20000])
SETINPUT(mdvar, case, 'ablstats_avgz', '57.19')
case.ABLpostpro_loadnetcdffile()
mdvar['img_postpro_statsfile']   = imagedir+'/postpro_statsfile.png'

if savefigs:
    screenshot.Xvfb_screenshot(mdvar['img_postpro_statsfile'], 
                               crop=(0, 0, 515, 435))

# Get the basic ABL report 
if docheap:
    mdvar['ablreport'] = """
Loading w'theta'_r
Loading theta
Loading u
Loading v'v'_r
Loading v
Loading u'u'_r
Loading w'w'_r
        z       Uhoriz      WindDir       TI_TKE     TI_horiz        Alpha     ObukhovL 
      ===         ====         ====         ====         ====         ====         ==== 
    57.19 6.129959e+00 2.300725e+02 6.614705e-02 1.035226e-01 2.969138e-02 -4.795765e+01 
"""
else:
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    case.ABLpostpro_printreport()
    sys.stdout=old_stdout
    mdvar['ablreport']=mystdout.getvalue()

print("ABL report: ")
print(mdvar['ablreport'])

###########################
# PLOT USTAR AND ABL FORCING
case.inputvars['ablstats_scalarplot'].tkentry.selection_set(2)
mdvar['img_postpro_selectustar']   = imagedir+'/postpro_selectustar.png'
if savefigs:
    screenshot.Xvfb_screenshot(mdvar['img_postpro_selectustar'], 
                               crop=(0, 250, 515, 425))
case.inputvars['ablstats_scalarplot'].tkentry.selection_clear(0, Tk.END)

# Set up the ustar plot and plot it
fig, ax = plt.subplots(figsize=(6,4), facecolor='w', dpi=150)
time.sleep(0.25)
case.ABLpostpro_plotscalars(ax=ax, plotvars=['ustar'])
ax.set_ylim([0, 0.25])

mdvar['img_postpro_ablustar']   = imagedir+'/postpro_ablustar.png'
# -- save the plot --
if savefigs:
    plt.savefig(mdvar['img_postpro_ablustar'])

# Set up the Tsurf plot and plot it
fig, ax = plt.subplots(figsize=(6,4), facecolor='w', dpi=150)
time.sleep(0.25)
case.ABLpostpro_plotscalars(ax=ax, plotvars=['abl_forcing_x', 'abl_forcing_y'])

mdvar['img_postpro_ablforcing']   = imagedir+'/postpro_ablforcing.png'
# -- save the plot --
if savefigs:
    plt.savefig(mdvar['img_postpro_ablforcing'])

###########################
# PLOT VELOCITY PROFILE

case.inputvars['ablstats_profileplot'].tkentry.selection_set(1)
mdvar['img_postpro_selectuhoriz']   = imagedir+'/postpro_selectuhoriz.png'
if savefigs:
    screenshot.Xvfb_screenshot(mdvar['img_postpro_selectuhoriz'], 
                               crop=(0, 115, 515, 280))
case.inputvars['ablstats_profileplot'].tkentry.selection_clear(0, Tk.END)

# Plot ustar
fig, ax = plt.subplots(figsize=(6,6), facecolor='w', dpi=150)
case.ABLpostpro_plotprofiles(ax=ax, plotvars=['Uhoriz'], 
                             avgt=mdvar['ablstats_avgt'])

mdvar['img_postpro_abluhoriz']   = imagedir+'/postpro_abluhoriz.png'
# -- save the plot --
if savefigs:
    plt.savefig(mdvar['img_postpro_abluhoriz'])

# Plot tke
fig, ax = plt.subplots(figsize=(6,6), facecolor='w', dpi=150)
case.ABLpostpro_plotprofiles(ax=ax, plotvars=['TKE'], 
                             avgt=mdvar['ablstats_avgt'])

mdvar['img_postpro_abltke']   = imagedir+'/postpro_abltke.png'
# -- save the plot --
if savefigs:
    plt.savefig(mdvar['img_postpro_abltke'])

###########################
# PLOT SAMPLING PLANE
screenshot.scrollcanvas(case.notebook._tab['Postpro'].canvas, 1.0)
SETINPUT(mdvar, case, 'sampling_file', 'post_processing/sampling00000.nc')
case.Samplepostpro_loadnetcdffile()

mdvar['img_postpro_samplefile1']   = imagedir+'/postpro_samplefile1.png'
if savefigs:
    screenshot.Xvfb_screenshot(mdvar['img_postpro_samplefile1'], 
                               crop=(0, 350, 515, scrheight))

case.inputvars['samplingprobe_groups'].tkentry.selection_set(0)
case.Samplepostpro_getvars()
mdvar['img_postpro_samplefile2']   = imagedir+'/postpro_samplefile2.png'
if savefigs:
    screenshot.Xvfb_screenshot(mdvar['img_postpro_samplefile2'], 
                               crop=(0, 350, 515, scrheight))

case.inputvars['samplingprobe_variables'].tkentry.selection_set(1)
SETINPUT(mdvar, case, 'samplingprobe_plotaxis1', 'X')
SETINPUT(mdvar, case, 'samplingprobe_plotaxis2', 'Y')
SETINPUT(mdvar, case, 'samplingprobe_plottimeindex', 400)
mdvar['img_postpro_samplefile3']   = imagedir+'/postpro_samplefile3.png'
if savefigs:
    screenshot.Xvfb_screenshot(mdvar['img_postpro_samplefile3'], 
                               crop=(0, 350, 515, scrheight))

# -- Create the plot --
levels = 41 #np.linspace(0,8,41)
from matplotlib import cm
fig, ax = plt.subplots(figsize=(6,5), facecolor='w', dpi=150)
im1 = case.plotSamplePlane('p_hub', 'velocityx', 400, 0, 'X','Y',
                           ax=ax, colorbar=False, levels=levels, cmap=cm.jet)
fig.colorbar(im1[0])

mdvar['img_postpro_phub_u']   = imagedir+'/postpro_phub_u.png'
# -- save the plot --
if savefigs:
    plt.savefig(mdvar['img_postpro_phub_u'])

###########################
# WRAP UP AND FINISH
# -------------------------

# Write the markdown file
logging.info("Main    : writing "+mdfile)
with open(mdfile, "w") as f:
    f.write(mdstr.format(**mdvar))

# Quit and clean up
time.sleep(2)
case.quit()
t1.join()
