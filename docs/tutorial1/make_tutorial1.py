# Note to self: run on skybridge with
#  $ module load canopy/2.1.9
#  $ export PYTHONPATH=~/.local/lib/python2.7/site-packages/
#

import logging
import threading
import os, sys, shutil
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
mdtemplate= 'tutorial1gui_template.md'
mdfile    = 'tutorial1gui.md'
nbfile    = 'tutorial1python.ipynb'
gennbfile = False
runjupyter= False
savefigs  = True
# ========================================

farmtab    = 8
samplingtab= 5
turbinetab = 4
refinetab  = 3
abltab     = 2
domaintab  = 1
simtab     = 0

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
# START CREATING THE TUTORIAL
# -------------------------

# Get the startup image
mdvar['img_startup']     = imagedir+'/amrwind_frontend_startup.png'
mdvar['img_timecontrol'] = imagedir+'/simulation_type_timecontrol.png'
mdvar['img_simprops']    = imagedir+'/simulation_propertiesvalues.png'

if savefigs:
    screenshot.Xvfb_screenshot(mdvar['img_startup'], 
                               crop=(0, 0, 1280, 800))
SETINPUT(mdvar, case, 'incflo.physics', ['FreeStream', 'Actuator'])
SETINPUT(mdvar, case, 'time.stop_time', 100)
SETINPUT(mdvar, case, 'time.fixed_dt',  0.1)
SETINPUT(mdvar, case, 'time_control',   ['const dt'])

if savefigs:
    screenshot.Xvfb_screenshot(mdvar['img_timecontrol'], 
                               crop=(0, 0, 400, 400))

SETINPUT(mdvar, case, 'ConstValue.density.value',  1.0)
SETINPUT(mdvar, case, 'ConstValue.velocity.value', [10.0,0.0,0.0])
SETINPUT(mdvar, case, 'turbulence.model',  ['Laminar'])

if savefigs:
    screenshot.scrollcanvas(case.notebook._tab['Simulation'].canvas, 1.0)
    screenshot.Xvfb_screenshot(mdvar['img_simprops'], 
                               crop=(0, 400, 450, 800))

###########################
# DOMAIN
# -------------------------
case.notebook.select(domaintab)
SETINPUT(mdvar, case, 'geometry.prob_lo', [-1000, -500, -500])
SETINPUT(mdvar, case, 'geometry.prob_hi', [ 1000,  500,  500])
SETINPUT(mdvar, case, 'amr.n_cell',       [ 128, 64, 64])
mdvar['img_domainmesh']    = imagedir+'/domain_domainmesh.png'
if savefigs:
    screenshot.Xvfb_screenshot(mdvar['img_domainmesh'], 
                               crop=(0, 0, 515, 310))

SETINPUT(mdvar, case, 'is_periodicx', False)
SETINPUT(mdvar, case, 'is_periodicy', True)
SETINPUT(mdvar, case, 'is_periodicz', True)

SETINPUT(mdvar, case, 'xlo.type', 'mass_inflow')
SETINPUT(mdvar, case, 'xhi.type', 'pressure_outflow')
SETINPUT(mdvar, case, 'xlo.density', 1.0)
SETINPUT(mdvar, case, 'xlo.velocity', [10.0, 0.0, 0.0])

mdvar['img_domainBC']    = imagedir+'/domain_BC.png'

if savefigs:
    case.toggledframes['frame_xBC'].setstate(True)
    screenshot.scrollcanvas(case.notebook._tab['Domain'].canvas, 1.0)
    screenshot.Xvfb_screenshot(mdvar['img_domainBC'], 
                               crop=(0, 200, 515, 680))

###########################
# ABL
# -------------------------
case.notebook.select(abltab)
SETINPUT(mdvar, case, 'incflo.velocity', [10.0, 0.0, 0.0])
case.ABL_calculateWDirWS()
mdvar['img_ABL_WS']    = imagedir+'/ABL_WS.png'

if savefigs:
    screenshot.Xvfb_screenshot(mdvar['img_ABL_WS'], 
                               crop=(0, 25, 515, 340))

###########################
# Plot the domain
# -------------------------
p2 = case.launchpopupwin('plotdomain', savebutton=False)
time.sleep(0.1)

mdvar['img_plotdomain_basic']    = imagedir+'/plotdomain_basic.png'
if savefigs:
    screenshot.Xvfb_screenshot(mdvar['img_plotdomain_basic'], 
                               crop=screenshot.getwinpos(p2))
# -- save the plot --
fig, ax = plt.subplots(figsize=(6,4), facecolor='w', dpi=100)
time.sleep(0.25)
case.plotDomain(ax=ax)
plt.tight_layout()
mdvar['img_plotdomain_level0'] = imagedir+'/plotdomain_domain_level0.png'
if savefigs:
    plt.savefig(mdvar['img_plotdomain_level0'])
time.sleep(0.2)
p2.okclose()

###########################
# Add turbine
# -------------------------
case.notebook.select(turbinetab)
mdvar['img_turbines_addturb']    = imagedir+'/turbines_addturbine.png'
SETINPUT(mdvar, case, 'ICNS.source_terms', ['ActuatorForcing'])

if savefigs:
    screenshot.Xvfb_screenshot(mdvar['img_turbines_addturb'], 
                               crop=(0, 0, 360, 370))
# Set up the turbine
turbine = case.get_default_actuatordict()
turbine['Actuator_name']          = mdvar['Actuator_name'] = 'turbine0'
turbine['Actuator_base_position'] = mdvar['Actuator_base_position'] = [0, 0, -90]
turbine['Actuator_yaw']           = mdvar['Actuator_yaw']     = 270.0
turbine['Actuator_density']       = mdvar['Actuator_density'] = 1.0
turbine['use_turbine_type']       = mdvar['use_turbine_type'] = 'NREL5MW ADM NOSERVO'

# Add the turbine to the simulation.
case.add_turbine(turbine, verbose=True)

mdvar['img_turb_newturb']   = imagedir+'/turbines_newturbine.png'
if savefigs:
    # Get pbub sampling plane
    case.listboxpopupwindict['listboxactuator'].tkentry.selection_set(0)
    p=case.listboxpopupwindict['listboxactuator'].edit()
    screenshot.Xvfb_screenshot(mdvar['img_turb_newturb'],
                               crop=(0,0, 500, 675))
    p.destroy()
    # Delete it
    case.listboxpopupwindict['listboxactuator'].tkentry.selection_set(0)
    p=case.listboxpopupwindict['listboxactuator'].remove()

# Set up the turbine (redo)
turbine2 = case.get_default_actuatordict()
turbine2['Actuator_name']          = mdvar['Actuator_name'] 
turbine2['Actuator_base_position'] = mdvar['Actuator_base_position'] 
turbine2['Actuator_yaw']           = mdvar['Actuator_yaw']     
turbine2['Actuator_density']       = mdvar['Actuator_density'] 

turbine2 = case.turbinemodels_applyturbinemodel(turbine2, mdvar['use_turbine_type'],
                                                docopy=True, updatefast=True)

# Add the turbine to the simulation.
case.add_turbine(turbine2, verbose=True)

mdvar['img_turb_newturb_filled']   = imagedir+'/turbines_newturb_filledout.png'
if savefigs:
    case.listboxpopupwindict['listboxactuator'].tkentry.selection_set(0)
    p=case.listboxpopupwindict['listboxactuator'].edit()
    screenshot.Xvfb_screenshot(mdvar['img_turb_newturb_filled'],
                               crop=(0,0, 500, 675))
    p.destroy()

###########################
# Add refinement region
# -------------------------
case.notebook.select(refinetab)
mdvar['img_refine_maxlevel']    = imagedir+'/refinement_maxlevel1.png'
mdvar['img_refine_box']         = imagedir+'/refinement_box.png'
mdvar['img_refine_box_bot']     = imagedir+'/refinement_box_bottom.png'
SETINPUT(mdvar, case, 'amr.max_level', 1)
if savefigs:
    screenshot.Xvfb_screenshot(mdvar['img_refine_maxlevel'],
                               crop=(0,0, 450, 220))

refinewin = case.get_default_taggingdict()
refinewin['tagging_name']         = mdvar['tagging_name'] = 'box1'
refinewin['tagging_shapes']       = mdvar['tagging_shapes'] = 'box1'
refinewin['tagging_type']         = mdvar['tagging_type'] = 'GeometryRefinement'
refinewin['tagging_level']        = mdvar['tagging_level'] = 0
refinewin['tagging_geom_type']    = mdvar['tagging_geom_type'] = 'box'
refinewin['tagging_geom_origin']  = mdvar['tagging_geom_origin'] = [-200, -200, -200]
refinewin['tagging_geom_xaxis']   = mdvar['tagging_geom_xaxis'] = [400, 0, 0]
refinewin['tagging_geom_yaxis']   = mdvar['tagging_geom_yaxis'] = [0, 400, 0]
refinewin['tagging_geom_zaxis']   = mdvar['tagging_geom_zaxis'] = [0, 0, 400]
case.add_tagging(refinewin)

if savefigs:
    case.listboxpopupwindict['listboxtagging'].tkentry.selection_set(0)
    p=case.listboxpopupwindict['listboxtagging'].edit()
    p.popup_toggledframes['geom_frame'].setstate(True)
    screenshot.Xvfb_screenshot(mdvar['img_refine_box'], 
                               crop=(0, 0, 500, 250))
    screenshot.scrollcanvas(p.scrolledframe.canvas, 1.0)
    screenshot.Xvfb_screenshot(mdvar['img_refine_box_bot'], 
                               crop=(0, 100, 500, 550))
    p.destroy()

###########################
# Add sampling plane
# -------------------------
SETINPUT(mdvar, case,  'incflo.post_processing',    ['sampling'])
SETINPUT(mdvar, case,  'sampling.output_frequency', 100)
SETINPUT(mdvar, case,  'sampling.fields',           ['velocity'])
case.notebook.select(samplingtab)

mdvar['img_io_sampling']         = imagedir+'/io_sampling.png'
mdvar['img_io_sampling_done']    = imagedir+'/io_sampling_done.png'
mdvar['img_io_xyplane_top']      = imagedir+'/io_xyplane_top.png'
mdvar['img_io_xyplane']          = imagedir+'/io_xyplane.png'
if savefigs:
    screenshot.Xvfb_screenshot(mdvar['img_io_sampling'], 
                               crop=(0, 210, 400, 500))
    
sampleplane = case.get_default_samplingdict()

# Modify the geometry
sampleplane['sampling_name'] = 'xyplane'
sampleplane['sampling_type'] = 'PlaneSampler'
sampleplane['sampling_p_num_points'] = [101, 51]
sampleplane['sampling_p_origin']     = [-1000, -500, 0]
sampleplane['sampling_p_axis1']      = [2000, 0, 0]
sampleplane['sampling_p_axis2']      = [0, 1000, 0]

# Add sampling plane to simuation
case.add_sampling(sampleplane, verbose=True)

if savefigs:
    case.listboxpopupwindict['listboxsampling'].tkentry.selection_set(0)
    p=case.listboxpopupwindict['listboxsampling'].edit()
    screenshot.Xvfb_screenshot(mdvar['img_io_xyplane_top'], 
                               crop=(0, 0, 450, 200))
    p.popup_toggledframes['sample_plane_frame'].setstate(True)
    screenshot.scrollcanvas(p.scrolledframe.canvas, 1.0)
    screenshot.Xvfb_screenshot(mdvar['img_io_xyplane'], 
                               crop=(0, 175, 450, 450))
    p.destroy()

if savefigs:
    screenshot.Xvfb_screenshot(mdvar['img_io_sampling_done'], 
                               crop=(0, 210, 400, 500))

###########################
# Plot the domain
# -------------------------
p3 = case.launchpopupwin('plotdomain', savebutton=False)
time.sleep(0.1)

mdvar['img_plotdomain_basicfilled']    = imagedir+'/plotdomain_basicfilled.png'
mdvar['img_plotdomain_selected1']      = imagedir+'/plotdomain_selected_box1turbine0.png'
mdvar['img_plotdomain_domain_turbrefine'] = imagedir+'/plotdomain_domain_turbrefine.png'
mdvar['img_plotdomain_domain_xyplane'] = imagedir+'/plotdomain_domain_xyplane.png'

if savefigs:
    screenshot.Xvfb_screenshot(mdvar['img_plotdomain_basicfilled'], 
                               crop=screenshot.getwinpos(p3))

p3.temp_inputvars['plot_refineboxes'].tkentry.selection_set(0)
p3.temp_inputvars['plot_turbines'].tkentry.selection_set(0)
if savefigs:
    screenshot.Xvfb_screenshot(mdvar['img_plotdomain_selected1'], 
                               crop=screenshot.getwinpos(p3))

time.sleep(0.2)
p3.okclose()

# -- Create a figure and axes to plot domain --
plt.rc('font', size=14)
fig, ax = plt.subplots(figsize=(6,4), facecolor='w',dpi=100)
# Set additional items to plot
#case.popup_storteddata['plotdomain']['plot_sampleprobes']    = ['xyplane']
case.popup_storteddata['plotdomain']['plot_turbines']        = ['turbine0']
case.popup_storteddata['plotdomain']['plot_refineboxes']     = ['box1']
# Plot the figure 
case.plotDomain(ax=ax)
plt.tight_layout()
if savefigs:
    plt.savefig(mdvar['img_plotdomain_domain_turbrefine'])

# -- Create a figure and axes to plot domain --
plt.rc('font', size=14)
fig, ax = plt.subplots(figsize=(6,4), facecolor='w',dpi=100)
# Set additional items to plot
case.popup_storteddata['plotdomain']['plot_sampleprobes']    = ['xyplane']
case.popup_storteddata['plotdomain']['plot_turbines']        = ['turbine0']
case.popup_storteddata['plotdomain']['plot_refineboxes']     = []
# Plot the figure 
case.plotDomain(ax=ax)
plt.tight_layout()
if savefigs:
    plt.savefig(mdvar['img_plotdomain_domain_xyplane'])


###########################
# Validate the inputs
# -------------------------
checkoutput=case.validate()
mdvar['checkoutput'] = checkoutput

###########################
# Saving the input file
# -------------------------
outstr=case.writeAMRWindInput('')

# Delete any empty lines
mdvar['inputfilestr'] = os.linesep.join([s for s in outstr.splitlines() if s])

outstr=case.writeAMRWindInput('tutorial1.inp')

###########################
# WRAP UP AND FINISH
# -------------------------

# Delete the openfast directory
if os.path.exists('turbine0_OpenFAST_NREL5MW'): 
    shutil.rmtree('turbine0_OpenFAST_NREL5MW')

# Write the markdown file
logging.info("Main    : writing "+mdfile)
with open(mdfile, "w") as f:
    f.write(mdstr.format(**mdvar))

# Quit and clean up
time.sleep(2)
case.quit()
time.sleep(2)
t1.join()
