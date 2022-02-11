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

getsubstring  = lambda x, s1, s2: x.split(s1)[-1].split(s2)[0]

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
mdtemplate= 'tutorial2guisetup_template.md'
mdfile    = 'tutorial2guisetup.md'
nbfile    = 'tutorial2python.ipynb'
gennbfile = False
runjupyter= True
savefigs  = False
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
# SIMULATION SETTINGS
case.notebook.select(0)

SETINPUT(mdvar, case, 'time.stop_time',           20000.0)
SETINPUT(mdvar, case, 'time.max_step',            40000)
SETINPUT(mdvar, case, 'time.fixed_dt',            0.5)
SETINPUT(mdvar, case, 'time.checkpoint_interval', 2000)            ## MOVELATER
mdvar['time_control']        = 'const dt'

SETINPUT(mdvar, case, 'transport.viscosity', 1.8375e-05)

SETINPUT(mdvar, case, 'incflo.physics', ['ABL'])
SETINPUT(mdvar, case, 'io.check_file', 'chk')

SETINPUT(mdvar, case, 'incflo.use_godunov', True)
SETINPUT(mdvar, case, 'incflo.godunov_type', 'weno')

SETINPUT(mdvar, case, 'turbulence.model',    ['OneEqKsgsM84'])
SETINPUT(mdvar, case, 'TKE.source_terms',    ['KsgsM84Src'])

mdvar['img_time_settings']   = imagedir+'/time_settings.png'
mdvar['img_turb_settings']   = imagedir+'/turb_settings.png'

if savefigs:
    screenshot.Xvfb_screenshot(mdvar['img_time_settings'], 
                               crop=(0, 0, 515, 500))
    screenshot.scrollcanvas(case.notebook._tab['Simulation'].canvas, 1.0)
    screenshot.Xvfb_screenshot(mdvar['img_turb_settings'], 
                               crop=(0, 300, 515, scrheight))

###########################
# DOMAIN AND BC
case.notebook.select(1)
SETINPUT(mdvar, case, 'geometry.prob_lo', [ 0.0, 0.0, 0.0 ])
SETINPUT(mdvar, case, 'geometry.prob_hi', [1536.0, 1536.0, 1920.0])
SETINPUT(mdvar, case, 'amr.n_cell',       [128, 128, 160])
SETINPUT(mdvar, case, 'is_periodicx', True)
SETINPUT(mdvar, case, 'is_periodicy', True)
SETINPUT(mdvar, case, 'is_periodicz', False)
mdvar['img_domain_settings']   = imagedir+'/domain_settings.png'
if savefigs:
    screenshot.Xvfb_screenshot(mdvar['img_domain_settings'], 
                               crop=(0, 0, 515, 425))

SETINPUT(mdvar, case, 'zlo.type',             'wall_model')
SETINPUT(mdvar, case, 'zlo.temperature_type', 'wall_model')          
SETINPUT(mdvar, case, 'zlo.tke_type',         'zero_gradient')       
SETINPUT(mdvar, case, 'zhi.type',             'slip_wall')           
SETINPUT(mdvar, case, 'zhi.temperature_type', 'fixed_gradient')      
SETINPUT(mdvar, case, 'zhi.temperature',      0.000974025974) 
mdvar['img_zBC_settings']   = imagedir+'/zBC_settings.png'

case.toggledframes['frame_zBC'].setstate(True)
screenshot.scrollcanvas(case.notebook._tab['Domain'].canvas, 1.0)
if savefigs:
    screenshot.Xvfb_screenshot(mdvar['img_zBC_settings'], 
                               crop=(0, 425, 515, scrheight))

###########################
# ABL SETTINGS
case.notebook.select(2)
SETINPUT(mdvar, case, 'ABLForcing',        True)
SETINPUT(mdvar, case, 'BoussinesqForcing', True)
SETINPUT(mdvar, case, 'CoriolisForcing',   True)

#SETINPUT(mdvar, case, 'incflo.velocity', [4.70059422901, 3.93463008353, 0.0])
SETINPUT(mdvar, case, 'useWSDir', True)
SETINPUT(mdvar, case, 'ABL_windspeed', 6.13)
SETINPUT(mdvar, case, 'ABL_winddir',   230.07)
case.ABL_calculateWindVector()
#SETINPUT(mdvar, case, 'useWSDir', False)

SETINPUT(mdvar, case, 'ABLForcing.abl_forcing_height',   57.19)
SETINPUT(mdvar, case, 'ABL.kappa',                       0.4) 

SETINPUT(mdvar, case, 'ABL.normal_direction',      2)
SETINPUT(mdvar, case, 'ABL.surface_roughness_z0',  0.0001)
SETINPUT(mdvar, case, 'ABL.reference_temperature', 288.15)
SETINPUT(mdvar, case, 'ABL.surface_temp_rate',     0.0)
SETINPUT(mdvar, case, 'ABL.surface_temp_flux',     0.0122096146646)

SETINPUT(mdvar, case, 'ABL.mo_beta_m',             16.0)
SETINPUT(mdvar, case, 'ABL.mo_gamma_m',            5.0)
SETINPUT(mdvar, case, 'ABL.mo_gamma_h',            5.0)

SETINPUT(mdvar, case, 'CoriolisForcing.latitude',  55.49)
SETINPUT(mdvar, case, 'BoussinesqBuoyancy.reference_temperature', 288.15) 

SETINPUT(mdvar, case, 'ABL.temperature_heights', '1050.0 1150.0 1920.0')
SETINPUT(mdvar, case, 'ABL.temperature_values',  '288.15 296.15 296.9')

SETINPUT(mdvar, case, 'ABL.perturb_ref_height', None)
SETINPUT(mdvar, case, 'ABL.Uperiods', None)
SETINPUT(mdvar, case, 'ABL.Vperiods', None)
SETINPUT(mdvar, case, 'ABL.deltaU',   None)
SETINPUT(mdvar, case, 'ABL.deltaV',   None)
SETINPUT(mdvar, case, 'ABL.theta_amplitude',   None)
SETINPUT(mdvar, case, 'ABL.cutoff_height',   None)

SETINPUT(mdvar, case, 'ABL.stats_output_frequency',   1)                   
SETINPUT(mdvar, case, 'ABL.stats_output_format',      'netcdf')            

mdvar['img_ABL_settings1']   = imagedir+'/ABL_settings1.png'
mdvar['img_ABL_settings2']   = imagedir+'/ABL_settings2.png'

# case.setAMRWindInput('ICNS.source_terms',     ['ABLForcing','BoussinesqBuoyancy', 'CoriolisForcing'])

if savefigs:
    screenshot.Xvfb_screenshot(mdvar['img_ABL_settings1'], 
                               crop=(0, 0, 515, scrheight))
    screenshot.scrollcanvas(case.notebook._tab['ABL'].canvas, 1.0)
    screenshot.Xvfb_screenshot(mdvar['img_ABL_settings2'], 
                               crop=(0, 0, 515, scrheight))

###########################
# SAMPLING SETTINGS
case.notebook.select(5)
SETINPUT(mdvar, case, 'incflo.verbose',            3)
SETINPUT(mdvar, case, 'time.plot_interval',        2000)
SETINPUT(mdvar, case, 'incflo.post_processing',    ['sampling'])            
SETINPUT(mdvar, case, 'sampling.output_frequency', 100)                 
SETINPUT(mdvar, case, 'sampling.fields',           ['velocity', 'temperature'])
mdvar['img_IO_settings']   = imagedir+'/IO_settings.png'

sampleplane = case.get_default_samplingdict()
# Modify the geometry
sampleplane['sampling_name']         = mdvar['phub_name']   = 'p_hub'
sampleplane['sampling_type']         = mdvar['phub_type']   = 'PlaneSampler'
sampleplane['sampling_p_num_points'] = mdvar['phub_Npts']   = [129, 129]
sampleplane['sampling_p_origin']     = mdvar['phub_origin'] = [0, 0, 0]
sampleplane['sampling_p_axis1']      = mdvar['phub_axis1']  = [1536, 0, 0]
sampleplane['sampling_p_axis2']      = mdvar['phub_axis2']  = [0, 1536, 0]
sampleplane['sampling_p_normal']     = mdvar['phub_normal'] = [0, 0, 1]
sampleplane['sampling_p_offsets']    = mdvar['phub_offset'] = '17        28.5      41        57        77        90'
case.add_sampling(sampleplane)
mdvar['img_phub_settings1']   = imagedir+'/phub_settings1.png'
mdvar['img_phub_settings2']   = imagedir+'/phub_settings2.png'

sampleplane = case.get_default_samplingdict()
sampleplane['sampling_name']         = mdvar['xbc_name']   = 'xbc'
sampleplane['sampling_type']         = mdvar['xbc_type']   = 'PlaneSampler'
sampleplane['sampling_p_num_points'] = mdvar['xbc_Npts']   = [257, 161]
sampleplane['sampling_p_origin']     = mdvar['xbc_origin'] = [0, 0, 0]
sampleplane['sampling_p_axis1']      = mdvar['xbc_axis1']  = [0, 1536, 0]
sampleplane['sampling_p_axis2']      = mdvar['xbc_axis2']  = [0, 0, 1920]
sampleplane['sampling_p_normal']     = mdvar['xbc_normal'] = [1, 0, 0]
sampleplane['sampling_p_offsets']    = mdvar['xbc_offset'] = '0.0 1536'
case.add_sampling(sampleplane)
mdvar['img_xbc_settings1']   = imagedir+'/xbc_settings1.png'
mdvar['img_xbc_settings2']   = imagedir+'/xbc_settings2.png'

sampleplane = case.get_default_samplingdict()
sampleplane['sampling_name']         = mdvar['ybc_name']   = 'ybc'
sampleplane['sampling_type']         = mdvar['ybc_type']   = 'PlaneSampler'
sampleplane['sampling_p_num_points'] = mdvar['ybc_Npts']   = [257, 161]
sampleplane['sampling_p_origin']     = mdvar['ybc_origin'] = [0, 0, 0]
sampleplane['sampling_p_axis1']      = mdvar['ybc_axis1']  = [1536, 0, 0]
sampleplane['sampling_p_axis2']      = mdvar['ybc_axis2']  = [0, 0, 1920]
sampleplane['sampling_p_normal']     = mdvar['ybc_normal'] = [0, 1, 0]
sampleplane['sampling_p_offsets']    = mdvar['ybc_offset'] = '0.0 1536'
case.add_sampling(sampleplane)
mdvar['img_ybc_settings1']   = imagedir+'/ybc_settings1.png'
mdvar['img_ybc_settings2']   = imagedir+'/ybc_settings2.png'


if savefigs:
    screenshot.Xvfb_screenshot(mdvar['img_IO_settings'], 
                               crop=(0, 0, 515, 500))

    # Get pbub sampling plane
    case.listboxpopupwindict['listboxsampling'].tkentry.selection_set(0)
    p=case.listboxpopupwindict['listboxsampling'].edit()
    p.popup_toggledframes['sample_plane_frame'].setstate(True)
    screenshot.Xvfb_screenshot(mdvar['img_phub_settings1'],
                               crop=(0,0, 450, 125))
    screenshot.scrollcanvas(p.scrolledframe.canvas, 1.0)
    screenshot.Xvfb_screenshot(mdvar['img_phub_settings2'],
                               crop=(0,200, 450, 450))
    p.destroy()

    # Get xbc sampling plane
    case.listboxpopupwindict['listboxsampling'].tkentry.selection_clear(0, Tk.END)
    case.listboxpopupwindict['listboxsampling'].tkentry.selection_set(1)
    p=case.listboxpopupwindict['listboxsampling'].edit()
    p.popup_toggledframes['sample_plane_frame'].setstate(True)
    screenshot.Xvfb_screenshot(mdvar['img_xbc_settings1'],
                               crop=(0,0, 450, 125))
    screenshot.scrollcanvas(p.scrolledframe.canvas, 1.0)
    screenshot.Xvfb_screenshot(mdvar['img_xbc_settings2'],
                               crop=(0,200, 450, 450))
    p.destroy()

    # Get ybc sampling plane
    case.listboxpopupwindict['listboxsampling'].tkentry.selection_clear(0, Tk.END)
    case.listboxpopupwindict['listboxsampling'].tkentry.selection_set(1)
    p=case.listboxpopupwindict['listboxsampling'].edit()
    p.popup_toggledframes['sample_plane_frame'].setstate(True)
    screenshot.Xvfb_screenshot(mdvar['img_ybc_settings1'],
                               crop=(0,0, 450, 125))
    screenshot.scrollcanvas(p.scrolledframe.canvas, 1.0)
    screenshot.Xvfb_screenshot(mdvar['img_ybc_settings2'],
                               crop=(0,200, 450, 450))
    p.destroy()

###########################
# EXPERT SETTINGS
case.notebook.select(7)

SETINPUT(mdvar, case, 'nodal_proj.mg_rtol'                       , 1e-06)
SETINPUT(mdvar, case, 'nodal_proj.mg_atol'                       , 1e-12)
SETINPUT(mdvar, case, 'mac_proj.mg_rtol'                         , 1e-06)
SETINPUT(mdvar, case, 'mac_proj.mg_atol'                         , 1e-12)
SETINPUT(mdvar, case, 'diffusion.mg_rtol'                        , 1e-06)
SETINPUT(mdvar, case, 'diffusion.mg_atol'                        , 1e-12)
SETINPUT(mdvar, case, 'temperature_diffusion.mg_rtol'            , 1e-10)
SETINPUT(mdvar, case, 'temperature_diffusion.mg_atol'            , 1e-13)

SETINPUT(mdvar, case, 'ABL.random_gauss_mean',     0.0)
SETINPUT(mdvar, case, 'ABL.random_gauss_var',      1.0)

mdvar['img_expert_settings']   = imagedir+'/expert_settings.png'

if savefigs:
    screenshot.Xvfb_screenshot(mdvar['img_expert_settings'], 
                               crop=(0, 0, 515, 425))

###########################
# -- Plot the domain --
p2 = case.launchpopupwin('plotdomain', savebutton=False)
time.sleep(0.1)
p2.temp_inputvars['plot_sampleprobes'].tkentry.select_set(0, Tk.END)

mdvar['img_plotDomainWin_samplingzone'] = imagedir+'/plotDomainWin_samplingzone.png'
if savefigs:
    screenshot.Xvfb_screenshot(mdvar['img_plotDomainWin_samplingzone'], 
                               crop=screenshot.getwinpos(p2))
time.sleep(0.2)
p2.okclose()

# -- save the plot --
fig, ax = plt.subplots(figsize=(5,5), facecolor='w', dpi=150)
time.sleep(0.25)
case.popup_storteddata['plotdomain']['plot_sampleprobes']    = case.listboxpopupwindict['listboxsampling'].getitemlist()
case.plotDomain(ax=ax)
plt.tight_layout()
mdvar['img_plotDomainFig_sampling'] = imagedir+'/plotDomainFig_sampling.png'
if savefigs:
    plt.savefig(mdvar['img_plotDomainFig_sampling'])

###########################
# -- Write the input file --
inputfile = case.writeAMRWindInput('')
mdvar['amrwindinput1'] = "\n".join([s for s in inputfile.split("\n") if s])


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
time.sleep(2)
t1.join()
