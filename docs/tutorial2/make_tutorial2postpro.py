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

# ========================================
# Set the tutorial properties
scrwidth  = 1280
scrheight = 800
imagedir  = 'images'
mdtemplate= 'tutorial2gui_template.md'
mdfile    = 'tutorial2gui.md'
nbfile    = 'tutorial2python.ipynb'
gennbfile = True
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
# WRAP UP AND FINISH
# -------------------------

# Quit and clean up
time.sleep(2)
case.quit()
t1.join()
