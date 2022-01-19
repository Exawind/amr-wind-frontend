import logging
import threading

import os, sys, tempfile
from PIL import Image
from xvfbwrapper import Xvfb
import yaml

"""
Note that this script relies on:
- xvfbwrapper
- xwd
- convert

On ubuntu systems, install with
sudo apt install x11-apps imagemagick-6.q16
"""

# Set the location of amrwind_frontend.py script
scriptpath=os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, scriptpath)
import amrwind_frontend as amrwind

# Set title and local configuration directory
title='AMR-Wind'
localconfigdir=os.path.join(scriptpath,'local')

def start_Xvfb(**kwargs):
    vdisplay = Xvfb(**kwargs)
    vdisplay.start()
    return vdisplay

def Xvfb_screenshot(filename, crop=None, log=True):
    """
    Saves a screen shot of the Xvfb screen buffer to file filename
    """
    # See 
    # https://unix.stackexchange.com/questions/365268/how-do-i-take-a-screen-shot-of-my-xvfb-buffer
    # https://www.geeksforgeeks.org/python-pil-image-crop-method/

    if crop is not None:
        tmp = tempfile.NamedTemporaryFile()
        filename1 = tmp.name
        #print(filename1)
    else:
        filename1 = filename

    # Dump the screenshot
    display=os.environ['DISPLAY']
    xwdcmd="xwd -display %s -root -silent | convert xwd:- png:%s"%(display, filename1)
    os.system(xwdcmd)

    if crop is not None:
        im  = Image.open(filename1)
        im1 = im.crop(crop)
        im1.save(filename)
    if log: 
        logging.info("genscreenshot: saving %s"%filename)
    return

def start_instance(name, casedict, lock):
    lock.acquire()
    logging.info("Thread %s: starting", name)
    case=amrwind.MyApp(configyaml=os.path.join(scriptpath,'config.yaml'), 
                       localconfigdir=localconfigdir, 
                       scriptpath=scriptpath,
                       title=title)

    # -- maximize window ---
    w,h=case.winfo_screenwidth(),case.winfo_screenheight()
    case.geometry("%dx%d+0+0" % (w, h))
    casedict[name]=case
    lock.release()

    case.mainloop()
    logging.info("Thread %s: finishing", name)

def getwinpos(win):
    """
    Returns the window position 
    """
    win.update()
    x0, y0 = win.winfo_rootx(), win.winfo_rooty()
    w,  h  = win.winfo_width(), win.winfo_height()
    return (x0, y0, w, h)

def scrollcanvas(canvas, ypos):
    canvas.update()
    canvas.yview_moveto(ypos)

def gethelpmesg(yamlfile, key, basekey='helpwindows', mesgkey='mesg'):
    with open(yamlfile) as stream:
        try:
            yamldata=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)   
    return yamldata[basekey][key][mesgkey]

def setAMRWindInputString(case, casename, key, extra='', comment=''):
    """
    Returns the string 
      casename.setAMRWindInput('key', value)
    """
    extraargs  = '' if extra=='' else ', '+extra
    addcomment = '' if comment=='' else ' # '+comment
    valstr = repr(case.getAMRWindInput(key))
    outstr = """%s.setAMRWindInput('%s', %s%s)%s\n"""
    return outstr%(casename, key, valstr, extraargs, addcomment)
