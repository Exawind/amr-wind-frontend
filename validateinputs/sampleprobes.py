from validateinputs import registerplugin
from validateinputs import CheckStatus as status
import numpy as np

"""
See README.md for details on the structure of classes here
"""

def isPointInside(p, corner1, corner2):
    inside = (corner1[0] <= p[0]) and (p[0] <= corner2[0]) and \
             (corner1[1] <= p[1]) and (p[1] <= corner2[1]) and \
             (corner1[2] <= p[2]) and (p[2] <= corner2[2]) 
    return inside

def checkLineProbeInside(name, pdict, corner1, corner2):
    checkstatus                = {}   # Dict containing return status
    checkstatus['subname']     = name
    checkstatus['mesg']        = 'is inside domain'
    checkstatus['result']      = status.PASS

    p1 = pdict['sampling_l_start']
    p2 = pdict['sampling_l_end']
    
    p1inside = isPointInside(p1, corner1, corner2)
    p2inside = isPointInside(p2, corner1, corner2)

    if (p1inside and p2inside): 
        # It's inside the domain, all good
        return checkstatus
    else:
        # It's outside, issue warning
        checkstatus['mesg']        = 'Line extends outside domain'
        checkstatus['result']      = status.WARN
        return checkstatus

def checkSamplePlaneInside(name, pdict, corner1, corner2):
    checkstatus                = {}   # Dict containing return status
    checkstatus['subname']     = name
    checkstatus['mesg']        = ''
    checkstatus['result']      = status.PASS

    Npts   = pdict['sampling_p_num_points']
    origin = np.array(pdict['sampling_p_origin'])
    axis1  = np.array(pdict['sampling_p_axis1'])
    axis2  = np.array(pdict['sampling_p_axis2'])
    #dx1    = axis1/(Npts[0]-1.0)
    #dx2    = axis2/(Npts[1]-1.0)

    if (pdict['sampling_p_offsets'] is not None) and \
       (pdict['sampling_p_offsets'] != 'None'):
        offsets = [float(x) for x in pdict['sampling_p_offsets'].split()]
    else:
        offsets = [0]
    
    offsetnormal = np.array(pdict['sampling_p_normal'])

    # Construct the list of offsets
    offsetvec = []
    if len(offsets)==0:
        offsetvec.append(np.zeros(3))
    else:
        for dx in offsets:
            offsetvec.append(offsetnormal*dx)

    # Test all corner points
    resultvec = []
    for doffset in offsetvec:
        # Make the corner points
        p1 = origin + doffset
        p2 = origin + axis1 + doffset
        p3 = origin + axis2 + doffset
        p4 = origin + axis1 + axis2 + doffset
        resultvec.append(isPointInside(p1, corner1, corner2))
        resultvec.append(isPointInside(p2, corner1, corner2))
        resultvec.append(isPointInside(p3, corner1, corner2))
        resultvec.append(isPointInside(p4, corner1, corner2))

    if False not in resultvec: 
        # It's inside, all good
        return checkstatus
    else:
        # It's outside, issue warning
        checkstatus['mesg']        = 'Plane extends outside domain'
        checkstatus['result']      = status.WARN
        return checkstatus


@registerplugin
class Check_sampleprobes_inside(): 
    name = "Sampling probes"

    def check(self, app):
        
        post_processing = app.inputvars['post_processing'].getval()
        allsamplingdata = app.listboxpopupwindict['listboxsampling']
        allprobes       = allsamplingdata.getitemlist()

        if (len(allprobes)==0) or ('sampling' not in post_processing):
            skipstatus                = {}   # Dict containing return status
            skipstatus['subname']     = ''   # Additional name info
            skipstatus['result']      = status.SKIP
            skipstatus['mesg']        = 'Not active or no sampling planes'
            return [skipstatus]
        
        # Get the domain corners
        corner1  = app.inputvars['prob_lo'].getval()
        corner2  = app.inputvars['prob_hi'].getval()

        # return 
        statuses = []

        for probe in allprobes:
            keystr = lambda n, d1, d2: d2.name
            pdict = allsamplingdata.dumpdict('AMR-Wind', subset=[probe], 
                                             keyfunc=keystr)
            if pdict['sampling_type'][0]=='LineSampler':
                statuses.append(checkLineProbeInside(probe, pdict, 
                                                     corner1, corner2))
            if pdict['sampling_type'][0]=='PlaneSampler':
                statuses.append(checkSamplePlaneInside(probe, pdict, 
                                                       corner1, corner2))
        return statuses
