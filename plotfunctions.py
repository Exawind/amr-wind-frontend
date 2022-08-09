"""
Plotting functions
"""
import numpy as np
from collections            import OrderedDict 
from matplotlib.collections import PatchCollection
from matplotlib.patches     import Rectangle
import matplotlib.pyplot    as plt
from matplotlib.lines       import Line2D

import OpenFASTutil as OpenFAST

# -------------------------------------------------------------
def readCartBoxFile(filename):
    """
    Read the Cartesian box file
    """
    allboxes = []
    fname    = open(filename, 'r')
    # read the number of levels
    Nlevels = int(fname.readline().strip())
    for i in range(Nlevels):
        # Read the number of boxes at this level
        Nboxes = int(fname.readline().strip())
        levelboxes=[]
        for b in range(Nboxes):
            # Read each box
            boxline  = fname.readline().strip().split()
            box      = [float(x) for x in boxline]
            if len(box)!=6:
                print("Line does not contain 6 floats:")
                print(" %s"%boxline)
            levelboxes.append(box)
        allboxes.append(levelboxes)
    fname.close()
    return allboxes

def plotRectangle(figax, corner1, corner2, ix, iy, **kwargs):
    """
    Plot a rectangle onto figax
    """
    x1 = corner1[ix]
    y1 = corner1[iy]
    x2 = corner2[ix]
    y2 = corner2[iy]
    Lx = x2-x1
    Ly = y2-y1
    rect=Rectangle((x1, y1), Lx, Ly, **kwargs)
    figax.add_patch(rect)
    return x1, y1, x2, y2

def plot3DBox(figax, origin, xaxis, yaxis, zaxis, ix, iy, **kwargs):
    """
    Plots a 3D box on figax.  Corner is at origin 
    """
    # Define the faces to plot
    plotfaces = [[origin, xaxis, yaxis],
                 [origin, xaxis, zaxis],
                 [origin, yaxis, zaxis]]
    # Build list for each of three faces
    for face in plotfaces:
        p1     = np.array(face[0])
        p2     = p1+np.array(face[1])
        p3     = p2+np.array(face[2])
        p4     = p3-np.array(face[1])
        ptlist = [p1, p2, p3, p4]
        xlist  = [p[ix] for p in ptlist]
        ylist  = [p[iy] for p in ptlist]
        figax.fill(xlist, ylist, **kwargs)
    return

def getCirclePts(inputorigin, inputnormal, R, Npts=10):
    origin = np.array(inputorigin)
    # Normalize
    normal = np.array(inputnormal)/np.linalg.norm(np.array(inputnormal))
    # Get the rhat vector
    v     = np.array([1,1,1])   # Arbitrary point
    dist  = np.dot(v, normal)
    dr    = v - dist*normal
    rhat  = dr/np.linalg.norm(dr)
    # Get the dtheta vector
    theta = 2.0*np.pi/Npts
    #ds    = 2.0*R*np.sin(theta/2)
    ds    = R*np.tan(theta)
    ptlist = [origin+rhat*R]
    for i in range(Npts):
        rhat  = ptlist[-1]-origin
        rhat  = rhat/np.linalg.norm(rhat)
        thhat = np.cross(rhat, normal)
        newpt = ptlist[-1]+ds*thhat
        newrhat = newpt - origin
        newrhat = newrhat/np.linalg.norm(newrhat)
        newpt   = origin + R*newrhat
        ptlist.append(newpt)
    return ptlist

def plotPtList(figax, ptlist, ix, iy, **kwargs):
    xlist = [p[ix] for p in ptlist]
    ylist = [p[iy] for p in ptlist]
    figax.fill(xlist, ylist, **kwargs)    
    return

def plotCylinderSurface(figax, circle1, circle2, ix, iy, **kwargs):
    Nsegs = len(circle1)
    if (len(circle2) != Nsegs):
        print("Circles have different edge counts!  Can't plot")
        return
    for i in range(Nsegs):
        ip1 = i+1 if i<Nsegs-1 else 0
        ptlist = [circle1[i], circle1[ip1], circle2[ip1], circle2[i]]
        plotPtList(figax, ptlist, ix, iy, **kwargs)
    return

def plotCylinder(figax, startpt, endpt, R1, R2, ix, iy, Npts=20, **kwargs):
    normal = np.array(endpt)-np.array(startpt)
    startR2 = getCirclePts(startpt, normal, R2, Npts=Npts)
    endR2   = getCirclePts(endpt, normal, R2, Npts=Npts)
    plotPtList(figax, startR2, ix, iy, **kwargs)
    plotPtList(figax, endR2,   ix, iy, **kwargs)
    plotCylinderSurface(figax, startR2, endR2, ix, iy, **kwargs)
    if R1 is not None:
        print("Cannot plot the inner radius of cylinders")
    return

def rotatepoint(pt, orig, theta):
    """
    Rotates a point pt about origin orig
    Here theta is measured w.r.t. the x-axis
    """
    dx = pt[0]-orig[0]
    dy = pt[1]-orig[1]
    p2=[0.0, 0.0, 0.0]
    p2[0] = dx*np.cos(theta) - dy*np.sin(theta) + orig[0]
    p2[1] = dx*np.sin(theta) + dy*np.cos(theta) + orig[1]
    p2[2] = pt[2]
    return p2

def plotTurbine(figax, basexyz, hubheight, turbD, nacelledir, ix, iy,
                thetaoffset=0.0, **kwargs):
    """
    Plot turbine on figax
    """
    turbR = 0.5*turbD
    Nsegs = 30  # Number of segments on rotor circumference
    # Construct the rotor diameter ring
    rotorpts = []
    for theta in np.linspace(0, 360, Nsegs+1):
        x = 0
        y = turbR*np.cos(theta*np.pi/180.0)
        z = turbR*np.sin(theta*np.pi/180.0)
        rotorpts.append([x,y,z])
    # Rotate the rotor ring to the right orientation
    rotatetheta = (270.0+thetaoffset-nacelledir)*np.pi/180.0
    rotatedring = [rotatepoint(p, [0.0,0.0,0.0], rotatetheta) for p in rotorpts]
    # Translate the right to the right location
    hhpt     = np.array([0.0, 0.0, hubheight])
    rotorpts = [np.array(p)+np.array(basexyz)+hhpt for p in rotatedring]
    #print(rotorpts)
    # Get the list of x and y points and plot them
    xlist = [p[ix] for p in rotorpts]
    ylist = [p[iy] for p in rotorpts]
    figax.fill(xlist, ylist, **kwargs)
    #figax.plot(xlist, ylist, **kwargs)
    
    # Plot the turbine nacelle
    nacelleW = 0.1*turbD   # nacelle width (lateral)
    nacelleH = 0.1*turbD   # nacelle height
    nacelleL = 0.2*turbD   # nacelle length (streamwise)
    nacellecorner = [0, -nacelleW/2, -nacelleH/2]
    nacelleLaxis  = [nacelleL, 0, 0]
    nacelleWaxis  = [0, nacelleW, 0]
    nacelleHaxis  = [0, 0, nacelleH]
    # Rotate the points
    nacelleLaxis  = rotatepoint(nacelleLaxis, [0,0,0], rotatetheta)
    nacelleWaxis  = rotatepoint(nacelleWaxis, [0,0,0], rotatetheta)
    # rotate and translate the corner
    nacellecorner = rotatepoint(nacellecorner, [0,0,0], rotatetheta)
    nacellecorner = np.array(nacellecorner) + np.array(basexyz) + hhpt
    plot3DBox(figax, nacellecorner, nacelleLaxis, nacelleWaxis, nacelleHaxis,
              ix, iy, **kwargs)
    return

# -------------------------------------------------------------
def plotDomain(self, ax=None):
    # Clear and resize figure
    if ax is None: ax=self.setupfigax()

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

    # Do the domain plot here
    x1, y1, x2, y2  = plotRectangle(ax, corner1, corner2, ix, iy,
                                    color='gray', alpha=0.25)
    Lx = x2-x1
    Ly = y2-y1
    Cx = 0.5*(x1+x2)
    Cy = 0.5*(y1+y2)
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
            ax.text(compasscenter[ix], 0.99*compasscenter[iy], 
                    'N', color='r', ha='right', va='top')

    # Plot the sample probes
    # ---------------------------
    if ((plotparams['plot_sampleprobes'] is not None) 
        and (len(plotparams['plot_sampleprobes'])>0)):
        allsamplingdata = self.listboxpopupwindict['listboxsampling']
        allprobes=allsamplingdata.getitemlist()
        keystr = lambda n, d1, d2: d2.name
        ms=2
        for p in plotparams['plot_sampleprobes']:
            pdict = allsamplingdata.dumpdict('AMR-Wind', subset=[p], keyfunc=keystr)
            if pdict['sampling_type'][0]=='LineSampler':
                Npts  = pdict['sampling_l_num_points']
                start = np.array(pdict['sampling_l_start'])
                end   = np.array(pdict['sampling_l_end'])
                dx    = (end-start)/(Npts-1.0)
                pts   = []
                for i in range(Npts):
                    pt = start + dx*i
                    pts.append(pt)
                pts = np.array(pts)
                ax.plot(pts[:,ix], pts[:,iy], '.', markersize=ms, label=p)
            if pdict['sampling_type'][0]=='PlaneSampler':
                Npts   = pdict['sampling_p_num_points']
                origin = np.array(pdict['sampling_p_origin'])
                axis1  = np.array(pdict['sampling_p_axis1'])
                axis2  = np.array(pdict['sampling_p_axis2'])
                dx1    = axis1/(Npts[0]-1.0)
                dx2    = axis2/(Npts[1]-1.0)

                # Construct the list of offsets
                if (pdict['sampling_p_offsets'] is not None) and \
                   (pdict['sampling_p_offsets'] != 'None'):
                    offsets =[float(x) for x in pdict['sampling_p_offsets'].split()]
                else:
                    offsets = [0.0]
                offsetnormal = np.array(pdict['sampling_p_normal'])
                offsetvec = []
                if len(offsets)==0:
                    offsetvec.append(np.zeros(3))
                else:
                    for dx in offsets:
                        offsetvec.append(offsetnormal*dx)

                pts    = []

                # Construct the list of all plane points
                for doffset in offsetvec:
                    for i in range(Npts[0]):
                        for j in range(Npts[1]):
                            pt = origin + i*dx1 + j*dx2 + doffset
                            pts.append(pt)
                pts = np.array(pts)
                ax.plot(pts[:,ix], pts[:,iy], '.', markersize=ms, label=p)
        legendprobes=ax.legend(title="Sampling probes", fontsize=10,
                               loc='upper right')
        for legend_handle in legendprobes.legendHandles:
            legend_handle._legmarker.set_markersize(9)
        plt.setp(legendprobes.get_title(),fontsize=10)
        ax.add_artist(legendprobes)

    # Plot the refinement boxes
    # ---------------------------
    if ((plotparams['plot_refineboxes'] is not None) and 
        (len(plotparams['plot_refineboxes'])>0)):
        #print(plotparams['plot_refineboxes'])
        allrefinements = self.listboxpopupwindict['listboxtagging']
        alltags        = allrefinements.getitemlist()
        keystr         = lambda n, d1, d2: d2.name

        # Need to validate maxlevel! Fix this!
        maxlevel       = self.inputvars['max_level'].getval()
        # Get the level colors
        try: 
            levelcolors=plt.rcParams['axes.color_cycle']
        except:
            levelcolors=plt.rcParams['axes.prop_cycle'].by_key()['color']

        for p in plotparams['plot_refineboxes']:
            pdict = allrefinements.dumpdict('AMR-Wind',
                                            subset=[p], keyfunc=keystr)
            # Plot the Cartesian Box Refinements
            if pdict['tagging_type'][0]=='CartBoxRefinement':
                filename = pdict['tagging_static_refinement_def']
                # Load the boxes
                allboxes = readCartBoxFile(filename)
                if len(allboxes)>maxlevel: maxlevel = len(allboxes)
                for ilevel, boxlevel in enumerate(allboxes):
                    for box in boxlevel:
                        corner1 = box[0:3]
                        corner2 = box[3:6]
                        color   = levelcolors[ilevel]
                        plotRectangle(ax, corner1, corner2, ix, iy,
                                      facecolor=color, ec='k', lw=0.5, 
                                      alpha=0.90)
            # Plot the Geometry Refinements
            if pdict['tagging_type'][0]=='GeometryRefinement':
                if pdict['tagging_geom_type'][0]=='box':
                    origin = pdict['tagging_geom_origin']
                    xaxis  = pdict['tagging_geom_xaxis']
                    yaxis  = pdict['tagging_geom_yaxis']
                    zaxis  = pdict['tagging_geom_zaxis']
                    ilevel = pdict['tagging_level']
                    if ilevel is not None: 
                        color   = levelcolors[ilevel]
                    else:
                        color   = levelcolors[0]
                    plot3DBox(ax, origin, xaxis, yaxis, zaxis, ix, iy,
                              lw=0.4, facecolor=color, alpha=0.90)
                if pdict['tagging_geom_type'][0]=='cylinder':
                    cylstart  = pdict['tagging_geom_start']
                    cylend    = pdict['tagging_geom_end']
                    outerR    = pdict['tagging_geom_outer_radius']
                    innerR    = pdict['tagging_geom_inner_radius']
                    ilevel = pdict['tagging_level']
                    if ilevel is not None: 
                        color   = levelcolors[ilevel]
                    else:
                        color   = levelcolors[0]
                    #print("cylinder geometry refinement plotting not supported")
                    print("plotting cylinder")
                    plotCylinder(ax, cylstart, cylend, innerR, outerR, ix, iy,
                                 facecolor=color)

        # Add a legend with the level labels
        legend_el = []
        legend_label = []
        legend_el.append(Line2D([0],[0], 
                                linewidth=0, marker='s', color='gray',
                                alpha=0.25, label='Level 0'))
        legend_label.append('Level 0')
        for i in range(maxlevel):
            legend_el.append(Line2D([0],[0], 
                                    linewidth=0, marker='s',
                                    color=levelcolors[i+0], 
                                    alpha=0.75, 
                                    label='Level %i'%(i+0)))
            legend_label.append('Level %i'%(i+1))
        legendrefine = ax.legend(legend_el, legend_label, 
                                 frameon=True, numpoints=1, 
                                 fontsize=10, loc='lower right')
        ax.add_artist(legendrefine)

    # Plot the turbines
    # ---------------------------
    if ((plotparams['plot_turbines'] is not None) and 
        (len(plotparams['plot_turbines'])>0)):
        print("Plotting turbines")
        allturbines  = self.listboxpopupwindict['listboxactuator']
        alltags      = allturbines.getitemlist()
        keystr       = lambda n, d1, d2: d2.name

        # Get the defaults
        default_type   = self.inputvars['Actuator_default_type'].getval()
        default_type   = None if len(default_type)==0 else default_type
        default_type   = default_type[0] if isinstance(default_type, list) else default_type
        if 'Actuator_%s_rotor_diameter'%default_type in self.inputvars:
            default_turbD  = self.inputvars['Actuator_%s_rotor_diameter'%default_type].getval()
        else:
            default_turbD  = 100.0
        if 'Actuator_%s_hub_height'%default_type in self.inputvars:
            default_hh     = self.inputvars['Actuator_%s_hub_height'%default_type].getval()
        else:
            default_hh     = None

        # Get the wind direction
        self.ABL_calculateWDirWS()
        winddir = self.inputvars['ABL_winddir'].getval()

        # Get any north offset
        thetaoffset = self.get_N_angle_to_Y()

        for turb in plotparams['plot_turbines']:
            tdict = allturbines.dumpdict('AMR-Wind',
                                         subset=[turb], keyfunc=keystr)
            turbtype = default_type if 'Actuator_type' not in tdict else tdict['Actuator_type']
            turbtype = turbtype[0] if isinstance(turbtype, list) else turbtype
            turbhh   = default_hh  if tdict['Actuator_hub_height'] is None else tdict['Actuator_hub_height']
            turbhh   = 0.0 if turbhh is None else turbhh
            turbD    = default_turbD if tdict['Actuator_rotor_diameter'] is None else tdict['Actuator_rotor_diameter']

            basepos  = tdict['Actuator_base_position']
            yaw      = winddir if tdict['Actuator_yaw'] is None else tdict['Actuator_yaw']  #270.0

            if turbtype in ['TurbineFastLine', 'TurbineFastDisk']:
                fstfile  = tdict['Actuator_openfast_input_file']
                EDfile   = OpenFAST.getFileFromFST(fstfile,'EDFile')
                EDdict   = OpenFAST.FASTfile2dict(EDfile)
                EDyaw    = float(EDdict['NacYaw'])
                yaw      = 270.0+thetaoffset-EDyaw

            plotTurbine(ax, basepos, turbhh, turbD, yaw, ix, iy,
                        lw=1, color='k', alpha=0.75)

    # --------------------------------
    # Set some plot formatting parameters
    ax.set_aspect('equal')
    ax.set_xlabel('%s [m]'%xstr)
    ax.set_ylabel('%s [m]'%ystr)
    ax.set_title(r'Domain')
    self.figcanvas.draw()
    #self.figcanvas.show()

    return

# -------------------------------------------------------------
def plotGenericProfile(self, xvar, yvar, useInputVar=True, ax=None):
    """
    Plots a profile given by the xvar and yvar string variables
    """
    # Clear and resize figure
    if ax is None: ax=self.setupfigax()

    # Get the strings
    xstr = self.getAMRWindInput(xvar) if useInputVar else xvar
    ystr = self.getAMRWindInput(yvar) if useInputVar else yvar

    # Convert the strings to arrays
    xarr = np.array([float(x) for x in xstr.split()])
    yarr = np.array([float(y) for y in ystr.split()])

    # Plto it
    ax.plot(xarr, yarr)
    
    self.figcanvas.draw()
    return
