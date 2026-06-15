"""\
A tool to generate native boundary planes
-----------------------------------------

This script is for generating boundary plane data.

"""

import argparse
import pathlib
import numpy as np
import amrex.space3d as amr
import native_boundary_plane as nbp
from amrex_plotfile import AmrexPlotFile
from scipy.interpolate import RegularGridInterpolator


def process(plt):
    """Interpolate from one plt file to another"""
    assert plt.nlevels == 1
    assert plt.ngrids[0] == 1
    ilev = 0
    for igrid in range(plt.ngrids[ilev]):
        print(ilev, igrid)
        original = plt.mfs[ilev].to_xp()[igrid]
        xg, yg, zg = plt.coordinates(ilev, igrid)

        for nc in range(plt.ncomp):
            data = original[:, :, :, nc]
            print(data.shape)
            print(f'mean [igrid={igrid}]',np.mean(data))

def interpolateBP(plt, xgi, ygi, zgi, ncomp):
    """Interpolate from one plt file to another"""
    assert plt.nlevels == 1
    assert plt.ngrids[0] == 1
    ilev = 0
    for igrid in range(plt.ngrids[ilev]):
        #print(ilev, igrid)
        original = plt.mfs[ilev].to_xp()[igrid]
        xg, yg, zg = plt.coordinates(ilev, igrid)

        data = original[:, :, :, ncomp]
        interp = RegularGridInterpolator(
            (xg[:, 0, 0], yg[0, :, 0], zg[0, 0, :]),
            data,
            bounds_error=False,
            fill_value=None,
        )
        #datai = plti.mfs[ilev].to_xp()[0]
        #datai[:, :, :, nc] =  interp((xgi, ygi, zgi))
        #print(data.shape)
        #print(f'mean [igrid={igrid}]',np.mean(data))
        interpbp = interp((xgi, ygi, zgi))
        #print(interpbp)
        #print(f'mean ',np.mean(interpbp))
        #print(f'min',  np.min(interpbp))
        #print(f'max',  np.max(interpbp))
        return interpbp

class Constant:
    """Constant functor."""

    def __init__(self, constant):
        self.constant = constant

    def __call__(self, xg, yg, zg, time):
        assert xg.shape == yg.shape
        assert xg.shape == zg.shape
        return self.constant * np.ones(xg.shape)

class FromBP:
    """Do stuff from another boundary plane"""

    def __init__(self, bpfile, ncomp):
        #self.constant = 0.0
        self.ncomp    = ncomp
        self.plt = AmrexPlotFile(bpfile)
        self.mfs = self.plt()
        #print(self.plt.prob_lo, self.plt.prob_hi)
        #print(self.plt.coordinates(0, 0))
        #print(self.plt.spacedim)
        #process(self.plt)
        return


    def __call__(self, xg, yg, zg, time):
        assert xg.shape == yg.shape
        assert xg.shape == zg.shape
        #return self.constant * np.ones(xg.shape)
        interpdat = interpolateBP(self.plt, xg, yg, zg, self.ncomp)
        #print(interpdat.shape, xg.shape)
        return interpdat


def ncomp_from_field(field):
    return 3 if field == "velocity" else 1

def makeBPfromfile(field, ncomp, ori, odir, fname, step, time, bpfile):
    functor_dict = {}
    if ncomp == 1:
        functor_dict[field] = FromBP(bpfile,0)
    elif ncomp == 3:
        functor_dict[field+"x"] = FromBP(bpfile,0)
        functor_dict[field+"y"] = FromBP(bpfile,1)
        functor_dict[field+"z"] = FromBP(bpfile,2)
    else:
        raise ValueError('ERROR: ncomp = %i, must be 1 or 3')
    # Write the boundary plane
    bp = nbp.NativeBoundaryPlane(field, ncomp, ori, odir)
    bp.define_from_file(fname, step, time)
    bp.evaluate(functor_dict)
    bp.write()
    return

def makeBPconst(field, ncomp, ori, odir, fname, step, time, constval):
    functor_dict = {}
    if ncomp == 1:
        functor_dict[field] = Constant(constval)
    elif ncomp == 3:
        functor_dict[field+"x"] = Constant(constval[0])
        functor_dict[field+"y"] = Constant(constval[1])
        functor_dict[field+"z"] = Constant(constval[2])
    else:
        raise ValueError('ERROR: ncomp = %i, must be 1 or 3')
    # Write the boundary plane
    bp = nbp.NativeBoundaryPlane(field, ncomp, ori, odir)
    bp.define_from_file(fname, step, time)
    bp.evaluate(functor_dict)
    bp.write()
    return

def main():
    parser = argparse.ArgumentParser(
        description="A tool to generate native boundary planes"
    )
    parser.add_argument(
        "-i",
        "--iname",
        help="Kynema-SGF input file for boundary data",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--srcdir",
        help="Source directory for boundary data",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--destdir",
        help="Destination directory for boundary data",
        default='bndry_file',
        type=str,
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Force overwrite of existing files",
    )
    args = parser.parse_args()

    amr.initialize([])

    timefile = 'time.dat'
    srcdir = pathlib.Path(args.srcdir)
    
    #bdir = pathlib.Path("bndry_file")
    bdir   = pathlib.Path(args.destdir)
    if bdir.exists() and not args.overwrite:
        errorstr = f"{bdir} exists in this directory. Skipping. Use -o to overwrite."
        raise Exception(errorstr)

    #field = "velocity"
    #ori = 0  # xlo = 0, ylo = 1
    xlo_ori, ylo_ori = 0, 1
    xhi_ori, yhi_ori = 3, 4
    ncomp = 3

    src_timedat = np.loadtxt(srcdir/timefile)
    #print(src_timedat)
    nsteps = len(src_timedat)
    steps = [int(x) for x in src_timedat[:,0]]
    times = src_timedat[:,1]
    #print(steps, times)
    #raise ValueError('stop here')

    #nsteps = 100
    #steps = [x for x in range(nsteps + 1)]
    #times = np.linspace(0, 5, nsteps + 1)
    
    bpvars = ['velocity',
              #'tke',
              'temperature',]
    constvars = ['tke']
    constval = 0.0
    
    surf_ori = [xlo_ori, xhi_ori, ylo_ori, yhi_ori]

    makebpfile = lambda sdir, step, ori, v: sdir/f'bndry_output{step:05d}'/f'Header_{ori}_{v}'
    #src_bp = srcdir/f'bndry_output{step:05d}'/f'Header_{xlo_ori}_velocity'
    
    for step, time in zip(steps, times):
        odir = bdir / f"bndry_output{step:05d}"
        print(f"Generating {odir} at time {time}: ", end='')
        # Go through vars from another boundary plane
        for bpvar in bpvars:
            #bpvar = v[0]
            ncomp = ncomp_from_field(bpvar)
            print(f'{bpvar}[', end='')
            for surf in surf_ori:
                print(f'{surf} ',end='')
                makeBPfromfile(bpvar, ncomp, surf,
                               odir, args.iname, step, time,
                               makebpfile(srcdir, step, surf, bpvar))
            print(f'] ',end='')
        # Go through any const vars
        for bpvar in constvars:
            ncomp = ncomp_from_field(bpvar)
            print(f'{bpvar}[', end='')
            for surf in surf_ori:
                print(f'{surf} ',end='')
                makeBPconst(bpvar, ncomp, surf,
                               odir, args.iname, step, time,
                               constval)
            print(f'] ',end='')
        print()
                
    np.savetxt(bdir / "time.dat", np.c_[steps, times], fmt="%.17g")


if __name__ == "__main__":
    main()
