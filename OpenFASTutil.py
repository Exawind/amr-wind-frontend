#!/usr/bin/env python

import sys, os, re
from collections            import OrderedDict 
import numpy as np
import fileinput

def is_number(s):
    try:
        complex(s) # for int, long, float and complex
    except ValueError:
        return False
    return True

def editFASTfile(FASTfile, replacedict):
    commentchars = ['=', '#']
    for line in fileinput.input(FASTfile, inplace=True, backup='.bak'):
        #sys.stdout.write("# "+line)
        linesplit=re.split('[, ;]+', line.strip())
        outline=""
        # Check to make sure the line doesn't start with comment char
        firstchar = ""
        if len(line.strip())>0: firstchar = line.strip()[0]
        if firstchar in commentchars: 
            outline=str(line)
        # Ignore any lines with less than two items
        if len(linesplit)<2:
            outline=str(line)
            
        # Check to make sure line is not all numbers
        allnums = [is_number(x) for x in linesplit]
        if False not in allnums:
            outline=str(line)

        # Handle list of nodes
        if outline=="":
            idx = 1
            if is_number(linesplit[idx]):
                # Find the right keyword
                idx = allnums.index(False)

            keyword = linesplit[idx]
        
            if keyword in replacedict.keys():
                outline  = '%10s '%repr(replacedict[keyword])
                outline += ' '.join(linesplit[idx:])
                outline += ' [EDITED]\n'
                sys.stderr.write(outline)
            else:
                outline=line

        # Write out the line
        sys.stdout.write(outline)
    return

def FASTfile2dict(FASTfile):
    """
    Reads the file FASTfile and returns a dictionary with parameters
    """
    commentchars = ['=', '#']
    d = OrderedDict()
    # go through the file line-by-line
    with open(FASTfile) as fp:
        line=fp.readline()
        while line:
            # Check to make sure the line doesn't start with comment char
            firstchar = ""
            if len(line.strip())>0: firstchar = line.strip()[0]
            if firstchar in commentchars: 
                line=fp.readline()
                continue
            #linesplit=line.strip().split(", ")
            linesplitX=re.split('[, ;]+', line.strip())
            # Remove any empty tokens in linesplit
            linesplit=[x.strip() for x in linesplitX if x.strip() != '']

            # Ignore any lines with less than two items
            if len(linesplit)<2:
                line=fp.readline()
                continue          

            # Check to make sure line is not all numbers
            allnums = [is_number(x) for x in linesplit]
            if False not in allnums:
                line=fp.readline()
                continue          
                
            # Handle the outlist
            if linesplit[0]=="OutList":
                outlistline = fp.readline()
                outlist     = []
                while outlistline.strip().split()[0] != "END":
                    outlist.append(outlistline.strip())
                    outlistline = fp.readline()
                    #print(outlistline.strip().split()[0] != "END")
                d["OutList"] = outlist
                line = fp.readline()
                continue

            # Handle list of nodes
            idx = 1
            if is_number(linesplit[idx]):
                # Find the right keyword
                idx = allnums.index(False)

            keyword = linesplit[idx]
            if idx==1:
                d[keyword] = linesplit[0]
            else:
                d[keyword] = linesplit[:idx]

            line=fp.readline()
    return d

def getFileFromFST(fstfile, key, fstdict=None):
    """
    Get the file referenced by key in fstfile
    """
    if fstdict is None:
        fstdict=FASTfile2dict(fstfile)
    keyfile = fstdict[key].strip('"').strip("'")
    # Now set up the path to keyfile correctly
    fstpath = os.path.dirname(os.path.abspath(fstfile))
    return os.path.join(fstpath, keyfile)

def getVarFromFST(fstfile, key, fstdict=None):
    """
    Get the file referenced by key in fstfile
    """
    if fstdict is None:
        fstdict=FASTfile2dict(fstfile)
    return fstdict[key]

def loadoutfile(filename):
    """
    Loads the FAST output file
    """
    # load the data file  
    dat=np.loadtxt(filename, skiprows=8)
    # get the headers and units
    with open(filename) as fp:
        fp.readline() # blank  
        fp.readline() # When FAST was run
        fp.readline() # linked with...   
        fp.readline() # blank            
        fp.readline() # Description of FAST input file
        fp.readline() # blank                         
        varsline=fp.readline()
        unitline=fp.readline()
        headers=varsline.strip().split()
        units  =unitline.strip().split()
    return dat, headers, units

def loadalldata(allfiles):
    """
    Load all data files given in allfiles
    """
    adat=[]
    header0=[]
    units0=[]
    names=[]
    for ifile, file in enumerate(allfiles):
        names.append(file)
        print("Loading file "+file)
        dat, headers, units = loadoutfile(file)
        adat.append(dat)
        if ifile==0:
            header0 = headers
            units0   = units
        else:
            if ((len(header0) != len(headers)) or (len(units0)!=len(units))):
                print("Data sizes doesn't match")
                sys.exit(1)
    return adat, header0, units0, names
