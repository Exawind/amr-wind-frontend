"""
Wind farm functions
"""
import numpy as np
import pandas as pd
import sys, csv
import re
from collections            import OrderedDict 

# Load the right version of StringIO
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

def loadcsv(f, stringinput=False, reqheaders=None, optheaders=None,
            **kwargs):
    """
    Load the csv input, whether from string or filename
    """
    # Get input f into a string
    if stringinput:
        s = f.lstrip()
    else:
        s = open(f, 'r').read().lstrip()

    # Remove comments 
    cleanstr = re.sub(r'(?m)^ *#.*\n?', '', s)

    # Check for header
    sniffer   = csv.Sniffer()
    hasheader = sniffer.has_header(cleanstr)

    #print("hasheader = %s"%repr(hasheader))
    
    header = None
    colheaders=reqheaders+optheaders
    if hasheader:
        header     = 0
        colheaders = None

    # Convert string to io stream
    finput = StringIO(cleanstr)
    df = pd.read_csv(finput, header=header, names=colheaders, **kwargs)

    # Double check headers to make sure it has the required headers
    if hasheader:
        # first rename the headers
        renamemap = {}
        for x in list(df.columns): renamemap[x] = str(x.strip().lower())
        df.rename(columns=renamemap, inplace=True)
        csvheaders=list(df.columns)
        #csvheaders=[x.strip().lower() for x in list(df.columns)]
        #print(csvheaders)
        for header in reqheaders:
            if header.lower() not in csvheaders:
                print('ERROR: required data column %s not present in data'%(header))
    return df

def dataframe2dict(df, reqheaders, optheaders, dictkeys=[]):
    """
    Convert dataframe to a list of dictionaries
    """
    listdict = []
    for index, row in df.iterrows():
        rowdict = OrderedDict()
        for key in reqheaders:
            rowdict[key] = row[key]
        # Add the optional headers
        for key in optheaders:
            if key in list(df.columns):
                rowdict[key] = parseoptions(row[key]) if key in dictkeys else row[key]  
            else:
                rowdict[key] = None
        listdict.append(rowdict)
    return listdict

def parseoptions(optionstr):
    """
    Splits an option string into a dictionary.  optionstr should be of
    the form "key1:val1 key2:val2"
    """
    sanitizestr = optionstr if isinstance(optionstr, str) else str(optionstr)
    allopts = sanitizestr.split()
    optdict = OrderedDict()
    for opt in allopts:
        optsplit = opt.split(':')
        key = optsplit[0]
        if len(optsplit)>1:
            val = optsplit[1]
        else: 
            val = None
        optdict[key] = val
    return optdict

def button_loadcsv(self, filenameinput, csvtextbox):
    """
    Button to load a CSV file
    """
    # Get the filename to load
    csvfile  = self.inputvars[filenameinput].getval()
    # Need to double check file exists

    # Load the filename and display it in the text box    
    csvstr = open(csvfile, 'r').read().lstrip()
    self.inputvars[csvtextbox].setval(csvstr)
    return

def runtest1():
    testdata="""
#
# This is comment 1
 A, b, C, d, e
 1, 2, 3, 4, x:1 b:2
5, 6, 7, 8, xy:2;4;5
# This is comment 2
 1, 2, 3, 4, 
  5, 6, 7, 8, ad
"""
    testdatanohead="""
1, 2, 3, 4
5, 6, 7, 8
1, 2, 3, 4
5, 6, 7, 8
"""

    reqheaders = ['a', 'b', 'c', 'd']
    optheaders = ['e']
    #df = loadcsv(testdata, stringinput=True)
    df = loadcsv(testdatanohead, stringinput=True, 
                 reqheaders=reqheaders, optheaders=optheaders)
    #df = loadcsv('csvtest.csv', stringinput=False)
    print(df)
    print("Test parseoptions:")
    print(parseoptions("key1:val1 key2:3 key3:0.01"))
    print("test dataframe2dict")
    for k in dataframe2dict(df, reqheaders, optheaders, dictkeys=['e']):
        print(k)

if __name__ == "__main__":
    runtest1()
