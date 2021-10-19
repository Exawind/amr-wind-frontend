"""
Wind farm functions
"""
import numpy as np
import pandas as pd
import sys, os, csv
import re
from collections            import OrderedDict 

# Load the right version of StringIO
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

# Load ruamel or pyyaml as needed
try:
    import ruamel.yaml as yaml
    #print("# Loaded ruamel.yaml")
    useruamel=True
    loaderkwargs = {'Loader':yaml.RoundTripLoader}
    dumperkwargs = {'Dumper':yaml.RoundTripDumper, 'indent':4} # 'block_seq_indent':2, 'line_break':0, 'explicit_start':True, 
except:
    import yaml as yaml
    #print("# Loaded yaml")
    useruamel=False
    loaderkwargs = {}
    dumperkwargs = {}

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

    # Check if file exists
    if not os.path.isfile(csvfile):
        print("ERROR: %s does not exist"%csvfile)
        return

    # Load the filename and display it in the text box    
    csvstr = open(csvfile, 'r').read().lstrip()
    self.inputvars[csvtextbox].setval(csvstr)
    return

def resetFarmSetup(self):
    """
    Resets all variables with 'farmsetup' in outputdef to their defaults
    """
    for key, var in self.inputvars.items():
        outputkey = 'farmsetup'
        if outputkey in var.outputdef:
            var.setdefault()
    return

def writeFarmSetupYAML(self, filename, verbose=True):
    """
    Write out the farm setup parameters into a YAML file
    """
    inputdict = dict(self.getDictFromInputs('farmsetup', onlyactive=False))

    if useruamel: yaml.scalarstring.walk_tree(inputdict)

    outfile = sys.stdout if filename == sys.stdout else open(filename, 'w')
    yaml.dump(inputdict, outfile, default_flow_style=False, 
              **dumperkwargs)
    if filename != sys.stdout: 
        outfile.close()
        print("Saved farm setup to %s"%filename)
    return

def loadFarmSetupYAML(self, loadfile, stringinput=False):
    """
    Load the farm setup from a YAML file
    """
    if useruamel: Loader=yaml.load
    else:         Loader=yaml.safe_load
    if stringinput:
        yamldict = Loader(loadfile, **loaderkwargs)
    else:
        # Check if file exists
        if not os.path.isfile(loadfile):
            print("ERROR: %s does not exist"%loadfile)
            return
        # Load the file
        with open(loadfile, 'r') as fp:
            yamldict = Loader(fp, **loaderkwargs)
        print("Loaded farm setup from %s"%loadfile)

    print(yamldict)

    # Set the values of each variable
    for key, val in yamldict.items():
        self.inputvars[key].setval(val, forcechange=True)
    return

def button_saveFarmSetupYAML(self):
    """
    Button to save the farm setup
    """
    farmfile  = self.inputvars['farm_setupfile'].getval()
    if len(farmfile)==0:
        print('Blank farm setup file provided.  Cannot save.')
        return
    if farmfile=='sys.stdout': farmfile=sys.stdout

    self.writeFarmSetupYAML(farmfile)
    return

def button_loadFarmSetupYAML(self):
    """
    Button to load the farm setup
    """
    farmfile  = self.inputvars['farm_setupfile'].getval()
    if len(farmfile)==0:
        print('Blank farm setup file provided.  Cannot load.')
        return
    # Check if file exists
    if not os.path.isfile(farmfile):
        print("ERROR: %s does not exist"%farmfile)
        return
    self.loadFarmSetupYAML(farmfile, stringinput=False)

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

    loadFarmSetupYAML(testinp, stringinput=True)    

if __name__ == "__main__":
    runtest1()
    #runtest2()
    #inputdict = {'a':'apple', 'b':'boy'}
    #yaml.dump(dict(inputdict), sys.stdout, default_flow_style=False)
