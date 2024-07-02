#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

# Get the location where this script is being run
import sys, os
scriptpath = os.path.dirname(os.path.realpath(__file__))
basepath   = os.path.dirname(scriptpath)

# Add any possible locations of amr-wind-frontend here
amrwindfedirs = ['../',
                 basepath]
import sys, os, shutil
for x in amrwindfedirs: sys.path.insert(1, x)
import postproengine as ppeng
import argparse

try:
    import argcomplete
    has_argcomplete = True
except:
    has_argcomplete = False

# Load ruamel or pyyaml as needed
try:
    import ruamel.yaml
    yaml = ruamel.yaml.YAML(typ='rt')
    useruamel=True
    loaderkwargs = {}
    dumperkwargs = {}
    
    Loader=yaml.load
except:
    import yaml as yaml
    print("# Loaded yaml")
    useruamel=False
    loaderkwargs = {}
    dumperkwargs = {'default_flow_style':False }
    Loader=yaml.safe_load


if useruamel:
    from ruamel.yaml.comments import CommentedMap 
    def comseq(d):
        """
        Convert OrderedDict to CommentedMap
        """
        if isinstance(d, OrderedDict):
            cs = CommentedMap()
            for k, v in d.items():
                cs[k] = comseq(v)
            return cs
        return d


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":
    helpstring = """
    Run the post-processing engine
    """
    
    # Handle arguments
    parser     = argparse.ArgumentParser(description=helpstring,
                                         formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.add_argument(
        "inputfile",
        help="input yaml file",
        type=str,
    )
    parser.add_argument(
        '--printinfo',
        help="print times and information about netcdf file",
        action='store_true',
    )        
    parser.add_argument(
        '-v', '--verbose', 
        action='count', 
        help="Verbosity level (multiple levels allowed)",
        default=0)

    # Load the options
    if has_argcomplete: argcomplete.autocomplete(parser)    
    args      = parser.parse_args()
    inputfile = args.inputfile
    verbose   = args.verbose
    printinfo = args.printinfo

    if printinfo:
        ppeng.print_inputs()
        exit()

    # --- Load the inputfile ---
    # Check if file exists
    if not os.path.isfile(inputfile):
        print("ERROR: %s does not exist"%inputfile)
        raise Exception('File does not exist')
    # Load the file
    with open(inputfile, 'r') as fp:
        yamldict = Loader(fp, **loaderkwargs)

    # Run the driver
    ppeng.driver(yamldict, verbose=verbose)
