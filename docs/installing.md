# Installing `amrwind_frontend`

## From github

You can obtain `amrwind-frontend` by downloading it from github directly with a `git clone` command:

```bash
$ git clone --recursive git@github.com:lawrenceccheung/amrwind-frontend.git
Cloning into 'amrwind-frontend'...
remote: Enumerating objects: 1340, done.
remote: Counting objects: 100% (173/173), done.
remote: Compressing objects: 100% (145/145), done.
remote: Total 1340 (delta 66), reused 118 (delta 27), pack-reused 1167
Receiving objects: 100% (1340/1340), 4.22 MiB | 11.56 MiB/s, done.
Resolving deltas: 100% (763/763), done.
Submodule 'tkyamlgui' (https://github.com/lawrenceccheung/tkyamlgui.git) registered for path 'tkyamlgui'
Cloning into '/gpfs/lcheung/2022/amrwind-frontend-tutorial/amrwind-frontend/tkyamlgui'...
remote: Enumerating objects: 331, done.        
remote: Total 331 (delta 0), reused 0 (delta 0), pack-reused 331        
Receiving objects: 100% (331/331), 61.82 KiB | 711.00 KiB/s, done.
Resolving deltas: 100% (212/212), done.
Submodule path 'tkyamlgui': checked out '856069ce5097a38c1eb4eeb6ad8ef1b2e13c44c4'
```

Then, if that command worked out properly, you should be able to do:  
```bash
$ cd amrwind-frontend/
$ $ ./amrwind_frontend.py  --help
usage: amrwind_frontend.py [-h] [--ablstatsfile ABLSTATSFILE]
                           [--farmfile FARMFILE] [--samplefile SAMPLEFILE]
                           [--outputfile OUTPUTFILE] [--validate]
                           [--calcmeshsize] [--localconfigdir LOCALCONFIGDIR]
                           [inputfile]

AMR-Wind

positional arguments:
  inputfile

optional arguments:
  -h, --help            show this help message and exit
  --ablstatsfile ABLSTATSFILE
                        Load the ABL statistics file [default: None]
  --farmfile FARMFILE   Load the farm layout YAML file [default: None]
  --samplefile SAMPLEFILE
                        Load the sample probe file [default: None]
  --outputfile OUTPUTFILE
                        Write the output file [default: None]
  --validate            Check input file for errors and quit [default: False]
  --calcmeshsize        Estimate the meshsize [default: False]
  --localconfigdir LOCALCONFIGDIR
                        Local configuration directory [default:
                        /gpfs/lcheung/2022/amrwind-frontend-tutorial/amrwind-
                        frontend/local]
```

## Dependencies

`amrwind-frontend` should work on both python 2.7 and 3+.  However, it
requires on the following libraries to be installed:

**Required libraries**
- numpy
- scipy
- matplotlib
- netcdf4
- pyyaml
- pandas

In addition, there are a few libraries which make life easier or are
nice to have:

**Optional libraries**
- ruamel (for better yaml handling)
- xvfbwrapper (for non-X11/headless operation)
- [utm](https://pypi.org/project/utm/) (for handling lat/long conversion, see https://github.com/Turbo87/utm for library)


If you on an Ubuntu system, you can install all of these using apt:

```bash
sudo apt-get install python-numpy python-scipy  python-netcdf4 
sudo apt-get install python-matplotlib
sudo apt-get install python-yaml
sudo apt-get install python-enum34
```

And for some of the optional libraries,
```bash
sudo apt-get install python-ruamel.yaml 
sudo apt-get install python-xvfbwrapper 
```
