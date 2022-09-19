# Installing `amrwind_frontend`

## From github

You can obtain `amrwind-frontend` by downloading it from github directly with a `git clone` command:

```bash
$ git clone --recursive git@github.com:Exawind/amr-wind-frontend.git
Cloning into 'amr-wind-frontend'...
Warning: the ECDSA host key for 'github.com' differs from the key for the IP address '140.82.113.4'
Offending key for IP in /ascldap/users/lcheung/.ssh/known_hosts:111
Matching host key in /ascldap/users/lcheung/.ssh/known_hosts:132
X11 forwarding request failed on channel 0
remote: Enumerating objects: 1460, done.
remote: Counting objects: 100% (1460/1460), done.
remote: Compressing objects: 100% (602/602), done.
remote: Total 1460 (delta 856), reused 1455 (delta 851), pack-reused 0
Receiving objects: 100% (1460/1460), 4.17 MiB | 0 bytes/s, done.
Resolving deltas: 100% (856/856), done.
Submodule 'tkyamlgui' (https://github.com/lawrenceccheung/tkyamlgui.git) registered for path 'tkyamlgui'
Cloning into 'tkyamlgui'...
remote: Enumerating objects: 334, done.
remote: Counting objects: 100% (3/3), done.
remote: Compressing objects: 100% (3/3), done.
remote: Total 334 (delta 0), reused 1 (delta 0), pack-reused 331
Receiving objects: 100% (334/334), 77.67 KiB | 0 bytes/s, done.
Resolving deltas: 100% (212/212), done.
Submodule path 'tkyamlgui': checked out 'd3a9d0543e5a8367d3a9c11980b33aeb7d0e902c'
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
