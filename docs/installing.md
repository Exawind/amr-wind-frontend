# Installing `amrwind_frontend`

## From github
You can obtain `amrwind-frontend` by downloading it from github directly:
```bash
$ git clone git@github.com:lawrenceccheung/amrwind-frontend.git
$ cd amrwind-frontend
$ git submodule init
$ git submodule update --recursive
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

In addition, there are a few libraries which make life easier or are
nice to have:

**Optional libraries**
- ruamel (for better yaml handling)
- xvfbwrapper (for non-X11/headless operation)


If you on an Ubuntu system, you can install all of these using apt:

```bash
sudo apt-get install python-numpy python-scipy  python-netcdf4 
sudo apt-get install python-matplotlib
sudo apt-get install python-yaml
sudo apt-get install python-enum34
```

And for the optional libraries,
```bash
sudo apt-get install python-ruamel.yaml 
sudo apt-get install python-xvfbwrapper 
```