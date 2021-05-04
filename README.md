# AMR-Wind front end tool

A tool to help setup, visualize, and postprocess AMR-Wind simulations.

Built using the [TK yaml library](https://github.com/lawrenceccheung/tkyamlgui)

What can it do?
- Load an AMR-Wind input file and change parameters interactively
- Plot the simulation domain, including refinement zones and sampling
  probes/planes
- Help visualize the sampling outputs (probes, lines, and planes)
- Help postprocess ABL statistics files.
- Use it in Jupyter notebooks or python scripts to automate
  processing.

## Downloading 
You can obtain `amrwind-frontend` by downloading it from github directly:
```bash
$ git clone git@github.com:lawrenceccheung/amrwind-frontend.git
$ cd amrwind-frontend
$ git submodule init
$ git submodule update --recursive
```
## Running
Once you've downloaded it, you can launch the interactive GUI using
the command:

```bash
$ ./amrwind_frontend.py
```

This is what it should look like:
![screenshot](docs/amrwind_frontend_splash.png).

## User guide

This section is forthcoming.

