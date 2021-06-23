# Local customization

Put any local customization of variables in this directory.  For
instance, the following can be used to override the locations of the
amr_wind executable and module loads.

```yaml
popupwindow:
  # === Set up a local run ===
  localrun:
    inputwidgets:
      - name:       localrun_exe
        defaultval: '/projects/wind_uq/lcheung/AMRWindBuilds/tcf.20210610/amr-wind.lcheung/build/amr_wind'
      - name:       localrun_modules
        defaultval: "module load cde/prod/gcc/7.2.0/openmpi/3.1.6 cde/prod/gcc/7.2.0/hdf5 cde/prod/gcc/7.2.0/netcdf-c/4.7.3"
```
