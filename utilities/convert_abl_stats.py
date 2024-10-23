### This is a helper script to read MMC forcing data from ABL stats file
### and convert it into a tendency forcing input file

import netCDF4 as nc
import argparse


## Read the relevant fields from ABL Stats
def read_abl_stats(filename):
    data = {}
    abl_stats = nc.Dataset(filename, "r")
    data["times"] = abl_stats.variables["time"][:]
    data["tflux"] = abl_stats.variables["Q"][:]
    mean_profiles = abl_stats["mean_profiles"]
    data["heights"] = mean_profiles.variables["h"][:]
    data["momentum_u"] = mean_profiles.variables["abl_meso_forcing_mom_x"][:].flatten()
    data["momentum_v"] = mean_profiles.variables["abl_meso_forcing_mom_y"][:].flatten()
    data["temperature"] = mean_profiles.variables["abl_meso_forcing_theta"][:].flatten()
    abl_stats.close()
    return data


## Create a tendency forcing netcdf file
def create_tendency_forcing(data, filename):
    tf = nc.Dataset(filename, "w")
    tf.createDimension("ntime", len(data["times"]))
    tf.createDimension("nheight", len(data["heights"]))
    tf.createDimension("datasize", len(data["heights"]) * len(data["times"]))

    for var in ["momentum_u", "momentum_v", "temperature"]:
        temp_var = tf.createVariable(var, "double", ("datasize",))
        temp_var[:] = data[var]
    for var in ["tflux", "times"]:
        temp_var = tf.createVariable(var, "double", ("ntime",))
        temp_var[:] = data[var]
    for var in ["heights"]:
        temp_var = tf.createVariable(var, "double", ("nheight",))
        temp_var[:] = data[var]
    tf.setncattr("coordinates", "heights")
    tf.close()


if __name__ == "__main__":
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--statsfile",
        help="ABL stats file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--outfile", help="Output file", type=str, default="tendency_forcing.nc"
    )
    args = parser.parse_args()
    print(f"Reading {args.statsfile}")
    data = read_abl_stats(args.statsfile)
    create_tendency_forcing(data, args.outfile)
