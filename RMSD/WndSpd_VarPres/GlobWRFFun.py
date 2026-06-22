import numpy as np
from netCDF4 import Dataset
from wrf import getvar, to_np, ll_to_xy, ALL_TIMES
from glob import glob

def MonthGlob(folders, Var, lat_point, lon_point, offset = 0):
    """
    Extract time series data from multiple WRF output folders for a specified variable and location.

    Args:
        folders (list of str): Paths to WRF output folders or file globs.
        Var (str): Variable to extract (e.g., 'T2').
        lat_point (float): Latitude of point.
        lon_point (float): Longitude of point.

    Returns:
        Time_all (np.ndarray): Time values (from the first dataset).
        Vals (list of np.ndarray): Extracted variable values for each folder.
    """
    Vals = []
    Time = []

    # Use the first file of the first folder to find the grid index
    sample_files = sorted(glob(folders[0]))
    sample_ds = Dataset(sample_files[0])
    x_idx, y_idx = map(int, ll_to_xy(sample_ds, lat_point, lon_point, as_int=True))

    # Loop through all folders / file globs
    for folder in folders:
        files = sorted(glob(folder))
        thisVal = []

        for f in files:
            ds = Dataset(f)
            var_data = getvar(ds, Var, timeidx=ALL_TIMES)
            thisVal.extend(to_np([var_data[y_idx, x_idx] + offset]))  # Kelvin to Celsius

            # Time info: only take once from the first folder
            if len(Time)<30*24:
                Time.append(var_data.Time.values)

        Vals.append(np.array(thisVal))

    return Time, Vals
