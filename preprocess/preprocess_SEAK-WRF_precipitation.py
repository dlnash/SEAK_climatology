"""
Filename:    preprocess_SEAK-WRF_precipitation.py
Author:      Deanna Nash, dlnash@ucsb.edu
Description: preprocess hourly precipitation from 40 years of SEAK-WRF data and save as yearly nc files
"""

## Imports
import os, sys
import glob
import numpy as np
import xarray as xr

# Necessary to add cwd to path when script run
# by SLURM (since it executes a copy)
sys.path.append(os.getcwd())

# Path to modules
sys.path.append('../modules')
# Import my modules
from wrf_preprocess import preprocess_PCPT

## Set up paths
path_to_wrf = '/cw3e/mead/projects/cwp140/scratch/dnash/data/downloads/SEAK-WRF/' # access raw SEAK-WRF files
path_to_out = '/cw3e/mead/projects/cwp140/scratch/dnash/data/preprocessed/SEAK-WRF-precip/' # save preprocessed precip

# command for selecting variables **include XTIME,XLONG,XLAT in var list
# ncks -v varname1,varname2 input_file output_file

### Input ###

### PRECIP ###
output_varname = 'PCPT'

### START PROGRAM ###
start_yr = 1980
end_yr = 2019

## Loop through all above years
for year in np.arange(start_yr, end_yr+1):
    print('Processing ...', year)
    
    # get list of filenames that contain data from that year from current year folder
    filenames = []
    for name in glob.glob(path_to_wrf + '{0}/WRFDS_*'.format(str(year))):
        filenames.append(name)
    # sort filenames so they are in chronological order
    filenames = sorted(filenames)

    ## create new ds with just precip
    ds = preprocess_PCPT(filenames)
    
    # write to netCDF
    fname = os.path.join(path_to_out, 'WRFDS_PCPT_{0}.nc').format(str(year))
    ds.to_netcdf(path=fname, mode = 'w', format='NETCDF4')