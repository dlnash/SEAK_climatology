#!/usr/bin/python3
"""
Filename:    wrf_preprocess.py
Author:      Deanna Nash, dlnash@ucsb.edu
Description: preprocess functions for AK 4 km WRF simulations downloaded from https://registry.opendata.aws/wrf-se-alaska-snap/
"""

## Imports
import numpy as np
import xarray as xr


def preprocess_PCPT(filenames):
    """preprocess_prec from SEAK WRF data
    
    Returns a ds object (xarray) including the ACCUMULATED TOTAL GRID SCALE PRECIPITATION (mm)
    
    Parameters
    ----------
    filenames : list
        list of wrf filenames to process
  
    Returns
    -------
    ds : ds object
        includes variables PCPT "ACCUMULATED TOTAL GRID SCALE PRECIPITATION" (mm)
    
    """
    # empty list to append datasets
    ds_lst = []

    for i, wrfin in enumerate(filenames):
        # open each file
        f = xr.open_dataset(wrfin)

        # start with fixing times
        f = f.assign(Time=f.time.values)
        f = f.drop(['time']) # drop time variable
        f = f.rename({'Time':'time'})

        # now let's fix coordinates
        f = f.assign_coords(lat=f.coords['XLAT'], lon=f.coords['XLONG']) # reassign lat and lon as coords
        f = f.rename({'south_north':'y', 'west_east':'x'}) # rename coords to be cf-compliant
        f = f.drop(['XLAT', 'XLONG']) # drop XLAT and XLONG coords

        # drop all other vars except precip
        varlst = list(f.keys())
        varlst.remove('PCPT')
        f = f.drop(varlst)
        f = f.drop(['interp_levels'])
        
        ds_lst.append(f)

        f.close()

    # concatenate datasets
    ds = xr.concat(ds_lst, dim='time')

    return ds