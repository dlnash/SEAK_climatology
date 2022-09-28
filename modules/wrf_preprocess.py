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
    """preprocess_prec from AK WRF data
    
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
    # arrays to append data
    prec_final = []
    da_time = []

    for i, wrfin in enumerate(filenames):
        f = xr.open_dataset(wrfin)

        # get lats and lons
        wrflats = f['XLAT'].isel(west_east=0).values
        wrflons = f['XLONG'].isel(south_north=0).values
         # extract the data we need
        pcpt  = f['PCPT'].values

        # get time steps from this file
        da_time.append(f.time.values)

        # put values into preassigned arrays
        prec_final.append(pcpt)

        f.close()

    # merge array of dates
    dates = np.concatenate(da_time, axis=0)
    
    # stack prec arrays
    prec_array = np.concatenate(prec_final, axis=0)

    # convert lats/lons to 4-byte floats (this alleviates striping issue)
    wrflats = np.float32(wrflats)
    wrflons = np.float32(wrflons)

    # put into a dataset
    var_dict = {'prec': (['time', 'lat', 'lon'], prec_array)}
    ds = xr.Dataset(var_dict,
                    coords={'time': (['time'], dates),
                            'lat': (['lat'], wrflats),
                            'lon': (['lon'], wrflons)})
    
    ds.prec.attrs['units'] = 'mm'

    return ds