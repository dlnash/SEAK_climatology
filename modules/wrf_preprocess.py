#!/usr/bin/python3
"""
Filename:    wrf_preprocess.py
Author:      Deanna Nash, dlnash@ucsb.edu
Description: preprocess functions for AK 4 km WRF simulations downloaded from https://registry.opendata.aws/wrf-se-alaska-snap/
"""

## Imports
import numpy as np
import xarray as xr


def preprocess_2Dvar(filenames, varname):
    """preprocess 2D var from SEAK WRF data
    
    Returns a ds object (xarray) including the ACCUMULATED TOTAL GRID SCALE PRECIPITATION (mm)
    
    Parameters
    ----------
    filenames : list
        list of wrf filenames to process
    varname : str
        str of 2D varname to process
  
    Returns
    -------
    ds : ds object
        includes 2D variable requested to preprocess
        ex: PCPT "ACCUMULATED TOTAL GRID SCALE PRECIPITATION" (mm)
        ex: T2 "TEMP at 2 M" (K)
    
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
        f = f.assign_coords(lat=f['lat'], lon=f['lon']) # reassign lat and lon as coords
        f = f.rename({'south_north':'y', 'west_east':'x'}) # rename coords to be cf-compliant

        # drop all other vars except precip
        varlst = list(f.keys())
        varlst.remove(varname)
        f = f.drop(varlst)
        f = f.drop(['interp_levels'])
        
        ds_lst.append(f)

        f.close()

    # concatenate datasets
    ds = xr.concat(ds_lst, dim='time')

    return ds

def preprocess_2Dvar_new(filenames, varname):
    """preprocess 2D var from SEAK WRF data
    
    Returns a ds object (xarray) including the indicated variable
    ex: varname = 'PCPT' - Preprocesses precipitation "ACCUMULATED TOTAL GRID SCALE PRECIPITATION" (mm)
    ex: varname = 'T2' - Preprocesses 2 m temperature "TEMP at 2 M" (K)
    
    Parameters
    ----------
    filenames : list
        list of wrf filenames to process
    varname : str
        str of 2D varname to process
  
    Returns
    -------
    ds : ds object
        includes 2D variable requested to preprocess
    
    """
    # arrays to append data
    var_final = []
    da_time = []

    for i, wrfin in enumerate(filenames):
        
        f = xr.open_dataset(wrfin)

        # get lats and lons
        wrflats = f['lat'].values
        wrflons = f['lon'].values
         # extract the data we need
        data  = f[varname].values

        # get time steps from this file
        da_time.append(f.time.values)

        # put values into preassigned arrays
        var_final.append(data)
        
        # get units and attributes from original
        attributes = f.attrs
        var_attrs = f[varname].attrs

        f.close()

    # merge array of dates
    dates = np.concatenate(da_time, axis=0)
    
    # stack prec arrays
    data_array = np.concatenate(var_final, axis=0)

    # convert lats/lons to 4-byte floats (this alleviates striping issue)
    wrflats = np.float32(wrflats)
    wrflons = np.float32(wrflons)

    # put into a dataset
    var_dict = {varname: (['time', 'y', 'x'], data_array)}
    ds = xr.Dataset(var_dict,
                    coords={'time': (['time'], dates),
                            'lat': (['y', 'x'], wrflats),
                            'lon': (['y', 'x'], wrflons)})
    
    # set dataset and dataarray attributes
    ds[varname].attrs = var_attrs
    ds.attrs = attributes

    return ds