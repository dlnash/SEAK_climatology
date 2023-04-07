#!/usr/bin/python3
"""
Filename:    wrf_preprocess.py
Author:      Deanna Nash, dlnash@ucsb.edu
Description: preprocess functions for AK 4 km WRF simulations downloaded from https://registry.opendata.aws/wrf-se-alaska-snap/
"""

## Imports
import numpy as np
import xarray as xr


def preprocess_UVwind(filenames, levlst):
    """preprocess 3D var from SEAK WRF data
    
    Returns a ds object (xarray) with u and v wind for the selected vertical levels
    ex: levlst = ['850'] - selects the 850 level 
    
    Parameters
    ----------
    filenames : list
        list of wrf filenames to process
    levlst : list
        list of pressure level to select (available levs: 100.,  200.,  300.,  400.,  500.,  700.,  850.,  925., 1000.)
  
    Returns
    -------
    ds : ds object
        includes U and V wind at the requested vertical level(s)
    
    """
    # arrays to append data
    u_final = []
    v_final = []
    da_time = []

    for i, wrfin in enumerate(filenames):
        
        f = xr.open_dataset(wrfin)

        # get lats and lons
        wrflats = f['lat'].values
        wrflons = f['lon'].values
         # extract the data we need
        udata  = f['U'].sel(interp_levels=levlst).values
        vdata  = f['V'].sel(interp_levels=levlst).values

        # get time steps from this file
        da_time.append(f.time.values)

        # put values into preassigned arrays
        u_final.append(udata)
        v_final.append(vdata)
        
        # get units and attributes from original
        attributes = f.attrs
        u_attrs = f['U'].attrs
        v_attrs = f['V'].attrs

        f.close()

    # merge array of dates
    dates = np.concatenate(da_time, axis=0)
    
    # stack prec arrays
    udata_array = np.concatenate(u_final, axis=0)
    vdata_array = np.concatenate(v_final, axis=0)

    # convert lats/lons to 4-byte floats (this alleviates striping issue)
    wrflats = np.float32(wrflats)
    wrflons = np.float32(wrflons)

    # put into a dataset
    var_dict = {'U': (['time', 'y', 'x'], udata_array),
                'V': (['time', 'y', 'x'], vdata_array)}
    ds = xr.Dataset(var_dict,
                    coords={'time': (['time'], dates),
                            'lat': (['y', 'x'], wrflats),
                            'lon': (['y', 'x'], wrflons)})
    
    # set dataset and dataarray attributes
    ds['U'].attrs = u_attrs
    ds['V'].attrs = v_attrs
    ds.attrs = attributes
    
    return ds


def preprocess_2Dvar(filenames, varname):
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

def lag_and_combine(ds, lags, dim="time"):
    """
    Adapted from: https://github.com/jbusecke/xarrayutils/blob/master/xarrayutils/utils.py
    Creates lagged versions of the input object,
    combined along new `lag` dimension.
    NOTE: Lagging produces missing values at boundary. Use `.fillna(...)`
    to avoid problems with e.g. xr_linregress.
    Parameters
    ----------
    ds : {xr.DataArray, xr.Dataset}
        Input object
    lags : np.Array
        Lags to be computed and combined. Values denote number of timesteps.
        Negative lag indicates a shift backwards (to the left of the axis).
    dim : str
        dimension of `ds` to be lagged
    Returns
    -------
    {xr.DataArray, xr.Dataset}
        Lagged version of `ds` with additional dimension `lag`
    """

    datasets = []
    for ll in lags:
        datasets.append(ds.shift(**{dim: ll}))
        
    final = xr.concat(datasets, dim=lags)
    final = final.rename({'concat_dim':'lag'}) # rename concat_dim to lag
    
    return final