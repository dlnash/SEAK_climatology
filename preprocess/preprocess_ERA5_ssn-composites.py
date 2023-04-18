# Standard Python modules
import os, sys
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
import metpy.calc as mpcalc
from metpy.units import units

# Set up paths

path_to_data = '/cw3e/mead/projects/cwp140/scratch/dnash/data/'      # project data -- read only
path_to_out  = '../out/'       # output files (numerical results, intermediate datafiles) -- read & write
path_to_figs = '../figs/'      # figures

## open ERA5 data

varname_lst = ['huv', 'ivt', 'mslp']

def preprocess_huv(ds):
    '''keep only selected variable and level'''
    ds = ds.drop(['u', 'v'])
    ds = ds.sel(level=250.)
    ds = ds.rename({'latitude':'lat', 'longitude':'lon'})
    return ds

def preprocess_mslp(ds):
    '''keep only selected variable and level'''
    ds = ds.rename({'latitude':'lat', 'longitude':'lon'})
    return ds

def preprocess_ivt(ds):
    '''keep only selected variable and level'''
    ds = ds.rename({'latitude':'lat', 'longitude':'lon', 'p71.162': 'IVTu', 'p72.162': 'IVTv'})
    return ds

ds_final = []
for i, varname in enumerate(varname_lst):
    print('Reading ...', varname)
    fname_pattern = path_to_data + 'downloads/ERA5/{0}/6hr/era5_ak_025dg_6hr_{0}_*.nc'.format(varname)
    if varname == 'huv':
        era = xr.open_mfdataset(fname_pattern, combine='by_coords', preprocess=preprocess_huv)
    elif varname == 'mslp':
        era = xr.open_mfdataset(fname_pattern, combine='by_coords', preprocess=preprocess_mslp)
    elif varname == 'ivt':
        era = xr.open_mfdataset(fname_pattern, combine='by_coords', preprocess=preprocess_ivt)
        era = era.assign(IVT=lambda era: np.sqrt(era.IVTu**2 + era.IVTv**2))

    ## calculate seasonal averages
    ds_ssn = era.resample(time='QS-DEC', keep_attrs=True).mean()                            

    # calculate average of seasonal avg
    clim = ds_ssn.groupby('time.season').mean(dim='time').compute()
    ds_final.append(clim)
    
ds_write = xr.combine_by_coords(ds_final)

# write to netCDF
fname = os.path.join(path_to_data, 'preprocessed/ERA5_ivt_250z_mslp_clim.nc')
ds_write.to_netcdf(path=fname, mode = 'w', format='NETCDF4')