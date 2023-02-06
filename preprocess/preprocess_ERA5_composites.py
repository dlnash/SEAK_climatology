# Standard Python modules
import os, sys
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
import metpy.calc as mpcalc
from metpy.units import units


# import personal modules
sys.path.append('../modules') # Path to modules
from preprocess_dataframes import combine_ivt_ar_prec_df

# Set up paths

path_to_data = '/cw3e/mead/projects/cwp140/scratch/dnash/data/'      # project data -- read only
path_to_work = '/work/dnash/SEAK_clim_data/preprocessed/ERA5-IVT/'
path_to_out  = '../out/'       # output files (numerical results, intermediate datafiles) -- read & write
path_to_figs = '../figs/'      # figures

## open precipitation and ivt dfs
## append precip to each community IVT df
option = 'a'
temporal_res = 'daily'
community_lst = ['Hoonah', 'Skagway', 'Klukwan', 'Yakutat', 'Craig', 'Kasaan']

df_lst = combine_ivt_ar_prec_df(option, temporal_res, community_lst) # combine dfs into list of dfs

## get list of dates that are Extreme Precip and AR for each community
ardate_lst = []
for i, df in enumerate(df_lst):
    prec_thres = df['prec'].describe(percentiles=[.95]).loc['95%'] # 95th percentile precipitation threshold
    # idx = (df.AR == 1) & (df.prec > prec_thres) 
    idx = (df.AR == 1) & (df.prec > prec_thres) & (df.index != '2008-02-29 00:00:00') # hack to get rid of the leap day (not in WRF data)
    tmp = df.loc[idx]
    
    ar_dates = tmp.time.values
    ardate_lst.append(tmp.time.values)

## load 250Z data
varname = 'huv'
output_varname = '250Z'
drop_var = ['u', 'v']
lev = 250.
rename_var = {'latitude':'lat', 'longitude':'lon'}

## load MSLP data
varname = 'mslp'
output_varname = 'mslp'
drop_var = None
lev = None
rename_var = {'latitude':'lat', 'longitude':'lon'}


## load IVT data
varname = 'ivt'
output_varname = 'ivt'
drop_var = None
lev = None
rename_var = {'latitude':'lat', 'longitude':'lon', 'p71.162': 'IVTu', 'p72.162': 'IVTv'}


def preprocess(ds):
    '''keep only selected variable and level'''
    ds = ds.drop(drop_var)
    ds = ds.sel(level=lev)
    ds = ds.rename(rename_var)
    return ds

fname_pattern = path_to_data + 'downloads/ERA5/{0}/6hr/era5_ak_025dg_6hr_{0}_*.nc'.format(varname)
era = xr.open_mfdataset(fname_pattern, combine='by_coords', preprocess=preprocess)
if temporal_res == 'hourly':
    era = era
elif temporal_res == 'daily':
    era = era.resample(time="1D").mean('time') # resample daily
    
## make a dataset for each community subset to its AR dates
ds_lst = []
for i, ar_dates in enumerate(ardate_lst):
    print('Processing {0}'.format(community_lst[i]))
    tmp = era.sel(time=ar_dates)
    tmp = tmp.mean('time')
    tmp = tmp.load()
    
    # write to netCDF
    fname = os.path.join(path_to_data, 'preprocessed/ERA5_{0}_daily_{1}.nc'.format(output_varname, community_lst[i]))
    tmp.to_netcdf(path=fname, mode = 'w', format='NETCDF4')