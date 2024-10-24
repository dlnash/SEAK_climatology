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
from wrf_preprocess import lag_and_combine

# Set up paths

path_to_data = '/home/dnash/comet_data/'      # project data -- read only
path_to_out  = '../out/'       # output files (numerical results, intermediate datafiles) -- read & write
path_to_figs = '../figs/'      # figures

## open precipitation and ivt dfs
## append precip to each community IVT df
option = 'a'
temporal_res = 'daily'
community_lst = ['Hoonah', 'Skagway', 'Klukwan', 'Yakutat', 'Craig', 'Kasaan']
lag_lst = [0]

df_lst = combine_ivt_ar_prec_df(option, temporal_res, community_lst) # combine dfs into list of dfs

## get list of dates that are:
### (middle) top 95th percentile precip (ALL IVT)

ardate_lst = []
for i, df in enumerate(df_lst):
    idx = (df.AR == 1) & (df.extreme == 1) & (df.index != '2008-02-29 00:00:00') # hack to get rid of the leap day (not in WRF data)
    tmp = df.loc[idx]
    
    ar_dates = tmp.time.values
    ardate_lst.append(tmp.time.values)
    
## merge ardate_lst into single list and remove duplicates
tmp = np.concatenate(ardate_lst, axis=0)
new_data = np.unique(tmp)
print(len(new_data))
## get list of dates that are:
### (left) top 5th percentile IVT, bottom 5th percentile of precip

ardate_lst = []
for i, df in enumerate(df_lst):
    prec_thres = df['prec'].describe(percentiles=[.05]).loc['5%'] # 5th percentile precipitation threshold
    ivt_thres = df['IVT'].describe(percentiles=[.95]).loc['95%'] # 95th percentile IVT threshold
    idx = (df.AR == 1) & (df.prec < prec_thres) & (df.IVT > ivt_thres) & (df.index != '2008-02-29 00:00:00') # hack to get rid of the leap day (not in WRF data)
    tmp = df.loc[idx]
    
    ar_dates = tmp.time.values
    ardate_lst.append(tmp.time.values)
    
## merge ardate_lst into single list and remove duplicates
tmp = np.concatenate(ardate_lst, axis=0)
new_data2 = np.unique(tmp)
print(len(new_data2))
## get list of dates that are:
### (right) top 95th percentile IVT, top 95th percentile precip

ardate_lst = []
for i, df in enumerate(df_lst):
    prec_thres = df['prec'].describe(percentiles=[.95]).loc['95%'] # 95th percentile precipitation threshold
    ivt_thres = df['IVT'].describe(percentiles=[.95]).loc['95%'] # 95th percentile IVT threshold
    idx = (df.AR == 1) & (df.prec > prec_thres) & (df.IVT > ivt_thres) & (df.index != '2008-02-29 00:00:00') # hack to get rid of the leap day (not in WRF data)
    tmp = df.loc[idx]
    
    ar_dates = tmp.time.values
    ardate_lst.append(tmp.time.values)
    
## merge ardate_lst into single list and remove duplicates
tmp = np.concatenate(ardate_lst, axis=0)
new_data3 = np.unique(tmp)
print(len(new_data3))
ardate_lst = [new_data, new_data2, new_data3]

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
special_lst = ['middle', 'left', 'right']
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


    if temporal_res == 'hourly':
        era = era
    elif temporal_res == 'daily':
        era = era.resample(time="1D").mean('time') # resample daily

    ## create lagged composites
    era = lag_and_combine(era, lags=lag_lst, dim='time')

    ## make a dataset for each community subset to its AR dates
    ds_lst = []
    for i, ar_dates in enumerate(ardate_lst):
        print('Processing {0}'.format(special_lst[i]))
        tmp = era.sel(time=ar_dates)
        tmp = tmp.mean('time')
        tmp = tmp.load()
        ds_lst.append(tmp) # append to list

    ## merge all communities into single DS with "community name" as axis
    ds_comp = xr.concat(ds_lst, dim=special_lst)
    ds_comp = ds_comp.rename({'concat_dim':'special'}) # rename concat_dim to community
    ds_final.append(ds_comp)
    
ds_write = xr.combine_by_coords(ds_final)

# write to netCDF
fname = os.path.join(path_to_data, 'preprocessed/ERA5_ivt_250z_mslp_daily_composite_special.nc')
ds_write.to_netcdf(path=fname, mode = 'w', format='NETCDF4')