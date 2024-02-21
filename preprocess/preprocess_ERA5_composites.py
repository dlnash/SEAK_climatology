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
from statistical_tests import xr_zscore_diff_mean

# Set up paths

path_to_data = '/home/dnash/comet_data/'      # project data -- read only
path_to_out  = '../out/'       # output files (numerical results, intermediate datafiles) -- read & write
path_to_figs = '../figs/'      # figures

## open precipitation and ivt dfs
## append precip to each community IVT df
option = 'a'
temporal_res = 'daily'
community_lst = ['Hoonah', 'Skagway', 'Klukwan', 'Yakutat', 'Craig', 'Kasaan']
lag_lst = [-4, -3, -2, -1, 0]

df_lst = combine_ivt_ar_prec_df(option, temporal_res, community_lst) # combine dfs into list of dfs

## get list of dates that are Extreme Precip and AR for each community
ardate_lst = []
for i, df in enumerate(df_lst):
    idx = (df.AR == 1) & (df.extreme == 1) & (df.index != '2008-02-29 00:00:00') # hack to get rid of the leap day (not in WRF data)
    tmp = df.loc[idx]
    
    ar_dates = tmp.time.values
    ardate_lst.append(tmp.time.values)
    
    
## merge ardate_lst into single list and remove duplicates
tmp = np.concatenate(ardate_lst, axis=0)
new_data = np.unique(tmp)

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

diff_ds_final = []
pval_ds_final = []
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
    
    all_extreme_ARs = era.sel(time=new_data)

    ## make a dataset for each community subset to its AR dates
    diff_lst = []
    pval_lst = []
    for i, ar_dates in enumerate(ardate_lst):
        print('Processing {0}'.format(community_lst[i]))
        tmp = era.sel(time=ar_dates)
        diff, pval = xr_zscore_diff_mean(tmp, all_extreme_ARs)
        diff = diff.load()
        pval = pval.load()
        diff_lst.append(diff) # append to list
        pval_lst.append(pval)

    ## merge all communities into single DS with "community name" as axis
    ds_comp_diff = xr.concat(diff_lst, dim=community_lst)
    ds_comp_diff = ds_comp_diff.rename({'concat_dim':'community'}) # rename concat_dim to community
    diff_ds_final.append(ds_comp_diff)
    
    ds_comp_pval = xr.concat(pval_lst, dim=community_lst)
    ds_comp_pval = ds_comp_pval.rename({'concat_dim':'community'}) # rename concat_dim to community
    pval_ds_final.append(ds_comp_pval)
    
diff_ds_write = xr.combine_by_coords(diff_ds_final)
pval_ds_write = xr.combine_by_coords(pval_ds_final)

# write to netCDF
fname = os.path.join(path_to_data, 'preprocessed/ERA5_ivt_250z_mslp_daily_diff_composite.nc')
diff_ds_write.to_netcdf(path=fname, mode = 'w', format='NETCDF4')

fname = os.path.join(path_to_data, 'preprocessed/ERA5_ivt_250z_mslp_daily_pval_composite.nc')
pval_ds_write.to_netcdf(path=fname, mode = 'w', format='NETCDF4')