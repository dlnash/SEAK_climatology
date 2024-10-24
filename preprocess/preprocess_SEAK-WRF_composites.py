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
varname = 'PCPT' # 'PCPT' or 'UV'

df_lst = combine_ivt_ar_prec_df(option, temporal_res, community_lst) # combine dfs into list of dfs

## get list of dates that are Extreme Precip and AR for each community
ardate_lst = []
for i, df in enumerate(df_lst):
    idx = (df.AR == 1) & (df.extreme == 1) & (df.index != '2008-02-29 00:00:00') # hack to get rid of the leap day (not in WRF data)
    tmp = df.loc[idx]
    
    ar_dates = tmp.time.values
    ardate_lst.append(tmp.time.values)

fname_pattern = path_to_data + 'preprocessed/SEAK-WRF-{0}/WRFDS_{0}_*.nc'.format(varname)
wrf = xr.open_mfdataset(fname_pattern, combine='by_coords')
if temporal_res == 'hourly':
    wrf = wrf
elif (temporal_res == 'daily') & (varname == 'PCPT'):
    wrf = wrf.resample(time="1D").sum('time') # resample WRF data to be mm per day
    
elif (temporal_res == 'daily') & (varname == 'UV'):
    wrf = wrf.sel(lev='1000')
    wrf = wrf.resample(time="1D").mean('time') # resample WRF data to be m s-1


## make a dataset for each community subset to its AR dates
ds_lst = []
for i, ar_dates in enumerate(ardate_lst):
    print('Processing {0}'.format(community_lst[i]))
    tmp = wrf.sel(time=ar_dates)
    tmp = tmp.mean('time') # mean or max
    ds_lst.append(tmp.load())

## merge all communities into single DS with "community name" as axis
ds_comp = xr.concat(ds_lst, dim=community_lst)
ds_comp = ds_comp.rename({'concat_dim':'community'}) # rename concat_dim to community

# write to netCDF
fname = os.path.join(path_to_data, 'preprocessed/SEAK-WRF_{0}_daily_composite.nc'.format(varname))
ds_comp.to_netcdf(path=fname, mode = 'w', format='NETCDF4')