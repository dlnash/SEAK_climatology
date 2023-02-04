# Standard Python modules
import os, sys
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
import metpy.calc as mpcalc
from metpy.units import units


# import personal modules
# Path to modules
sys.path.append('../modules')

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

# open precipitation data
fname = path_to_out + 'SEAK_precip_max_{0}_{1}.csv'.format(option, temporal_res)
prec_df = pd.read_csv(fname)
prec_df = prec_df.replace(0, np.NaN) # replace any instance of zero with nan to ignore dates with no precipitation

# open IVT data
df_lst = []
for i, community in enumerate(community_lst):
    fname = path_to_out + 'IVT_ERA5_{0}.csv'.format(community)
    ivt_df = pd.read_csv(fname)
    
    ## calculate IVT direction
    uvec = units.Quantity(ivt_df['uIVT'].values, "m/s")
    vvec = units.Quantity(ivt_df['vIVT'].values, "m/s")
    ivtdir = mpcalc.wind_direction(uvec, vvec)
    ivt_df['ivtdir'] = ivtdir
    ivt_df = ivt_df.drop(['Unnamed: 0'], axis=1) # drop unnecessary vars
    
    if temporal_res == 'hourly':
        ivt_df = ivt_df
    elif temporal_res == 'daily':
        ivt_df.index = ivt_df['time']
        ivt_df = ivt_df.set_index(pd.to_datetime(ivt_df['time'])) # reset the index as "date" 
        ivt_df = ivt_df.resample('1D').mean(numeric_only=True)
        ivt_df = ivt_df.reset_index() # remove the index
    
    ## append AR data
    fname = path_to_out + 'SEAK_ardates_{0}.csv'.format(temporal_res)
    ar_df = pd.read_csv(fname) # read in AR dates

    # append AR dates to current df
    ivt_df['AR'] = ar_df.AR
    
    ## append impact data
    fname = path_to_out + 'SEAK_impactdates_{0}.csv'.format(temporal_res)
    impact_df = pd.read_csv(fname) # read in impact dates
    # append impact dates to current df
    ivt_df['impact'] = impact_df.IMPACT
    
    # ## append community precipitation data
    ivt_df['prec'] = prec_df[community]
    
    # reset the index as "time"
    ivt_df = ivt_df.set_index(pd.to_datetime(ivt_df['time']))
    
    ## select the 00, 06, 12, and 18 hour timesteps
    idx = (ivt_df.index.hour == 0) | (ivt_df.index.hour == 6) | (ivt_df.index.hour == 12) | (ivt_df.index.hour == 18)
    ivt_df = ivt_df.loc[idx]
    
    df_lst.append(ivt_df)
    

## get list of dates that are Extreme Precip and AR for each community
ardate_lst = []
for i, df in enumerate(df_lst):
    prec_thres = df['prec'].describe(percentiles=[.95]).loc['95%'] # 95th percentile precipitation threshold
    # idx = (df.AR == 1) & (df.prec > prec_thres) 
    idx = (df.AR == 1) & (df.prec > prec_thres) & (df.index != '2008-02-29 00:00:00') # hack to get rid of the leap day (not in WRF data)
    tmp = df.loc[idx]
    
    ar_dates = tmp.time.values
    ardate_lst.append(tmp.time.values)

## load MSLP
def preprocess(ds):
    '''rename lats/lons'''
    ds = ds.rename({'latitude':'lat', 'longitude':'lon'})
    return ds

fname_pattern = path_to_data + 'downloads/ERA5/mslp/6hr/era5_ak_025dg_6hr_mslp_*.nc'
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
    fname = os.path.join(path_to_data, 'preprocessed/ERA5_mslp_daily_{0}.nc'.format(community_lst[i]))
    tmp.to_netcdf(path=fname, mode = 'w', format='NETCDF4')