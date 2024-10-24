"""
Filename:    preprocess_AR_6-hourly.py
Author:      Deanna Nash, dnash@ucsd.edu
Description: This script generates a .csv 6-hourly and daily time series of days an AR makes landfall in southeast Alaska based on tARget v3 AR detection algorithm.
"""
# Import Python modules
import os, sys
import yaml
import numpy as np
import pandas as  pd
import xarray as xr
from datetime import timedelta, date

# Set up paths
server = "skyriver"
if server == "comet":
    path_to_data = '/cw3e/mead/projects/cwp140/scratch/dnash/data/'      # project data -- read only
elif server == "skyriver":
    path_to_data = '/work/dnash/data/'
path_to_out  = '../out/'       # output files (numerical results, intermediate datafiles) -- read & write
path_to_figs = '../figs/'      # figures

# Select lat/lon grid
# bnds = [360-175., 360-120., 50., 60.] # extend of AK
bnds = [360-141., 360-130., 54., 61.] # extent of SEAK
ext1 = [-141.5, -130.0, 54., 61.5] # extent of SEAK
lonmin, lonmax, latmin, latmax = bnds

# set start and end dates to match WRF data
start_date = '1980-01-01 0:00'
end_date = '2019-12-31 23:00'

## Read data
ar_filename =  '/home/dnash/comet_data/downloads/AR_Catalog/globalARcatalog_ERA-Interim_1979-2019_v3.0.nc'
ds = xr.open_dataset(ar_filename)
da = ds.sel(time=slice(start_date, end_date), lat=slice(latmin, latmax), lon=slice(lonmin, lonmax))


## Option 1: AR days based on if they make "landfall" in SE AK
# option = 'landfall'

## Option 2: AR days based on if the AR is within SEAK bounding box
option = 'bounding-box'

lfloc = da.lflocmap.squeeze()
lfloc_ct = lfloc.count('time')
clim_freq = lfloc_ct.squeeze().values

index_clim = np.flatnonzero(clim_freq)
landfall_clim = clim_freq.ravel()[index_clim]


clim_lons, clim_lats = np.meshgrid(lfloc_ct.lon, lfloc_ct.lat)
clim_lat_list = clim_lats.ravel()[index_clim]
clim_lon_list = clim_lons.ravel()[index_clim]

## manually adding points to land mask
lats_add = [58.5, 58.5, 57., 57., 55.5, 55.5]
lons_add = [-139.5, -138.0, -136.5, -135., -135., -133.5]

for i, (xs1, ys1) in enumerate(zip(lons_add, lats_add)):
    da.islnd.loc[dict(lat=ys1, lon=xs1+360.)] = 1
    
    
lfdates = lfloc.where(lfloc > 0, drop=True).time.values # only pulls the dates where an AR made landfall in SE AK
print('No. of AR landfalls in SE AK:', len(lfdates))

kidmap = da.kidmap.squeeze()
kidmap = kidmap.where(da.islnd == 1) # mask kidmap where is ocean
ardates = kidmap.where(kidmap > 0, drop=True).time.values # pulls the dates when an AR object is within bbox 
print('No. of AR objects near SE AK:', len(ardates))

# create list of lists of trackIDs for each time step
if option == 'landfall':
    date_lst = lfdates
elif option == 'bounding-box':
    date_lst = ardates
final_lst = []
for i, ids in enumerate(date_lst):
#     print(ids)
    # pull track ID list for each time step
    x = kidmap.sel(time=ids).values.flatten()
    result = x[np.logical_not(np.isnan(x))]
    trackID = np.unique(result)
    # # get landfalling latitude and longitude
    # tmp = lfloc.sel(time=ids)
    # tmp = tmp.where(tmp > 0, drop=True)
    # lflat = tmp.lat.values
    # lflon = tmp.lon.values - 360.0 # to get degrees W value

    for j in range(len(trackID)):
        final_lst.append([ids, trackID[j]])
    
# put final_lst into df
track_df = pd.DataFrame(final_lst, columns=['date', 'trackID'])
track_df
track_ids = track_df.trackID.unique()
# create df with trackID, start date, end date, and duration of AR
data = []
for i, ids in enumerate(track_ids):
    idx = (track_df.trackID == ids)
    test = track_df.loc[idx]
    start = test.date.min()
    stop = test.date.max() + timedelta(hours=6)
    tmp = (stop-start)
    duration = tmp.total_seconds()/(3600) # convert to number of hours
    
    data.append([ids, start, stop, duration])
    
duration_df = pd.DataFrame(data, columns=['trackID', 'start_date', 'end_date', 'duration'])

## save to csv
outfile = path_to_out + 'AR_track_duration_SEAK.csv'
duration_df.to_csv(outfile)

def get_ar_dates_from_duration_df(duration_df, freq):
    # Get list of AR time steps from the start-stop dates
    dt_lst = []
    for index, row in duration_df.iterrows():
        dt = pd.date_range(row['start_date'], row['end_date'], freq='1H')
        dt_lst.append(dt)
        
    ardates = dt_lst[0].union_many(dt_lst[1:])
    # put into dataframe
    d = {'ar_dates': ardates}
    df = pd.DataFrame(data=d)

    # keep only unique dates
    df.index = pd.to_datetime(df.ar_dates)
    df_new = df.ar_dates.unique()
    d = {'ar_dates': df_new}
    df = pd.DataFrame(data=d)
    
    if freq == '1D':
        ## this method takes the hourly dataset and converts it to daily
        df_hourly = df
        df_hourly.index = pd.to_datetime(df_hourly.ar_dates)
        idx = (df_hourly.index.hour == 0) | (df_hourly.index.hour == 6) | (df_hourly.index.hour == 12) | (df_hourly.index.hour == 18)
        test_daily = df_hourly.loc[idx]
        test_daily.index = test_daily.ar_dates.dt.normalize()
        df_new = test_daily.index.unique()
        d = {'ar_dates': df_new}
        df_daily = pd.DataFrame(data=d)
        df = df_daily
    else:
        df = df
        
    ## clean up dataframe to have 1's where AR == True
    df.index = pd.to_datetime(df.ar_dates)
    df = df.drop(['ar_dates'], axis=1)
    df = df.reset_index()
    df = df.rename(columns={"ar_dates": "dates"})
    df['ar'] = 1

    # date array with all days
    dates_allDays = pd.date_range(start=start_date, end=end_date, freq=freq)
    arr_allDays = np.zeros(len(dates_allDays), dtype=int)

    # Loop over ar days and match to ar_full 
    for i, date in enumerate(df['dates'].values):
        idx = np.where(dates_allDays == date)
        arr_allDays[idx] = 1

    # Create dataframe
    data = {'AR':arr_allDays}
    df_all = pd.DataFrame(data, index=dates_allDays)
    
    return df_all


ardates_hourly = get_ar_dates_from_duration_df(duration_df, '1H')
ardates_daily = get_ar_dates_from_duration_df(duration_df, '1D')

## save to csv file
ardates_hourly.to_csv(path_to_out + 'SEAK_ardates_hourly.csv')
ardates_daily.to_csv(path_to_out + 'SEAK_ardates_daily.csv')