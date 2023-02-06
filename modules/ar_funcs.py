"""
Filename:    ar_funcs.py
Author:      Deanna Nash, dlnash@ucsb.edu
Description: Functions for getting AR metrics
"""

# Import Python modules
import os, sys
import numpy as np
import xarray as xr
import pandas as pd
from itertools import groupby


## FUNCTIONS
def get_topo_mask(regridlats, regridlons):
    '''
    Gets topography mask from ETOPO1 data based on regridding (downscaled) lats and lons
    '''
    fname = '/home/sbarc/students/nash/data/elevation_data/ETOPO1_Bed_g_gmt4.grd'
    version = 'bedrock'
    grid = xr.open_dataset(fname)
    # Add more metadata and fix some names
    names = {"ice": "Ice Surface", "bedrock": "Bedrock"}
    grid = grid.rename(z=version, x="lon", y="lat")
    grid[version].attrs["long_name"] = "{} relief".format(names[version])
    grid[version].attrs["units"] = "meters"
    grid[version].attrs["vertical_datum"] = "sea level"
    grid[version].attrs["datum"] = "WGS84"
    grid.attrs["title"] = "ETOPO1 {} Relief".format(names[version])
    grid.attrs["doi"] = "10.7289/V5C8276M"
    
    # select matching domain
    grid = grid.sel(lat=slice(regridlats.min(), regridlats.max()), lon=slice(regridlons.min(), regridlons.max()))

    # regrid topo to match horizontal resolution
    regrid_topo = grid.interp(lon=regridlons, lat=regridlats)
    
    return regrid_topo

def resample_track_id(df):
    '''
    Returns an array that has a single AR track ID for each 24 hr
    
    '''
    # stack up 3 subregions id numbers
    d = {'r01id': df['R01_id'],
         'r02id': df['R02_id'],
         'r03id': df['R03_id'],
         'r04id': df['R04_id']}

    df_tmp = pd.DataFrame(data=d)
    ## combine into single series
    df_tmp = df_tmp.stack()

    ## resample to 1D taking maximum of new column
    level_values = df_tmp.index.get_level_values
    df_tmp = df_tmp.groupby([level_values(i) for i in [0,1]]+[pd.Grouper(freq='1D', level=0)]).max()


    # date array with all days
    start_date = str(df.index.year[0])+'-'+ str(df.index.month[0]) + '-' + str(df.index.day[0])
    end_date = str(df.index.year[-1])+'-'+ str(df.index.month[-1]) + '-' + str(df.index.day[-1])
    dates_allDays = pd.date_range(start=start_date, end=end_date, freq='1D')
    arr_allDays = np.zeros(len(dates_allDays), dtype=np.float)
    arr_allDays[:] = np.nan

    # Loop over AR days ID and match to list of ALL days 
    for i, date in enumerate(df_tmp.index.get_level_values(2)):
        idx = np.where(dates_allDays == date)
        arr_allDays[idx] = df_tmp.values[i]

    return arr_allDays

def preprocess_ar_area_subregions(df, thres):
    '''
    Returns dataframe encoded for AR Days and identifies subregion where AR is present.
    
    Parameters
    ----------
    df : pandas dataframe
        dataframe that has subregion where percentage of area of AR is given for each day
    threshold: number, float
        singular threshold used for the percentage of area covered by an AR
        
    Returns
    -------
    df : pandas dataframe
        df that indicates whether each time step is an AR day and the location of the AR
    '''
    ## drop lev and ens cols
    df = df.drop(columns=['lev', 'ens'])
    ## Get single AR ID for each day
    track_ids = resample_track_id(df)
    
    # resample to daily
    df = df.resample('1D').mean()
    ## manually add column back into resampled df with area covered
    df['track_id'] = track_ids
    df = df.drop(columns=['R01_id', 'R02_id', 'R03_id', 'R04_id'])
    
    # Add column of AR days based on threshold
    # (no AR day eq 0; AR day eq 1)
    df['ar'] = 0
    idx = (df['R01'] > thres) | (df['R02'] > thres) | (df['R03'] > thres)
    df.loc[idx, 'ar'] = 1

    # Add column of AR locations 
    # ('R01', 'R02', 'R03', 'R01/R02', 'R02/R03', 'R01/R03', 'R01/R02/R03', nan)
    df['location'] = np.nan

    idx = (df['R01'] >= thres) & (df['R02'] < thres) & (df['R03'] < thres)
    df.loc[idx, 'location'] = 'R01'

    idx = (df['R01'] < thres) & (df['R02'] >= thres) & (df['R03'] < thres)
    df.loc[idx, 'location'] = 'R02'

    idx = (df['R01'] < thres) & (df['R02'] < thres) & (df['R03'] >= thres)
    df.loc[idx, 'location'] = 'R03'

    idx = (df['R01'] >= thres) & (df['R02'] >= thres) & (df['R03'] < thres)
    df.loc[idx, 'location'] = 'R01/R02'

    idx = (df['R01'] < thres) & (df['R02'] >= thres) & (df['R03'] >= thres)
    df.loc[idx, 'location'] = 'R02/R03'

    idx = (df['R01'] >= thres) & (df['R02'] < thres) & (df['R03'] >= thres)
    df.loc[idx, 'location'] = 'R01/R03'

    idx = (df['R01'] >= thres) & (df['R02'] >= thres) & (df['R03'] >= thres)
    df.loc[idx, 'location'] = 'R01/R02/R03'
    
    return df

def preprocess_ar_SASIA(df, thres):
    '''
    Returns dataframe encoded for AR Days and identifies trackID of AR.
    
    Parameters
    ----------
    df : pandas dataframe
        dataframe that has subregion where percentage of area of AR is given for each day
    threshold: number, float
        singular threshold used for the percentage of area covered by an AR
        
    Returns
    -------
    df : pandas dataframe
        df that indicates whether each time step is an AR day and the location of the AR
    '''
    ## drop lev and ens cols
    df = df.drop(columns=['lev', 'ens'])
    ## Get single AR ID for each day
    track_ids = resample_track_id(df)
    
    # resample to daily
    df = df.resample('1D').mean()
    ## manually add column back into resampled df with area covered
    df['track_id'] = track_ids
    df = df.drop(columns=['R01_id', 'R02_id', 'R03_id', 'R04_id'])
    
    # Add column of AR days based on threshold
    # (no AR day eq 0; AR day eq 1)
    df['ar'] = 0
    idx = (df['R04'] > thres)
    df.loc[idx, 'ar'] = 1
    
    return df

def preprocess_ar_bbox(ds, start_date, end_date):
        
        # convert dataset to dataframe
        df = ds.kidmap.to_dataframe(dim_order=['time', 'lat', 'lon'])
        df = df.dropna(axis='rows')
        # keep only rows that have trackID
        new_df = df.groupby('time').kidmap.unique()
        trackID = new_df

        # create list of days that have an AR
        date_lst = new_df.index.normalize().unique()
        date_lst = date_lst + pd.DateOffset(hours=9)
        
        # Create AR dataframe
        arr_ARDays = np.ones(len(date_lst), dtype=np.int)
        tmp = pd.DataFrame({'ar':arr_ARDays, 'time':date_lst})

        # Create dataframe with all days
        dates_allDays = pd.date_range(start=start_date, end=end_date, freq='1D')
        arr_allDays = np.empty(len(dates_allDays), dtype=np.int)*np.nan
        ar_dates = pd.DataFrame({'time':dates_allDays}, index=dates_allDays)

        # merge dfs together
        df = pd.merge(ar_dates, tmp, how='outer', on='time')
        df['ar'] = df['ar'].fillna(0)
        # set time as index
        df = df.set_index(df.time)
        ar_dates = df.drop(['time'], axis=1)
        
        return trackID, ar_dates
        
def get_ar_days(reanalysis, start_date, end_date, bbox=None, thresh=None, elev_thres=None):
    path_to_data = '/work/dnash/SEAK_clim_data/downloads/AR_catalog/'
    
    if reanalysis == 'era5':
        filename =  'globalARcatalog_ERA-Interim_1979-2019_v3.0.nc'
    elif reanalysis == 'merra2':
        filename = 'globalARcatalog_MERRA2_1980-2019_v3.0.nc'

    # open ds
    ds = xr.open_dataset(path_to_data + filename, chunks={'time': 1460}, engine='netcdf4')
    ds = ds.squeeze()
    # remove lev and ens coords
    ds = ds.reset_coords(names=['lev', 'ens'], drop=True)

    # select lats, lons, and dates within start_date, end_date and months
    lat1, lat2, lon1, lon2 = bbox
    ds = ds.sel(time=slice(start_date, end_date), lat=slice(lat1,lat2), lon=slice(lon1,lon2))

    # add topo mask
    mask = get_topo_mask(ds.lat, ds.lon)
    ds = ds.where(mask.bedrock >= elev_thres)
    trackID, ar_dates = preprocess_ar_bbox(ds, start_date, end_date)

    return trackID, ar_dates

def ar_climatology(dataarray, threshold):
    '''
    Returns list array of dates considered AR days based on the input subregion.
    
    Parameters
    ----------
    datarray : xarray dataarray object
        subregion where percentage of area of AR is given for each day
    threshold: number, float
        singular threshold used for the percentage of area covered by an AR
        
    Returns
    -------
    day_list : 1D array, float
        list of datetime objects that an AR covered threshold*100% of the subregion's area
    mon_ct : 1D array, float
        array of count of ARs per month in the time series
    clim_ct : 1D array, float
        array of size 12 with the total number of ARs per month
    '''
    mask = dataarray.where(dataarray >= threshold).dropna(dim='time')
    mask = mask.resample(time='1D').mean()
    day_list = mask.dropna(dim='time').time
    mon_ct = mask.resample(time='1MS').count()
    clim_ct = mask.groupby('time.month').count('time')
                           
    return day_list, mon_ct, clim_ct

def add_ar_time_series(ds, df):
    '''Add AR time series to ds; set as coordinate variables'''
    ds['ar'] = ('time', df.ar)
    ds = ds.set_coords('ar')
#     ds['location'] = ('time', df.location)
#     ds = ds.set_coords('location')
    
    return ds

def calc_seasonal_contribution(ds_list, df):
    '''
    For a list of ds, calculate the average total seasonal contribution of ARs for the given prec_vars
    
    '''
    ds_clim_lst = []
    ds_frac_lst = []
    ds_std_lst = []

    for k, ds in enumerate(ds_list):
        # Add AR time series to ds; set as coordinate variables
        ds = add_ar_time_series(ds, df)
        
        # Select AR days
        idx = (ds.ar >= 1)
        ds_ar = ds.sel(time=idx)

        # calculate seasonal totals
        ds_ssn_sum = ds.resample(time='QS-DEC', keep_attrs=True).sum()
        ds_ar_ssn_sum = ds_ar.resample(time='QS-DEC', keep_attrs=True).sum()                                 

        # calculate average of seasonal totals
        clim = ds_ssn_sum.groupby('time.season').mean(dim='time').compute()
        ds_ar_clim = ds_ar_ssn_sum.groupby('time.season').mean(dim='time').compute()
        
        # calculate standard deviation of AR Precip
        std = ds_ar_ssn_sum.groupby('time.season').std(dim='time').compute()
        
        # things to output/append to final lists
        ds_frac_lst.append((ds_ar_clim/clim)*100.)
        ds_std_lst.append(std)
        ds_clim_lst.append(clim)
    
    return ds_clim_lst, ds_frac_lst, ds_std_lst

def ar_daily_df(ssn, nk, out_path):
    
    filepath = out_path + 'AR-types_ALLDAYS.csv'
    df = pd.read_csv(filepath)

    # set up datetime index
    df = df.rename(columns={'Unnamed: 0': 'date'})
    df = df.set_index(pd.to_datetime(df.date))
    
    ## Break up columns into different AR Types
    keys = []
    for k in range(nk):
        keys.append("AR_CAT{:1d}".format(k+1,))

    values = np.zeros((len(df.index)))
    dicts = dict(zip(keys, values))

    df_cat = pd.DataFrame(dicts, index=df.index)

    for k in range(nk):
        idx = (df['AR_CAT'] == k+1)
        col = "AR_CAT{:1d}".format(k+1,)
        df_cat.loc[idx, col] = 1
        
    # get total of all AR types
    df_cat['AR_ALL'] = df_cat['AR_CAT1'] + df_cat['AR_CAT2'] + df_cat['AR_CAT3']
    df_cat['AR_CAT'] = df['AR_CAT']
    
    return df_cat

def duration_stats(x, bins):
    '''
    Count number of independent AR events and their duration in days
    '''
    duration = x
    nevents = len(x)
    duration_freq = np.array(np.unique(duration, return_counts=True))

    sizes = []
    freq = []
    for i, b in enumerate(bins):
        idx = np.where((duration_freq[0] >= b[0]) & (duration_freq[0] < b[1]))
        sizes.append((duration_freq[1][idx].sum()))
        freq.append((duration_freq[1][idx].sum()/nevents)*100)
    
    return sizes, freq

def calc_seasonal_contribution_ar_mask(ds_list):
    ## attach ar-mask to ds and use to calculate seasonal precip
    ar_filename = path_to_data + 'ar_catalog/globalARcatalog_ERA-Interim_1979-2019_v3.0.nc'
    ar_ds = xr.open_dataset(ar_filename)
    ar_ds = ar_ds.squeeze()
    ar_ds = ar_ds.sel(time=slice(start_date, end_date), lat=slice(latmin, latmax), lon=slice(lonmin, lonmax))
    ar_ds = ar_ds.shape.load()
    # resample AR catalog to daily
    ar_ds = ar_ds.resample(time="1D").mean('time')

    clim_lst = []
    frac = []
    std_lst = []

    for i, ds in enumerate(ds_lst):
        # regrid AR catalog to match ds
        ar_tmp = ar_ds.interp(lon=ds.lon.values, lat=ds.lat.values)
        print(ar_tmp)
        # add AR mask to ds prec
        mask = ds.prec.where(ar_tmp > 0)
        ds = ds.assign(ar_prec=lambda ds: mask)
        # sum total precip per season
        tmp = ds.resample(time='QS-DEC', keep_attrs=True).sum()
        # Summarize each array into one single (mean) value per season
        clim = tmp.groupby('time.season').mean(dim='time').compute()
        clim_lst.append(clim)
        # Summarize each array into one single (std) value per season
        std = tmp.groupby('time.season').std(dim='time').compute()
        std_lst.append(std)
        # calculate fraction of total seasonal precipitation
        frac.append((clim.ar_prec/clim.prec)*100)
        
    return clim_lst, frac, std_lst

def AR_rank(df):
    """
    Super simple AR rank determiner based on Ralph et al., 2019. 
    This ranks the AR based on the duration of AR conditions and maximum IVT during the AR.
    Really only works if you know the start and stop of each AR
    
    Parameters
    ----------
    df : 
        pandas dataframe with IVT
    
    Returns
    -------
    rank : int
        AR rank
    
    Notes
    -----
    - This currently only functions with hourly data.
    
    """    
    # get AR preliminary rank
    max_IVT = df.IVT.max()
    
    if (max_IVT >= 250.) & (max_IVT < 500.):
        prelim_rank = 1
    elif (max_IVT >= 500.) & (max_IVT < 750.):
        prelim_rank = 2
    elif (max_IVT >= 750.) & (max_IVT < 1000.):
        prelim_rank = 3
    elif (max_IVT >= 1000.) & (max_IVT < 1250.):
        prelim_rank = 4
    elif (max_IVT >= 1250):
        prelim_rank = 5
    else:
        prelim_rank = np.nan()
        
        
    # get maximum AR duration with IVT >=250 kg m-1 s-1
    duration = max([len(list(g)) for k, g in groupby(df['IVT']>=250) if k==True])
    
    if duration >= 48:
        rank = prelim_rank + 1
    elif duration < 24:
        rank = prelim_rank - 1
    else: 
        rank = prelim_rank
        
    return max_IVT, rank


def AR_rank_df(df):
    """
    AR rank determiner based on Ralph et al., 2019. 
    This ranks the AR based on the duration of AR conditions and maximum IVT during the AR.
    Really only works if you know the start and stop of each AR
    
    Parameters
    ----------
    duration : int
        duration of AR in hours
        
    max_IVT : float
        maximum IVT for the duration of the AR event
    
    Returns
    -------
    rank : int
        AR rank
    
    Notes
    -----
    - This currently only functions with hourly data.
    
    """    
    
    if (df['max_IVT'] >= 250.) & (df['max_IVT'] < 500.):
        prelim_rank = 1
    elif (df['max_IVT'] >= 500.) & (df['max_IVT'] < 750.):
        prelim_rank = 2
    elif (df['max_IVT'] >= 750.) & (df['max_IVT'] < 1000.):
        prelim_rank = 3
    elif (df['max_IVT'] >= 1000.) & (df['max_IVT'] < 1250.):
        prelim_rank = 4
    elif (df['max_IVT'] >= 1250):
        prelim_rank= 5
    else:
        prelim_rank = np.nan()
        
    
    if df['duration'] >= 48:
        rank = prelim_rank + 1
    elif df['duration'] < 24:
        rank = prelim_rank - 1
    else: 
        rank = prelim_rank
        
    df['rank'] = rank
        
    return df