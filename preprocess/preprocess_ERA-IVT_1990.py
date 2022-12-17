"""
Filename:    preprocess_ERA5-IVT.py
Author:      Deanna Nash, dlnash@ucsb.edu
Description: preprocess hourly IVT from ERA5 data and save as yearly csv files for each community
"""

# Standard Python modules
import os, sys
import numpy as np
import pandas as pd
import xarray as xr
from functools import reduce

# Set up paths

path_to_data = '/home/dnash/SEAK_clim_data/'      # project data -- read only
path_to_work = '/work/dnash/SEAK_clim_data/preprocessed/ERA5-IVT/'
path_to_out  = '../out/'       # output files (numerical results, intermediate datafiles) -- read & write
path_to_figs = '../figs/'      # figures

### to pull time series from data
## choose which option
# (a) select the grid cell closest to each of the communities
# (b) select the 9 closest grid cells to each of the communities - take maximum value
# (c) select the 25 closest grid cells to each of the communities- take maximum value
option = 'a'

### choose which temporal resolution for the precipitation data (hourly or daily)
temporal_res = 'hourly'

### set the years you are processing
start_yr = 1990
end_yr = 2000

### TODO: make a yaml dict
ext1 = [-141., -130., 54., 61.] # extent of SEAK

lonmin, lonmax, latmin, latmax = ext1
xs = [-135.4519, -135.3277, -135.8894, -139.671, -133.1358, -132.4009]
ys = [58.1122, 59.4538, 59.3988, 59.5121, 55.4769, 55.5400]
lbl1 = ['Hoonah', 'Skagway', 'Klukwan', 'Yakutat', 'Craig', 'Kasaan']


def generate_IVT_tseries(era, xs, ys, community_lst, option, year):
    
    for i, (slon, slat) in enumerate(zip(xs, ys)):
        community = community_lst[i]
        print('Processing ', community)
        
        if option == 'a':
            ## select nearest grid cell to station
            ds = era.sel(lat=slat, lon=slon, method="nearest").load()
            df = ds.to_dataframe()
        ##TODO fix option b    
        elif option == 'b':
            scale = 1.5 ## "nearest neighbor" grid cells
            # get the dates for the current year to save in df later
            dates = era.time.values
            
            # get the latitude/longitude resolution
            diff_lat = era.lat.values[2] - era.lat.values[1]
            diff_lon = era.lon.values[2] - era.lon.values[1]
            
            # select the nearest neighbor grid cells
            ds = era.sel(lat=slice(slat-diff_lat*scale, slat+diff_lat*scale), lon=slice(slon-diff_lon*scale, slon+diff_lon*scale))
            
            ### localized IVT maxima at each time step
            event_max = ds.where(ds.IVT==ds.IVT.max(['lat', 'lon']), drop=True).squeeze().load()
            ## pull IVT and IVTDIR where ivt is max
            uvec = event_max.uIVT.values
            vvec = event_max.vIVT.values
            # ## if metpy installed
            # uvec = units.Quantity(uvec, "m/s")
            # vvec = units.Quantity(vvec, "m/s")
            # ivtdir = mpcalc.wind_direction(uvec, vvec)
            # ivtdir_vals = ivtdir.item()

            ivt_vals = event_max.IVT.values.tolist()
            iwv_vals = event_max.IWV.values.tolist()

            ## make a pandas df and put the info in 
            d ={"IVT": ivt_vals,
                "uIVT": uvec,
                "vIVT": vvec,
                "IWV": iwv_vals}
            df = pd.DataFrame(data=d)
            
        print('Writing ', community, year, 'to csv')
        ## save to csv file
        df.to_csv(path_to_work + '{0}_IVT_{1}.csv'.format(community, year))
        
## This is a lot of data so pulling from each year and processing one year at a time
def preprocess(ds):
    '''keep only selected lats and lons'''
    return ds.sel(lat=slice(latmin, latmax), lon=slice(lonmin, lonmax))

ds_lst = []
for i, yr in enumerate(range(start_yr, end_yr)):
    print('Reading ', yr)
    filenames = '/data/downloaded/Reanalysis/ERA5/IVT/{0}/ERA5_IVT_*.nc'.format(yr) 
    era = xr.open_mfdataset(filenames, combine='by_coords', preprocess=preprocess)
    if temporal_res == 'hourly':
        era = era
    elif temporal_res == 'daily':
        era = era.resample(time="1D").mean('time')
    
    # calculate time series for each community based on grid option
    generate_IVT_tseries(era, xs, ys, lbl1, option, yr)