# Standard Python modules
import os, sys
import numpy as np
import pandas as pd
import xarray as xr
from functools import reduce
import xoak
import metpy.calc as mpcalc
from metpy.units import units

# Set up paths

path_to_data = '/cw3e/mead/projects/cwp140/scratch/dnash/data/'      # project data -- read only
path_to_work = '/cw3e/mead/projects/cwp140/scratch/dnash/data/preprocessed/'
path_to_out  = '../out/'       # output files (numerical results, intermediate datafiles) -- read & write
path_to_figs = '../figs/'      # figures

### to pull time series from data
## choose which option
# (a) select the grid cell closest to each of the communities
# (b) select the 9 closest grid cells to each of the communities - take maximum value
# (c) select the 25 closest grid cells to each of the communities- take maximum value
option = 'a'

### choose which temporal resolution for the precipitation data (hourly or daily)
temporal_res = 'daily'

### variable name (PCPT, T2, UV)
varname = 'UV'

### TODO: make a yaml dict
ext1 = [-141., -130., 54., 61.] # extent of SEAK

lonmin, lonmax, latmin, latmax = ext1
xs = [-135.4519, -135.3277, -135.8894, -139.671, -133.1358, -132.4009]
ys = [58.1122, 59.4538, 59.3988, 59.5121, 55.4769, 55.5400]
lbl1 = ['Hoonah', 'Skagway', 'Klukwan', 'Yakutat', 'Craig', 'Kasaan']
        
fname_pattern = path_to_work + 'SEAK-WRF-{0}/WRFDS_{0}_*.nc'.format(varname)
wrf = xr.open_mfdataset(fname_pattern, combine='by_coords')
        
if varname == 'UV':
    wrf = wrf.sel(lev='1000')
      
if (temporal_res == 'daily') & (varname == 'PCPT'):
    wrf = wrf.resample(time="1D").sum('time')

elif (temporal_res == 'daily') & (varname != 'PCPT'):
    wrf = wrf.resample(time="1D").mean('time')

elif (temporal_res == 'hourly'):
    wrf = wrf
        
df_lst2 = []
row_lbl2 = []
for i, (slon, slat) in enumerate(zip(xs, ys)):
    
    if option == 'a':
        ## select nearest grid cell to station
        points = xr.Dataset({"lat": slat, "lon": slon})
        wrf.xoak.set_index(["lat", "lon"], "sklearn_geo_balltree")
        ds = wrf.xoak.sel(lat=points.lat, lon=points.lon)
        
        if varname == 'UV':
            ## calculate UV direction
            uvec = units.Quantity(ds['U'].values, "m/s")
            vvec = units.Quantity(ds['V'].values, "m/s")
            uvdir = mpcalc.wind_direction(uvec, vvec)
            ds = ds.assign(UV=lambda ds: uvdir)
        
    df = ds[varname].to_dataframe()
    df['time'] = ds.time.values
    df = df.rename(columns={varname: lbl1[i]}) # rename from varname to the name of the community
    df_lst2.append(df)
        
## merge all dfs to one
df_merged = reduce(lambda x, y: pd.merge(x, y, on = 'time'), df_lst2)
## hack for weird behavior for daily df option a
if (option == 'a') & (varname == 'UV'):
    df_merged = df_merged.drop(['lat_x', 'lat_y', 'lon_x', 'lon_y', 'lev_y', 'lev_x'], axis=1)
elif (option == 'a'):
    df_merged = df_merged.drop(['lat_x', 'lat_y', 'lon_x', 'lon_y'], axis=1)
    
## save to csv file
df_merged.to_csv(path_to_out + 'SEAK_{0}_{1}_{2}.csv'.format(varname, option, temporal_res))