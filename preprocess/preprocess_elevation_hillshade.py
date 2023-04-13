"""
Filename:    preprocess_elevation_hillshade.py
Author:      Deanna Nash, dnash@ucsd.edu
Description: preprocess elevation from 7.5 arc second data to include hillshade
"""

# Standard Python modules
import os, sys
from pathlib import Path
import xarray as xr

# extras
import earthpy.spatial as es
import rasterio

# Set up paths
server = "comet"
if server == "comet":
    path_to_data = '/cw3e/mead/projects/cwp140/scratch/dnash/data/'      # project data -- read only
elif server == "skyriver":
    path_to_data = '/work/dnash/data/'
path_to_out  = '../out/'       # output files (numerical results, intermediate datafiles) -- read & write
path_to_figs = '../figs/'      # figures

## open file and get lats/lons
## adapted from: https://gis.stackexchange.com/questions/388047/get-coordinates-of-all-pixels-in-a-raster-with-rasterio

filename = path_to_data + 'downloads/GMTED/50n150w_20101117_gmted_mea075.tif'
with rasterio.open(filename) as src:
    elev = src.read()
    print('Elevation tif has shape', elev.shape)
    height = elev[0].shape[0]
    width = elev[0].shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(src.transform, rows, cols)
    lons= np.array(xs)
    lats = np.array(ys)
    print('lons shape', lons.shape)
    
## crop to extent of SEAK
# hacked this: TODO write function to find nearest neighbor idx for lat lon
elev = elev[0, 4500:7500, 4000:9750]
lats = lats[4500:7500, 4000:9750]
lons = lons[4500:7500, 4000:9750]
print(elev.shape, lats.shape, lons.shape)

hillshade = es.hillshade(elev, azimuth=270, altitude=1)

# put into a xarray dataset
var_dict = {'elev': (['lat', 'lon'], elev),
            'hillshade': (['lat', 'lon'], hillshade)}
ds = xr.Dataset(var_dict,
                coords={'lat': (['lat'], lats[:, 0]),
                        'lon': (['lon'], lons[0, :])})

## add some metadata
ds['elev'].attrs["long_name"] = "Global Multi-resolution Terrain Elevation Data 2010"
ds['elev'].attrs["units"] = "meters"
ds['elev'].attrs["vertical_datum"] = "sea level"
ds['elev'].attrs["datum"] = "WGS84"
ds.attrs["title"] = "USGS GMTED2010 7.5 arc second elevation"
ds.attrs["doi"] = "10.5066/F7J38R2N"
ds['hillshade'].attrs["long_name"] = "Elevation hillshade by Earthpy, Azimuth=270 degrees"

# write to netCDF
fname = path_to_data + 'preprocessed/seak_gmted_mea075.nc'
ds.to_netcdf(path=fname, mode = 'w', format='NETCDF4')