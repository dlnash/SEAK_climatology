"""
Filename:    precip_composte_AMS.py
Author:      Deanna Nash, dnash@ucsd.edu
Description: Run precipitation composites for AMS - since comet is down
"""

# Standard Python modules
import os, sys
import numpy as np
import pandas as pd
import xarray as xr

# plot styles/formatting
import seaborn as sns
import cmocean.cm as cmo
import cmocean

# matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# cartopy
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
import cartopy.feature as cfeature

# extras
import metpy.calc as mpcalc
from metpy.units import units

# import personal modules
# Path to modules
sys.path.append('../modules')
import nclcmaps as nclc
from plotter import draw_basemap

pd.options.display.float_format = "{:,.2f}".format # makes it so pandas tables display only first two decimals

# Set up paths

path_to_data = '/cw3e/mead/projects/cwp140/scratch/dnash/data/'      # project data -- read only
path_to_work = '/cw3e/mead/projects/cwp140/scratch/dnash/data/preprocessed/SEAK-WRF-precip/'
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
    
## load WRF data
fname_pattern = path_to_work + 'WRFDS_PCPT_*.nc'
wrf = xr.open_mfdataset(fname_pattern, combine='by_coords')
if temporal_res == 'hourly':
        wrf = wrf
elif temporal_res == 'daily':
    wrf = wrf.resample(time="1D").sum('time') # resample WRF data to be mm per day

## make a dataset for each community subset to its AR dates
ds_lst = []
for i, ar_dates in enumerate(ardate_lst):
    print('Processing {0}'.format(community_lst[i]))
    tmp = wrf.sel(time=ar_dates)
    tmp = tmp.mean('time')
    ds_lst.append(tmp.load())

## Plot figures
# Set up projection
mapcrs = ccrs.Mercator()
datacrs = ccrs.PlateCarree()

# Set tick/grid locations
lats = wrf.lat.values
lons = wrf.lon.values
dx = np.arange(lons.min().round(),lons.max().round()+1,1)
dy = np.arange(lats.min().round(),lats.max().round()+1,1)

ext1 = [-141., -130., 54., 61.] # extent of SEAK

for i, ds in enumerate(ds_lst):
    community = community_lst[i]
    # Create figure
    fig = plt.figure(figsize=(8, 12))
    fig.dpi = 300
    fname = path_to_figs + 'extreme-AR_prec_composite_{0}'.format(community)
    fmt = 'png'

    nrows = 1
    ncols = 1

    # Set up Axes Grid
    axes_class = (GeoAxes,dict(projection=mapcrs))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(nrows, ncols), axes_pad = 0.45,
                    cbar_location='bottom', cbar_mode='single',
                    cbar_pad=0.05, cbar_size='3%',label_mode='')


    for k, ax in enumerate(axgr):
        ax = draw_basemap(ax, extent=ext1, xticks=dx, yticks=dy,left_lats=True, right_lats=False, mask_ocean=False)

        # Contour Filled
        prec = ds.prec.values
        print(np.nanmax(prec))
        if temporal_res == 'hourly':
            clevs = np.arange(0.1, 2.2, 0.1)
            clabel = 'precipitation (mm hour$^{-1}$)'
        elif temporal_res == 'daily':
            clevs = np.arange(0.1, 165, 15)
            clabel = 'precipitation (mm day$^{-1}$)'
        cf = ax.contourf(lons, lats, prec, transform=datacrs,
                         levels=clevs, cmap=nclc.cmap('WhiteBlueGreenYellowRed'), alpha=0.9, extend='max')
        
        ax.set_title(community, loc='left')


    # Colorbar (single)
    cb = fig.colorbar(cf, axgr.cbar_axes[0], orientation='horizontal', drawedges=False)
    cb.set_label(clabel, fontsize=11)
    cb.ax.tick_params(labelsize=12)

    fig.savefig('%s.%s' %(fname, fmt), bbox_inches='tight', dpi=fig.dpi)