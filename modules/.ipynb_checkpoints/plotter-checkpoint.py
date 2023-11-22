"""
Filename:    plotter.py
Author:      Deanna Nash, dnash@ucsd.edu
Description: Functions for plotting
"""

# Import Python modules

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import colorsys
from matplotlib.colors import LinearSegmentedColormap # Linear interpolation for color maps
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections import get_projection_class
import pandas as pd
import matplotlib.gridspec as gridspec
import seaborn as sns
import cmocean.cm as cmo

# Import my modules
sys.path.append('../modules') # Path to modules
from constants import ucsd_colors

def draw_basemap(ax, datacrs=ccrs.PlateCarree(), extent=None, xticks=None, yticks=None, grid=False, left_lats=True, right_lats=False, bottom_lons=True, mask_ocean=False, coastline=True):
    """
    Creates and returns a background map on which to plot data. 
    
    Map features include continents and country borders.
    Option to set lat/lon tickmarks and draw gridlines.
    
    Parameters
    ----------
    ax : 
        plot Axes on which to draw the basemap
    
    datacrs : 
        crs that the data comes in (usually ccrs.PlateCarree())
        
    extent : float
        Set map extent to [lonmin, lonmax, latmin, latmax] 
        Default: None (uses global extent)
        
    grid : bool
        Whether to draw grid lines. Default: False
        
    xticks : float
        array of xtick locations (longitude tick marks)
    
    yticks : float
        array of ytick locations (latitude tick marks)
        
    left_lats : bool
        Whether to add latitude labels on the left side. Default: True
        
    right_lats : bool
        Whether to add latitude labels on the right side. Default: False
        
    Returns
    -------
    ax :
        plot Axes with Basemap
    
    Notes
    -----
    - Grayscale colors can be set using 0 (black) to 1 (white)
    - Alpha sets transparency (0 is transparent, 1 is solid)
    
    """

    # Use map projection (CRS) of the given Axes
    mapcrs = ax.projection    
    
    ## Map Extent
    # If no extent is given, use global extent
    if extent is None:        
        ax.set_global()
        extent = [-180., 180., -90., 90.]
    # If extent is given, set map extent to lat/lon bounding box
    else:
        ax.set_extent(extent, crs=datacrs)
    
    # Add map features (continents and country borders)
    ax.add_feature(cfeature.LAND, facecolor='0.9')      
    ax.add_feature(cfeature.BORDERS, edgecolor='0.4', linewidth=0.8)
    if coastline == True:
        ax.add_feature(cfeature.COASTLINE, edgecolor='0.4', linewidth=0.8)
    if mask_ocean == True:
        ax.add_feature(cfeature.OCEAN, edgecolor='0.4', zorder=12, facecolor='white') # mask ocean
        
    ## Tickmarks/Labels
    ## Add in meridian and parallels
    if mapcrs == ccrs.NorthPolarStereo():
        gl = ax.gridlines(draw_labels=False,
                      linewidth=.5, color='black', alpha=0.5, linestyle='--')
    elif mapcrs == ccrs.SouthPolarStereo():
        gl = ax.gridlines(draw_labels=False,
                      linewidth=.5, color='black', alpha=0.5, linestyle='--')
        
    else:
        gl = ax.gridlines(crs=datacrs, draw_labels=True,
                      linewidth=.5, color='black', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.left_labels = left_lats
        gl.right_labels = right_lats
        gl.bottom_labels = bottom_lons
        gl.xlocator = mticker.FixedLocator(xticks)
        gl.ylocator = mticker.FixedLocator(yticks)
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 10, 'color': 'gray', 'fontweight': 'light'}
        gl.ylabel_style = {'size': 10, 'color': 'gray', 'fontweight': 'light'}
    
    ## Gridlines
    # Draw gridlines if requested
    if (grid == True):
        gl.xlines = True
        gl.ylines = True
    if (grid == False):
        gl.xlines = False
        gl.ylines = False
            

    # apply tick parameters    
    ax.tick_params(direction='out', 
                   labelsize=10, 
                   length=4, 
                   pad=2, 
                   color='black')
    
    return ax

def add_subregion_boxes(ax, subregion_xy, width, height, ecolor, datacrs):
    '''This function will add subregion boxes to the given axes.
    subregion_xy 
    [[ymin, xmin], [ymin, xmin]]
    '''
    for i in range(len(subregion_xy)):
        ax.add_patch(mpatches.Rectangle(xy=subregion_xy[i], width=width[i], height=height[i],
                                        fill=False,
                                        edgecolor=ecolor,
                                        linewidth=1.0,
                                        transform=datacrs,
                                        zorder=100))
        
    return ax

        
def plot_maxmin_points(lon, lat, data, extrema, nsize, symbol, color='k',
                       plotValue=True, transform=None):
    """
    This function will find and plot relative maximum and minimum for a 2D grid. The function
    can be used to plot an H for maximum values (e.g., High pressure) and an L for minimum
    values (e.g., low pressue). It is best to used filetered data to obtain  a synoptic scale
    max/min value. The symbol text can be set to a string value and optionally the color of the
    symbol and any plotted value can be set with the parameter color
    lon = plotting longitude values (2D)
    lat = plotting latitude values (2D)
    data = 2D data that you wish to plot the max/min symbol placement
    extrema = Either a value of max for Maximum Values or min for Minimum Values
    nsize = Size of the grid box to filter the max and min values to plot a reasonable number
    symbol = String to be placed at location of max/min value
    color = String matplotlib colorname to plot the symbol (and numerica value, if plotted)
    plot_value = Boolean (True/False) of whether to plot the numeric value of max/min point
    The max/min symbol will be plotted on the current axes within the bounding frame
    (e.g., clip_on=True) 
    
    ^^^ Notes from MetPy. Function adapted from MetPy.
    """
    from scipy.ndimage.filters import maximum_filter, minimum_filter

    if (extrema == 'max'):
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif (extrema == 'min'):
        data_ext = minimum_filter(data, nsize, mode='nearest')
    else:
        raise ValueError('Value for hilo must be either max or min')

    mxy, mxx = np.where(data_ext == data)

    for i in range(len(mxy)):
        ax.text(lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]], symbol, color=color, size=13,
                clip_on=True, horizontalalignment='center', verticalalignment='center',
                fontweight='extra bold',
                transform=transform)
        ax.text(lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]],
                '\n \n' + str(np.int(data[mxy[i], mxx[i]])),
                color=color, size=8, clip_on=True, fontweight='bold',
                horizontalalignment='center', verticalalignment='center', 
                transform=transform, zorder=10)
        
    return ax

def loadCPT(path):
    """A function that loads a .cpt file and converts it into a colormap for the colorbar.
    
    This code was adapted from the GEONETClass Tutorial written by Diego Souza, retrieved 18 July 2019. 
    https://geonetcast.wordpress.com/2017/06/02/geonetclass-manipulating-goes-16-data-with-python-part-v/
    
    Parameters
    ----------
    path : 
        Path to the .cpt file
        
    Returns
    -------
    cpt :
        A colormap that can be used for the cmap argument in matplotlib type plot.
    """
    
    try:
        f = open(path)
    except:
        print ("File ", path, "not found")
        return None
 
    lines = f.readlines()
 
    f.close()
 
    x = np.array([])
    r = np.array([])
    g = np.array([])
    b = np.array([])
 
    colorModel = 'RGB'
 
    for l in lines:
        ls = l.split()
        if l[0] == '#':
            if ls[-1] == 'HSV':
                colorModel = 'HSV'
                continue
            else:
                continue
        if ls[0] == 'B' or ls[0] == 'F' or ls[0] == 'N':
            pass
        else:
            x=np.append(x,float(ls[0]))
            r=np.append(r,float(ls[1]))
            g=np.append(g,float(ls[2]))
            b=np.append(b,float(ls[3]))
            xtemp = float(ls[4])
            rtemp = float(ls[5])
            gtemp = float(ls[6])
            btemp = float(ls[7])
 
        x=np.append(x,xtemp)
        r=np.append(r,rtemp)
        g=np.append(g,gtemp)
        b=np.append(b,btemp)
 
    if colorModel == 'HSV':
        for i in range(r.shape[0]):
            rr, gg, bb = colorsys.hsv_to_rgb(r[i]/360.,g[i],b[i])
        r[i] = rr ; g[i] = gg ; b[i] = bb
 
    if colorModel == 'RGB':
        r = r/255.0
        g = g/255.0
        b = b/255.0
 
    xNorm = (x - x[0])/(x[-1] - x[0])
 
    red   = []
    blue  = []
    green = []
 
    for i in range(len(x)):
        red.append([xNorm[i],r[i],r[i]])
        green.append([xNorm[i],g[i],g[i]])
        blue.append([xNorm[i],b[i],b[i]])
 
    colorDict = {'red': red, 'green': green, 'blue': blue}
    # Makes a linear interpolation
    cpt = LinearSegmentedColormap('cpt', colorDict)
    
    return cpt


def make_cmap(colors, position=None, bit=False):
    '''
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    import matplotlib as mpl
    import numpy as np
    bit_rgb = np.linspace(0,1,256)
    if position == None:
        position = np.linspace(0,1,len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red':[], 'green':[], 'blue':[]}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap',cdict,256)
    return cmap

def nice_intervals(data, nlevs):
    '''
    Purpose::
        Calculates nice intervals between each color level for colorbars
        and contour plots. The target minimum and maximum color levels are
        calculated by taking the minimum and maximum of the distribution
        after cutting off the tails to remove outliers.
    Input::
        data - an array of data to be plotted
        nlevs - an int giving the target number of intervals
    Output::
        clevs - A list of floats for the resultant colorbar levels
    '''
    # Find the min and max levels by cutting off the tails of the distribution
    # This mitigates the influence of outliers
    data = data.ravel()
    mn = mstats.scoreatpercentile(data, 5)
    mx = mstats.scoreatpercentile(data, 95)
    # if min less than 0 and or max more than 0 put 0 in center of color bar
    if mn < 0 and mx > 0:
        level = max(abs(mn), abs(mx))
        mnlvl = -1 * level
        mxlvl = level
    # if min is larger than 0 then have color bar between min and max
    else:
        mnlvl = mn
        mxlvl = mx

    # hack to make generated intervals from mpl the same for all versions
    autolimit_mode = mpl.rcParams.get('axes.autolimit_mode')
    if autolimit_mode:
        mpl.rc('axes', autolimit_mode='round_numbers')

    locator = mpl.ticker.MaxNLocator(nlevs)
    clevs = locator.tick_values(mnlvl, mxlvl)
    if autolimit_mode:
        mpl.rc('axes', autolimit_mode=autolimit_mode)

    # Make sure the bounds of clevs are reasonable since sometimes
    # MaxNLocator gives values outside the domain of the input data
    clevs = clevs[(clevs >= mnlvl) & (clevs <= mxlvl)]
    return clevs


def _drawmap(fig, lons, lats, VO1, VO2, VO3, cmap1, cmap2, cmap3, clevs1, clevs2, clevs3, title, ext,
             datacrs, mapcrs, ndeg=10.):
    '''Draw contour map for create_animation.'''
    # Clear current axis to overplot next time step
    ax = fig.gca()
    ax.clear()
    # Add subplot, title, and set extent
    ax = fig.add_subplot(1,1,1, projection=mapcrs)
    xticks = np.arange(ext[0], ext[1]+ndeg, ndeg)
    yticks = np.arange(ext[2], ext[3]+ndeg, ndeg)
    
#     ax = draw_basemap(ax, datacrs=datacrs, 
#                  extent=ext, xticks=None, yticks=None, 
#                  grid=False, left_lats=True, right_lats=False, 
#                  bottom_lons=True, mask_ocean=False)
    
    ax.set_extent(ext, crs=mapcrs)
    
    # Add Border Features
    coast = ax.coastlines(linewidths=1.0)
    ax.add_feature(cfeature.BORDERS)
    
    # Add grid lines
    gl = ax.gridlines(crs=datacrs, draw_labels=True,
                      linewidth=.5, color='black', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.left_labels = True
    gl.right_labels = False
    gl.bottom_labels = True
    gl.xlocator = mticker.FixedLocator(xticks)
    gl.ylocator = mticker.FixedLocator(yticks)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'gray'}
    gl.ylabel_style = {'size': 10, 'color': 'gray'}
    
    # Add contour plot (line)
    cs = ax.contour(lons, lats, VO2, transform=datacrs,
                    levels=clevs2, colors='grey', linewidths=0.7, linestyles='solid', zorder=10)
    kw_clabels = {'fontsize': 8, 'inline': True, 'inline_spacing': 10, 'fmt': '%i',
                  'rightside_up': True, 'use_clabeltext': True}
    plt.clabel(cs, **kw_clabels)
    
    # Add contour plot (shaded precip)
    cf2 = ax.contourf(lons, lats, VO3, transform=datacrs, cmap=cmap3, levels=clevs3, zorder=5, extend='max', alpha=0.5)
    
    # Add contour plot (shaded)
    cf = ax.contourf(lons, lats, VO1, transform=datacrs, cmap=cmap1, levels=clevs1, zorder=0, extend='max', alpha=0.8)
    
    ax.set_title(title, fontsize=14)
    
    # Add a color bar
    cbar = fig.colorbar(cf, orientation='vertical', cmap=cmap1, shrink=0.99)
#     cbar.set_label(units, fontsize=12)
    
#     # add second colorbar
#     rect_loc = [1.02, 0.08, 0.03, 0.87]  # define position 
#     cax2 = fig.add_axes(rect_loc)       # left | bottom | width | height
#     cbar2  = plt.colorbar(second_contour, cax=cax2)
    
    return cf, cf2, ax

def _myanimate(i, fig, DS, var1, var2, var3, lats, lons, cmap1, cmap2, cmap3, clevs1, clevs2, clevs3, ext, datacrs, mapcrs):
    '''Loop through time steps for create_animation.'''
    # Clear current axis to overplot next time step
    ax = fig.gca()
    ax.clear()
    # Loop through time steps in ds
    VO1 = DS[var1].values[i]
    VO2 = DS[var2].values[i]
    VO3 = DS[var3].values[i]
    # Set title based on long name and current time step
    ts = pd.to_datetime(str(DS.time.values[i])).strftime("%Y-%m-%d %H:%M")
    long_name = DS[var1].long_name
    title = '{0} at {1}'.format(long_name, ts)
    # Add next contour map
    new_contour, new_contour2, new_ax = _drawmap(fig, lons, lats, VO1, VO2, VO3, cmap1, cmap2, cmap3, clevs1, clevs2, clevs3, title, ext, datacrs, mapcrs) 
    
    return new_ax

def create_animation(DS, var1, var2, var3, clevs1, clevs2, clevs3, cmap1, cmap2, cmap3, 
                     ext=[-180.0, 180.0, -90., 90.], datacrs=ccrs.PlateCarree(), mapcrs=ccrs.PlateCarree()):
    '''Create an mp4 animation using an xarray dataset with lat, lon, and time dimensions.
    
        Parameters
        ----------
        DS: xarray dataset object
        
        var: string
            Variable name to plot
        clevs: int
            Contour levels to plot
        cmap: string
            Colormap for plotting
            
        Returns
        -------
        filename, mp4 file of animation
        
        '''
    
    # Get information from ds
    lats = DS.lat
    lons = DS.lon
    long_name = DS[var1].long_name
    units = DS[var1].units
    t0 = pd.to_datetime(str(DS.time.values[0])).strftime("%Y-%m-%d %H:%M")
    title = '{0} at {1}'.format(long_name, t0)
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title=title,
                    comment='')
    writer = FFMpegWriter(fps=20, metadata=metadata)
    
    # Create a new figure window
    fig = plt.figure(figsize=[8,4])
#     # Draw first timestep
    first_contour, second_contour, first_ax = _drawmap(fig, lons, lats, DS[var1].values[0], DS[var2].values[0], DS[var3].values[0], cmap1, cmap2, cmap3, clevs1, clevs2, clevs3, title, ext, datacrs, mapcrs)

    # Loop through animation
    ani = animation.FuncAnimation(fig, _myanimate, frames=np.arange(len(DS[var1])),
                                  fargs=(fig, DS, var1, var2, var3, lats, lons, cmap1, 
                                         cmap2, cmap3, clevs1, clevs2, clevs3, ext, datacrs, mapcrs), interval=50)
    filename = long_name + ".mp4"
    ani.save(long_name + ".mp4")
    
    # save animation at 30 frames per second 
    ani.save(long_name + ".gif", writer='imagemagick', fps=10)
    
    return filename

##'''from https://stackoverflow.com/questions/35042255/how-to-plot-multiple-seaborn-jointplot-in-subplot'''
class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())
        
def range_labels(bins):
    '''
    a function that gives nice labels for precipitation ranges
    '''
    labels = []
    for left, right in zip(bins[:-1], bins[1:]):
        if left == bins[0]:
            labels.append('<{}'.format(right))
        elif right == bins[-1]:
            labels.append('>{}'.format(left))
        else:
            labels.append('{}-{}'.format(str(left[:-1]), right))

    return list(labels)

def assign_percentiles(df, prec_bins, ivt_bins, perc_labels):
    '''
    ### Determine the percentile group for precipitation and ivt

    - assign a precipitation bin for each row with pandas.cut
    - assign a ivt bin for each row with pandas.cut
    '''
    
    df = (df.assign(prec_bins=lambda df: pd.cut(df['prec'], bins=prec_bins, labels=perc_labels, right=True))
            .assign(ivt_bins=lambda df: pd.cut(df['IVT'], bins=ivt_bins, labels=perc_labels, right=True))
         )
    
    df = df.rename(columns={"prec_bins": "Precipitation Percentiles", "ivt_bins": "IVT Percentiles"})
        
    return df

def community_heatmap_values(df, perc_lbl):
    denom = np.empty((6, 6), float)
    num = np.empty((6, 6), float)
    extreme_prec_AR = np.empty((6), float)
    extreme_ivt_AR = np.empty((6), float)
    for i, perc_i in enumerate(perc_lbl):
        ## get total number of precipitation days within each percentile bin
        idx = (df['Precipitation Percentiles'] == perc_i)
        denom_val = len(df.loc[idx])
        

        ## get total number of IVT days within the same percentile bin
        
        for j, perc_j in enumerate(perc_lbl):
            idx = (df['Precipitation Percentiles'] == perc_i) & (df['IVT Percentiles'] == perc_j)
            num_val = len(df.loc[idx])
            num[j, i] = num_val # put numerator in array
            
            # put denominator in array
            denom[j, i] = denom_val # put denominator in array
            
            
        ## get fraction of AR days that are also 95th percentile precip
        idx = (df['Precipitation Percentiles'] == perc_i) & (df['AR'] == 1)
        AR_num = len(df.loc[idx])
        extreme_prec_AR[i] = (AR_num/denom_val)*100

        ## get fraction of AR days that are also 95th percentile IVT
        idx = (df['IVT Percentiles'] == perc_i)
        denom_val = len(df.loc[idx])
        idx = (df['IVT Percentiles'] == perc_i) & (df['AR'] == 1)
        AR_num = len(df.loc[idx])
        extreme_ivt_AR[i] = (AR_num/denom_val)*100
        
    heatmap_vals = (num/denom)*100
    
    return heatmap_vals, extreme_prec_AR, extreme_ivt_AR


def create_heatmap_plot(heatmap_vals, extreme_prec_AR, extreme_ivt_AR, ax, ax_histx, ax_histy, tck_lblx, tck_lbly, bar_tck):

    # no labels, remove spines
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histx.spines['top'].set_visible(False)
    ax_histx.spines['right'].set_visible(False)
    ax_histx.set_ylim(0, 99)

    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.spines['top'].set_visible(False)
    ax_histy.spines['right'].set_visible(False)
    ax_histy.set_ylim(0, 99)

    ax_histx.bar(x = bar_tck, height=extreme_prec_AR, align='edge', color='#DAE6E6')
    ax_histy.barh(range(len(bar_tck)), np.flip(extreme_ivt_AR), align='edge', color='#DAE6E6')

    ## add heatmap
    g = sns.heatmap(np.flipud(heatmap_vals), cmap=cmo.dense, annot=True, linewidth=.5, xticklabels=tck_lblx, yticklabels=tck_lbly, ax=ax, cbar=False)
    # apply tick parameters    
    ax.tick_params(direction='out', 
                   labelsize=8, 
                   length=4, 
                   pad=2, 
                   color='black',
                   labelrotation=0.0)

## Define a function to convert centered angles to left-edge radians
def _convert_dir(directions, N=None):
    if N is None:
        N = directions.shape[0]
    barDir = directions * np.pi/180. - np.pi/N
    barWidth = 2 * np.pi / N
    return barDir, barWidth

## define wind rose function
def wind_rose(ax, rosedata, wind_dirs, legend_req, palette=None):
    if palette is None:
        palette = sns.color_palette('inferno', n_colors=rosedata.shape[1])
    else:
        palette = sns.color_palette(palette, n_colors=rosedata.shape[1])

    bar_dir, bar_width = _convert_dir(wind_dirs)

    
    ax.set_theta_direction('clockwise')
    ax.set_theta_zero_location('N')

    for n, (c1, c2) in enumerate(zip(rosedata.columns[:-1], rosedata.columns[1:])):
        if n == 0:
            # first column only
            ax.bar(bar_dir, rosedata[c1].values, 
                   width=bar_width,
                   color=palette[0],
                   edgecolor='none',
                   label=c1,
                   linewidth=0,
                   alpha=0.8)

        # all other columns
        ax.bar(bar_dir, rosedata[c2].values, 
               width=bar_width, 
               bottom=rosedata.cumsum(axis=1)[c1].values,
               color=palette[n+1],
               edgecolor='none',
               label=c2,
               linewidth=0,
               alpha=0.8)
        
    # xticks = ax.set_xticks(np.pi/180. * np.linspace(180,  -180, 8, endpoint=False))
    # xtl = ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']) # turns on cardinal direction labels
    xtl = ax.set_xticklabels([]) # removes the cardinal direction tick labels

    if legend_req == True:
        leg = ax.legend(loc=(0.75, 0.1), ncol=1, fontsize=12, title='Percentile')
    
    return ax

def calc_rose(df, prec_bins, prec_labels, dir_bins, dir_labels):
    '''
    ### Determine the relative percentage of observation in each speed and direction bin
    # Adapted from: https://gist.github.com/phobson/41b41bdd157a2bcf6e14 as an example
    Here's how we do it:

    - assign a precipitation bin for each row with pandas.cut
    - assign a direction bin for each row (again, pandas.cut)
    - unify the 360° and 0° bins under the 0° label
    - group the data simultaneously on both precipitation and direction bins
    - compute the size of each group
    - unstack (pivot) the speed bins into columns
    - fill missing values with 0
    - sort the columns -- they are a catgerical index, so "calm" will be first (this is awesome!)
    - convert all of the counts to percentages of the total number of observations
    '''
    total_count = df.shape[0]
    calm_thres = prec_bins[1]
    idx = (df.prec < calm_thres)
    
    calm_count = len(df.loc[idx])
    print('Of {} total observations, {} have less than {} mm of precipitation.'.format(total_count, calm_count, calm_thres))

    df = (df.assign(prec_bins=lambda df: pd.cut(df['prec'], bins=prec_bins, labels=prec_labels, right=False)) 
          .assign(ivtdir_bins=lambda df: pd.cut(df['ivtdir'], bins=dir_bins, labels=dir_labels, right=False))
          .replace({'ivtdir_bins': {360: 0}})
          .groupby(by=['prec_bins', 'ivtdir_bins'])
          .size()
          .unstack(level='prec_bins')
          .fillna(0)
          # .assign(calm=lambda df: calm_count / df.shape[0])
          .sort_index(axis=1)
          .applymap(lambda x: x / total_count * 100)
         )
        
    return df


def build_rose(mapx, mapy, ax, width, rose, directions, legend_req, rad_ticks, transform):
    '''
    adapted from: https://stackoverflow.com/questions/55854988/subplots-onto-a-basemap/55890475#55890475
    and: https://stackoverflow.com/questions/46262749/plotting-scatter-of-several-polar-plots/46263911#46263911
    Function to create inset axes and plot wind rose on it

    '''
    lbldict = {'fontsize': 7,
               'fontweight': 'normal',
               'verticalalignment': 'bottom',
               'horizontalalignment': 'center'}
    
    ax_h = inset_axes(ax, width=width, height=width, loc=10,
                      # projection='polar',
                      bbox_to_anchor=(mapx, mapy), 
                      bbox_transform=transform, 
                      borderpad=0, 
                      axes_kwargs={'alpha': 0.35, 'visible': True},
                      axes_class=get_projection_class("polar"))
    
    wind_rose(ax_h, rose, directions, legend_req, palette=[ucsd_colors['yellow'], ucsd_colors['blue'], ucsd_colors['aqua']])
    ax_h.set_rticks(rad_ticks, labelsize=5)  # Less radial ticks
    ax_h.set_rlabel_position(90.0)  # Move radial labels away from plotted line
    tmp = list(map(str, rad_ticks[:-1]))
    ytcklbls = list(map("{}%".format, tmp)) + [''] # sets radial tick labels
    ax_h.set_yticklabels(ytcklbls, fontdict=lbldict) 
    ax_h.yaxis.grid(linewidth=0.5, linestyle='--')
    # ax_h.axis('off') # this turns off the axis grid completely
    ax_h.patch.set_alpha(0.01) # this sets the face color of the axis grid to transparent
    ax_h.spines['polar'].set_visible(False) # this turns the outer edge of the polar plot off
    
    return ax_h