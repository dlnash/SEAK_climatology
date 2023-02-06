"""
Filename:    preprocess_dataframes.py
Author:      Deanna Nash, dnash@ucsd.edu
Description: Functions for preprocessing dataframes
"""
import os, sys
import numpy as np
import pandas as pd
import metpy.calc as mpcalc
from metpy.units import units

def combine_ivt_ar_prec_df(option, temporal_res, community_lst):
    '''
    Returns list of dataframes (based on community_lst) encoded for ARs, Impacts, IVT, and Precipitation
    
    Parameters
    ----------
    option : string
        'a', 'b', or 'c' option for how to handle precipitation preprocessing
        
    temporal_res : string
        'daily' or 'hourly' temporal resolution
        
    community_lst : list
        list of strings of community names
        
    Returns
    -------
    list : list of pandas dataframes
        df that indicates whether each time step is an AR, impact, precipitation and IVT
    '''
    path_to_out = '../out/'
    
    # open precipitation data
    fname = path_to_out + 'SEAK_precip_max_{0}_{1}.csv'.format(option, temporal_res)
    prec_df = pd.read_csv(fname)
    prec_df = prec_df.replace(0, np.NaN) # replace any instance of zero with nan to ignore dates with no precipitation
    
    df_lst = []
    for i, community in enumerate(community_lst):
        fname = path_to_out + 'IVT_ERA5_{0}.csv'.format(community)
        df = pd.read_csv(fname) # open IVT data

        ## calculate IVT direction
        uvec = units.Quantity(df['uIVT'].values, "m/s")
        vvec = units.Quantity(df['vIVT'].values, "m/s")
        ivtdir = mpcalc.wind_direction(uvec, vvec)
        df['ivtdir'] = ivtdir
        df = df.drop(['Unnamed: 0'], axis=1) # drop unnecessary vars

        if temporal_res == 'hourly':
            df = df
        elif temporal_res == 'daily':
            df = df.set_index(pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')) # reset the index as "date" 
            df = df.resample('1D').max()
            df = df.drop(['time'], axis=1)
            df = df.reset_index() # remove the index

        ## append AR data
        fname = path_to_out + 'SEAK_ardates_{0}.csv'.format(temporal_res)
        ar_df = pd.read_csv(fname) # read in AR dates
        df['AR'] = ar_df.AR # append AR dates to current df

        ## append impact data
        fname = path_to_out + 'SEAK_impactdates_{0}.csv'.format(temporal_res)
        impact_df = pd.read_csv(fname) # read in impact dates        
        df['impact'] = impact_df.IMPACT # append impact dates to current df

        ## append community precipitation data
        df['prec'] = prec_df[community]
        # reset the index as "time"
        df = df.set_index(pd.to_datetime(df['time']))
        
        df_lst.append(df)        
        
    return df_lst
    
def precipitation_annual_clim(df_lst, community_lst):
    '''
    Returns list of dataframes (based on community_lst) that has the annual climatology for precipitation
    
    Parameters
    ----------
    list : list of pandas dataframes
        list of daily or hourly pandas dataframes with precipitation data
        
    community_lst : list
        list of strings of community names

    Returns
    -------
    df : pandas dataframe
        df that has annual climatology of precipitation for each community

    '''
    prec_clim_lst = []    
    for i, df in enumerate(df_lst):
        community = community_lst[i]
        # reset the index as "time"
        df = df.set_index(pd.to_datetime(df['time']))

        # create day of year column
        df['day_of_year'] = df.index.dayofyear

        prec_clim = df.groupby(['day_of_year'])['prec'].mean()
        prec_clim = prec_clim.rename('prec_{0}'.format(community))
        prec_clim_lst.append(prec_clim)
        
    prec_clim = pd.concat(prec_clim_lst, axis=1)
        
    return prec_clim
        
def calculate_ivt_prec_percentiles(df_lst, community_lst):
    '''
    Returns two lists of dataframes (based on community_lst) that has the percentile value of ivt or precipitation 
    
    Parameters
    ----------
    list : list of pandas dataframes
        list of daily or hourly pandas dataframes with precipitation and ivt data
        
    community_lst : list
        list of strings of community names

    Returns
    -------
    df : pandas dataframe
        df that has percentile value of precipitation for each community
        
    df : pandas dataframe
        df that has percentile value of ivt for each community
    '''
    percentile_lst_ivt = []
    percentile_lst_prec = []
    
    for i, df in enumerate(df_lst):
        community = community_lst[i]
        # calculate percentile rank for IVT and prec
        df['prec_{0}'.format(community)] = df['prec'].rank(pct=True)
        percentile_lst_prec.append(df['prec_{0}'.format(community)])
        df['ivt_{0}'.format(community)] = df['IVT'].rank(pct=True)
        percentile_lst_ivt.append(df['ivt_{0}'.format(community)])
        
    ## make new_df that is just the percentile rank of each df
    ivt_percentile = pd.concat(percentile_lst_ivt, axis=1)
    prec_percentile = pd.concat(percentile_lst_prec, axis=1)
        
    return prec_percentile, ivt_percentile
