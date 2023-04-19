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
    
    # open UVdir data
    fname = path_to_out + 'SEAK_UV_{0}_{1}.csv'.format(option, temporal_res)
    UV_df = pd.read_csv(fname)
    
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
        
        ## append community UV data
        df['UVdir'] = UV_df[community]
        
        # reset the index as "time"
        df = df.set_index(pd.to_datetime(df['time']))
        
        df_lst.append(df)        
        
    return df_lst
    
def df_annual_clim(df_lst, community_lst, varname='prec'):
    '''
    Returns list of dataframes (based on community_lst) that has the annual climatology for precipitation/IVT
    
    Parameters
    ----------
    list : list of pandas dataframes
        list of daily or hourly pandas dataframes with precipitation/IVT data
        
    community_lst : list
        list of strings of community names

    Returns
    -------
    df : pandas dataframe
        df that has annual climatology of precipitation/IVT for each community

    '''
    clim_mean_lst = []
    clim_std_lst = []
    for i, df in enumerate(df_lst):
        community = community_lst[i]
        # reset the index as "time"
        df = df.set_index(pd.to_datetime(df['time']))
        
        if varname == 'prec':
            
            ## calculate monthly sum
            df_mon = df[varname].resample('1M').sum()  # monthly sum
            clim_mean = df_mon.groupby(df_mon.index.month).mean() # get mean of monthly sum
            clim_std = df_mon.groupby(df_mon.index.month).std() # get std of monthly sum
        else:
            # create month column
            df['month'] = df.index.month
            clim_mean = df.groupby(['month'])[varname].mean() # get mean
            clim_std = df.groupby(['month'])[varname].std() # get std
        
        # rename and append to list
        clim_mean = clim_mean.rename('{0}_{1}'.format(varname, community))
        clim_mean_lst.append(clim_mean)
        clim_std = clim_std.rename('{0}_{1}'.format(varname, community))
        clim_std_lst.append(clim_std)
    
    rename_dict = {'{0}_Hoonah'.format(varname): 'Hoonah',
                   '{0}_Skagway'.format(varname): 'Skagway',
                   '{0}_Klukwan'.format(varname): 'Klukwan',
                   '{0}_Yakutat'.format(varname): 'Yakutat',
                   '{0}_Craig'.format(varname): 'Craig',
                   '{0}_Kasaan'.format(varname): 'Kasaan'}
    clim_mean_final = pd.concat(clim_mean_lst, axis=1)
    clim_mean_final = clim_mean_final.rename(columns=rename_dict)
    clim_std_final = pd.concat(clim_std_lst, axis=1)
    clim_std_final = clim_std_final.rename(columns=rename_dict)
        
    return clim_mean_final, clim_std_final


def df_AR_annual_clim(df_lst, community_lst, varname='AR'):
    '''
    Returns list of dataframes (based on community_lst) that has the annual climatology for AR frequency
    
    Parameters
    ----------
    list : list of pandas dataframes
        list of daily or hourly pandas dataframes with AR data
        
    community_lst : list
        list of strings of community names

    Returns
    -------
    df : pandas dataframe
        df that has annual climatology of AR frequency for each community

    '''
    clim_mean_final = []

    for i, df in enumerate(df_lst):
        community = community_lst[i]
        # reset the index as "time"
        df = df.set_index(pd.to_datetime(df['time']))

        ## make new column that is 0 - non-AR non-extreme, 1 - AR non-extreme, 2 - non-AR extreme, 3 - AR extreme
        df['extremeAR'] = df['AR']
        df.loc[(df['AR']==1) & (df['prec_{0}'.format(community)] >=0.95), 'extremeAR'] = 2 # AR extreme
        df.loc[(df['AR']==0) & (df['prec_{0}'.format(community)] >=0.95), 'extremeAR'] = 3 # non-AR extreme
        
        # create day of year column
        df['month'] = df.index.month
        prec_type_lst = [1, 2, 3, 0]
        prec_name_lst = ['AR non-extreme', 'AR extreme', 'non-AR extreme', 'non-AR non-extreme']
        clim_mean_lst = []
        for j, prec_type in enumerate(prec_type_lst):
            idx = (df.extremeAR == prec_type)
            ardates_daily = df.loc[idx] # get only AR dates

            mon_ar = ardates_daily['extremeAR'].resample("M").count()  # count number of ARs per month
            clim_ct = mon_ar.groupby(mon_ar.index.month).mean() # get average number of ARs per month
            clim_ct = clim_ct.rename(prec_name_lst[j])
            clim_mean_lst.append(clim_ct)

        clim_mean_final.append(pd.concat(clim_mean_lst, axis=1))
        
    # concatenate all communities
    df_concat = pd.concat(clim_mean_final)
    by_row_index = df_concat.groupby(df_concat.index)
    df_means = by_row_index.mean()
        
    return df_means
        
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

def df_AR_precip_contribution(df_lst, community_lst, varname='prec'):
    '''
    Returns dataframe that contribution to annual precipitation from:
    a) non-AR non-extreme precipitation (0)
    b) AR non-extreme precipitation (1)
    c) AR extreme precipitation (2)
    d) non-AR extreme precipitation (3)
    
    
    Parameters
    ----------
    list : list of pandas dataframes
        list of daily or hourly pandas dataframes with AR and precipitation data
        
    community_lst : list
        list of strings of community names

    Returns
    -------
    df : pandas dataframe
        df with contribution of precipitation to annual climatology from the 4 types for each community

    '''
    prec_name_lst = ['non-AR non-extreme', 'AR non-extreme', 'AR extreme', 'non-AR extreme']
    df_final = pd.DataFrame(columns=prec_name_lst, index=community_lst)
    df_total = pd.DataFrame(columns=['total'], index=community_lst)

    for i, df in enumerate(df_lst):
        community = community_lst[i]
        # reset the index as "time"
        df = df.set_index(pd.to_datetime(df['time']))
        
        yr_ar = df['prec'].resample("Y").sum()  # resample to annual total
        clim = yr_ar.mean() # get average amount of precipitation per year in that category
        df_total.iloc[i, :] = clim
        
        ## make new column that is 0 - non-AR non-extreme, 1 - AR non-extreme, 2 - non-AR extreme, 3 - AR extreme
        df['extremeAR'] = df['AR'] # AR non-extreme
        df.loc[(df['AR']==1) & (df['prec_{0}'.format(community)] >=0.95), 'extremeAR'] = 2 # AR extreme
        df.loc[(df['AR']==0) & (df['prec_{0}'.format(community)] >=0.95), 'extremeAR'] = 3 # non-AR extreme

        prec_type_lst = [0, 1, 2, 3]
        for j, prec_type in enumerate(prec_type_lst):

            idx = (df.extremeAR == prec_type)
            ardates_daily = df.loc[idx] # get only type dates

            yr_ar = ardates_daily['prec'].resample("Y").sum()  # resample to annual total
            clim = yr_ar.mean() # get average amount of precipitation per year in that category
            total = df_total.iloc[i].values.item()
            df_final.iloc[i, j] = (clim/total)*100.
            
        
    return df_final