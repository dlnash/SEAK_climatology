"""
Filename:    statistical_tests.py
Author:      Deanna Nash, dnash@ucsd.edu
Description: Functions for running different significance tests on tseries or ds objects
"""

# Import Python modules

import os, sys
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.stats import ttest_1samp, t, pearsonr, norm, linregress, ttest_ind_from_stats
import scipy.stats.distributions as dist

def test_diff_means(mean1, std1, nobs1, mean2, std2, nobs2):
    '''
    Calculate the zscore for testing 
    the differences in means in two samples
    
    mean1 : sample 1 mean
    std1 : sample 1 standard deviation
    nobs1 : number of observations in sample 1
    
    return 
    Z : the test statistic
    p : the p-value
    
    '''
    
    diff = mean1-mean2 
    var1 = (std1**2) # sample 1 variance
    var2 = (std2**2) # sample 2 variance
    std_diff = np.sqrt((var1/nobs1)+(var2/nobs2)) 
    
    Z = diff/std_diff
    
    return Z

def _test_diff_means_ufunc(mean1, std1, nobs1, mean2, std2, nobs2, dims=['lat', 'lon']):
    """ufunc to wrap test_diff_means for xr_zscore_diff_mean"""
    return xr.apply_ufunc(test_diff_means, # function
                          mean1, std1, nobs1, mean2, std2, nobs2, # now arguments in order
                          input_core_dims=[dims, dims, [], dims, dims, []],  # list with one entry per arg
                          output_core_dims=[dims, dims], # size out output 
                          dask='allowed')

def _test_diff_means_pval_ufunc(Z, dims=['lat', 'lon']):
    """ufunc to wrap test_diff_means for xr_zscore_diff_mean"""
    return xr.apply_ufunc(norm.sf, # function
                          abs(Z), # now arguments in order
                          input_core_dims=[dims],  # list with one entry per arg
                          output_core_dims=[dims], # size out output 
                          dask='allowed')

def xr_zscore_diff_mean(data1, data2):
    '''
       Adapted from Schaums Outline of Statistics 4th edition Chapter 10
       
    Parameters
    ----------
    data1 : xarray ds
        sample 1
    data2: xarray ds
        sample 2
        
    Returns
    -------
    tvalue : array_like, float
        xarray ds object with same vars as data1 that has the tvalue for the two-sample t-test
    pvalue : array_like, float
        xarray ds object with same vars as data1 that has the pvalue for the two-sample t-test 
        
    '''
    mean1 = data1.mean('time')
    mean2 = data2.mean('time')
    std1 = data1.std('time')
    std2 = data2.std('time')
    nobs1 = float(len(data1.time))
    nobs2 = float(len(data2.time))
    
    diff = mean1 - mean2
    
    zscore = test_diff_means(mean1, std1, nobs1, mean2, std2, nobs2)
    # Calculate the  p-value
    # based on the standard normal distribution z-test
    pvalue = _test_diff_means_pval_ufunc(zscore)*2 # Multiplied by two indicates a two tailed testing.
    
    return diff, pvalue