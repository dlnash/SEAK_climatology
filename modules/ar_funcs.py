"""
Filename:    ar_funcs.py
Author:      Deanna Nash, dlnash@ucsb.edu
Description: Functions for getting AR metrics
"""

# Import Python modules
import pandas as pd
from itertools import groupby

def AR_rank(df):
    """
    Super simple AR rank determiner based on Ralph et al., 2019. 
    This ranks the AR based on the duration of AR conditions and maximum IVT during the AR.
    
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
        
    return rank