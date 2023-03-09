## Preprocessing Scripts for SEAK Climatology Analysis

1. preprocess_SEAK-WRF_2Dvar.py

This script extracts identified variables from the SEAK WRF precipitation data for analysis. Run this with `output_varname = 'PCPT'`.

2. SEAK-WRF_precipitation_tseries.ipynb

This script generates a .csv hourly time series of precipitation values for the grid cell closest to each community.

3. preprocess_ERA-IVT_hourly.py

This script generates a .csv hourly time series of IVT values for the grid cell closest to each community.

4. preprocess_AR_6-hourly.py

This script generates a .csv 6-hourly and daily time series of days an AR makes landfall in southeast Alaska 

5. preprocess_ERA5_composites.py

Combines the precipitation, IVT, and AR time series .csv files
Generates average composites for all ERA5 variables for each community at different time lags for >95th percentile precipitation AR days

6. preprocess_SEAK-WRF_composites.py

Combines the precipitation, IVT, and AR time series .csv files
Generates average composites for precipitation for each community at different time lags for >95th percentile precipitation AR days