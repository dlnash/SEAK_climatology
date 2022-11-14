#!/bin/bash
######################################################################
# Filename:    download_SEAK-WRF.sh
# Author:      Deanna Nash dnash@ucsd.edu
# Description: Script to download SEAK-WRF data
# https://registry.opendata.aws/wrf-se-alaska-snap/ (data link)
#
######################################################################

### Activate python
source /home/dnash/miniconda3/etc/profile.d/conda.sh 

### Activate conda env that has AWS-CLI installed
conda activate SEAK-clim

# ### List out files from a S3 directory
# aws s3 ls --no-sign-request s3://wrf-se-ak-ar5/cfsr/4km/hourly/1980/

# ### Copy a single file from S3 to local
# aws s3 cp s3://wrf-se-ak-ar5/cfsr/4km/hourly/1980/WRFDS_1980-01-01.nc /work/dnash/SEAK_clim_data/downloads/wrf-AK/WRFDS_1980-01-01.nc --no-sign-request

### Copy entire directory from S3 to local
PATH_TO_OUT='/work/dnash/SEAK_clim_data/downloads/wrf-AK/'
aws s3 cp s3://wrf-se-ak-ar5/cfsr/4km/hourly/ ${PATH_TO_OUT} --recursive --no-sign-request





