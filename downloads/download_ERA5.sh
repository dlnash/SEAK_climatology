#!/bin/bash
######################################################################
# Filename:    download_ERA5.sh
# Author:      Deanna Nash dlnash@ucsb.edu
# Description: Script to download all the necessary ERA5 files
# To run, activate conda env with "conda activate cds", then run the script with "bash download_ERA5.sh"
######################################################################

### Activate python

source /home/dnash/miniconda3/etc/profile.d/conda.sh 

### Activate conda env

# conda activate cds

# names of configuration dictionaries to loop through
array=(
case_study_202012 # hourly pressure level data, dec 1-4, 2020
# prec_case_202012 # hourly precip dec 1-3, 2020
# case_study_202011 # hourly pressure level data, nov 28-30, 2020
# prec_case_202011 # hourly precip nov 28-30, 2020
# case_study_201801 # hourly pressure level data jan 11-16, 2018
# mslp # hourly mslp for entire clim
# huv # 6 hourly huv data at 850, 500, and 250 hPa for entire clim
# mslp # 6 hourly mslp data
# ivt # 6 hourly ivt data
)

# now loop through each configuration dictionary to download the ERA5 data

##TODO figure out how to input argument into python run file

for i in ${!array[*]}
do 
    inconfig="${array[$i]}"
    python getERA5_batch.py ${inconfig}
done