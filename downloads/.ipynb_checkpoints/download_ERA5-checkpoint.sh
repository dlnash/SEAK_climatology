#!/bin/bash
######################################################################
# Filename:    download_ERA5.sh
# Author:      Deanna Nash dlnash@ucsb.edu
# Description: Script to download all the necessary ERA5 files
#
######################################################################

### Activate python

source /home/dnash/miniconda3/etc/profile.d/conda.sh 

### Activate conda env

conda activate cds

# names of configuration dictionaries to loop through
array=(
case_study_202012 # hourly pressure level data, dec 1-3, 2020
# prec_case_202012 # hourly precip dec 1-3, 2020
case_study_202011 # hourly pressure level data, nov 28-30, 2020
# prec_case_202011 # hourly precip nov 28-30, 2020
)

# now loop through each configuration dictionary to download the ERA5 data

##TODO figure out how to input argument into python run file

for i in ${!array[*]}
do 
    inconfig="${array[$i]}"
    python getERA5_batch.py ${inconfig}
done