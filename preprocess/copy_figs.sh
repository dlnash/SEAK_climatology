#!/bin/bash
######################################################################
# Filename:    copy_figs.sh
# Author:      Deanna Nash dlnash@ucsb.edu
# Description: Script to copy final figures to one folder and save as pdf
#
######################################################################

# Input parameters
maindir="../figs/" # main figure folder
finaldir="../figs/final_figs/" # final figure folder

# fig names in main folder
array=(
elevation_7.5arcsec_with_inset
ar_barplot
percentile_bins_daily
extreme-AR_IVT-250Z_composite_final
extreme-AR_windrose_elev-overlay_daily
extreme-AR_prec_composite_all_daily
)

# new names to be fig<name given in array2>
array2=(
1
2
3
4
5
6
)



for i in ${!array[*]}
do 
    infile="${maindir}${array[$i]}.png"
    outfile="${finaldir}fig${array2[$i]}.png"
#     echo "${infile} to ${outfile}"
    cp -v ${infile} ${outfile}
done

# ### supplemental figs
# supp_array=(
# composite_ar_types_bias
# )

# ## new names to be given
# supp_array2=(
# 1
# )
# for i in ${!supp_array[*]}
# do 
#     infile="${maindir}${supp_array[$i]}.png"
#     outfile="${finaldir}figS${supp_array2[$i]}.png"
# #     echo "${infile} to ${outfile}"
#     cp -v ${infile} ${outfile}
# done

## convert png to pdf
# python png_to_pdf.py

## zip to single file
cd ../figs/final_figs
zip figs.zip fig*