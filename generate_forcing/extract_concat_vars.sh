#!/bin/bash
# Script for extracting variables and concatenating them
# in NETCDF format for use in python script 'generate_forcing_from_mnh'
# in order to create FORCING.nc file in SURFEX offline

#modelfolder=$1
#modelfolder=${modelfolder:-'1.15_std_d1_15-30'}
modelfolder=${1?Need a value}

folder='/cnrm/surface/lunelt/NO_SAVE/nc_out/'
output_name='LIAIS.1.CAT_'$modelfolder

cd $folder/$modelfolder
mkdir -p ./EXTRACTED_FILES

for file in LIAIS.1.SEG??.0??dg.nc; do
  name_red=${file}.red
  ncks -O -v T2M_ISBA,Q2M_ISBA,PRES,RAINF_ISBA,SNOWF_ISBA,UT,VT,LWD,DIRFLASWD,SCAFLASWD,latitude,longitude -d level,1,1 $file ./EXTRACTED_FILES/$name_red

done

cd EXTRACTED_FILES/
ncecat LIAIS.1.SEG*.red $output_name

