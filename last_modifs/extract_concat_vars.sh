#!/bin/bash
# Script for extracting variables and concatenating them
# in NETCDF format for use in python script 'generate_forcing_from_mnh'
# in order to create FORCING.nc file in SURFEX offline

folder='/cnrm/surface/lunelt/NO_SAVE/nc_out/7.15_irrlagrip30_d1_15-30'
output_name='LIAIS.1.CAT_irrlagrip30_7.15'

cd $folder
mkdir -p ./EXTRACTED_FILES

for file in LIAIS.1.SEG??.0??dg.nc; do
  outname=${file}.red
  ncks -O -v T2M_ISBA,Q2M_ISBA,PRES,RAINF_ISBA,SNOWF_ISBA,UT,VT,LWD,DIRFLASWD,SCAFLASWD -d level,1,1 $file ./EXTRACTED_FILES/$outname

done

cd EXTRACTED_FILES/
ncecat LIAIS.1.SEG*.red $output_name

