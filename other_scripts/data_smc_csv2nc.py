#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 15:58:11 2023

@author: lunelt

Script for converting the data from SMC from CSV to netcdf
"""

import pandas as pd
import xarray as xr
import os
import global_variables as gv
import numpy as np

folder_path = '/home/lunelt/Data/data_LIAISE/SMC/ALL_stations_july/csv_files_orig/'

for filename in os.listdir(folder_path):
#filename = 'C6.csv'

    station_name = filename.replace('.csv', '')        
    # Define the input CSV file path and output NetCDF file path
    csv_file_path = f'{folder_path}/{filename}'
    netcdf_file_path = f'{folder_path}/../{station_name}.nc'
    
    if os.path.isfile(csv_file_path):
        print(f'station: {station_name}')
    
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path) # assumes that the first row contains the column names
        
        df = df.rename(columns={'data':'datetime'})
        df.index = df.datetime
        df['datetime'] = [str(i) for i in df['datetime']]
        
        # Convert the DataFrame to an xarray Dataset
        dataset = xr.Dataset.from_dataframe(df)
        station_prop = gv.stations_SMC_coords[gv.stations_SMC_coords['name']==station_name]
        dataset['lon'] = station_prop['lon']
        dataset['lat'] = station_prop['lat']
        dataset['altitude'] = station_prop['altitude']
        dataset['station_name'] = station_name
        
        # get the height of wind measurement
        temp_obs_height = []
        for nb in [2, 6, 10]:
            if not np.isnan(dataset[f'VV{nb}']).all():
                temp_obs_height.append(nb)
        dataset['obs_wind_height'] = temp_obs_height
        print(temp_obs_height)
        
        # Save the xarray Dataset to a NetCDF file
        dataset.to_netcdf(netcdf_file_path)


