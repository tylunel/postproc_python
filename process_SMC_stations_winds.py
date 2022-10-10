#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:33:15 2022

@author: lunelt
"""

import xarray as xr
import os
import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np


save_plot = False

#%% SET DATAFOLDER
datafolder = '/cnrm/surface/lunelt/data_LIAISE/SMC_22stations/'
#filename_ex = 'LIAISE_C6_SMC_MTO-1MN_L0_20210910_V01.nc'    


#%% OLD WAY - AS FOR PRECIP
fname_list = []
read_data = True
if read_data:
    wind_list = []
    stations = {}

# iterate over files in that directory
for filename in os.listdir(datafolder):
    if '2021072' in filename:
        fname_chunks = filename.split('_')
        fname_list.append(fname_chunks)    
    
        #create dataframe of data - takes 1-2min
        if read_data:
            station_name = fname_chunks[1]
            print(station_name)
            date = fname_chunks[-2]    
            ds = xr.open_dataset(datafolder + filename)
            stations[station_name] = ds
            try:
                wind_list.append([station_name, date, float(ds.lat), float(ds.lon),
                                  ds.WS.data, len(ds.WS.data), ds.WS.attrs,
                                  ds.WD.data, len(ds.WD.data), ds.WD.attrs])
            except:
                pass

#%%
if read_data:
    dfwind = pd.DataFrame(wind_list,
                            columns=['station', 'date', 'latitude', 'longitude', 
                                     'ws', 'length_ws', 'ws_attrs',
                                     'wd', 'length_wd', 'wd_attrs',])

    
stations = dfwind.drop_duplicates(subset='station')[
        ['station', 'latitude', 'longitude']]
stations.set_index('station', inplace=True)

#%% date refining - keep summer period only
#refine dfwind before other computation

dfwind1 = dfwind[(dfwind.date.values.astype(float) > 20210720) & \
                     (dfwind.date.values.astype(float) < 20210725)]

#dfwind = dfwind[dfwind.date.values.astype(float) == 20210722]

dfwind_refined = dfwind1

#%% Get mean wind
total_wind = {}
frac_missing = {}

for id_station in stations.index :
    print(id_station)
    df1 = dfwind_refined[dfwind_refined.station == id_station]
    df1.sort_values('date', inplace=True)
    valid_df1 = df1[df1.length_ws == 1440]
    frac_missing_values = (len(df1)-len(valid_df1))/len(df1)
    
    try:
#        total1=_wind = (valid_df1.ws.sum().sum())/len(valid_df1)
        mean_wind = valid_df1.ws.mean().mean()
    except:
        if valid_df1.ws.sum() == 0:
            mean_wind = None
        else:
            print('Type of valid_df')
            print(type(valid_df1.ws))
            raise ValueError('Type issue')
    
    frac_missing[id_station] = frac_missing_values
    total_wind[id_station] = mean_wind



df3 = pd.Series(total_wind, name='total_wind')
df4 = pd.Series(frac_missing, name='frac_missing')
stations = pd.concat([stations, df3, df4], axis=1)

#exclude station if fraction of missing values is too high
threshold_frac_missing = 0.5

valid_stations = stations[
        stations.frac_missing < threshold_frac_missing].dropna()

#%%
fig, ax = plt.subplots(1, figsize=(10,8))

# Load PGD for background of maps
DS = xr.open_dataset(
    '/cnrm/surface/lunelt/NO_SAVE/nc_out/2.01_pgds_irr/' + \
    'PGD_400M_CovCor_v26_ivars.nc')

ax.pcolormesh(DS.longitude.data, 
              DS.latitude.data, 
              DS.ZS.data, 
              cmap='binary')

ax.contour(DS.longitude.data, 
          DS.latitude.data, 
          DS.COVER369.data,
          levels=0,
          linestyles='dashed',
          linewidths=1,)

# Stations DATA
temp = ax.scatter(valid_stations.longitude,
                  valid_stations.latitude, 
                  s=30,
                  c=valid_stations.total_wind, 
                  cmap='seismic_r')


plot_title = 'Average wind speed on 202107 on SMC stations'
plt.title(plot_title)
cbar = plt.colorbar(temp)
cbar.set_label('wind speed m/s')
plt.show()


