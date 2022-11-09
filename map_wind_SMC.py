#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:33:15 2022

@author: lunelt

Pb for comparing values of wind_speed!:
    most of stations inside irrgated area are 2m high, and 10m high outside
    of irrigated zone.
    For relevant comparison, check C6 vs C7 (10m high each)
"""

import xarray as xr
import os
import pandas as pd
import matplotlib.pyplot as plt
import tools
#import numpy as np

##########################################
date_begin = 20210630  #keep it as float, even if it is a date
date_end = 20210801
#Be careful for conflicts with partial_filename

datafolder = '/cnrm/surface/lunelt/data_LIAISE/SMC_22stations/'
#filename_ex = 'LIAISE_C6_SMC_MTO-1MN_L0_20210910_V01.nc'    
partial_filename = '202107'

zoom_on = 'urgell'  #None for no zoom, 'liaise' or 'urgell'

if zoom_on == 'liaise':
    lat_range = [41.45, 41.8]
    lon_range = [0.7, 1.2]
elif zoom_on == 'urgell':
    lat_range = [41.1, 42.1]
    lon_range = [0.2, 1.7]

save_plot = True
save_folder = './figures/winds/corr/'

selected_stations_height = 'all'  #'2m', '10m' or 'all'

##############################################

# not all SMC stations are 0m high:
stations_2m = ['VH', 'WK', 'V1', 'WB', 'VM', 'WX', 'WA', 'WC', 'V8', 'XI',
               'XM', 'WL', 'UM', 'WI', 'VE']
stations_10m = ['VK', 'C6', 'C7', 'C8', 'D1', 'XD', 'XR', 'XA', 'VP', 'VB',
                'VQ']
stations_unk = ['YJ', 'CW', 'MR', 'VM', 'WV', 'VD', 'YD', 'XX', 'YJ', ]


#%% Get stations availables in folder
fname_list = []
read_data = True
if read_data:
    wind_list = []
    stations = {}

# iterate over files in that directory
for filename in os.listdir(datafolder):
    if partial_filename in filename:
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

dfwind1 = dfwind[(dfwind.date.values.astype(float) > date_begin) & \
                     (dfwind.date.values.astype(float) < date_end)]

#dfwind = dfwind[dfwind.date.values.astype(float) == 20210722]


dfwind_refined = dfwind1

#dfwind_10m = dfwind1[
#        dfwind1.ws_attrs['long_name'][-4::].replace(" ", "") == '10m']

#%% Get mean wind
total_wind = {}
frac_missing = {}

if selected_stations_height == '2m':
    list_stations = stations_2m
elif selected_stations_height == '10m':
    list_stations = stations_10m
else:
    list_stations = stations.index




for id_station in stations.index :
    print(id_station)
#    if id_station not in stations.index:
#        continue
    
    df1 = dfwind_refined[dfwind_refined.station == id_station]
    
    if selected_stations_height == 'all':
        pass
    elif (df1.iloc[0].ws_attrs['long_name'][-4::].replace(" ", "") != \
        selected_stations_height):
        continue
    
    df1.sort_values('date', inplace=True)
    # Check if data available for every minute of day:
    valid_df1 = df1[df1.length_ws == 1440]
    # compute fraction of missing values
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
    
    # put in dict fraction of missing values and mean_wind
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
stations_data = ax.scatter(valid_stations.longitude,
                  valid_stations.latitude, 
                  s=50,
                  c=valid_stations.total_wind, 
                  cmap='seismic',
                  vmin=0.5,
                  vmax=4)

if zoom_on is not None:
    plt.ylim(lat_range)
    plt.xlim(lon_range)

plot_title = 'Average wind speed from {0} to {1} on {2} SMC stations'.format(
        date_begin, date_end, selected_stations_height)
plt.title(plot_title)
cbar = plt.colorbar(stations_data)
cbar.set_label('wind speed m/s')
plt.show()

#%% Save_plot
if save_plot:
    tools.save_figure(plot_title, save_folder)

