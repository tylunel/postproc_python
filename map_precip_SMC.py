#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:33:15 2022

@author: lunelt
"""

import xarray as xr
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
#import numpy as np


save_plot = False

#%% SET DATAFOLDER
#datafolder = '/cnrm/surface/lunelt/data_LIAISE/SMC_22stations/'
directory = '/cnrm/surface/lunelt/data_LIAISE/SMC_22stations/' 
fname_list = []


#%% OLD - FOR PRECIP

read_data = True
if read_data:
    precip_list = []
    stations = {}

# iterate over files in that directory
for filename in os.listdir(directory):
    if '202107' in filename:
        fname_chunks = filename.split('_')
        fname_list.append(fname_chunks)    
    
        #create dataframe of data - takes 1-2min
        if read_data:
            station_name = fname_chunks[1]
            print(station_name)
            date = fname_chunks[-2]    
            ds = xr.open_dataset(directory + filename)
            stations[station_name] = ds
            try:
                precip_list.append([station_name, date, float(ds.lat), float(ds.lon),
                                    ds.PCP.data, len(ds.PCP.data), ds.PCP.attrs])
            except:
                pass

#if read_bool:
if read_data:
    dfprecip = pd.DataFrame(precip_list,
                            columns=['station', 'date', 'latitude', 
                                     'longitude', 'precipitation', 
                                     'length_var', 'details'])
    
dfname = pd.DataFrame(fname_list,
              columns=['expe', 'station', 'organisation', 
                       'type', 'unknown', 'date', 'version'])
    

#%% place stations on map
    
stations = dfprecip.drop_duplicates(subset='station')[
        ['station', 'latitude', 'longitude']]
stations.set_index('station', inplace=True)



#%% date refining - keep summer period only
#refine dfprecip before other computation
dfprecip1 = dfprecip[(dfprecip.date.values.astype(float) > 20210600) & \
                     (dfprecip.date.values.astype(float) < 20210900)]

dfprecip2 = dfprecip[dfprecip.date.values.astype(float) == 20210722]

dfprecip_refined = dfprecip

#%% Get total amount of rain
total_precip = {}
frac_missing = {}

for id_station in stations.index :
    print(id_station)
    df1 = dfprecip_refined[dfprecip_refined.station == id_station]
    df1.sort_values('date', inplace=True)
    valid_df1 = df1[df1.length_var == 1440]
    frac_missing_values = (len(df1)-len(valid_df1))/len(df1)
    
    try:
        mean_precip = (valid_df1.precipitation.sum().sum())/len(valid_df1)
    except:
        if valid_df1.precipitation.sum() == 0:
            mean_precip = None
        else:
            print('Type of valid_df')
            print(type(valid_df1.precipitation))
            raise ValueError('Type issue')
    
    frac_missing[id_station] = frac_missing_values
    total_precip[id_station] = mean_precip



df3 = pd.Series(total_precip, name='total_precip')
df4 = pd.Series(frac_missing, name='frac_missing')
stations = pd.concat([stations, df3, df4], axis=1)

#exclude station if fraction of missing values is too high
threshold_frac_missing = 0.5

valid_stations = stations[
        stations.frac_missing < threshold_frac_missing].dropna()


#%% plot station on map with geopandas

#gdf = gpd.GeoDataFrame(
#    stations, 
#    geometry=gpd.points_from_xy(stations.longitude, stations.latitude),
#    crs="EPSG:4326")
#
#world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
#europe = world[world.continent == "Europe"]
#spain = europe[europe.name == 'Spain']
#
#ax = spain.plot(color='white', edgecolor='black')
#gdf.plot(
#        ax=ax, 
#        column='margin', cmap='viridis', zorder=2)
#ax.set_xlim(0, 3)
#ax.set_ylim(40, 43)
##ax.annotate('test', xy=(1, 41), xytext=(3,3), textcoords='truc')
#plt.show()

#%% Load PGD for background of maps
DS = xr.open_dataset(
    '/cnrm/surface/lunelt/NO_SAVE/nc_out/2.01_pgds_irr/' + \
    'PGD_400M_CovCor_v26_ivars.nc')

#%% with Matplotlib subplots

fig, ax = plt.subplots(1, figsize=(10,8))
#fig.suptitle('total precipitation per station')

ax.pcolormesh(DS.longitude.data, 
              DS.latitude.data, 
              DS.ZS.data, 
              cmap='binary')
#
#ax.pcolormesh(DS.longitude.data, 
#              DS.latitude.data, 
#              DS.COVER369.data, 
#              cmap='YlGn', alpha=0.3)

from scipy.ndimage.filters import gaussian_filter
sigma = 0.2
dataCOV369_filtered = gaussian_filter(DS.COVER369.data, sigma)

ax.contour(DS.longitude.data, 
          DS.latitude.data, 
          dataCOV369_filtered,
          levels=0,
          linestyles='dashed',
          linewidths=1,
#          colors=['None'],
#          hatches='-'
          )

temp = ax.scatter(valid_stations.longitude,
                  valid_stations.latitude, 
                  s=30,
                  c=valid_stations.total_precip, 
                  cmap='RdYlBu')

plt.colorbar(temp)
ax.set_title('mean precipitation in mm/day from 2021-05 to 2022-04')
plt.show()


if save_plot == True:
    plt.savefig('./figures/'+'SMC_stations_ACPPR_202105-202204.png')
