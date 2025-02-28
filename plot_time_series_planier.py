#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot planier data: lidar wind and tke, buoy sst and wave height

@author: lunelt
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import xarray as xr
import tools
import global_variables as gv


global_data_nemo = gv.global_data_nemo
save_plot = True

#%% LIDAR parameters
site = 'planier'
varname_list_lidar = [
    'ws', 
    # 'wd', 
    # 'tke',
    # 'diss_prod_ratio',
    # 'tke_bud_prod',
    # 'tke_bud_diss',
    ]
height_wind_list = [
    # 120,
    140, 
    # 200, 220, 
    # 240
    ]

#%% MF station parameters
station_mf = 'marignane'
varname_list_mf = [ # checkout gv.dict_station_mf_metadata for exhaustive list
    # 'INS', 
    # 'GLO', 
    'NBAS', 
    # 'B1'
    ]

#%% BUOY parameters - planier and solemio 
varname_list_buoy = [
    'buoy',
    ]  # 'buoy' allows to plot sst and awve height

#%% Global parameters
date_start, date_end = '2023-01-25 00:15:00', '2023-01-29 00:15:00'

plot_type = 'std'  #'barbs' or 'std'
skip_time_barbs = 2
barb_size_option = 'weak_winds'

#%% LOAD LIDAR data
lidar_filepath_dict = {
    'ws': f'{global_data_nemo}/{site}/30-min/wind_speed.csv',
    'wd': f'{global_data_nemo}/{site}/30-min/wind_direction.csv',
    'tke_along': f'{global_data_nemo}/{site}/30-min/TKE_along_wind_standard_method.csv',
    'tke_cross': f'{global_data_nemo}/{site}/30-min/TKE_cross_wind_standard_method.csv',
    'tke_verti': f'{global_data_nemo}/{site}/30-min/TKE_vertical_wind_standard_method.csv',
    'tke_bud_prod': f'{global_data_nemo}/{site}/30-min/production_rate.csv',
    'tke_bud_diss': f'{global_data_nemo}/{site}/30-min/dissipation_rate.csv',
    'tke_bud_diff_trans': f'{global_data_nemo}/{site}/30-min/diffusive_transport.csv',
    }

lidar_data_dict = {}

for var in lidar_filepath_dict.keys():
    data_temp = pd.read_csv(lidar_filepath_dict[var], 
                            sep='\t')
    data_temp.columns = ['time', 60, 80, 100, 120, 140, 
                  160, 180, 200, 220, 240]
    data_temp.index = pd.DatetimeIndex(data_temp.time)
    data_temp = data_temp.drop(columns=['time'])
    lidar_data_dict[var] = data_temp

try:
    lidar_data_dict['tke'] = lidar_data_dict['tke_along'] + \
        lidar_data_dict['tke_cross'] + lidar_data_dict['tke_verti']
    lidar_data_dict['diss_prod_ratio'] = \
        lidar_data_dict['tke_bud_diss'] / lidar_data_dict['tke_bud_prod']
    
except KeyError:
    pass

# preprocess outliers in tke values, i.e. values above 6
lidar_data_dict['tke'] = lidar_data_dict['tke'].clip(0, 6)


#%% LOAD STATION MF data

folder_path = f"{global_data_nemo}/in-situ/{station_mf}/"
prefix_file_station_mf = 'FRNOR'
filename = None  # if None, take the csv file that begins with prefix
remove_commas_by_dots = True

if filename == None:
    files = [f for f in os.listdir(folder_path) if f.startswith(prefix_file_station_mf) and f.endswith('.csv')]
    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected one CSV file starting with '{prefix_file_station_mf}', but found {len(files)}")
    file_path = os.path.join(folder_path, files[0])
else:
    file_path = f"{folder_path}/{filename}"

# Preprocess the CSV file by replacing commas with dots
if remove_commas_by_dots:
    with open(file_path, 'r') as file:
        content = file.read()
    content = content.replace(',', '.')
    with open(file_path, 'w') as file:
        file.write(content)

# Read the CSV file with the semicolon separator
data = pd.read_csv(file_path, sep=";", dtype=str)

# Convert the 'DATE' column (second column) to datetime format
data['DATE'] = pd.to_datetime(data['DATE'], format="%Y%m%d%H")

# Convert numeric columns to appropriate data types (assuming they are float or int)
data[varname_list_mf] = data[varname_list_mf].apply(pd.to_numeric, errors='coerce')


#%% LOAD BUOY data

# buoy = xr.open_dataset(f'{global_data_nemo}/bouee/GL_TS_MO_6100021_202307.nc')
buoy_planier = xr.open_dataset(f'{global_data_nemo}/bouee/GL_TS_MO_6100289.nc')     # Planier buoy
# buoy = xr.open_dataset(f'{global_data_nemo}/bouee/GL_TS_MO_6100284.nc')
# buoy = xr.open_dataset(f'{global_data_nemo}/bouee/GL_TS_MO_6100431_202303.nc')
buoy_solemio = xr.open_dataset(f'{global_data_nemo}/bouee/GL_TS_MO_SOLEMIO.nc')     # SST partially available on 2023

# sst_product = pd.read_csv(
#     f'{global_data_nemo}/sst_product/clim_lon_4.375_lat_43.375_1982-2011.csv', 
#     sep=',')
# sst_product = pd.read_csv(
#     f'{global_data_nemo}/sst_product/event_lon_5.125_lat_43.125_1982-2011.csv', 
#     sep=',')

# print('files loaded')
#%% PLOTs
varname_list_total = varname_list_lidar + varname_list_mf + varname_list_buoy

if plot_type == 'std':
    fig, ax = plt.subplots(len(varname_list_total), 1, 
                           figsize=(10, 1+1.5*len(varname_list_total)),
                           sharex=True)
    
    for i, var in enumerate(varname_list_total):
        if var in varname_list_lidar:
            
            for height_wind in height_wind_list:
                ax[i].plot(lidar_data_dict[var].index, 
                           lidar_data_dict[var][height_wind], 
                           label=f'{var}_{height_wind}')
            try:
                ax[i].set_ylabel(gv.dict_lidar_metadata[var])
            except KeyError:
                ax[i].set_ylabel(var)
                
            if var == 'wd':
                ax[i].set_yticks([0, 90, 180, 270, 360], ['N', 'E', 'S', 'W', 'N'])
            elif var == 'tke_bud_prod':
                ax[i].set_ylim([0, 0.005])
            elif var == 'tke_bud_diss':
                ax[i].set_ylim([0, 0.1])
            
        
        if var in varname_list_mf:
            ax[i].plot(data['DATE'], data[var], label=var)
            ax[i].set_ylabel(var)
            ax[i].set_title(gv.dict_station_mf_metadata['variables'][var]['label'],
                             y=0.94, size=9)
            if gv.dict_station_mf_metadata['variables'][var]['unit'] == 'OCTAS':
                ax[i].set_yticks([0, 2, 4, 6, 8], ['Clear', 'FEW', 'SCT', 'BKN', 'OVC'])
    
        if var == 'buoy':
    # depth_buoy_index = 0  # 0->0.0m or 1->0.4m
    
            for depth_buoy_index in [0,1]:
                sst_planier = buoy_planier.TEMP[:, depth_buoy_index]
                ax[i].scatter(sst_planier.TIME, sst_planier,
                              color='r', marker='+', s=10,
                              label=f'sst_planier_{depth_buoy_index}')
                
                sst_solemio = buoy_solemio.TEMP[:,depth_buoy_index]
                ax[i].scatter(sst_solemio.TIME, sst_solemio,
                              color='orange', marker='+', s=10,
                              label=f'sst_solemio_{depth_buoy_index}')
        
            ax[i].set_ylabel('SST [°C]')    
    # ax[i].set_ylim([15, 30])

            wave_height = buoy_planier.VAVH[:, 0]
            ax_sec = ax[i].twinx()
            ax_sec.scatter(wave_height.TIME, wave_height,
                           color='b', marker='x', s=10,
                           label='wave_height_planier')
            ax_sec.set_ylabel('wave height [m]')
            ax_sec.legend()
    
    for axi in ax:
        axi.legend(loc='best')
        axi.grid()

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
    plt.xlim(pd.Timestamp(date_start), pd.Timestamp(date_end))


elif plot_type == 'barbs':
    fig, ax = plt.subplots(2,1, figsize=(10,9))
    
    datirange=lidar_data_dict['ws'].index
    subset_datirange = datirange.where(
        np.logical_and(pd.Timestamp(date_start) < datirange, 
                       datirange < pd.Timestamp(date_end))
        ).dropna()[::skip_time_barbs]
    subset_datirange
    
    cmap = matplotlib.cm.get_cmap('viridis')

    
    for time_i in subset_datirange:
        print(time_i)
        u, v = tools.calc_u_v(lidar_data_dict['ws'].loc[time_i], 
                              lidar_data_dict['wd'].loc[time_i])
        for height in u.index:
            
            ax[0].barbs(time_i, height, u.loc[height], v.loc[height],              
                        barb_increments=gv.barb_size_increments[barb_size_option])
            sc = ax[1].scatter(time_i, height,
                          # c=cmap(lidar_data_dict['tke'].loc[time_i].loc[height]),
                          c=lidar_data_dict['tke'].loc[time_i].loc[height],
                          vmin=0, vmax=2, s=100, cmap=cmap)
            
    plt.annotate(gv.barb_size_description[barb_size_option],
                   fontsize=9,
                   xy=(0.1, 0.9),
                   xycoords='subfigure fraction'
                   )
    
    for axi in ax:
        # axi.set_xlim(pd.Timestamp('20230724'), pd.Timestamp('20230727'))
        axi.set_ylabel('height asl [m]')
        axi.set_ylim([0, 280])
        axi.legend(loc='lower left')
        axi.grid()
    plt.colorbar(sc, location='bottom', aspect=40, fraction=0.08)    
    # plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap), 
                 # values=[0,3])

vars_list = varname_list_lidar + varname_list_mf + varname_list_buoy
plot_title = f'{date_start}_time-series_planier_{vars_list}'
save_folder = './figures/time_series/planier/'

if save_plot:
    tools.save_figure(plot_title, save_folder)
