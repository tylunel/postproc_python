#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021
    
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import tools
import global_variables as gv
import os

############# Independant Parameters (TO FILL IN):


global_data_nemo = gv.global_data_nemo
station_mf = 'marignane'

folder_path = f"{global_data_nemo}/in-situ/{station_mf}/"
prefix_file_station_mf = 'FRNOR'
filename = None  # if None, take the csv file that begins with prefix

varname_list = ['INS', 'GLO', 'NBAS', 'B1']   # checkout gv.dict_station_mf_metadata for exhaustive list
# T, N

save_plot = False
save_folder = './figures/time_series/{0}/'.format(station_mf)
# save_folder = 'article3/fig/time_series/{0}/'.format(station_mf)

remove_commas_by_dots = True
kelvin_to_celsius = True

xmin = pd.Timestamp('20230108T00')
xmax = pd.Timestamp('20230112T00')


figsize = (6, 2.5) #small for presentation: (6,6), big: (15,9), paper:(7, 7)
plt.rcParams.update({'font.size': 11})

#%% LOAD
# Specify the path to your CSV file

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
data[varname_list] = data[varname_list].apply(pd.to_numeric, errors='coerce')

#%% PLOT
# Plot time series of three variables into three subplots
fig, axs = plt.subplots(len(varname_list), 1, 
                        figsize=(10, 2+1.5*len(varname_list)), 
                        sharex=True)

# Plot each variable
for i, var in enumerate(varname_list):
    axs[i].plot(data['DATE'], data[var], label=var)
    axs[i].set_ylabel(var)
    axs[i].legend()
    axs[i].grid()
    axs[i].set_title(gv.dict_station_mf_metadata['variables'][var]['label'],
                     y=0.94, size=9)
    if gv.dict_station_mf_metadata['variables'][var]['unit'] == 'OCTAS':
        axs[i].set_yticks([0, 2, 4, 6, 8], ['Clear', 'FEW', 'SCT', 'BKN', 'OVC'])

# Set the x-axis label and format
plt.xlim([xmin, xmax])
plt.xlabel('time UTC')

plt.xticks(rotation=30)
plt.subplots_adjust(top=0.92)

plot_title = f'{station_mf}-{varname_list}'
plt.suptitle(plot_title)

#%% Save figure
if save_plot:
    tools.save_figure(plot_title, save_folder)
