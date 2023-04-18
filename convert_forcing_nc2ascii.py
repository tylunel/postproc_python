#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 09:05:40 2022

@author: lunelt

More info here:
http://www.umr-cnrm.fr/surfex/spip.php?article214
http://www.umr-cnrm.fr/surfex/spip.php?article215
"""

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

############################

## Data path
work_dir = '/cnrm/surface/lunelt/data_LIAISE/irta-corn/seb/'

fn = 'FORCING_irta-corn_obs.nc'
#fn = 'eddypro_LACENDROSA_Alfalfa_CNRM_30min_full_output.csv'

ds = xr.open_dataset(work_dir + fn)

plot_data = True

# CREATE Forc_... files
#name_correspondance_dict = {
#        'Tair': 'TA',
#        'Qair': 'QA',
#        'PSurf': 'PS',
#        'Rainf': 'RAIN',
#        'Snowf': 'SNOW',
#        'DIR_SWdown': 'DIR_SW',
#        'SCA_SWdown': 'SCA_SW',
#        'LWdown': 'LW',
#        'Wind': 'WIND',
#        'Wind_DIR': 'DIR',
#        'CO2air': 'CO2',
#        }
#
#for var in name_correspondance_dict:
#    df_temp = ds[var].to_dataframe()
#    df_temp.to_csv(work_dir + 'Forc_{0}.txt'.format(name_correspondance_dict[var]),
#                   sep='\n', header=False, index=False,
#                   encoding='ascii')
#
## CREATE Params_config.txt
#with open(work_dir + "Param_config_test.txt", "w") as file:
##     Y/N (only in binary case) to specify if the forcing data must be swapped
##     number of geographical points
#    file.write(str(1) +'\n')
##    number of simulation steps
#    file.write(str(len(ds.time)) +'\n')
##    forcing time step (seconds)
#    file.write(str(float(ds['FRC_TIME_STP'])) +'\n')
#
#    # Get timestamp of beginning of simulation
#    timestamp = pd.Timestamp(ds.time[0].values)
##    year
#    file.write(str(timestamp.year) +'\n')
##    month
#    file.write(str(timestamp.month) +'\n')
##    day
#    file.write(str(timestamp.day) +'\n')
##    hour (seconds)
#    file.write(str(timestamp.hour*3600 + timestamp.minute*60 + timestamp.second) +'\n')
##    longitude for each point of the domain (degrees)
#    file.write(str(float(ds['LON'])) +'\n')
##    latitude for each point of the domain (degrees)
#    file.write(str(float(ds['LAT'])) +'\n')
##    altitude of each point of the domain (m)
#    file.write(str(float(ds['ZS'])) +'\n')
##    height of temperature forcing for each point of the domain (m)
#    file.write(str(float(ds['ZREF'])) +'\n')
##    height of wind forcing for each point of the domain (m)
#    file.write(str(float(ds['UREF'])) +'\n')


#%% Plot Data to check consistency

if plot_data:
    for key in ['Qair', 
#                'Tair', 'PSurf',"DIR_SWdown", "SCA_SWdown",
                "LWdown", "Rainf", 
#                "Snowf", "Wind", "Wind_DIR", "CO2air"
                ]:
        plt.figure()
        plt.plot(ds["time"][:], ds[key][:], 'black')
        plt.ylabel(key)
        
