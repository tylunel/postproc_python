#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot radiosounding from MF

@author: lunelt
"""

import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
# import xarray as xr
import tools
import time

global_data_nemo = '/home/lunelt/Data/data_NEMO/rs/'

filename = 'DALTI2024000515169.csv'  # NIMES-COURBESSAC2023
site = 'nimes'
wanted_date = '202307260000'

#%% LOAD RS
rs_low = tools.extract_profile_rs_mf(
    site, wanted_date, 
    max_height=3000, folder_path=global_data_nemo, filename=filename)

#%% PLOT RS

fig, ax = plt.subplots(1,3)
# Temp
ax[0].plot(rs_low['theta'], rs_low['ALTI'], label='theta', 
           marker='+', linestyle='--', color='r')
ax[0].plot(rs_low['theta_v'], rs_low['ALTI'], label='theta_v', 
           marker='+', linestyle='--', color='b')
ax[0].set_ylabel('alti [m]')
ax[0].set_xlabel('temperature [K]')
ax[0].legend()

# Wind speed
ax[1].plot(rs_low['FF'], rs_low['ALTI'], label='wind_speed', 
           marker='+', linestyle='--', color='k')
ax[1].set_xlim([0, 20])

# Wind speed
ax[2].plot(rs_low['DD'], rs_low['ALTI'], label='wind_dir', 
           marker='+', linestyle='--', color='k')
ax[2].set_xticks([0, 90, 180, 270, 360], ['N', 'E', 'S', 'W', 'N'])


for axi in ax:
    axi.grid()
    
fig.suptitle(f'RS for {wanted_date}')




            
            