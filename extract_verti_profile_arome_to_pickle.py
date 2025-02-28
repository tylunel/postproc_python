#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for extracting vertical profiles from AROME .fa files
and saving it in a pickle for later postprocessing

@author: lunelt
"""

import numpy as np
import os
import re
import pandas as pd
# import xarray as xr
# from datetime import datetime
# import metpy.calc as mcalc  # found in ~/Logiciels/MetPy-PBL-estimation/src/metpy/calc
import pickle
import global_variables as gv
import tools


folder_path = '/home/lunelt/Data/analysis/'

site = 'salin-giraud'

varname_prog_list = ['t', 'q', 'tke', 'u', 'v']

max_height = 3000  # [m]

# month = '01'
# month_next = str(int(month)+1).zfill(2)
# date_start, date_end = f'2023-{month}-01 00:15:00', f'2023-{month_next}-01 00:15:00'
date_start, date_end = '2023-01-21 00:00:00', '2023-01-22 23:15:00'

save_folder = '/home/lunelt/Data/data_NEMO/arome_profiles/'

# Regular expression to match filenames and extract date and hour
pattern = r"arome\.AN\.(\d{4})(\d{2})(\d{2})\.(\d{2})"  # to have all date
# pattern = r"arome\.AN\.(\d{4})(\d{2})(\d{2})\ .12"  # only midday

# List to store datetime objects
datetimes = []
# Iterate over files in the folder
for filename in os.listdir(folder_path):
    match = re.match(pattern, filename)
    if match:
        if len(match.groups()) == 4:
            year, month, day, hour = map(int, match.groups())
            datetimes.append(pd.Timestamp(
                year=year, month=month, day=day, hour=hour))
        elif len(match.groups()) == 3:
            year, month, day = map(int, match.groups())
            datetimes.append(pd.Timestamp(
                year=year, month=month, day=day, hour=int(pattern[-2::])))
# Sort the datetimes (optional, ensures chronological order)
datetimes.sort()

# subset the datetimes of interest
df = pd.DatetimeIndex(datetimes)
df_temp = df[df > date_start]
datetimes = df_temp[df_temp < date_end]

#%% LOAD the verti profiles

for wanted_date in datetimes:
    print(wanted_date)
    
    verti_profile = {}
    
    # load from arome GRIB
    for varname in varname_prog_list:
        try:
            temp = tools.extract_profile_arome(
                site, wanted_date, varname, max_height=max_height, 
                verbose=False)
            verti_profile[varname] = temp[varname]
            
            # -- Save profile to pickle
            datetime_formatted = wanted_date.strftime('%Y%m%d.%H')
            filename_pickle = f"verti_profile_arome_{site}_{datetime_formatted}"
            
            # Format pd.DataFrame:
            df = pd.DataFrame(verti_profile)
            df.to_pickle(f'{save_folder}/{site}/{filename_pickle}.pickle')
            # Format dict: (heavier than pd.DataFrame)
            # with open(f'{save_folder}/{filename_pickle}.pickle', 'wb') as handle:
            #     pickle.dump(verti_profile, handle)
            #     print(f'{filename_pickle}.pickle saved in {save_folder}')
        except:
            continue
    

    



#%% Opening the pickle
# obj = pd.read_pickle(f'{save_folder}/{filename_pickle}.pickle')
    
# days = {date: pd.DataFrame(profile).to_xarray() for date, profile in obj.items()}

# ds = xr.concat(list(days.values()), dim=xr.Variable('time', list(days.keys())))


