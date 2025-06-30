#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot planier data: lidar wind and tke, buoy sst and wave height


#To ad:
# stability (ri_gradient)
# LLJ obs and model
# shear
# maximum shear height



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
import scipy
import re
from scipy.optimize import curve_fit

global_data_nemo = gv.global_data_nemo


# %% LIDAR parameters
site = 'planier'
varname_list_lidar = [
    'ws',
    'wd',
    'tke',
    'diss_prod_ratio',
    'tke_bud_diss',
]


# %% MF station parameters
station_mf = 'marignane'
varname_list_mf = [  # checkout gv.dict_station_mf_metadata for exhaustive list
    # 'INS',
    'GLO',
    'NBAS',
    # 'B1'
]

# %% BUOY parameters - planier and solemio
varname_list_buoy = [
    'buoy',
]  # 'buoy' allows to plot sst and awve height

# %% AROME variables
varname_list_arome = [
    't',
    'ws',
    'wd',
]

# %% Global parameters
date_start, date_end = '2023-01-01 00:15:00', '2023-12-31 00:15:00'

plot_type = 'std'  # 'barbs' or 'std'
skip_time_barbs = 2
barb_size_option = 'weak_winds'

# %% LOAD LIDAR data
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
    lidar_data_dict['ws_shear'] = pd.DataFrame(
        (np.gradient(lidar_data_dict['ws'], axis=1) /
         np.gradient(lidar_data_dict['ws'].columns)),
        index=lidar_data_dict['ws'].index,
        columns=lidar_data_dict['ws'].columns)

except KeyError:
    pass

# preprocess outliers in tke values, i.e. values above 6
lidar_data_dict['tke'] = lidar_data_dict['tke'].clip(0, 6)

datetimes_avail_lidar = lidar_data_dict['ws'].index

temp_data_dict = {}
temp_data_dict['llj_ws'] = []
temp_data_dict['llj_height'] = []
temp_data_dict['max_ws_shear'] = []
temp_data_dict['max_ws_shear_height'] = []
temp_data_dict['ws_shear_exponent'] = []

for timestamp in datetimes_avail_lidar:
    print(f'Obs diags: {timestamp}')
    # LLJ analysis
    ws_profile = lidar_data_dict['ws'].loc[timestamp]
    llj = tools.diag_lowleveljet(ws_profile)
    temp_data_dict['llj_height'].append(llj[0])
    temp_data_dict['llj_ws'].append(llj[1])
    # wind shear analysis
    max_height_analysis_lidar = 240
    ws_shear_profile = lidar_data_dict['ws_shear'].loc[timestamp]
    ws_shear_profile = ws_shear_profile[
        ws_shear_profile.index < max_height_analysis_lidar]
    max_shear = ws_shear_profile.max()
    temp_data_dict['max_ws_shear'].append(max_shear)
    try:
        temp_data_dict['max_ws_shear_height'].append(float(ws_shear_profile[
            ws_shear_profile == max_shear].index.values))
    except TypeError:  # if error on float conversion (typically because max_shear is nan)
        temp_data_dict['max_ws_shear_height'].append(np.nan)
    # wind profile law fitting
    ws_profile = ws_profile.dropna()
    try:
        h_0 = ws_profile.index[0]
        ws_0 = ws_profile.values[0]
        # def func_powerlaw(h, alpha):
        #     return ws_0 * (h/h_0)**alpha
        func_powerlaw = lambda h, alpha: ws_0 * (h/h_0)**alpha
        params = curve_fit(func_powerlaw, 
                           np.array(ws_profile.index), np.array(ws_profile.values))
        temp_data_dict['ws_shear_exponent'].append(float(params[0]))
    except IndexError:
        temp_data_dict['ws_shear_exponent'].append(np.nan)
    # temp_data_dict['ws_shear_exponent_var'].append(params[1])

for key in temp_data_dict:
    lidar_data_dict[key] = pd.DataFrame(
        temp_data_dict[key], index=datetimes_avail_lidar)

lidar_data_dict['llj_presence'] = ~lidar_data_dict['llj_ws'].isna()
# llj_days = lidar_data_dict['llj_ws'].where(
#     lidar_data_dict['llj_ws'] == True).dropna().index
# llj_height = lidar_data_dict['llj_height'].where(
#     lidar_data_dict['llj_presence'] == True).dropna()

# %% LOAD STATION MF data

folder_path = f"{global_data_nemo}/in-situ/{station_mf}/"
prefix_file_station_mf = 'FRNOR'
filename = None  # if None, take the csv file that begins with prefix
remove_commas_by_dots = True

if filename == None:
    files = [f for f in os.listdir(folder_path) if f.startswith(
        prefix_file_station_mf) and f.endswith('.csv')]
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
data_station_mf = pd.read_csv(file_path, sep=";", dtype=str)

# Convert the 'DATE' column (second column) to datetime format
data_station_mf['DATE'] = pd.to_datetime(
    data_station_mf['DATE'], format="%Y%m%d%H")
data_station_mf.index = data_station_mf['DATE']

# Convert numeric columns to appropriate data types (assuming they are float or int)
data_station_mf[varname_list_mf] = data_station_mf[varname_list_mf].apply(
    pd.to_numeric, errors='coerce')


# %% LOAD BUOY data

# buoy = xr.open_dataset(f'{global_data_nemo}/bouee/GL_TS_MO_6100021_202307.nc')
buoy_planier = xr.open_dataset(
    f'{global_data_nemo}/bouee/GL_TS_MO_6100289.nc')     # Planier buoy
# buoy = xr.open_dataset(f'{global_data_nemo}/bouee/GL_TS_MO_6100284.nc')
# buoy = xr.open_dataset(f'{global_data_nemo}/bouee/GL_TS_MO_6100431_202303.nc')
# SST partially available on 2023
buoy_solemio = xr.open_dataset(f'{global_data_nemo}/bouee/GL_TS_MO_SOLEMIO.nc')

# sst_product = pd.read_csv(
#     f'{global_data_nemo}/sst_product/clim_lon_4.375_lat_43.375_1982-2011.csv',
#     sep=',')
# sst_product = pd.read_csv(
#     f'{global_data_nemo}/sst_product/event_lon_5.125_lat_43.125_1982-2011.csv',
#     sep=',')

buoy_planier = buoy_planier.where(
    buoy_planier.TIME > pd.Timestamp(date_start), drop=True)
buoy_planier = buoy_planier.where(
    buoy_planier.TIME < pd.Timestamp(date_end), drop=True)
sst_planier = buoy_planier['TEMP'][:, 1].to_dataframe()['TEMP']
wh_planier = buoy_planier['VAVH'][:, 0].to_dataframe()['VAVH']

buoy_solemio = buoy_solemio.where(
    buoy_solemio.TIME > pd.Timestamp(date_start), drop=True)
buoy_solemio = buoy_solemio.where(
    buoy_solemio.TIME < pd.Timestamp(date_end), drop=True)
sst_solemio = buoy_solemio['TEMP'][:, 0].to_dataframe()['TEMP']

# %% LOAD AROME profiles
ref_time_series = lidar_data_dict['tke'][140].index

arome_profiles_folder = '/home/lunelt/Data/data_NEMO/arome_profiles/'

# pattern = r"arome\.AN\.(\d{4})(\d{2})(\d{2})\.(\d{2})"  # to have all date
# pattern = r"arome\.AN\.(\d{4})(\d{2})(\d{2})\.12"  # only midday
# to have all date
pattern = r"verti_profile_arome_planier_(\d{4})(\d{2})(\d{2})\.(\d{2}).pickle"

# List to store datetime objects
datetimes_avail_arome = []
# Iterate over files in the folder
for filename in os.listdir(arome_profiles_folder):
    match = re.match(pattern, filename)
    if match:
        if len(match.groups()) == 4:
            year, month, day, hour = map(int, match.groups())
            datetimes_avail_arome.append(pd.Timestamp(
                year=year, month=month, day=day, hour=hour))
        elif len(match.groups()) == 3:
            year, month, day = map(int, match.groups())
            datetimes_avail_arome.append(pd.Timestamp(
                year=year, month=month, day=day, hour=int(pattern[-2::])))
# Sort the datetimes_avail_arome (optional, ensures chronological order)
datetimes_avail_arome.sort()
# subset the datetimes of interest
df = pd.DatetimeIndex(datetimes_avail_arome)
df_temp = df[df > date_start]
datetimes_avail_arome = df_temp[df_temp < date_end]

# Store AROME charac in a dict with list (lists are along time dim)
dict_arome = {}
# -- For variables defined all along the vertical, keep specific level value:
var_verti_arome_list = ['ws', 'wd', 'ri_bulk', 'tke']
height_arome_list = [60, 100, 140, ]
for var in var_verti_arome_list:
    for height in height_arome_list:
        dict_arome[f'arome_{var}_{height}'] = []
# -- Single value variables for a given site
# dict_arome['arome_ri_grad_surf'] = []  # ri_grad bad for stability characterization
dict_arome['arome_ri_bulk_surf'] = []
dict_arome['arome_abl_top_ri'] = []
dict_arome['arome_abl_top_tke'] = []
dict_arome['arome_abl_top_dtheta'] = []
dict_arome['arome_llj_height'] = []
dict_arome['arome_llj_ws'] = []
dict_arome['arome_max_ws_shear'] = []
dict_arome['arome_max_ws_shear_height'] = []

for wanted_date in datetimes_avail_arome:
    print(f'Arome diags: {wanted_date}')
    # wanted_date  = pd.Timestamp('2023-02-14 00:00:00')
    datetime_formatted = wanted_date.strftime('%Y%m%d.%H')
    filename_pickle = f"verti_profile_arome_{site}_{datetime_formatted}.pickle"
    verti_profile = pd.read_pickle(
        f'{arome_profiles_folder}/{filename_pickle}')

    # AROME Diags
    # vertical diag
    verti_profile['p'] = tools.height_to_pressure_std(verti_profile.index)
    verti_profile['theta'] = tools.potential_temperature_from_temperature(
        verti_profile['p'], verti_profile['t'])
    verti_profile['ws'], verti_profile['wd'] = tools.calc_ws_wd(
        verti_profile['u'], verti_profile['v'])
    verti_profile['ri_bulk'] = tools.bulk_richardson_number(
        verti_profile.index, verti_profile.theta.values,
        verti_profile.u.values, verti_profile.v.values,
        q=verti_profile.q.values)
    verti_profile['ri_grad'] = tools.gradient_richardson_number(
        verti_profile.index, verti_profile.theta.values,
        verti_profile.u.values, verti_profile.v.values,
        q=verti_profile.q.values, pres=verti_profile.p.values)
    verti_profile['ws_shear'] = pd.DataFrame(
        (np.gradient(verti_profile['ws'], axis=0) /
         np.gradient(verti_profile['ws'].index)),
        index=verti_profile['ws'].index)

    # Single value diags in dict
    # direct from profiles:
    for var in var_verti_arome_list:
        for height in height_arome_list:
            interp_val = np.interp(
                height, verti_profile.index, verti_profile[var])
            dict_arome[f'arome_{var}_{height}'].append(interp_val)

    # other diag:
    dict_arome['arome_ri_bulk_surf'].append(
        verti_profile['ri_bulk'].iloc[1])
    dict_arome['arome_abl_top_ri'].append(
        tools.abl_top_from_bulk_ri(verti_profile.index,
                                   verti_profile['ri_bulk'].values))
    dict_arome['arome_abl_top_tke'].append(
        tools.abl_top_from_tke(verti_profile.index,
                               verti_profile['tke'].values))
    dict_arome['arome_abl_top_dtheta'].append(
        tools.abl_top_from_potential_temperature(verti_profile.index,
                                                 verti_profile['theta'].values))
    llj_arome = tools.diag_lowleveljet(verti_profile['ws'], max_height=220)
    dict_arome['arome_llj_height'].append(llj_arome[0])
    dict_arome['arome_llj_ws'].append(llj_arome[1])
    # Wind shear analysis
    max_height_shear_analysis = 300
    max_shear = verti_profile['ws_shear'][
        verti_profile['ws_shear'].index < max_height_shear_analysis].max()
    dict_arome['arome_max_ws_shear'].append(max_shear)
    dict_arome['arome_max_ws_shear_height'].append(float(verti_profile['ws_shear'][
        verti_profile['ws_shear'] == max_shear].index.values))

df_arome = pd.DataFrame(dict_arome, index=datetimes_avail_arome)
df_arome['arome_llj_presence'] = ~df_arome['arome_llj_ws'].isna()


# %% Global Dataframe

# varname_list_total = varname_list_lidar + varname_list_mf + varname_list_buoy

lidar_height_wind_list = [  # between 60 and 240
    # 120,
    140,
    200,
    # 220,
    # 240
]

# Populate the dataframe with data from different sources:
df = pd.DataFrame()
for height in lidar_height_wind_list:
    df[f'tke_{height}'] = lidar_data_dict['tke'][height]
    df[f'ws_{height}'] = lidar_data_dict['ws'][height]
    df[f'wd_{height}'] = lidar_data_dict['wd'][height]
    df[f'tke_bud_diss_{height}'] = lidar_data_dict['tke_bud_diss'][height]
df['llj_ws'] = lidar_data_dict['llj_ws']
df['llj_height'] = lidar_data_dict['llj_height']
df['llj_presence'] = lidar_data_dict['llj_presence']
df['llj_presence'] = df['llj_presence'].astype(bool)
df['max_ws_shear'] = lidar_data_dict['max_ws_shear']
df['max_ws_shear_height'] = lidar_data_dict['max_ws_shear_height']
df['ws_shear_exponent'] = lidar_data_dict['ws_shear_exponent']

df['sst_solemio'] = np.interp(df.index, sst_solemio.index, sst_solemio)
df['sst_planier'] = np.interp(df.index, sst_planier.index, sst_planier)
df['wh_planier'] = np.interp(df.index, wh_planier.index, wh_planier)

df['nebul_marignane'] = np.interp(df.index, data_station_mf.index,
                                  data_station_mf['NBAS'])

for key in df_arome:
    df[f'{key}'] = np.interp(df.index, df_arome.index,
                             df_arome[f'{key}'])

corr_matrix = df.corr()

print('llj freq in obs: ',
      len(lidar_data_dict['llj_ws'].dropna())/len(lidar_data_dict['llj_ws']))
print('llj freq in arome: ',
      len(df_arome['arome_llj_ws'].dropna())/len(df_arome['arome_llj_ws']))

df.to_pickle(f'{global_data_nemo}/{site}/metocean_conditions_planier.pickle')

# %% A few quick plots / analysis

# # distribution of a variable
# plt.figure()
# plt.hist(df['max_ws_shear_height'])
# # plt.hist(df['ws_140'], bins=30)

# time series of ABL height
plt.figure()
# plt.plot(df['arome_ri_grad_140'].clip(-2,0.5), label='ri_grad_140')
plt.plot(df['arome_ri_bulk_surf'].clip(-2,0.5), label='ri_bulk_surf')
plt.plot(df['arome_ri_bulk_140'].clip(-2,0.5), label='ri_bulk_140')
plt.legend()
plt.grid()

# # percentage of time whent the ABL is below a given height
# given_height = 300
# df['arome_abl_top_tke'][df['arome_abl_top_tke'] < given_height]
# percent = len(df['arome_abl_top_tke'][df['arome_abl_top_tke']
#               < given_height])/len(df['arome_abl_top_tke'])
# print('percentage: ', percent)


# %% curve fit
# ws_profile = lidar_data_dict

ws = np.array(ws_profile.values)
h = np.array(ws_profile.index)

def func_powerlaw(h, alpha):
    return ws[0] * (h/h[0])**alpha

params = curve_fit(func_powerlaw, h, ws)
alpha = params[0]
#%%
corr = df.corr()
fig, ax = plt.subplots(figsize=(10, 10))
ax.matshow(corr,cmap='coolwarm' )
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)


# %% COMPARISON with linta & maxime work

# from LLJ_detection_functions import detect_llj

# # format the data
# ws = np.array(lidar_data_dict['ws'])
# wd = np.random.rand(lidar_data_dict['ws'].shape[0],
#                     lidar_data_dict['ws'].shape[1])
# time = np.array(lidar_data_dict['ws'].index, dtype='datetime64[ns]')
# heights = ['60','80','100','120','140','160','180','200','220','240']  # Heights in meters

# # Create the dataset
# data = {}
# data['ws'] = xr.DataArray(
#     data=ws,
#     dims=['time', 'level'],
#     coords=dict(time=(['time'], time),
#                 level=(['level'], heights)),
#     attrs=dict(description='Wind speed', units='m/s'))
# data['wd'] = xr.DataArray(
#     data=wd,
#     dims=['time', 'level'],
#     coords=dict(time=(['time'], time),
#                 level=(['level'], heights)),
#     attrs=dict(description='Wind direction', units='°'))
# ds = xr.Dataset(data_vars=data,
#                 coords=dict(time=(['time'], time),
#                             level=(['level'], heights)))

# #Generate 30-mins averaged wind speed and direction data (or we can choose time='10Min to get 10mins averaged)
# #    ds_mean=ds.rolling(time=15,center=True,min_periods=1).mean().resample(time='30Min').asfreq()
# ds_mean = ds

# #Remove the wind profiles with at least one NaN value inside
# ds_mean = ds_mean.where(ds_mean.ws.dropna(dim='time',how='any'))
# # ds_mean = ds_mean.sel(time=[
# #     pd.Timestamp('2023-01-03T16:15:00'),
# #     # pd.Timestamp('2023-01-04T03:45:00'),
# #     ])

# lljs_xarray = detect_llj(ds_mean.ws)
# lljs_linta = pd.DatetimeIndex(lljs_xarray.time)

# # compare
# lljs_perso = lidar_data_dict['llj_ws'].dropna().index

# set_perso = set(lljs_perso)
# set_linta = set(lljs_linta)

# uniques_to_linta = set_linta.difference(set_perso)
# uniques_to_perso = set_perso.difference(set_linta)

# dtime = list(uniques_to_perso)[0]
# ws_prof = lidar_data_dict['ws'].loc[pd.Timestamp(dtime)]
# print(tools.diag_lowleveljet(ws_prof))
