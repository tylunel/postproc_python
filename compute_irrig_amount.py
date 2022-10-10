#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021

Script for computing quantity of water added during irrigation, or rain.
    
"""
#import os
import numpy as np
import pandas as pd
#import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from tools import indices_of_lat_lon
import xarray as xr


#%% Parameters -----------------------------
dati = pd.Timestamp('2021-07-23 22:30')
dati_start = pd.Timestamp('2021-07-23 22:30')
dati_end = pd.Timestamp('2021-07-24 02:30')

site = 'cendrosa'

varname_obs_prefix = 'soil_moisture'   #options are: soil_moisture, soil_temp

simu_folders = {
#        'irr': '2.13_irr_2021_22-27', 
#        'std': '1.11_ECOII_2021_ecmwf_22-27'
         }

minute_data = False

start_day = 21      #first day in simu
plot_title = 'Soil moisture profile before and after irrigation at {0}'.format(site)

save_plot = True
save_folder = './figures/soil_moisture/'

#%% Automatic variable assignation:
if varname_obs_prefix == 'soil_moisture':
    xlabel = 'humidité du sol [m3/m3]'
    constant_obs = 0
    sfx_letter = 'W'    #in surfex, corresponding variables will start by this
elif varname_obs_prefix == 'soil_temp':
    xlabel = 'temperature du sol [K]'
    constant_obs = 273.15
    sfx_letter = 'T'
else:
    raise ValueError('Unknown value')
    

if site == 'cendrosa':
    lat = 41.6925905
    lon = 0.9285671
    varname_sim_suffix = 'P9'
    datafolder = \
        '/cnrm/surface/lunelt/data_LIAISE/cendrosa/30min/'
    filename_prefix = \
         'LIAISE_LA-CENDROSA_CNRM_MTO-FLUX-30MIN_L2_'
    filename_date = '2021-07-{0}_V2.nc'.format(dati.day)
    if minute_data:
        datafolder = \
            '/cnrm/surface/lunelt/data_LIAISE/cendrosa/1min/'
        filename_prefix = \
             'LIAISE_LA-CENDROSA_CNRM_MTO-1MIN_L2_'
        filename_date = '202107{0}_V1.nc'.format(dati.day)
elif site == 'preixana':
    lat = 41.59373 
    lon = 1.07250 
#    lon = 1.15000 
    alt = 'undef'
    datafolder = \
        '/cnrm/surface/lunelt/data_LIAISE/preixana/30min/'
    filename_prefix = \
        'LIAISE_PREIXANA_CNRM_MTO-FLUX-30MIN_L2_'
    filename_date = '2021-07-{0}_V2.nc'.format(dati.day)
else:
    raise ValueError('Site name not known')


#%% ---- Method 1 - integral of profile ----

#%% OBS dataset before irrig

filename_date = '2021-07-{0}_V2.nc'.format(dati_start.day)
if minute_data:
    filename_date = '202107{0}_V1.nc'.format(dati_start.day)

obs_start = xr.open_dataset(datafolder + filename_prefix + filename_date)
obs = obs_start
obs_arr_start = []
obs_depth = [-0.05, -0.1, -0.3]

for level in [1, 2, 3]:
    varname_obs = varname_obs_prefix + '_' + str(level)
    val = float(obs[varname_obs].sel(time = dati_start)) + constant_obs
    obs_arr_start.append(val)

#%% OBS dataset after irrig

filename_date = '2021-07-{0}_V2.nc'.format(dati_end.day)
if minute_data:
    filename_date = '202107{0}_V1.nc'.format(dati_end.day)

obs_end = xr.open_dataset(datafolder + filename_prefix + filename_date)
obs = obs_end
obs_arr_end = []
obs_depth = [-0.05, -0.1, -0.3]

for level in [1, 2, 3]:
    varname_obs = varname_obs_prefix + '_' + str(level)
    val = float(obs[varname_obs].sel(time = dati_end)) + constant_obs
    obs_arr_end.append(val)

#%% Water added computation
sm_diff = np.array(obs_arr_end) - np.array(obs_arr_start)
layer_thickness = np.array([0.05, 0.05, 0.2]) #if soil moisture attributed to above layer
#layer_thickness = np.array([0.075, 0.125, 0.1]) # if soil moist attributed to closest layer

#water added on the 30 first cm of soil [m]
water_added_30cm = (sm_diff*layer_thickness).sum()
print('''Quantity of water added in the 30 first cm of soil 
      without consideration of drainage [m]: ''' + str(water_added_30cm))


#%% Plots of volumetric moist profile in ground

fig = plt.figure()
ax = plt.gca()

ax.set_xlim([0, 0.5])
ax.set_ylim([-0.5, 0])

ax.set_xlabel(xlabel)
ax.set_ylabel('depth (m)')

plt.title(plot_title)
#
#for key in val_simu:
#    plt.plot(val_simu[key], sim_depth, marker='+', 
#             label='simu_{0}_d{1}h{2}'.format(key, dati.day, dati.hour))

plt.plot(obs_arr_start, obs_depth, marker='x', 
         label='obs_d{0}h{1}m{2}'.format(dati_start.day, dati_start.hour,
                     dati_start.minute),
         color='k',
         linestyle=':')
plt.plot(obs_arr_end, obs_depth, marker='x', 
         label='obs_d{0}h{1}m{2}'.format(dati_end.day, dati_end.hour,
                     dati_end.minute),
         color='k',
         linestyle='--')

plt.legend()
plt.grid()
plt.show()

#%% ---- Method 2 - via drainage ----

#obs = xr.open_dataset(datafolder + 'CAT_202107LIAISE_LA-CENDROSA_CNRM_MTO-1MIN_L2_.nc')
#
##evaluation of hydraulic flux through data
## convert volumetric moisture to water content in layer
##obs['soil_moisture_1'].data = obs['soil_moisture_1'].data*layer_thickness[0]
#obs['soil_moisture_1'].plot(label='-5cm')
##obs['soil_moisture_2'].data = obs['soil_moisture_2'].data*layer_thickness[1]
#obs['soil_moisture_2'].plot(label='-10cm')
##obs['soil_moisture_3'].data = obs['soil_moisture_3'].data*layer_thickness[2]
#obs['soil_moisture_3'].plot(label='-30cm')
#
#dati1 = pd.Timestamp('2021-07-24 02:50')
#wc1 = float(obs['soil_moisture_1'].sel(time = dati1))
#dati2 = pd.Timestamp('2021-07-24 03:14')
#wc2 = float(obs['soil_moisture_1'].sel(time = dati2))
#dt = (dati2 - dati1).seconds
## hydraulic flux [m/s]:
#dsm_dt = (wc2 - wc1)/dt
## hydraulic flux [mm/h]:
#dsm_dt_mmh = dsm_dt * 3600 * 1000
#
#ax = plt.gca()
##ax.set_ylabel('water content in layer [m]')
##ax.set_xlim([pd.Timestamp('2021-07-23 12:00'), 
##             pd.Timestamp('2021-07-24 12:00')])
#plt.grid()
#plt.legend()


#%% Save figure

if save_plot:
    filename = (plot_title)
    filename = filename.replace('=', '').replace('(', '').replace(')', '')
    filename = filename.replace(' ', '_').replace(',', '').replace('.', '_')
    filename = filename.replace('/', 'over')
    plt.savefig(save_folder+filename)

