#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021

Script for comparing obs and simulation in the soil.

Fonctionnement:
    Seule première section a besoin d'être remplie, le reste est automatique.
    
"""
#import os
import numpy as np
import pandas as pd
#import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from find_xy_from_latlon import indices_of_lat_lon
import xarray as xr


##-------------------------------------------------------------
dati = pd.Timestamp('2021-07-24 01:00')
     #first day of simulation

site = 'cendrosa'

varname_obs_prefix = 'soil_moisture'   #options are: soil_moisture, soil_temp

simu_folders = {
#        'irr': '2.13_irr_2021_22-27', 
#        'std': '1.11_ECOII_2021_ecmwf_22-27'
         }

start_day = 21      #first day in simu
plot_title = 'Ground profile at {0} on {1}'.format(site, dati)
save_plot = False

#%%------------------
#Automatic variable assignation:
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
    alt = 'undef'
    datafolder = \
        '/cnrm/surface/lunelt/data_LIAISE/cendrosa_50m/30min/'
    filename_prefix = \
            'LIAISE_LA-CENDROSA_CNRM_MTO-FLUX-30MIN_L2_'
    filename_date = '2021-07-{0}_V2.nc'.format(dati.day)
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

#%% OBS dataset

obs = xr.open_dataset(datafolder + filename_prefix + filename_date)
obs_arr = []
obs_depth = [-0.05, -0.1, -0.3]

for level in [1, 2, 3]:
    varname_obs = varname_obs_prefix + '_' + str(level)
    val = float(obs[varname_obs].sel(time = dati)) + constant_obs
    obs_arr.append(val)

#%% SIMU datasets 

cisba = 'dif'

if cisba == 'dif':      # if CISBA = DIF in simu
    nb_layer = 14
    sim_depth = [-0.01, -0.04, -0.1, -0.2, -0.4, -0.6, -0.8, 
                 -1, -1.5, -2, -3, -5, -8, -12]     # in meters

val_simu = {}
for key in simu_folders:
    ds = xr.open_dataset(
        '/cnrm/surface/lunelt/NO_SAVE/nc_out/{0}/LIAIS.2.SEG{1}.001.nc'.format(
                simu_folders[key], (dati.day - start_day)*24 + dati.hour))

    index_lat, index_lon = indices_of_lat_lon(ds, lat, lon)

    val_simu[key] = []
    for level in range(1, nb_layer+1):
        var2d = ds['{0}G{1}P9'.format(sfx_letter, str(level))]
        val = var2d.data[index_lat, index_lon]
        if val == 999:
            val = np.NaN
        val_simu[key].append(val)
    

#%% PLOTs

#fig = plt.figure()
ax = plt.gca()

ax.set_xlim([0, 0.5])
ax.set_ylim([-0.5, 0])

ax.set_xlabel(xlabel)
ax.set_ylabel('depth (m)')

plt.title(plot_title)

for key in val_simu:
    plt.plot(val_simu[key], sim_depth, marker='+', 
             label='simu_{0}_d{1}h{2}'.format(key, dati.day, dati.hour))

plt.plot(obs_arr, obs_depth, marker='x', 
         label='obs_d{0}h{1}'.format(dati.day, dati.hour))
plt.legend()
plt.grid()
#plt.show()

if save_plot:
    filename = (plot_title + ' for ' + xlabel)
    filename = filename.replace('=', '').replace('(', '').replace(')', '')
    filename = filename.replace(' ', '_').replace(',', '').replace('.', '_')
    plt.savefig(filename)

