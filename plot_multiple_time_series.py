#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021

Fonctionnement:
    Seule premier bloc peut être rempli, le reste est automatique.
    
Noms des variables d'obs dispos: (last digit is the max number available)
ta_5, hus_5, hur_5, soil_moisture_3, soil_temp_3, u_var_3, w_var_3, swd,
lmon_3, u_star_3, 


"""
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from tools import indices_of_lat_lon 


#%% Independant Parameters (TO FILL IN):
    
site = 'cendrosa'

minute_data = False

date = '2021-07'
if minute_data:
    date = '202107'

save_plot = False

varname_obs_list = ['ta_2', 'hur_2']
colormap = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
linestylemap = ['-', '-', '--', '--', '-.', '-.', ':']


#%% Dependant Parameters

#if varname_obs in ['soil_moisture_1', 'soil_moisture_2', 'soil_moisture_3']:
#    ylabel = 'soil moisture [m3/m3]'
#    constant_obs = 0
#    coeff_obs = 1
#    varname_sim_prefix = 'WG3'    #corresponding variables in SFX
#    #other options: 'WG3_ISBA', 'WG4P9'
#    #N.B.: layers depth for diff: 
#    #    [-0.01, -0.04, -0.1, -0.2, -0.4, -0.6,
#    #     -0.8, -1, -1.5, -2, -3, -5, -8, -12]
#elif varname_obs in ['soil_temp_1', 'soil_temp_2', 'soil_temp_3']:
#    ylabel = 'soil temperature [K]'
#    constant_obs = 273.15
#    coeff_obs = 1
#    varname_sim_prefix = 'TG2'
#elif varname_obs in ['swd']:
#    ylabel = 'shortwave downward radiation [W/m2]'
#    constant_obs = 0
#    coeff_obs = 1
#    varname_sim_prefix = ''
#elif varname_obs in ['lmon_1', 'lmon_2', 'lmon_3']:
#    ylabel = 'monin-obukhov length [m]'
#    constant_obs = 0
#    coeff_obs = 1
#    varname_sim_prefix = ''
#else:
#    raise ValueError('Unknown value')


if site == 'cendrosa':
    lat = 41.6925905
    lon = 0.9285671
    varname_sim_suffix = 'P9'
    datafolder = \
        '/cnrm/surface/lunelt/data_LIAISE/cendrosa/30min/'
    filename_prefix = \
         'LIAISE_LA-CENDROSA_CNRM_MTO-FLUX-30MIN_L2_'
    if minute_data:
        datafolder = \
            '/cnrm/surface/lunelt/data_LIAISE/cendrosa/1min/'
        filename_prefix = \
             'LIAISE_LA-CENDROSA_CNRM_MTO-1MIN_L2_'
    in_filenames = filename_prefix + date
elif site == 'preixana':
    lat = 41.59373 
    lon = 1.07250
    varname_sim_suffix = 'P2'
    datafolder = \
        '/cnrm/surface/lunelt/data_LIAISE/preixana/30min/'
    filename_prefix = \
        'LIAISE_PREIXANA_CNRM_MTO-FLUX-30MIN_L2_'
    in_filenames = filename_prefix + date
elif site == 'elsplans':
    lat = 41.590111 
    lon = 1.029363  
    varname_sim_suffix = '_ISBA'
else:
    raise ValueError('Site name not known')

#varname_sim = varname_sim_prefix + varname_sim_suffix


#%% OBS - LOAD DATA

out_filename = 'CAT_' + date + filename_prefix + '.nc'

# Concatenate multiple days
if not os.path.exists(datafolder + out_filename):
    os.system('''
        cd {0}
        ncrcat {1}* -o {2}
        '''.format(datafolder, in_filenames, out_filename))

# Plot:
obs = xr.open_dataset(datafolder + out_filename)
#(obs[varname_obs]*coeff_obs).plot(label='obs_{0}'.format(varname_obs))

#%% PLOT multiple

for i, varname in enumerate(varname_obs_list):
    lv = len(varname_obs_list)    
    twins = {}
    plots = {}
    #tick_kw = dict(size=4, width=1.5)  #tick properties
    tick_kw = {}
    
    if i == 0:
        fig, ax = plt.subplots(figsize=(8 + 1*lv, 6))
        fig.subplots_adjust(right=1 - 0.1*lv)
        plots[i], = obs[varname].plot(ax=ax, color=colormap[i])
        ax.yaxis.label.set_color(plots[i].get_color())
        ax.tick_params(axis='y', colors=plots[i].get_color(), **tick_kw)
    else:
        twins[i] = ax.twinx()
        yaxe_position = 0.8 + 0.2*i
        twins[i].spines['right'].set_position(("axes", yaxe_position))
        plots[i], = obs[varname].plot(ax=twins[i], 
             color=colormap[i],
             linestyle=linestylemap[i]
             )
        twins[i].yaxis.label.set_color(plots[i].get_color())
        twins[i].tick_params(axis='y', colors=plots[i].get_color(), **tick_kw)

plt.grid()
plt.show()

#%% SIMU:
#
#simu_folders = {
##        'irr': '2.13_irr_2021_22-27/', 
##        'std': '1.11_ECOII_2021_ecmwf_22-27/'
#         }
#father_folder = '/cnrm/surface/lunelt/NO_SAVE/nc_out/'
#in_filenames = 'LIAIS.2.SEG*.001.nc'  # use of wildcard allowed
#out_filename = 'LIAIS.2.{0}.nc'.format(varname_sim)
#
#for key in simu_folders:
#    datafolder = father_folder + simu_folders[key]
#    if not os.path.exists(datafolder + out_filename):
#        os.system('''
#            cd {0}
#            ncecat -v {1} {2} {3}
#            '''.format(datafolder, varname_sim, in_filenames, out_filename))
#    #command 'cdo -select,name={1} {2} {3}' may work as well, but not always...
#
#    ds1 = xr.open_dataset(datafolder + out_filename)
#    
#    # find indices from lat,lon values 
#    index_lat, index_lon = indices_of_lat_lon(ds1, lat, lon)
#    
#    var3d = ds1[varname_sim]
#    
#    # Set time abscisse axis
#    start = np.datetime64('2021-07-21T01:00')
#    dati_arr = np.array([start + np.timedelta64(i, 'h') for i in np.arange(0, var3d.shape[0])])
#    
#    # PLOT d1
#    #var_1d = var4d.data[:, 2,index_lat, index_lon] #1st index is time, 2nd is Z,..
#    var_1d = var3d.data[:, index_lat, index_lon]
#    
#    #fig = plt.figure()
#    ax = plt.gca()
#    ax.set_ylabel(ylabel)
#    #ax.set_ylim([0, 0.4])
#    
##    ax.set_xlim([np.min(obs.time), np.max(obs.time)])
#    ax.set_xlim([np.min(dati_arr), np.max(dati_arr)])
#    
#    plt.plot(dati_arr, var_1d, 
#             label='simu_{0}'.format(key))


#plot_title = '{0} at {1}'.format(ylabel, site)
##plot_title = 'Temporal series \n at lat={0}, lon={1}, alt={2}'.format(
##        lat, lon, alt)
#
#plt.title(plot_title)
#plt.legend()
#plt.grid()


#%% Save figure

if save_plot:
    filename = (plot_title + ' for ' + ylabel)
    filename = filename.replace('=', '').replace('(', '').replace(')', '')
    filename = filename.replace(' ', '_').replace(',', '').replace('.', '_')
    plt.savefig(filename)
