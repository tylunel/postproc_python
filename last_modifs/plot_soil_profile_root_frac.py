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
import matplotlib.pyplot as plt
import tools
import xarray as xr
import global_variables as gv


##########################################
wanted_date = '20210715-1500'

site = 'cendrosa'

varname_obs_prefix = 'ROOT_FRC'   #options are: soil_moisture, soil_temp

models = [
#        'irr_d2_old', 
#        'std_d2_old',
#        'irr_d2', 
#        'std_d2', 
#        'irr_d1', 
#        'std_d1',
        'lagrip100_d1',
         ]

plot_title = 'Ground profile at {0} on {1}'.format(site, wanted_date)
save_plot = False
save_folder = './figures/soil_profiles/{0}/'.format(site)

wilt_and_fc = True
root_frac = True

########################################

simu_folders = {key:gv.simu_folders[key] for key in models}
father_folder = gv.global_simu_folder


colordict = {'irr_d2': 'g', 
             'std_d2': 'r',
             'irr_d1': 'g--', 
             'lagrip100_d1': 'b--',
             'std_d1': 'r--', 
             'irr_d2_old': 'g:', 
             'std_d2_old': 'r:', 
             'obs': 'k'}

lat = gv.sites[site]['lat']
lon = gv.sites[site]['lon']

#Automatic variable assignation:
if varname_obs_prefix == 'soil_moisture':
    xlabel = 'volumetric soil moisture [m3/m3]'
    constant_obs = 0
    sfx_letter = 'W'    #in surfex, corresponding variables will start by this
elif varname_obs_prefix == 'soil_temp':
    xlabel = 'soil temperature [K]'
    constant_obs = 273.15
    sfx_letter = 'T'
else:
    raise ValueError('Unknown value')
    

#%% OBS dataset

#obs = xr.open_dataset(datafolder + filename_prefix + filename_date)
#obs_arr = []
#obs_depth = [-0.05, -0.1, -0.3]
#
#for level in [1, 2, 3]:
#    varname_obs = varname_obs_prefix + '_' + str(level)
#    val = float(obs[varname_obs].sel(time = dati)) + constant_obs
#    obs_arr.append(val)
#
#plt.plot(obs_arr, obs_depth, marker='x', 
#         label='obs_d{0}h{1}'.format(dati.day, dati.hour))

#%% SIMU datasets 

cisba = 'dif'

if cisba == 'dif':      # if CISBA = DIF in simu
    nb_layer = 14
    sim_depth = [-0.01, -0.04, -0.1, -0.2, -0.4, -0.6, -0.8, 
                 -1, -1.5, -2, -3, -5, -8, -12]     # in meters

val_simu = {}

for model in simu_folders:
    datafolder = father_folder + simu_folders[model]
    filename = tools.get_simu_filename(model, wanted_date)
    
    # load dataset, default datetime okay as pgd vars are all the same along time
    ds = xr.open_dataset(filename)
    
    index_lat, index_lon = tools.indices_of_lat_lon(ds, lat, lon)
    
    val_simu[model] = []
    val_simu[model+'_wilt'] = []
    val_simu[model+'_fc'] = []
    val_simu[model+'_root'] = []
    
    for level in range(1, nb_layer+1):
        var2d = ds['{0}G{1}P9'.format(sfx_letter, level)]
        val = var2d.data[index_lat, index_lon]
        if val == 999:
            val = np.NaN
        val_simu[model].append(val)
        #get wilting pt and field capa
        if wilt_and_fc:
            wilt_pt = ds['WWILT{0}'.format(level)][index_lat, index_lon]
            val_simu[model+'_wilt'].append(float(wilt_pt))
            fc = ds['WFC{0}'.format(level)][index_lat, index_lon]
            val_simu[model+'_fc'].append(float(fc))
        if root_frac:
            root_frac = ds['ROOTFRAC{0}P9'.format(level)][index_lat, index_lon]
            if root_frac == 999:
                root_frac = np.nan
            val_simu[model+'_root'].append(float(root_frac))
            
    

#%% PLOTs

fig = plt.figure()

plt.title(plot_title)

for model in simu_folders:
    plt.plot(val_simu[model], sim_depth, marker='+', 
             label='simu_{0}_{1}'.format(model, wanted_date))
    if wilt_and_fc:
#        plt.vlines()
        plt.plot(val_simu[model+'_wilt'], sim_depth, marker='+', 
                 label='simu_{0}_wilt'.format(model))
        plt.plot(val_simu[model+'_fc'], sim_depth, marker='+', 
                 label='simu_{0}_fc'.format(model))
    if root_frac:
        #cumulated root frac
        plt.plot(val_simu[model+'_root'], sim_depth, marker='+',
                 linestyle='--',
                 color='k',
                 label='simu_{0}_root'.format(model))
        
        #density of roots
#        cumulated_root_frac = [0] + val_simu['lagrip100_d1_root']
#        sim_layer_thickness = [0] + sim_depth
#        
#        root_distribution = np.diff(cumulated_root_frac)  # in %/layer
#        normalized_root_density = root_distribution/(np.diff(sim_layer_thickness)*(-1))  # in %/cm
#        plt.plot(normalized_root_density, sim_depth, marker='+', 
#                 label='simu_{0}_root_density'.format(model))


ax = plt.gca()

#ax.set_xlim([0, 0.5])
ax.set_ylim([-2, 0.5])

ax.set_xlabel(xlabel)
ax.set_ylabel('depth (m)')
plt.legend()
plt.grid()
#plt.show()

if save_plot:
    tools.save_figure(plot_title + ' for ' + xlabel, save_folder)

