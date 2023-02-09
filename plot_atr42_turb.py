#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:55:16 2022

@author: Guylaine Canut, lunelt
"""
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import xarray as xr
import tools


########## Independant parameters ###############

foldername = '/cnrm/surface/lunelt/data_LIAISE/turb_atr42_raw/'
filename = foldername + 'LIAISE_ATR_turbulent_moments_20210721_RF05_L2_v1.0.nc'

# Simulation to show: 'irr' or 'std'
model = 'irr_d1'
# Datetime - UTC
date = '20210721-1400'
# Simu variable to show
var_name = 'TKET'
# Measure site
site_list = ['BL2', 'BL1']

# maximum level (height AGL) to plot
toplevel = 3000

save_plot = False
save_folder = 'figures/turb_atr42/'
##################################################


ds_obs = xr.open_dataset(filename) # perso lunelt

ds_BL1 = ds_obs.where(ds_obs['legtype']=='BL1', drop=True)
ds_BL2 = ds_obs.where(ds_obs['legtype']=='BL2', drop=True)

#%%

plt.plot(ds_BL1['TKE'],
         ds_BL1['alt'],
         'o', color='green', label='obs_atr42_BL1')
plt.plot(ds_BL2['TKE'],
         ds_BL2['alt'],
         'o', color='orange', label='obs_atr42_BL2')

    
    
#%% SIMU

#TODO: faire boucle pour print de 15h à la cendrosa et pour 13h à elsplans (meilleurs résultats, mais légère triche)


for site in site_list:
    # Dependant parameters
    filename = tools.get_simu_filename(model, date)
    
    #% load dataset and set parameters
    ds = xr.open_dataset(filename)
    #datetime = pd.DatetimeIndex([ds.time.values[0]])
    datetime = pd.Timestamp(date)
    
    var3d = ds[var_name]
    
    
    
    if site in ['BL1', 'BL2']:
        if site == 'BL2':    # line BL2 - arid
            start_pt = (41.41, 0.84)
            end_pt = (41.705, 1.26)
        elif site == 'BL1':  # line BL1 - humid zone
            start_pt = (41.53, 0.73)
            end_pt = (41.73, 1.04)
        
        line_dict = tools.line_coords(ds, start_pt, end_pt, 
                              nb_indices_exterior=0)
        ni_line = line_dict['ni_range']
        nj_line = line_dict['nj_range']
        
        section = []
        abscisse_coords = []
        abscisse_sites = {}
        
        for i, ni in enumerate(ni_line):
            print(i)
            #interpolation of all variables on ni_range
            profile = var3d.interp(ni=ni, nj=nj_line[i]).expand_dims({'i_sect':[i]})
            section.append(profile)
            
            #store values of lat-lon for the horiz axis
            lat = np.round(profile.latitude.values, decimals=3)
            lon = np.round(profile.longitude.values, decimals=3)
            latlon = str(lat) + '\n' + str(lon)
            abscisse_coords.append(latlon)
        
        #concatenation of all profile in order to create the 2D section dataset
        section_ds = xr.concat(section, dim="i_sect")
        section_ds_low = section_ds.where(section_ds.level<toplevel, drop=True)
        
        var1d = section_ds_low.mean(dim='i_sect').squeeze()
        h_agl = section_ds_low.level
    
    if site in ['cendrosa', 'elsplans', 'preixana']:
        if site == 'cendrosa':
            lat = 41.6925905
            lon = 0.9285671
        elif site == 'preixana':
            lat = 41.59373 
            lon = 1.07250
        elif site == 'elsplans':
            lat = 41.590111 
            lon = 1.029363  
        # find indices from lat,lon values of site
        index_lat, index_lon = tools.indices_of_lat_lon(ds, lat, lon)
        # keep only low layer of atmos (~ABL)
        var3d_low = var3d.where(var3d.level<toplevel, drop=True)
        var1d = var3d_low.data[0, :, index_lat, index_lon] #1st index is time, 2nd is Z,..
        h_agl = var3d_low.level
    
    
    if site in ['cendrosa', 'BL1']:
        col = 'g'
    elif site in ['elsplans', 'preixana', 'BL2']:
        col = 'orange'
    
    #% PLOT of SIMU
    
    plt.plot(var1d, h_agl, 
    #         label = '{0}h_{1}_{2}'.format(str(datetime.hour), site, model),
             label = 'simu_{0}'.format(site),
             color = col)
    #fig = plt.figure()
    ax = plt.gca()
    #
    ax.set_xlabel(var3d.long_name + '['+var3d.units+']')
    #ax.set_xlim([np.min(var1d), np.max(var1d)])
    
    ax.set_ylabel('Height AGL (m)')
    ax.set_ylim([-10, toplevel])
    plot_title = 'Turbulence over humid and arid zone \n on {0}'.format(
                                          datetime)
    plt.title(plot_title)
    plt.legend()

#%%
if save_plot:
    tools.save_figure(plot_title, save_folder)