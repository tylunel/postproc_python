#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:55:16 2022

@author: Guylaine Canut, lunelt
"""
import numpy as np
import pandas as pd
import netCDF4
import datetime
import matplotlib.pyplot as plt
#import glob
import xarray as xr
from tools import indices_of_lat_lon, get_simu_filename, line_coords


##############################lecture avion fichier .nc pour DB
#############################creation dataframe pour tous les vols
foldername = '/cnrm/surface/lunelt/data_LIAISE/turb_atr42_raw/'
filename = foldername + 'LIAISE_ATR_turbulent_moments_20210721_RF05_L2_v1.0.nc'

#ListFiles = np.sort(glob.glob(foldername + '*.nc'))
ListFiles = [filename,]
print(ListFiles)

datefile = [i[-24:-16] for i in ListFiles]
nbreflight = len(datefile)

ds_obs = xr.open_dataset(filename) # perso lunelt

data_atr= pd.DataFrame([])

nc=netCDF4.Dataset(filename, "r")
datevar = netCDF4.num2date(nc.variables['time'][:],nc.variables['time'].getncattr('units'))
# /!\ horaire en heure local et non pas UTC !!
datet = [datetime.datetime(i.year, i.month, i.day, i.hour, i.minute, i.second) for i in datevar]
df = pd.DataFrame()
liste_var = nc.variables.keys()
for p in liste_var :
    var = nc.variables[p]
    print(p)
        # on recupere la variable


    ts = pd.Series( var[:], index = datet )
    df[p] = ts
print(df)
data_atr = data_atr.append(df)
    
#data_atr = ds
    
#ds_df = ds.to_dataframe()
#ds_df == data_atr

#%%
#################plot COVARIANCE

zi_BL1=[
#        750,600,450,1750,300,
        1700,
#        1250,600
        ]
zi_BL2=[
#        700,700,650,1350,400,
        1750,
#        1200,1500
        ]

data_atr_BL1 = data_atr[data_atr['legtype']=='BL1']
data_atr_BL2 = data_atr[data_atr['legtype']=='BL2']
datefile_atr = [
#        '20210715','20210716','20210717','20210720','20210721',
        '20210722',
#        '20210727', '20210728'
        ]

ds_BL1 = ds_obs.where(ds_obs['legtype']=='BL1', drop=True)
ds_BL2 = ds_obs.where(ds_obs['legtype']=='BL2', drop=True)

#%%
for j, elt in enumerate(datefile_atr):
#    covariance=plt.figure('variance'+str(elt),
#                          figsize=(8, 8))
#    axa=covariance.add_subplot(1,1,1)
#    axa.set_title(str(elt))
    
    ## TKE
#    axb=covariance.add_subplot(1,1,1)    
    # 
    plt.plot(ds_BL1['TKE'],
             ds_BL1['alt'],
             'o', color='green', label='obs_atr42_BL1')
    plt.plot(ds_BL2['TKE'],
             ds_BL2['alt'],
             'o', color='orange', label='obs_atr42_BL2')
#    axb.plot(data_atr_BL2[elt]['TKE'],
#             data_atr_BL2[elt]['alt'],
#             'o', color='orange', label='ATR42 arid')
#    axb.set_ylim(0,3000)
    # add y-axis
#    axb.plot([0,0], [0,3000],'k',linewidth=3)
    # add lines of zi (eq. ABL height)
#    axb.plot([-0.1, np.max(data_atr_BL2[elt]['TKE'])], 
#              [zi_BL2[j], zi_BL2[j]],
#             '--',color='orange')
#    axb.plot([-0.1, np.max(data_atr_BL1[elt]['TKE'])], 
#              [zi_BL1[j], zi_BL1[j]],
#             '--',color='green')    
    # set labels
#    axb.set_ylabel('Altitude [m]')
#    axb.set_xlabel('TKE [m²/s²]') 
    
    
    
#%% SIMU

#TODO: faire boucle pour print de 15h à la cendrosa et pour 13h à elsplans (meilleurs résultats, mais légère triche)

    
########## Independant parameters ###############
# Simulation to show: 'irr' or 'std'
model = 'irr'
# Datetime - UTC
date = '20210721-1400'
# Simu variable to show
var_name = 'TKET'
# Measure site
site_list = ['BL2', 'BL1']

# maximum level (height AGL) to plot
toplevel = 3000

save_plot = False
##################################################

for site in site_list:
    # Dependant parameters
    filename = get_simu_filename(model, date)
    
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
        
        ni_line, nj_line = line_coords(ds, start_pt, end_pt, 
                                       nb_indices_exterior=0)
        
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
        index_lat, index_lon = indices_of_lat_lon(ds, lat, lon)
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


if save_plot:
    filename = (plot_title + ' for ' + var_name)
    filename = filename.replace('=', '').replace('(', '').replace(')', '')
    filename = filename.replace(' ', '_').replace(',', '').replace('.', '_')
    plt.savefig(save_folder+filename)