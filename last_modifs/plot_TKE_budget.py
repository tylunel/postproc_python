#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021

Cf also plot_verti_profile.py

example for skewT graph here:
    https://unidata.github.io/MetPy/latest/examples/plots/Skew-T_Layout.html#sphx-glr-examples-plots-skew-t-layout-py
"""
#import os
#import numpy as np
import pandas as pd
#import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from tools import indices_of_lat_lon, get_simu_filename
#from metpy.plots import SkewT
#from metpy.units import units
#import metpy.calc as mpcalc
import xarray as xr


########## Independant parameters ###############
wanted_date = pd.Timestamp('20210722-2000')
site = 'cendrosa'

# variable name from MNH files: 'THT', 'RVT'
vars_simu = ['DP', 'DISS', 'TP', 'TR']

simu_list = [
            'irr', 
#             'std'
             ]

# highest level AGL plotted
toplevel = 2500

save_plot = True
save_folder = 'figures/tke_budget/{0}/'.format(site)
##################################################

if site == 'cendrosa':
    lat = 41.6925905
    lon = 0.9285671
    vminmax_tke_budget = [-0.005, 0.005]
elif site == 'elsplans':
    lat = 41.590111 
    lon = 1.029363  
    vminmax_tke_budget = [-0.02, 0.02]
else:
    raise ValueError('Site without radiosounding')


fig = plt.figure(figsize=(8, 6))

colordict = {'irr': 'g', 'std': 'orange', 'obs': 'k'}


#%% LOAD SIMU DATASET

for model in simu_list:     # model will be 'irr' or 'std'
    # retrieve and open file
    filename_simu = get_simu_filename(model, wanted_date)
    ds = xr.open_dataset(filename_simu)
    
    # find indices from lat,lon values 
    index_lat, index_lon = indices_of_lat_lon(ds, lat, lon)
    
    var_sum = None
    for var_simu in vars_simu:
        # keep only variable of interest
        var3d = ds[var_simu]
        # keep only low layer of atmos (~ABL)
        var3d_low = var3d.where(var3d.level<toplevel, drop=True)
        var1d = var3d_low[0, 1::, index_lat, index_lon] #1st index is time, 2nd is Z,..
        
        if var_sum is None:
            var_sum = var1d
        else:
            var_sum += var1d
        
        # SIMU PLOT
        plt.plot(var1d.data, var1d.level, 
                 ls='--', 
#                 color=colordict[model], 
                 label=model + '_' + var_simu
                 )
        
    plt.plot(var_sum.data, var_sum.level, 
             ls='--',
             color='k', 
             label=model + '_SUM'
             )
    
#TODO: add wind barbs
#TODO: add CAPE and CIN ?


#%%GRAPH ESTHETICS
plot_title = 'TKE budget at {0} on {1}'.format(site, wanted_date)
plt.title(plot_title)
plt.ylabel('height AGL (m)')
plt.xlabel('TKE production' + '_[' + var1d.units + ']')
plt.xlim(vminmax_tke_budget)
plt.legend()
#add vertical line at x=0
plt.axvline(x=0, linewidth=0.75)

plt.show()

#%% Save plot
if save_plot:
    filename = (plot_title)
    filename = filename.replace('=', '').replace('(', '').replace(')', '')
    filename = filename.replace(' ', '_').replace(',', '').replace('.', '_')
    plt.savefig(save_folder+filename)



#%% PLOT TKE TOTAL
#data selection
var_simu = 'TKET'
var3d = ds[var_simu]
# keep only low layer of atmos (~ABL)
var3d_low = var3d.where(var3d.level<toplevel, drop=True)
var1d = var3d_low[0, 1::, index_lat, index_lon] #1st index is time, 2nd is Z,..

# SIMU PLOT
#new figure
fig = plt.figure(figsize=(8, 6))
#plot
plt.plot(var1d.data, var1d.level, 
#         ls='--', 
         color=colordict[model], 
         label=model + '_' + var_simu
         )

# REPRESENTATIVITY OF BUDGET AT T TIME?
# Idea is to ckeck the representavity of TKE budget profiles by integrating it
# over next big timestep to see if global TKE change the right way
var_tke_extrapolat = var1d + var_sum*1800
plt.plot(var_tke_extrapolat, var1d.level, 
         ls='--', 
         color=colordict[model], 
         label=model + '_tke_extrapol'
         )

#%%GRAPH ESTHETICS
plot_title = 'TKE at {0} on {1}'.format(site, wanted_date)
plt.title(plot_title)
plt.xlim([0, 3])
plt.ylabel('height AGL (m)')
plt.xlabel('TKE' + '_[' + var1d.units + ']')
plt.legend()

#%% Save plot
if save_plot:
    filename = (plot_title)
    filename = filename.replace('=', '').replace('(', '').replace(')', '')
    filename = filename.replace(' ', '_').replace(',', '').replace('.', '_')
    plt.savefig(save_folder+filename)
   
    
