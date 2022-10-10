#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021

Last modifications
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from find_xy_from_latlon import indices_of_lat_lon
from matplotlib import cm

#%% Parameters:
    
site = 'preixana'

save_plot = False

varname_obs_prefix = 'ta_'   # options are: (last digit is the max number available)
# ta_5, hus_5, hur_5, soil_moisture_3, soil_temp_3, u_var_3, w_var_3, ... 
level = 2
varname_obs = varname_obs_prefix + str(level)
varname_plot = varname_obs
plot_title = '{0} at {1}'.format(varname_plot, site)
#plot_title = 'Temporal series \n at lat={0}, lon={1}, alt={2}'.format(
#        lat, lon, alt)

viridis = cm.get_cmap('viridis', 10)

if site == 'cendrosa':
    lat = 41.6925905
    lon = 0.9285671
    alt = 'undef'
    datafolder = \
        '/cnrm/surface/lunelt/data_LIAISE/cendrosa/30min/'
#        '/cnrm/tramm/products/LIAISE/data/AERIS/LA-CENDROSA/MTO-FLUX-30MIN/'
    filename_prefix = \
            'LIAISE_LA-CENDROSA_CNRM_MTO-FLUX-30MIN_L2_'
#        'LIAISE_LA-CENDROSA_CNRM_MTO-1MIN_L2_'   
    date = '2021-07-21_V1.nc'
    filename = filename_prefix + date
elif site == 'preixana':
    lat = 41.59373 
#    lon = 1.07250 
    lon = 1.15000 
    alt = 'undef'
    datafolder = \
        '/cnrm/surface/lunelt/data_LIAISE/preixana/30min/'
#        '/cnrm/tramm/products/LIAISE/data/AERIS/PREIXANA/MTO-FLUX-30MIN/'
    filename_prefix = \
        'LIAISE_PREIXANA_CNRM_MTO-FLUX-30MIN_L2_'
#        'LIAISE_PREIXANA_CNRM_MTO-1MIN_L2_'   
elif site == 'elsplans':
    lat = 41.590111 
    lon = 1.029363  
    alt = 'undef'
else:
    raise ValueError('Site name not known')

#in_filenames = filename_prefix + '202107*'  # use of wildcard * allowed
in_filenames = filename_prefix + '2021-07-*'  # use of wildcard * allowed
out_filename = 'CAT_2021-07_' + filename_prefix + '.nc'

#concatenate multiple days
if not os.path.exists(datafolder + out_filename):
    os.system('''
        cd {0}
        ncrcat {1} -o {2}
        '''.format(datafolder, in_filenames, out_filename))


#%% PLOT OF OBS DATA
obs = xr.open_dataset(datafolder + out_filename)

#for level in [1, 2, 3]:
for level in [2,]:
    varname_obs = varname_obs_prefix + str(level)
    (obs[varname_obs]*1+273.15).plot(label=varname_obs + 'obs', color=viridis(level*2))


#%% COMPARISON OVER MULTIPLE LAYERS

#datafolder1 = '/cnrm/surface/lunelt/NO_SAVE/nc_out/1.11_ECOII_2021_ecmwf_22-27/'
#datafolder2 = '/cnrm/surface/lunelt/NO_SAVE/nc_out/1.12_ECOII_2021_arpifs/'
#datafolder3 = '/cnrm/surface/lunelt/NO_SAVE/nc_out/1.13_ECOII_2021_arom/'
#datafolder4 = '/cnrm/surface/lunelt/NO_SAVE/nc_out/2.13_irr_2021_22-27/'
#
#datafolder = datafolder3
#
#for layer in range(2, 5): #other layer are not extrapolated, eq to layer2 or 6
#    varname_sim = 'WG{0}P9'.format(str(layer))     #options are:
#    #Q2M, T2M_ISBA, WG2P9
#    #depth of DIF layers: 0.01, 0.04, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 3, 5, 8, and 12m
#    in_filenames = 'PREP_400M_202107*.nc'  # use of wildcard allowed
#    out_filename = 'PREP.{0}.nc'.format(varname_sim)
#    
#    #if not os.path.exists(datafolder3 + out_filename):
#    #    os.system('''
#    #        cd {0}
#    #        ncecat -v {1} {2} {3}
#    #        '''.format(datafolder3, varname_sim, in_filenames, out_filename))
#    #command 'cdo -select,name={1} {2} {3}' may work as well, but not always...
#        
#    if not os.path.exists(datafolder + out_filename):
#        os.system('''
#            cd {0}
#            ncecat -v {1} {2} {3}
#            '''.format(datafolder, varname_sim, in_filenames, out_filename))
##command 'cdo -select,name={1} {2} {3}' may work as well, but not always...
#    
#    #% load dataset and set parameters
#    ds1 = xr.open_dataset(datafolder + out_filename)
#    
#    # find indices from lat,lon values 
#    index_lat, index_lon = indices_of_lat_lon(ds1, lat, lon)
#    print(index_lat, index_lon)
#    
#    var4d = ds1[varname_sim]
#    
#    # Set time abscisse axis
#    start = np.datetime64('2021-07-21T01:00')
#    dati_arr = np.array([start + np.timedelta64(i*6, 'h') for i in np.arange(0, var4d.shape[0])])
#    
#    # PLOT d1
#    
#    #var_1d = var4d.data[:, 2,index_lat, index_lon] #1st index is time, 2nd is Z,..
#    var_1d = var4d.data[:, index_lat, index_lon]
#    
#    plt.plot(dati_arr, var_1d, label='prep_cep_layer'+str(layer), 
#             linestyle="dashed", color=(viridis(layer)))
#    
#
##fig = plt.figure()
#ax = plt.gca()
#ax.set_ylabel(varname_plot)
##ax.set_xlim([np.min(obs.time), np.max(obs.time)])
#ax.set_xlim([np.datetime64('2021-07-20T01:00'), 
#             np.datetime64('2021-07-27T01:00')])
#
#
##ax.set_xlim([np.min(var_1d), np.max(var_1d)])
##ax.set_ylabel('Height AGL (m)')
##ax.set_ylim([0, 0.4])
#plt.title(plot_title)
#plt.legend()
#
##plt.plot(dati_arr, var_1d, label='prep_cep_layer'+str(layer))

#%% COMPARISON OVER DIFF MODELS
    
datafolders = [
        '/cnrm/surface/lunelt/NO_SAVE/nc_out/1.11_ECOII_2021_ecmwf_22-27/',
        '/cnrm/surface/lunelt/NO_SAVE/nc_out/1.12_ECOII_2021_arpifs/',
        '/cnrm/surface/lunelt/NO_SAVE/nc_out/1.13_ECOII_2021_arom/']
modelnames = ['cep', 'arp', 'aro']

#options are:
#Q2M, WG2P9
#depth DIF layers: 0.01, 0.04, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 3, 5, 8, 12m
varprefix = 'THT'    #'WG', 'TG', 'Q2M'...
layer = ''
patch = ''    #'P9', P4', '_ISBA'

for i, datafolder in enumerate(datafolders):
    varname_sim = (varprefix + str(layer) + patch)      
    if modelnames[i] == 'aro':
        in_filenames = 'PREP_400M_*.nc'  # use of wildcard allowed
    else: 
        in_filenames = 'PREP_2KM_202107*.nc'  # use of wildcard allowed
    out_filename = 'PREP_{0}.nc'.format(varname_sim)
        
    if not os.path.exists(datafolder + out_filename):
        os.system('''
            cd {0}
            ncecat -v {1} {2} {3}
            '''.format(datafolder, varname_sim, in_filenames, out_filename))
#command 'cdo -select,name={1} {2} {3}' may work as well, but not always...
    
    #% load dataset and set parameters
    ds1 = xr.open_dataset(datafolder + out_filename)
    
    # find indices from lat,lon values 
    index_lat, index_lon = indices_of_lat_lon(ds1, lat, lon)
    print(index_lat, index_lon)
    
    var_md = ds1[varname_sim]
    
    # Set time abscisse axis
    start = np.datetime64('2021-07-21T01:00')
    dati_arr = np.array([start + np.timedelta64(i*6, 'h') for i in np.arange(0, var_md.shape[0])])
    
    # PLOT d1
    
    if len(var_md.shape) == 5:
        ilevel = 1   #0 is Halo, 1->2m, 2->6.12m, 3->10.49m
        var_1d = var_md.data[:, :, ilevel ,index_lat, index_lon] #1st index is time, 2nd is ?, 3rd is Z,..
    elif len(var_md.shape) == 4:
        ilevel = 1   #0 is Halo, 1->2m, 2->6.12m, 3->10.49m
        var_1d = var_md.data[:, ilevel ,index_lat, index_lon] #1st index is time, 2nd is Z,..
    elif len(var_md.shape) == 3:
        var_1d = var_md.data[:, index_lat, index_lon]
    
    plt.plot(dati_arr, var_1d, label='prep_{0}_layer{1}'.format(
            modelnames[i], layer), 
             linestyle="dashed", 
#             color=(viridis(layer))
             )
    

#fig = plt.figure()
ax = plt.gca()
ax.set_ylabel(varname_plot)
#ax.set_xlim([np.min(obs.time), np.max(obs.time)])
ax.set_xlim([np.datetime64('2021-07-20T01:00'), 
             np.datetime64('2021-07-27T01:00')])


#ax.set_xlim([np.min(var_1d), np.max(var_1d)])
#ax.set_ylabel('Height AGL (m)')
#ax.set_ylim([0, 0.4])
plt.title(plot_title)
plt.legend()

#plt.plot(dati_arr, var_1d, label='prep_cep_layer'+str(layer))

#%% load dataset 2
#ds2 = xr.open_dataset(datafolder3 + out_filename)
##datetime = ds.time.values[0]
#
#var4d2 = ds2[varname_sim]
## Set time abscisse axis
#start = np.datetime64('2021-07-21T01:00')
#dati_arr = np.array([start + np.timedelta64(i, 'h') for i in np.arange(0, var4d2.shape[0])])
#
## PLOT d2
#
##var_1d = var4d.data[:, 2,index_lat, index_lon] #1st index is time, 2nd is Z,..
#var_1d2 = var4d2.data[:, index_lat, index_lon]
#
#plt.plot(dati_arr, var_1d2,label='irrig')


#%%


if save_plot:
    filename = (plot_title + ' for ' + varname_plot)
    filename = filename.replace('=', '').replace('(', '').replace(')', '')
    filename = filename.replace(' ', '_').replace(',', '').replace('.', '_')
    plt.savefig(filename)






