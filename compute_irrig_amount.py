#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021

Script for computing quantity of water added during irrigation, or rain.
    
"""
#import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tools
import xarray as xr
import global_variables as gv


#%% Parameters -----------------------------
dati = pd.Timestamp('2021-07-23 22:30')
irr_dati_pre = pd.Timestamp('2021-07-23 22:30')
irr_dati_post = pd.Timestamp('2021-07-24 02:30')

site = 'cendrosa'

varname_obs_prefix = 'soil_moisture'   #options are: soil_moisture, soil_temp

simu_folders = {
#        'irr': '2.13_irr_2021_22-27', 
#        'std': '1.11_ECOII_2021_ecmwf_22-27'
         }


save_plot = True
save_folder = './figures/soil_moisture/'

#%% Automatic variable assignation:
if varname_obs_prefix == 'soil_moisture':
    xlabel = 'Volumetric soil water content [m$^{3}$ m$^{-3}$]'
    constant_obs = 0
    sfx_letter = 'W'    #in surfex, corresponding variables will start by this
elif varname_obs_prefix == 'soil_temp':
    xlabel = 'soil temperature [K]'
    constant_obs = 273.15
    sfx_letter = 'T'
else:
    raise ValueError('Unknown value')
    

if site == 'cendrosa':
    datafolder = \
        '/cnrm/surface/lunelt/data_LIAISE/cendrosa/30min/'
    filename_prefix = \
         'LIAISE_LA-CENDROSA_CNRM_MTO-FLUX-30MIN_L2_'
#    filename_date = '2021-07-{0}_V2.nc'.format(dati.day)
elif site == 'preixana':
    datafolder = \
        '/cnrm/surface/lunelt/data_LIAISE/preixana/30min/'
    filename_prefix = \
        'LIAISE_PREIXANA_CNRM_MTO-FLUX-30MIN_L2_'
#    filename_date = '2021-07-{0}_V2.nc'.format(dati.day)
elif site == 'irta-corn':
    datafolder = \
        '/cnrm/surface/lunelt/data_LIAISE/irta-corn/seb/'
    filename_prefix = \
         'LIAISE_IRTA-CORN_UIB_SEB-10MIN_L2.nc'
else:
    raise ValueError('Site name not known')


#%% ---- Method 1 - integral of profile ----

#%% OBS dataset before irrig
obs_arr_start = []
    
if site in ['cendrosa', 'preixana']:
    filename_date = '2021-07-{0}_V2.nc'.format(irr_dati_pre.day)
    obs = xr.open_dataset(datafolder + filename_prefix + filename_date)
    obs_depth = [-0.05, -0.1, -0.3]
elif site == 'irta-corn':
    obs = xr.open_dataset(datafolder + filename_prefix)
    irr_dati = tools.get_irrig_time(obs.VWC_40cm_Avg)[2]
    irr_dati_pre = irr_dati - pd.Timedelta(1,'h')
    obs_depth = [-0.05, -0.15, -0.25, -0.35, -0.45]

for i, level in enumerate(obs_depth):
    if site in ['cendrosa', 'preixana']:
        varname_obs = varname_obs_prefix + '_' + str(i+1)
    elif site == 'irta-corn':
        varname_obs = 'VWC_{0}0cm_Avg'.format(i+1)
    val = float(obs[varname_obs].sel(time = irr_dati_pre)) + constant_obs
    obs_arr_start.append(val)

#%% OBS dataset after irrig

obs_arr_end = []
    
if site in ['cendrosa', 'preixana']:
    filename_date = '2021-07-{0}_V2.nc'.format(irr_dati_post.day)
    obs = xr.open_dataset(datafolder + filename_prefix + filename_date)
    obs_depth = [-0.05, -0.1, -0.3]
elif site == 'irta-corn':
    obs = xr.open_dataset(datafolder + filename_prefix)
    irr_dati = tools.get_irrig_time(obs.VWC_40cm_Avg)[2]
    irr_dati_post = irr_dati + pd.Timedelta(2,'h')
    obs_depth = [-0.05, -0.15, -0.25, -0.35, -0.45]

for i, level in enumerate(obs_depth):
    if site in ['cendrosa', 'preixana']:
        varname_obs = varname_obs_prefix + '_' + str(i+1)
    elif site == 'irta-corn':
        varname_obs = 'VWC_{0}0cm_Avg'.format(i+1)
    val = float(obs[varname_obs].sel(time = irr_dati_post)) + constant_obs
    obs_arr_end.append(val)


#%% Water added computation
sm_diff = np.array(obs_arr_end) - np.array(obs_arr_start)
if site in ['cendrosa', 'preixana']:
    layer_thickness = np.array([0.05, 0.05, 0.2]) #if soil moisture attributed to above layer
    #layer_thickness = np.array([0.075, 0.125, 0.1]) # if soil moist attributed to closest layer
elif site == 'irta-corn':
    layer_thickness = np.array([0.05, 0.1, 0.1, 0.1, 0.1])


#water added on the 30 first cm of soil [m]
water_added_30cm = (sm_diff*layer_thickness).sum()
print('''Quantity of water added in the 30 first cm of soil 
      without consideration of drainage [m]: ''' + str(water_added_30cm))


#%% Plots of volumetric moist profile in ground

fig = plt.figure()
ax = plt.gca()

ax.set_xlim([0, 0.6])
ax.set_ylim([-0.5, 0])

ax.set_xlabel(xlabel)
ax.set_ylabel('Depth [m]')

# plot_title = 'Soil moisture profile before and after irrigation at {0}'.format(site)

#
#for key in val_simu:
#    plt.plot(val_simu[key], sim_depth, marker='+', 
#             label='simu_{0}_d{1}h{2}'.format(key, dati.day, dati.hour))

irr_dati_pre = pd.Timestamp(irr_dati_pre)
irr_dati_post = pd.Timestamp(irr_dati_post)

plt.plot(obs_arr_start, obs_depth, marker='x', 
         # label='obs_d{0}h{1}m{2}'.format(irr_dati_pre.day, irr_dati_pre.hour,
         #             irr_dati_pre.minute),
         label='obs_{0}:{1}'.format(
             str(irr_dati_pre.hour).zfill(2),
             str(irr_dati_pre.minute).zfill(2)),
         color='k',
         linestyle=':')
plt.plot(obs_arr_end, obs_depth, marker='x', 
         label='obs_{0}:{1}'.format(
             str(irr_dati_post.hour).zfill(2),
             str(irr_dati_post.minute).zfill(2)),
         color='k',
         linestyle='--')

plot_title = gv.sites[site]['longname'] + f' on {irr_dati_pre.day} July 2021' 
plt.title(plot_title)
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

save_title = plot_title
if save_plot:
    tools.save_figure(save_title, save_folder)

