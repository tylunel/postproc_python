#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021

Fonctionnement:
    Seule plusieurs sections ont besoin d'être remplies, à automatiser.
    Works fine for scalar variables, not great for wind
    
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from tools import indices_of_lat_lon, apparent_temperature


############# Independant Parameters (TO FILL IN):
    
site = 'cendrosa'

domain_nb = 2

#If varname_sim is 3D:
ilevel = 1   #0 is Halo, 1->2m, 2->6.12m, 3->10.49m

save_plot = True
save_folder = './figures/time_series/{0}/'.format(site)

######################################################

varname_obs = 'ta_2'   # options are: (last digit is the max number available)
#-- For CNRM:
# ta_5, hus_5, hur_5, soil_moisture_3, soil_temp_3, u_var_3, w_var_3, swd,... 
# w_h2o_cov, h2o_flux[_1], shf_1
# from données lentes: 1->0.2m, 2->2m, 3->10m, 4->25m, 5->50m
# from eddy covariance measures: 1->3m, 2->25m, 3->50m
#-- For UKMO (elsplans):
# TEMP, RHO, WQ, WT, UTOT, DIR, ... followed by _2m, _10mB, _25m, _50m, _rad, _subsoil

varname_sim = 'T2M_ISBA'      # simu varname to compare obs with
#T2M_ISBA, LE_P4, EVAP_P9, GFLUX_P4
#leave None for automatic attribution

simu_folders = {
        'irr': '2.13_irr_2021_22-27/', 
        'std': '1.11_ECOII_2021_ecmwf_22-27/'
         }
father_folder = '/cnrm/surface/lunelt/NO_SAVE/nc_out/'

date = '2021-07'

colordict = {'irr': 'g', 'std': 'orange', 'obs': 'k'}

secondary_axis_latent_heat = False

#%% Dependant Parameters

if varname_obs in ['swd']:
    ylabel = 'shortwave downward radiation [W/m2]'
    offset_obs = 0
    coeff_obs = 1
    varname_sim_prefix = ''
elif varname_obs in ['ta_1', 'ta_2', 'ta_3', 'ta_4', 'ta_5', 'TEMP_2m']:
    ylabel = 'air temperature [K]'
    offset_obs = 273.15
    coeff_obs = 1
elif varname_obs in ['hus_1', 'hus_2', 'hus_3', 'hus_4', 'hus_5', 'RHO_2m']:
    ylabel = 'specific humidity [kg/kg]'
    offset_obs = 0
    coeff_obs = 0.001
else:
    ylabel = varname_obs
    offset_obs = 0
    coeff_obs = 1
    pass
#    raise ValueError("nom de variable d'observation inconnue"), 'WQ_2m', 'WQ_10m'


if site == 'cendrosa':
    lat = 41.6925905
    lon = 0.9285671
    datafolder = '/cnrm/surface/lunelt/data_LIAISE/cendrosa/30min/'
    filename_prefix = 'LIAISE_LA-CENDROSA_CNRM_MTO-FLUX-30MIN_L2_'
    in_filenames_obs = filename_prefix + date
elif site == 'preixana':
    lat = 41.59373 
    lon = 1.07250
    datafolder = '/cnrm/surface/lunelt/data_LIAISE/preixana/30min/'
    filename_prefix = 'LIAISE_PREIXANA_CNRM_MTO-FLUX-30MIN_L2_'
    in_filenames_obs = filename_prefix + date
elif site == 'elsplans':
    lat = 41.590111 
    lon = 1.029363
    datafolder = '/cnrm/surface/lunelt/data_LIAISE/elsplans/mat_50m/5min/'
    filename_prefix = 'LIAISE_'
    date = date.replace('-', '')
    in_filenames_obs = filename_prefix + date
else:
    raise ValueError('Site name not known')


#%% OBS: Concatenate and plot data

out_filename_obs = 'CAT_' + date + filename_prefix + '.nc'

# Concatenate multiple days
if not os.path.exists(datafolder + out_filename_obs):
    print("creation of file: ", out_filename_obs)
    os.system('''
        cd {0}
        ncrcat {1}*.nc -o {2}
        '''.format(datafolder, in_filenames_obs, out_filename_obs))

## PLOT ta_2
fig = plt.figure(figsize=(10, 6.5))
    
obs = xr.open_dataset(datafolder + out_filename_obs)

if site == 'elsplans':
    dati_arr = pd.date_range(start=obs.time.min().values, 
                             periods=len(obs[varname_obs]), 
                             freq='5T')
    #turn outliers into NaN
    obs_var_filtered = obs[varname_obs].where(
            (obs[varname_obs]-obs[varname_obs].mean()) < (4*obs[varname_obs].std()), 
            np.nan)
    plt.plot(dati_arr, ((obs_var_filtered+offset_obs)*coeff_obs), 
             label='obs_'+varname_obs,
             color=colordict['obs'])
else:
    ((obs[varname_obs])*coeff_obs).plot(label='obs_'+varname_obs,
                                                   color=colordict['obs'],
                                                   linewidth=1)


## PLOT THERMAL COMFORT
    
##diff with orig:
apparent_temp_obs = []
for i in range(len(obs.time)):
    apparent_temp_obs.append(apparent_temperature(
            obs['ta_2'].data[i], 
            obs['hur_2'].data[i], 
            obs['ws_2'].data[i]))

plt.plot(obs.time, 
         apparent_temp_obs, 
         label='obs_apparent_temp',
         color=colordict['obs'],
         linestyle = ':')

#%% SIMU:

varname_sim_list = [
                    'T2M_ISBA', 
                    'REHU', 
                    'UT',
                    'VT',
                    ]

## CONCATENATE DATA
for varname_sim in varname_sim_list:
    
    in_filenames_sim = 'LIAIS.{0}.SEG*.001dg.nc'.format(domain_nb)  # use of wildcard allowed
    out_filename_sim = 'LIAIS.{0}.{1}.nc'.format(domain_nb, varname_sim)
    
    for model in simu_folders:
        datafolder = father_folder + simu_folders[model]
        if not os.path.exists(datafolder + out_filename_sim):
            print("creation of file: ", out_filename_sim)
            os.system('''
                cd {0}
                ncecat -v {1} {2} {3}
                '''.format(datafolder, varname_sim, 
                       in_filenames_sim, out_filename_sim))

## RETRIEVE AND PLOT SIMU DATA 

#dict to collect results for the two models
apparent_temp_sim = {}

for model in simu_folders:
    #dict to collect data from the simu
    vars_sim = {}
    
    for varname_sim in varname_sim_list:
        datafolder = father_folder + simu_folders[model]
        ds1 = xr.open_dataset(datafolder + 'LIAIS.2.{0}dg.nc'.format(varname_sim))
        
        # find indices from lat,lon values 
        index_lat, index_lon = indices_of_lat_lon(ds1, lat, lon)
        
        var_md = ds1[varname_sim]
        
        # Set time abscisse axis
        start = np.datetime64('2021-07-21T01:00')
        dati_arr = np.array([start + np.timedelta64(i, 'h') for i in np.arange(0, var_md.shape[0])])
        
        # PLOT d1
        if len(var_md.shape) == 5:
            var_1d = var_md[:, :, ilevel, index_lat, index_lon].data #1st index is time, 2nd is ?, 3rd is Z,..
        elif len(var_md.shape) == 4:
            var_1d = var_md[:, ilevel, index_lat, index_lon].data #1st index is time, 2nd is Z,..
        elif len(var_md.shape) == 3:
            var_1d = var_md[:, index_lat, index_lon].data
        
        #Put in result dict
        vars_sim[varname_sim] = var_1d

    vars_sim['WS'] = np.sqrt(vars_sim['UT']**2 + \
                                vars_sim['VT']**2)

    ## Compute thermal comfort
    apparent_temp_sim[model] = []
    for i in range(len(var_1d)):
        apparent_temp_sim[model].append(
            apparent_temperature(
                float(vars_sim['T2M_ISBA'][i])-273.15,
                float(vars_sim['REHU'][i]), 
                float(vars_sim['WS'][i])
                )
            )

    #plot T2M
    plt.plot(dati_arr, vars_sim['T2M_ISBA']-273.15, 
             label='simu_{0}_{1}'.format(model, 'T2M_ISBA'),
             color=colordict[model])
    #plot apparent temperature
    plt.plot(dati_arr, apparent_temp_sim[model], 
             label='simu_{0}_{1}'.format(model, 'apparent_temp'),
             color=colordict[model],
             linestyle=':')

#%% FIGURE ESTHETICS

ylabel = 'apparent_temperature in degC'

plot_title = '{0} at {1}'.format(ylabel, site)

# add secondary axis on the right, relative to the left one
if secondary_axis_latent_heat:
    axes = plt.gca()
    secax = axes.secondary_yaxis("right",                              
        functions=(lambda evap: evap*2264000,
                   lambda le: le/2264000))
    secax.set_ylabel('latent heat flux [W/m²]')


ax = plt.gca()
ax.set_ylabel(ylabel)
#ax.set_ylim([0, 0.4])
#ax.set_xlim([np.min(obs.time), np.max(obs.time)])
ax.set_xlim([np.min(dati_arr), np.max(dati_arr)])

plt.title(plot_title)
plt.legend()
plt.grid()


#%% Save figure

if save_plot:
    filename = (plot_title)
    filename = filename.replace('=', '').replace('(', '').replace(')', '')
    filename = filename.replace(' ', '_').replace(',', '').replace('.', '_')
    filename = filename.replace('/', 'over')
    plt.savefig(save_folder+filename)
