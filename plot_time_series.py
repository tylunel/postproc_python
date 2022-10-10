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
from tools import indices_of_lat_lon 


############# Independant Parameters (TO FILL IN):
    
site = 'cendrosa'

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

#If varname_sim is 3D:
ilevel = 1   #0 is Halo, 1->2m, 2->6.12m, 3->10.49m

longname_as_label = True  #does not work for elsplans
if site == 'elsplans':
    longname_as_label = False  #does not work for elsplans

save_plot = True
save_folder = './figures/time_series/{0}/'.format(site)

######################################################

simu_folders = {
        'irr': '2.13_irr_2021_22-27/', 
        'std': '1.11_ECOII_2021_ecmwf_22-27/'
         }
father_folder = '/cnrm/surface/lunelt/NO_SAVE/nc_out/'

date = '2021-07'

colordict = {'irr': 'g', 'std': 'orange', 'obs': 'k'}

secondary_axis_latent_heat = False

#%% Dependant Parameters

if varname_obs in ['soil_moisture_1', 'soil_moisture_2', 'soil_moisture_3']:
    ylabel = 'soil moisture [m3/m3]'
    offset_obs = 0
    coeff_obs = 1
    varname_sim_prefix = 'WG3'    #corresponding variables in SFX
    #other options: 'WG3_ISBA', 'WG4P9'
    #N.B.: layers depth for diff: 
    #    [-0.01, -0.04, -0.1, -0.2, -0.4, -0.6,
    #     -0.8, -1, -1.5, -2, -3, -5, -8, -12]
elif varname_obs in ['soil_temp_1', 'soil_temp_2', 'soil_temp_3']:
    ylabel = 'soil temperature [K]'
    offset_obs = 273.15
    coeff_obs = 1
    varname_sim_prefix = 'TG2'
elif varname_obs in ['swd']:
    ylabel = 'shortwave downward radiation [W/m2]'
    offset_obs = 0
    coeff_obs = 1
    varname_sim_prefix = ''
elif varname_obs in ['lmon_1', 'lmon_2', 'lmon_3']:
    ylabel = 'monin-obukhov length [m]'
    offset_obs = 0
    coeff_obs = 1
    varname_sim_prefix = ''
elif varname_obs in ['h2o_flux_1', 'h2o_flux_2', 'h2o_flux']:  #this includes Webb Pearman Leuning correction on w_h2o_cov
    ylabel = 'h2o flux [kg.m-2.s-1]'
    offset_obs = 0
    coeff_obs = 0.001
    secondary_axis_latent_heat = True
    varname_sim_prefix = 'EVAP'
elif varname_obs in ['w_h2o_cov_1', 'w_h2o_cov_2', 'w_h2o_cov']:
    ylabel = 'h2o turbulent flux [kg.m-2.s-1]'
    offset_obs = 0
    coeff_obs = 0.001
    secondary_axis_latent_heat = True
    varname_sim_prefix = 'EVAP'
elif varname_obs in ['WQ_2m', 'WQ_10m']:
    ylabel = 'h2o turbulent flux [kg.m-2.s-1]'
    offset_obs = 0
    coeff_obs = 1
    secondary_axis_latent_heat = True
    varname_sim_prefix = 'EVAP'
elif varname_obs in ['ta_1', 'ta_2', 'ta_3', 'ta_4', 'ta_5', 'TEMP_2m']:
    ylabel = 'air temperature [K]'
    offset_obs = 273.15
    coeff_obs = 1
    varname_sim_prefix = 'T2M'
elif varname_obs in ['hus_1', 'hus_2', 'hus_3', 'hus_4', 'hus_5', 'RHO_2m']:
    ylabel = 'specific humidity [kg/kg]'
    offset_obs = 0
    coeff_obs = 0.001
    varname_sim_prefix = 'Q2M'
elif varname_obs in ['WT_2m']:
    ylabel = 'turbulent sensible heat flux [W/m²]'
    offset_obs = 0
    coeff_obs = 1005*1.20  #heat capacity * density of DRY air at 20°C
    varname_sim_prefix = 'Q2M'
else:
    ylabel = varname_obs
    offset_obs = 0
    coeff_obs = 1
    pass
#    raise ValueError("nom de variable d'observation inconnue"), 'WQ_2m', 'WQ_10m'


if site == 'cendrosa':
    lat = 41.6925905
    lon = 0.9285671
#    varname_sim_suffix = 'P9'
    varname_sim_suffix = '_ISBA'
    datafolder = '/cnrm/surface/lunelt/data_LIAISE/cendrosa/30min/'
    filename_prefix = 'LIAISE_LA-CENDROSA_CNRM_MTO-FLUX-30MIN_L2_'
    in_filenames_obs = filename_prefix + date
elif site == 'preixana':
    lat = 41.59373 
    lon = 1.07250
    varname_sim_suffix = '_ISBA'
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
    varname_sim_suffix = '_ISBA'
else:
    raise ValueError('Site name not known')

if varname_sim is None:
    varname_sim = varname_sim_prefix + varname_sim_suffix


#%% OBS: Concatenate and plot data

out_filename_obs = 'CAT_' + date + filename_prefix + '.nc'

# Concatenate multiple days
if not os.path.exists(datafolder + out_filename_obs):
    print("creation of file: ", out_filename_obs)
    os.system('''
        cd {0}
        ncrcat {1}*.nc -o {2}
        '''.format(datafolder, in_filenames_obs, out_filename_obs))

# Plot:

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
    ((obs[varname_obs]+offset_obs)*coeff_obs).plot(label='obs_'+varname_obs,
                                                   color=colordict['obs'],
                                                   linewidth=1)



#%% SIMU:

in_filenames_sim = 'LIAIS.1.SEG*.001.nc'  # use of wildcard allowed
out_filename_sim = 'LIAIS.1.{0}.nc'.format(varname_sim)

for model in simu_folders:
    datafolder = father_folder + simu_folders[model]
    if not os.path.exists(datafolder + out_filename_sim):
        print("creation of file: ", out_filename_sim)
        os.system('''
            cd {0}
            ncecat -v {1} {2} {3}
            '''.format(datafolder, varname_sim, 
                       in_filenames_sim, out_filename_sim))
    #command 'cdo -select,name={1} {2} {3}' may work as well, but not always...

    ds1 = xr.open_dataset(datafolder + out_filename_sim)
    
    # find indices from lat,lon values 
    index_lat, index_lon = indices_of_lat_lon(ds1, lat, lon)
    
    var_md = ds1[varname_sim]
    
    # Set time abscisse axis
    start = np.datetime64('2021-07-21T01:00')
    dati_arr = np.array([start + np.timedelta64(i, 'h') for i in np.arange(0, var_md.shape[0])])
    
    # PLOT d1
    if len(var_md.shape) == 5:
        var_1d = var_md.data[:, :, ilevel, index_lat, index_lon] #1st index is time, 2nd is ?, 3rd is Z,..
    elif len(var_md.shape) == 4:
        var_1d = var_md.data[:, ilevel, index_lat, index_lon] #1st index is time, 2nd is Z,..
    elif len(var_md.shape) == 3:
        var_1d = var_md.data[:, index_lat, index_lon]

    
    #fig = plt.figure()
    ax = plt.gca()
    ax.set_ylabel(ylabel)
    #ax.set_ylim([0, 0.4])
    
#    ax.set_xlim([np.min(obs.time), np.max(obs.time)])
    ax.set_xlim([np.min(dati_arr), np.max(dati_arr)])
    
    plt.plot(dati_arr, var_1d, 
             label='simu_{0}_{1}'.format(model, varname_sim),
             color=colordict[model])

if longname_as_label:
    ylabel = obs[varname_obs].long_name

plot_title = '{0} at {1}'.format(ylabel, site)

# add secondary axis on the right, relative to the left one
if secondary_axis_latent_heat:
    axes = plt.gca()
    secax = axes.secondary_yaxis("right",                              
        functions=(lambda evap: evap*2264000,
                   lambda le: le/2264000))
    secax.set_ylabel('latent heat flux [W/m²]')

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
