#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021

Fonctionnement:
    Seule plusieurs sections ont besoin d'être remplies, à automatiser.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from tools import indices_of_lat_lon, sm2swi 

#################### Independant Parameters (TO FILL IN):
    
site = 'cendrosa'

varname_obs = 'soil_moisture_3'   # options are: (last digit is the max number available)
# ta_5, hus_5, hur_5, soil_moisture_3, soil_temp_3, u_var_3, w_var_3, swd,... 

varname_sim = 'SWI4_ISBA'      # simu varname to compare obs with
#T2M_ISBA, LE_P4, EVAP_P9, GFLUX_P4, WG3_ISBA, WG4P9, SWI4_ISBA, SWI14_P12
#leave None for automatic attribution

date = '2021-07'

save_plot = False
save_folder = './figures/time_series/{0}/'.format(site)

###################################################


plot_title = '{0} at {1}'.format(varname_obs, site)
#plot_title = 'Temporal series \n at lat={0}, lon={1}, alt={2}'.format(
#        lat, lon, alt)


simu_folders = {
        'irr': '2.13_irr_2021_22-27/', 
        'std': '1.11_ECOII_2021_ecmwf_22-27/'
         }
father_folder = '/cnrm/surface/lunelt/NO_SAVE/nc_out/'

colordict = {'irr': 'g', 'std': 'orange', 'obs': 'k'}


#%% Dependant Parameters

if varname_obs in ['soil_moisture_1', 'soil_moisture_2', 'soil_moisture_3']:
    ylabel = 'soil water index'
    constant_obs = 0
    coeff_obs = 1
    varname_sim_prefix = 'WG3'    #corresponding variables in SFX
    #other options: 'WG3_ISBA', 'WG4P9', 'SWI4_ISBA', 'SWI14_P12'
    #N.B.: layers depth for diff: 
    #    [-0.01, -0.04, -0.1, -0.2, -0.4, -0.6,
    #     -0.8, -1, -1.5, -2, -3, -5, -8, -12]
else:
    raise ValueError('Unknown value')


if site == 'cendrosa':
    lat = 41.6925905
    lon = 0.9285671
    alt = 'undef'
    wilt_pt ={1: 0.141, 2: 0.07, 3: 0.125}       # To determine via plot on August
    field_capa = {1: 0.28, 2: 0.18, 3: 0.23}   # resp. layers 1, 2 & 3
    varname_sim_suffix = 'P9'
    datafolder = \
        '/cnrm/surface/lunelt/data_LIAISE/cendrosa/30min/'
    filename_prefix = \
         'LIAISE_LA-CENDROSA_CNRM_MTO-FLUX-30MIN_L2_'
    in_filenames = filename_prefix + date
elif site == 'preixana':
    lat = 41.59373 
    lon = 1.07250
    alt = 'undef'
    wilt_pt ={1: 0.065, 2: 0.135, 3: 0.115}       # To determine via plot
    field_capa = {1: 0.25, 2: 0.30, 3: 0.187}    # on May
    varname_sim_suffix = 'P2'
    datafolder = \
        '/cnrm/surface/lunelt/data_LIAISE/preixana/30min/'
    filename_prefix = \
        'LIAISE_PREIXANA_CNRM_MTO-FLUX-30MIN_L2_'
    in_filenames = filename_prefix + date
elif site == 'elsplans':
    lat = 41.590111 
    lon = 1.029363  
    alt = 'undef'
    varname_sim_suffix = '_ISBA'
else:
    raise ValueError('Site name not known')

if varname_sim is None:
    varname_sim = varname_sim_prefix + varname_sim_suffix


#%% OBS: Concatenate and plot data

out_filename = 'CAT_' + date + filename_prefix + '.nc'

# Concatenate multiple days
if not os.path.exists(datafolder + out_filename):
    os.system('''
        cd {0}
        ncrcat {1}* -o {2}
        '''.format(datafolder, in_filenames, out_filename))

# Plot:
fig = plt.figure(figsize=(10, 7.5))    

obs = xr.open_dataset(datafolder + out_filename)
#(obs[varname_obs]*coeff_obs).plot(label='obs_{0}'.format(varname_obs))

varplot = 'swi'
# FOR SWI
if varplot == 'swi':
    swi = {}
    #for i in [1, 2, 3]:
    for i in [3]:
        swi[i] = sm2swi(obs['soil_moisture_{0}'.format(str(i))], 
                        wilt_pt=wilt_pt[i], field_capa=field_capa[i])
        swi[i].plot(label='swi_{0}'.format(str(i)),
                    color=colordict['obs']
                    )
# FOR SM:  # mainly for debug, exploratory understanding
elif varplot == 'sm':
    obs[varname_obs].plot(label='varname_obs')


#%% SIMU:

in_filenames_sim = 'LIAIS.2.SEG*.001dg.nc'  # use of wildcard allowed
out_filename_sim = 'LIAIS.2.{0}dg.nc'.format(varname_sim)

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

    if len(var_md.shape) == 3:
        var_1d = var_md.data[:, index_lat, index_lon]
    else:
        raise ValueError('Issue on data dimensions, should be 3D')

    
    #fig = plt.figure()
    ax = plt.gca()
    ax.set_ylabel(ylabel)
    #ax.set_ylim([0, 0.4])
    
#    ax.set_xlim([np.min(obs.time), np.max(obs.time)])
#    ax.set_xlim([np.min(dati_arr), np.max(dati_arr)])
    
    plt.plot(dati_arr, var_1d, 
             label='simu_{0}'.format(model),
             color=colordict[model])


plot_title = '{0} at {1}'.format(ylabel, site)


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




