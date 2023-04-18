#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:55:16 2022

@author: Guylaine Canut, lunelt
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import tools
import global_variables as gv

########## Independant parameters ###############

varname_obs = 'MEAN_MR'
offset_obs = 0
coeff_obs = 0.001

# Simulations to show
models = [
        'irr_d1',
        'std_d1'
        ]
# Datetime - UTC
wanted_day = '20210716_RF02'  # '20210720_RF04', '20210721_RF05'
# Simu variable to show
varname_sim = 'RVT'

# maximum level (height AGL) to plot
toplevel = 3000

save_plot = True
save_folder = 'figures/flight_series/'
##################################################

foldername = '/cnrm/surface/lunelt/data_LIAISE/turb_atr42_raw/'
filename_obs = foldername + 'LIAISE_ATR_turbulent_moments_{0}_L2_v1.0.nc'.format(wanted_day)

simu_folders = {key:gv.simu_folders[key] for key in models}
father_folder = gv.global_simu_folder

colordict = {'irr_d2': 'g', 
             'std_d2': 'r',
             'irr_d1': 'g', 
             'std_d1': 'r', 
             'irr_d2_old': 'g', 
             'std_d2_old': 'r', 
             'obs': 'k'}


#%% OBS
ds_obs = xr.open_dataset(filename_obs)
ds_BL1 = ds_obs.where(ds_obs['legtype'] == 'BL1', drop=True)
ds_BL2 = ds_obs.where(ds_obs['legtype'] == 'BL2', drop=True)
ds_GLO = ds_obs.where(ds_obs['legtype'] == 'GLO', drop=True)


#fig, ax = plt.subplots(nrows=2, 
#                       gridspec_kw={'height_ratios': [7, 3]},
#                       figsize=(10, 10),)
# creating grid for subplots
fig = plt.figure(figsize=(10, 10))
ax1 = plt.subplot2grid(shape=(3, 2), loc=(0, 0), rowspan=2, colspan=2)
ax2 = plt.subplot2grid(shape=(3, 2), loc=(2, 0), colspan=1)
ax3 = plt.subplot2grid(shape=(3, 2), loc=(2, 1), colspan=1)

dati_arr = pd.to_datetime(ds_obs['time']).time

obs_var_corr = (ds_obs[varname_obs]+offset_obs)*coeff_obs

# VARNAME_OBS plot
ax1.plot(
#        dati_arr,
        ds_obs['time'],
         obs_var_corr,
         'o', 
         color='k', 
         label='obs_{0}'.format(varname_obs))

# ALTITUDE plot
ax2.plot(
#        dati_arr,
        ds_obs['time'],
         ds_obs['alt'],
         'o', 
#         ls='--',
         color='k', 
         label='obs_{0}'.format(varname_obs))

# MAP zone and flights

filename_pgd = tools.get_simu_filename('irr_d2', '20210722T1200')
# load dataset, default datetime okay as pgd vars are all the same along time
ds1 = xr.open_dataset(filename_pgd)
varNd = ds1['LAI_ISBA']
#remove single dimensions
var2d = varNd.squeeze()
# remove 999 values, and replace by nan
var2d = var2d.where(~(var2d == 999))
ax3.pcolormesh(var2d.longitude, var2d.latitude, var2d,
               cmap='RdYlGn',  # 'RdYlGn'
               vmin=0, vmax=4,)

for i, leg in enumerate(ds_obs.leg):
    traj_start = [float(ds_obs.lon_start[i]), 
                  float(ds_obs.lon_end[i])]
    traj_end = [float(ds_obs.lat_start[i]), 
                float(ds_obs.lat_end[i])]
    ax3.plot(traj_start, traj_end, marker='+')
    ax3.text(ds_obs.lon[i], ds_obs.lat[i], str(ds_obs.leg[i].data))

ax3.set_xlim([0.6, 1.3])
ax3.set_ylim([41.4, 41.85])
    
#%% SIMU

for model in models:
    
    in_filenames_sim = gv.format_filename_simu[model]
    out_filename_sim = 'LIAIS.{0}.{1}.nc'.format(
            in_filenames_sim[6], varname_sim)
    
    datafolder = father_folder + simu_folders[model]
    tools.concat_simu_files_1var(datafolder, varname_sim, 
                                 in_filenames_sim, out_filename_sim)
    ds_sim = xr.open_dataset(datafolder + out_filename_sim)
    # replace record by time
    dati_arr = pd.date_range(ds_sim.time.data[0],
                             periods=len(ds_sim[varname_sim]), 
                             freq='1H')
    ds_sim['record'] = dati_arr
    ds_sim = ds_sim.squeeze()  # delete 'time' dim
#    ds_sim = ds_sim.rename({'record': 'time'})
    
    #init result variables
    sim_series = {}
    sim_series[model] = []
    
    for i, time in enumerate(ds_obs.time):
        
        # get indices of plane trajectory
        start_pt = (float(ds_obs.lat_start[i]), float(ds_obs.lon_start[i]))
        end_pt = (float(ds_obs.lat_end[i]), float(ds_obs.lon_end[i]))
        line_dict = tools.line_coords(ds_sim, start_pt, end_pt, 
                              nb_indices_exterior=0)
        ni_line = line_dict['ni_range']
        nj_line = line_dict['nj_range']
        
        var4d = ds_sim[varname_sim]
        
        #Get data all along the trajectory of the leg and compute the mean
        traj_series = []
        for i_line, ni in enumerate(ni_line):
            #interpolation of all variables on ni_range
            traj_series.append(float(var4d.interp(ni=ni, 
                                           nj=nj_line[i_line],
                                           record=time,
                                           level=ds_obs['alt'][i]
                                           )
                                    )
                                )
        sim_series[model].append(np.mean(traj_series))
    
    #% PLOT of SIMU
    ax1.scatter(ds_obs.time, sim_series[model],
                color = colordict[model],
                marker='*',
                label = 'simu_{0}'.format(model),
                )

#
#ax.set_xlabel(var3d.long_name + '['+var3d.units+']')
#ax.set_xlim([np.min(var1d), np.max(var1d)])
ax1.set_ylabel('{0} [{1}]'.format(ds_sim[varname_sim].long_name, 
                                    ds_sim[varname_sim].units))
ax1.legend()
ax1.grid()
ax2.set_ylabel('Height AGL (m)')
ax2.grid()

plot_title = 'ATR42 and simu data - {0} on {1}'.format(varname_sim, wanted_day)
fig.suptitle(plot_title)



#%%
if save_plot:
    tools.save_figure(plot_title, save_folder)