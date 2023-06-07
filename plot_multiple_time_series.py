#!/usr/bin/env python3
"""
@author: Tanguy LUNEL

Available variable names:
    
    from CNRM: (last digit is the max number available)
ta_5, hus_5, hur_5, soil_moisture_3, soil_temp_3, u_var_3, w_var_3, swd,
lmon_3, u_star_3, 
...

    from UKMO: (needs work)
'TEMP_2m', 'TEMP_10m', 'TEMP_25m', 'UTOT_2m', 'UTOT_10m', 'UTOT_25m',
'DIR_2m', 'DIR_10m', 'DIR_25m', 'RHUM_2m', 'RHUM_10m', 'RHUM_25m',
'WQ_2m', 
...

"""
#import os
import matplotlib.pyplot as plt
import xarray as xr
import tools

#######################%% Independant Parameters (TO FILL IN):

site = 'irta-corn'

minute_data = False

date = '2021-07'
if minute_data:
    date = '202107'

varname_obs_list = ['TEMP_2m']

save_plot = False
save_folder = './figures/time_series_obs/'

####################################################

colormap = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
linestylemap = ['-', '-', '--', '--', '-.', '-.', ':']

#%% Dependant Parameters

if site == 'cendrosa':
#    varname_sim_suffix = 'P9'
#    varname_sim_suffix = '_ISBA'
    datafolder = '/cnrm/surface/lunelt/data_LIAISE/cendrosa/30min/'
    filename_prefix = 'LIAISE_LA-CENDROSA_CNRM_MTO-FLUX-30MIN_L2_'
    in_filenames_obs = filename_prefix + date
elif site == 'preixana':
#    varname_sim_suffix = '_ISBA'
    datafolder = '/cnrm/surface/lunelt/data_LIAISE/preixana/30min/'
    filename_prefix = 'LIAISE_PREIXANA_CNRM_MTO-FLUX-30MIN_L2_'
    in_filenames_obs = filename_prefix + date
elif site == 'elsplans':
    freq = '30'  # '5' min or '30'min
    datafolder = '/cnrm/surface/lunelt/data_LIAISE/elsplans/mat_50m/{0}min/'.format(freq)
    filename_prefix = 'LIAISE_'
    date = date.replace('-', '')
    in_filenames_obs = filename_prefix + date
#    varname_sim_suffix = '_ISBA'  # or P7, but already represents 63% of _ISBA
elif site in ['irta-corn', 'irta-corn-real']:
    datafolder = '/cnrm/surface/lunelt/data_LIAISE/irta-corn/seb/'
    in_filenames_obs = 'LIAISE_IRTA-CORN_UIB_SEB-10MIN_L2.nc'
else:
    raise ValueError('Site name not known')


#%% OBS - LOAD DATA

out_filename = 'CAT_' + date + filename_prefix + '.nc'
        
# Concatenate multiple days
tools.concat_obs_files(datafolder, in_filenames_obs, out_filename)

# Load data:
obs = xr.open_dataset(datafolder + out_filename)
#(obs[varname_obs]*coeff_obs).plot(label='obs_{0}'.format(varname_obs))

#%% PLOT 

for i, varname in enumerate(varname_obs_list):
    lv = len(varname_obs_list)    
    twins = {}
    plots = {}
    #tick_kw = dict(size=4, width=1.5)  #tick properties
    tick_kw = {}
    
    if i == 0:
        fig, ax = plt.subplots(figsize=(8 + 1*lv, 6))
        fig.subplots_adjust(right=1 - 0.1*lv)
        plots[i], = obs[varname].plot(ax=ax, color=colormap[i])
        ax.yaxis.label.set_color(plots[i].get_color())
        ax.tick_params(axis='y', colors=plots[i].get_color(), **tick_kw)
    else:
        twins[i] = ax.twinx()
        yaxe_position = 0.8 + 0.2*i
        twins[i].spines['right'].set_position(("axes", yaxe_position))
        plots[i], = obs[varname].plot(ax=twins[i], 
             color=colormap[i],
             linestyle=linestylemap[i]
             )
        twins[i].yaxis.label.set_color(plots[i].get_color())
        twins[i].tick_params(axis='y', colors=plots[i].get_color(), **tick_kw)


plot_title = '{0} at {1}'.format(varname_obs_list, site)
plt.title(plot_title)
plt.grid()
plt.show()

#%% Save figure

if save_plot:
    tools.save_figure(plot_title, save_folder)