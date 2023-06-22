#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021
    
"""

#import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import tools
import global_variables as gv

############# Independant Parameters (TO FILL IN):
    
site = 'ivars-lake'

models = ['irrlagrip30_d1']  # or std_d1
model = models[0]

file_suffix = 'dg'  # '' or 'dg'
hour_interval = [0, 3]

list_variables = [
        # AT GRID POINT
#        'SWU_P9', 
#        'Q2M_ISBA', 'T2M_ISBA', 
#        'WINDSPEED.OUT', 
#        'SWD', 'LWD',
#        'dQdZ_ISBA', 'dTdZ_ISBA',
        
        # PER PATCH
#        'TSRAD_P9', 'TG1P9',
#        'SWU_P9', 'LWU_P9', 'RN_P9', 'H_P9', 'LE_P9', 'GFLUX_P9', 'U_STAR_P9', 'TG1P9', 'TG2P9',
#        'SWU_P1', 'LWU_P1', 'RN_P1', 'H_P1', 'LE_P1', 'GFLUX_P1', 'U_STAR_P1', 'TG1P1', 'TG2P1',
#        'SWU_P4', 'LWU_P4', 'RN_P4', 'H_P4', 'LE_P4', 'GFLUX_P4', 'U_STAR_P4', 'TG1P4', 'TG2P4'        
#       'SWU_WAT', 
       'LWU_WAT', 'RN_WAT', 
#       'H_WAT', 'LE_WAT', 'GFLUX_WAT', 'U_STAR_WAT',
        
#        'Z0VEGP9', 'Z0VEGP10', 'Z0VEGP4', 'Z0_ISBA', 'Z0_WAT',
#        'TG1P1', 'TG2P1', 'TG3P1', 'TG4P1', 'TG5P1', 'TG6P1', 'TG7P1',
#        'SWI1_P1', 'SWI2_P1', 'SWI3_P1', 'SWI4_P1', 'SWI5_P1', 'SWI6_P1', 'SWI7_P1'
          ]

meas_height = {
        'preixana': 7.3,
        'cendrosa': 3,
        'elsplans': 2.5,
        'irta-corn': 2.5,
        'irta-apple': 3.5,
        'ivars-lake': 3,
        }

displacement_height = {
        'preixana': 0,
        'cendrosa': 0.1,
        'elsplans': 0,
        'irta-corn': 1,
        'irta-apple': 0,
        'ivars-lake': 0,
        }

#N.B.: layers depth for diff:
#    [-0.01, -0.04, -0.1, -0.2, -0.4, -0.6,
#     -0.8, -1, -1.5, -2, -3, -5, -8, -12]

#If varname_sim is 3D:
ilevel = 1   #0 is Halo, 1->2m, 2->6.12m, 3->10.49m

#If extrapolation or interpolation needed: [m]
new_height = 0.2

figsize = (6.5, 6) #small for presentation: (6,6), big: (15,9)
save_plot = False
save_folder = './figures/time_series/{0}/domain2/'.format(site)

add_seb_residue = False
######################################################

simu_folders = {key:gv.simu_folders[key] for key in models}
father_folder = gv.global_simu_folder

date = '2021-07'

colordict = {'H_WAT': 'r', 
             'LE_WAT': 'b',
             'GFLUX_WAT': 'brown',
             'H_ISBA': 'r',
             'HC_ISBA': 'r--', 
             'LE_ISBA': 'b',
             'GFLUX_ISBA': 'brown',
             'RAINFC_ISBA': 'cyan',
             'H_P9': 'r', 
             'LE_P9': 'b',
             'GFLUX_P9': 'brown',
             'LWD': 'b',
             'SWU': 'b',
#             'GFLUXC_WAT', 'GFLUXC_ISBA', 'HC_ISBA', 'LEC_ISBA', 'GFLUXC_P9', 'HC_P9', 'LEC_P9'
             'obs': 'k'}
    
# init of dict of results
dict_res = {}
dict_mean = {}

# Dependant Parameters
lat = gv.sites[site]['lat']
lon = gv.sites[site]['lon']

# PLOT:
fig = plt.figure(figsize=figsize)

# Loop
for varname_sim in list_variables:
    # LWU      
    if varname_sim == 'LWU_P9':
        varname_sim_list = ['TSRAD_P9', 'EMIS_ISBAP9']
    elif varname_sim == 'LWU_P1':
        varname_sim_list = ['TSRAD_P1', 'EMIS_ISBAP1']
    elif varname_sim == 'LWU_P10':
        varname_sim_list = ['TSRAD_P10', 'EMIS_ISBAP10']
    elif varname_sim == 'LWU_P4':
        varname_sim_list = ['TSRAD_P4', 'EMIS_ISBAP4']
    elif varname_sim == 'LWU_ISBA':
        varname_sim_list = ['TSRAD_ISBA', 'EMIS']
    elif varname_sim == 'LWU_WAT':
        varname_sim_list = ['TSRAD', 'EMIS']
    # SWU
    elif varname_sim == 'SWU_P9':
        varname_sim_list = ['SWD', 'TALB_P9']
    elif varname_sim == 'SWU_P4':
        varname_sim_list = ['SWD', 'TALB_P4']
    elif varname_sim == 'SWU_P1':
        varname_sim_list = ['SWD', 'TALB_P1']
    elif varname_sim == 'SWU_P10':
        varname_sim_list = ['SWD', 'TALB_P10']
    elif varname_sim == 'SWU_WAT':
        varname_sim_list = ['SWD', 'TALB_SEA']
    
    # U_STAR
    elif varname_sim == 'U_STAR':
        varname_sim_list = ['FMU_ISBA', 'FMV_ISBA']
    elif varname_sim == 'U_STAR_WAT':
        varname_sim_list = ['FMU_WAT', 'FMV_WAT']
    elif varname_sim == 'U_STAR_P9':
        varname_sim_list = ['FMU_P9', 'FMV_P9']
    elif varname_sim == 'U_STAR_P4':
        varname_sim_list = ['FMU_P4', 'FMV_P4']
    elif varname_sim == 'U_STAR_P10':
        varname_sim_list = ['FMU_P10', 'FMV_P10']
    elif varname_sim == 'U_STAR_P1':
        varname_sim_list = ['FMU_P1', 'FMV_P1']
    # other
    elif varname_sim == 'BOWEN_ISBA':
        varname_sim_list = ['H_ISBA', 'LE_ISBA']
    elif varname_sim == 'WINDSPEED':
        varname_sim_list = ['UT', 'VT']
    elif varname_sim == 'WINDSPEED.OUT':
        varname_sim_list = ['UT.OUT', 'VT.OUT']
    elif varname_sim == 'dTdZ_ISBA':
        varname_sim_list = ['Z0_ISBA', 'CD_ISBA', 'CH_ISBA', 'RI_ISBA', 
                            'T2M_ISBA', 'TG1_ISBA']
    elif varname_sim == 'dQdZ_ISBA':
        varname_sim_list = ['Z0_ISBA', 'CD_ISBA', 'CH_ISBA', 'RI_ISBA', 
                            'T2M_ISBA', 'TG1_ISBA', 'Q2M_ISBA', 'QS_ISBA',
                            'PABST']
    else:
        varname_sim_list = [varname_sim,]
    
    # LOAD the files
#    varname_sim_list = ['H_ISBA',] + varname_sim_list  # add 'H_ISBA' because it has latitude and longitude values
#    for i, varname_item in enumerate(varname_sim_list):
#        # get format of file to concatenate
#        in_filenames_sim = gv.format_filename_simu[model]
#        # set name of concatenated output file
#        out_filename_sim = 'LIAIS.{0}.{1}.nc'.format(
#                in_filenames_sim[6], varname_item)
#        # concatenate multiple days
#        datafolder = father_folder + simu_folders[model]
#        tools.concat_simu_files_1var(datafolder, varname_item, 
#                                     in_filenames_sim, out_filename_sim)
#        if i == 0:      # if first data, create new dataset
#            ds = xr.open_dataset(datafolder + out_filename_sim)
#        else:           # append data to existing dataset
#            ds_temp = xr.open_dataset(datafolder + out_filename_sim)
#            ds = ds.merge(ds_temp)
    
    ds = tools.load_dataset(varname_sim_list, model)
    
    # find indices from lat,lon values 
    try:
        index_lat, index_lon = tools.indices_of_lat_lon(ds, lat, lon)
    except AttributeError:  #if the data does not have lat-lon data, merge with another that have it
        ds = tools.load_dataset(['H_ISBA',] + varname_sim_list, model)
        # and now, try again:
        index_lat, index_lon = tools.indices_of_lat_lon(ds, lat, lon)
        
    # Remove useless dimension
    ds = ds.squeeze()
    
    # FIX/SET time abscisse axis
    try:
        start = ds.time.data[0]
    except IndexError:
        start = ds.time.data
    except AttributeError:
        print('WARNING! datetime array is hard coded')
        start = np.datetime64('2021-07-21T01:00')
    
    if varname_sim == 'WINDSPEED.OUT':
        dati_arr = np.array([start + np.timedelta64(i*30, 'm') for i in np.arange(0, ds['record'].shape[0])]) 
    else:
        dati_arr = np.array([start + np.timedelta64(i, 'h') for i in np.arange(0, ds['record'].shape[0])]) 
    ds['record'] = pd.DatetimeIndex(dati_arr)
    
  

  
    # Compute other diag variables

    # LWU
    if varname_sim == 'LWU_P9':
        ds[varname_sim] = tools.calc_longwave_up_sim(
                ds['TSRAD_P9'], ds['EMIS_ISBAP9'])
    elif varname_sim == 'LWU_P1':
        ds[varname_sim] = tools.calc_longwave_up_sim(
                ds['TSRAD_P1'], ds['EMIS_ISBAP1'])
    elif varname_sim == 'LWU_P10':
        ds[varname_sim] = tools.calc_longwave_up_sim(
                ds['TSRAD_P10'], ds['EMIS_ISBAP10'])
    elif varname_sim == 'LWU_P4':
        ds[varname_sim] = tools.calc_longwave_up_sim(
                ds['TSRAD_P4'], ds['EMIS_ISBAP4'])
    elif varname_sim == 'LWU_ISBA':
        ds[varname_sim] = tools.calc_longwave_up_sim(
                ds['TSRAD_ISBA'], ds['EMIS'])
    elif varname_sim == 'LWU_WAT':
        ds[varname_sim] = tools.calc_longwave_up_sim(
                ds['TSRAD'], ds['EMIS'])
    # SWU
    elif varname_sim == 'SWU_P9':
        ds[varname_sim] = ds['TALB_P9'] * ds['SWD']
    elif varname_sim == 'SWU_P4':
        ds[varname_sim] = ds['TALB_P4'] * ds['SWD']
    elif varname_sim == 'SWU_P1':
        ds[varname_sim] = ds['TALB_P1'] * ds['SWD']
    elif varname_sim == 'SWU_P10':
        ds[varname_sim] = ds['TALB_P10'] * ds['SWD']
    elif varname_sim == 'SWU_WAT':
        ds[varname_sim] = ds['TALB_SEA'].values.min() * ds['SWD']
    # U_STAR
    elif varname_sim == 'U_STAR':
        ds[varname_sim] = tools.calc_u_star_sim(ds['FMU'], ds['FMV'])
    elif varname_sim == 'U_STAR_P9':
        ds[varname_sim] = tools.calc_u_star_sim(ds['FMU_P9'], ds['FMV_P9'])
    elif varname_sim == 'U_STAR_P4':
        ds[varname_sim] = tools.calc_u_star_sim(ds['FMU_P4'], ds['FMV_P4'])
    elif varname_sim == 'U_STAR_P10':
        ds[varname_sim] = tools.calc_u_star_sim(ds['FMU_P10'], ds['FMV_P10'])
    elif varname_sim == 'U_STAR_P1':
        ds[varname_sim] = tools.calc_u_star_sim(ds['FMU_P1'], ds['FMV_P1'])
    elif varname_sim == 'U_STAR_WAT':
        ds[varname_sim] = tools.calc_u_star_sim(ds['FMU_WAT'], ds['FMV_WAT'])
    # other
    elif varname_sim == 'BOWEN_ISBA':
        ds[varname_sim] = tools.calc_bowen_sim(ds)
    # gradients
    elif varname_sim == 'dTdZ_ISBA':
        ds['T02M_ISBA'] = tools.cls_t(
                ds['T2M_ISBA'],
                gv.layers_height_MNH_LIAISE[ilevel],
                ds['Z0_ISBA'],
                ds['CD_ISBA'],
                ds['CH_ISBA'], 
                ds['RI_ISBA'],
                ds['TG1_ISBA'],
                new_height
                )
        ds[varname_sim] = (ds['T2M_ISBA'] - ds['T02M_ISBA']) / \
                          (gv.layers_height_MNH_LIAISE[ilevel] - new_height)
    elif varname_sim == 'dQdZ_ISBA':
        ds['T02M_ISBA'], ds['Q02M_ISBA'] = tools.cls_tq(
                ds['T2M_ISBA'][:, index_lat, index_lon],
                ds['Q2M_ISBA'][:, index_lat, index_lon],
                gv.layers_height_MNH_LIAISE[ilevel],
                ds['Z0_ISBA'][:, index_lat, index_lon],
                ds['CD_ISBA'][:, index_lat, index_lon],
                ds['CH_ISBA'][:, index_lat, index_lon], 
                ds['RI_ISBA'][:, index_lat, index_lon],
                ds['TG1_ISBA'][:, index_lat, index_lon],
                ds['QS_ISBA'][:, index_lat, index_lon],
                new_height,
                ds['PABST'][:, ilevel, index_lat, index_lon],
                )
        ds['dQdZ_ISBA'] = (ds['Q2M_ISBA'][:, index_lat, index_lon] - ds['Q02M_ISBA']) / \
                          (gv.layers_height_MNH_LIAISE[ilevel] - new_height)
        ds['dTdZ_ISBA'] = (ds['T2M_ISBA'][:, index_lat, index_lon] - ds['T02M_ISBA']) / \
                          (gv.layers_height_MNH_LIAISE[ilevel] - new_height)
#%%
    # EXTRACT variables of interest
    if varname_sim in ['WINDSPEED', 'WINDSPEED.OUT']:
        ds = ds.squeeze()
        if len(ds['UT'].shape) == 4:
            ut_1d = ds['UT'][:, ilevel, index_lat, index_lon]
            vt_1d = ds['VT'][:, ilevel, index_lat, index_lon]
        elif len(ds['UT'].shape) == 3:
            ut_1d = ds['UT'][:, index_lat, index_lon]
            vt_1d = ds['VT'][:, index_lat, index_lon]
        else:
            raise ValueError("Weird shape for ds['UT']")
        
        var_1d = tools.calc_ws_wd(
                ut_1d, 
                vt_1d)[0]
        
        # Compute windspeed at other height in CLS with log profile
#        ds_z0 = xr.open_dataset(datafolder + 'LIAIS.1.Z0_ISBA.nc')
#        zo_1d = ds_z0['Z0_ISBA'][:, index_lat, index_lon]
        
#        ilevel_height = atmo_level_height[ilevel]
        
#        var_1d_meas_height = tools.log_wind_profile(
#                var_1d, ilevel_height,
#                meas_height[site],
#                z_0=0.1,
#                displacement_height[site])
    elif varname_sim == 'dQdZ_ISBA':
        var_1d = ds[varname_sim]
        
    else:
        # keep variable of interest
        var_md = ds[varname_sim]
        # keep variable for specified location
        if len(var_md.shape) == 5:
            var_1d = var_md[:, :, ilevel, index_lat, index_lon] #1st index is time, 2nd is ?, 3rd is Z,..
        elif len(var_md.shape) == 4:
            var_1d = var_md[:, ilevel, index_lat, index_lon] #1st index is time, 2nd is Z,..
        elif len(var_md.shape) == 3:
            var_1d = var_md[:, index_lat, index_lon]
        
    # PLOT
    try:
        colorstyleline = colordict[varname_sim]
    except KeyError:
        colorstyleline = 'k'
        
    plt.plot(dati_arr, var_1d, 
#             color=colordict[model],
#             colorstyleline,
#             label='simu_{0}_{1}'.format(model, varname_sim),
             label='simu_{0}'.format(varname_sim),
             )

    ax = plt.gca()
    ax.set_xlim([np.min(dati_arr), np.max(dati_arr)])
    
        
    # CONCATENATION around noon (11 to 14)
    dict_res[varname_sim] = {}
    
    # create list object for storing the hourly data for each variable
    for val in var_1d:
        datetime_pd = pd.Timestamp(val.record.values)
        dict_res[varname_sim][str(datetime_pd.day)] = []
    
    # store the hourly values in a list
    for val in var_1d:
        datetime_pd = pd.Timestamp(val.record.values)
        if hour_interval[0] <= datetime_pd.hour <= hour_interval[1]:
            dict_res[varname_sim][str(datetime_pd.day)].append(float(val.values))
    
    # compute the mean of the hourly values
    dict_mean[varname_sim] = {}
    for day in dict_res[varname_sim]:
        dict_mean[varname_sim][day] = (np.array(dict_res[varname_sim][day])).mean()


# remove the value nan of the 31
for varname in dict_mean:
    dict_mean[varname].pop('31')
    

#%% Plot esthetics

try:
    ylabel = ds[varname_sim].comment
except (AttributeError, KeyError):
    ylabel = varname_sim


plot_title = '{0} at {1}'.format(ylabel, site)
ax = plt.gca()
ax.set_ylabel(ylabel)

# add grey zones for night
days = np.arange(1,30)
for day in days:
    # zfill(2) allows to have figures with two digits
    sunrise = pd.Timestamp('202107{0}-1930'.format(str(day).zfill(2)))
    sunset = pd.Timestamp('202107{0}-0500'.format(str(day+1).zfill(2)))
    ax.axvspan(sunset, sunrise, ymin=0, ymax=1, 
               color = '0.9'  #'1'=white, '0'=black, '0.8'=light gray
               )

plt.legend(loc='best')

plt.title(plot_title)
#plt.title('test')
plt.grid()

# keep only hours as X axis
#plt.xticks(dati_arr[1:25:2], labels=np.arange(2,25,2))
#plt.tick_params(rotation=0)

#%% Save figure

print(dict_mean)

#if save_plot:
#    tools.save_figure(plot_title, save_folder)
#    tools.save_figure(plot_title, '/d0/images/lunelt/figures/')
