#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for plotting vertical profiles from AROME .fa / MNH output / obs, etc files

@author: lunelt
"""

import matplotlib.pyplot as plt
# from scipy import integrate
import numpy as np
import os
import re
import pandas as pd
import xarray as xr
# from datetime import datetime
#import copy
import global_variables as gv
import tools
from scipy.stats import circmean


folder_path = '/home/lunelt/Data/analysis/'
arome_profiles_folder = '/home/lunelt/Data/data_NEMO/arome_profiles/'

site = 'planier'
varname = 'wind'     # tke, t, u, v, q, ws, wd

max_height = 500  # [m]
date_start, date_end = '2023-01-27 02:00:00', '2023-01-27 12:30:00'
base_of_datetimes_array = 'mnh-1km'

mnh_expe = '3_planier_0127'

sources_var =  {
    'lidar': {
        0: ['tke', 
            'tke_var',
            ],
        1: ['ws'],
        },
    # 'arome': {
    #     0: ['tke', 
    #         # 'TKE_EQ',
    #         ], 
    #     1: ['ws'],
    #     },
    'mnh_04_1km_COARE': {
        0: [
            # 'TKET', 
            'TKE_EQ',
            ], 
        1: ['WS'],
        },
    # 'mnh_0211_40m':{
    'mnh_0411_40m':{
    # # 'mnh_0414_40m_eastD3':{
        0: [
            # 'SBG_TKE', 
            # # 'RES_TKE',
            # 'U2ME', 'V2ME', 'W2ME',
            'TOT_TKE',
            ], 
        1: ['WS'],
        },
    }
profile_sources = list(sources_var.keys())

error_computation = False
plot_figure = True
save_figure = True
add_res_tke_frac = False

# last_folder_name = '-'.join(profile_sources)
save_folder = f'figures/verti_profiles_{site}_mean/0-{max_height}/{date_start}-to-{date_end}/'

# Regular expression to match filenames and extract date and hour
pattern = r"arome\.AN\.(\d{4})(\d{2})(\d{2})\.(\d{2})"  # to have all date
# pattern = r"arome\.AN\.(\d{4})(\d{2})(\d{2})\.12"  # only midday
# pattern = r"verti_profile_arome_planier_(\d{4})(\d{2})(\d{2})\.(\d{2}).pickle"  # to have all date

### Define array of datetime to plot:
if base_of_datetimes_array == 'arome':
    # List to store datetime objects
    datetimes = []
    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        match = re.match(pattern, filename)
        if match:
            if len(match.groups()) == 4:
                year, month, day, hour = map(int, match.groups())
                datetimes.append(pd.Timestamp(
                    year=year, month=month, day=day, hour=hour))
            elif len(match.groups()) == 3:
                year, month, day = map(int, match.groups())
                datetimes.append(pd.Timestamp(
                    year=year, month=month, day=day, hour=int(pattern[-2::])))
    # Sort the datetimes (optional, ensures chronological order)
    datetimes.sort()
    
    # subset the datetimes of interest
    df = pd.DatetimeIndex(datetimes)
    df_temp = df[df > date_start]
    datetimes = df_temp[df_temp < date_end]
elif base_of_datetimes_array == 'mnh-1km':
    datetimes_temp = np.arange(pd.Timestamp(date_start), 
                          pd.Timestamp(date_end), 
                          pd.Timedelta('1h'))
    datetimes = pd.DatetimeIndex(datetimes_temp)
else:
    datetimes = np.arange(pd.Timestamp(date_start), 
                          pd.Timestamp(date_end), 
                          pd.Timedelta('30min'))
    datetimes = pd.DatetimeIndex(datetimes_temp)

# datetimes = [pd.Timestamp('20230105T1400')]

color_dict_source = {
    'arome': 'b',
    'lidar': 'k',
    'mnh_02_1km': 'r',
    'mnh_021_200m': 'orange',
    # 'mnh_022_40m': 'olive',
    'mnh_04_1km_COARE': 'r',
    'mnh_021_200m': 'orange',
    # 'mnh_0411_40m': 'olive',
    'rs': 'grey',
    'default': 'green',
    }


#%% LOAD the verti profiles
profiles_all_dates = {}

for wanted_date in datetimes:

    verti_profiles = {}
    # profile_sources_orig = profile_sources.deep
    # time_avail_profile_sources = copy.deepcopy(profile_sources)
    
    for source in profile_sources:
        print(source)
        varname_dict = sources_var[source]
        varname_list = [item for sublist in varname_dict.values() for item in sublist]

        print(f'vars: {varname_list}')
        
        # ---- AROME
        if source == 'arome':
            verti_profiles[source] = {}
            
            datetime_formatted = wanted_date.strftime('%Y%m%d.%H')
            filename_pickle = f"verti_profile_arome_{site}_{datetime_formatted}.pickle"
            
            if filename_pickle in os.listdir(f'{arome_profiles_folder}/{site}/'):
                verti_profiles[source] = pd.read_pickle(
                    f'{arome_profiles_folder}/{site}/{filename_pickle}')
            else:
                continue
                #/!\ better to run "extract_verti_profile_arome_tp_pickle" first
                # instead of loading it here --
        #         for varname in varname_load_list:
        #             temp = tools.extract_profile_arome(
        #                 site, wanted_date, varname, max_height=max_height,
        #                 save_to_pickle=False,
        #                 verbose=True)
        #             verti_profiles[source][varname] = temp[varname] 
            
            df = verti_profiles['arome']  # note that verti_profiles contains pd.Dataframe
            df['ws'], df['wd'] = tools.calc_ws_wd(df['u'], df['v'])
            df['p'] = tools.height_to_pressure_std(df.index)
            df['theta'] = tools.potential_temperature_from_temperature(
                df['p'], df['t'])
            
        # ---- MNH
        elif source[:4] == 'mnh_':  # MNH
            if source.split('_')[2] == '1km':
                output_type = 'diag'
            else:
                output_type = 'backup'
                
            try:
                filename = tools.get_simu_filepath(
                        f'{mnh_expe}/{source[4:]}', 
                        wanted_date, output_type=output_type,
                        global_simu_folder=gv.global_simu_folder)
            except (KeyError, ValueError, FileNotFoundError):
            # except (KeyError,):
                # time_avail_profile_sources.remove(source)
                print(f'{source[4:]} does not have required timestep')
                continue
                
            ds = xr.open_dataset(filename)
            
            index_lat, index_lon = tools.get_indices_of_lat_lon(
                ds, lat=gv.sites[site]['lat'], lon=gv.sites[site]['lon'], 
                verbose=False)
                        
            # change variable to adapt to MNH
            # vars_mnh  = [gv.variable_correspondance_arome_mnh[var] for var in varname_load_list]
            vars_mnh = [
                'THT', 'RVT', 'TKET', 
                'WT', 'UT', 'VT',
                'UMME', 'VMME', 'WMME',
                'TKEMME', 'U2ME', 'V2ME', 'W2ME',
                'TKE_DISS', 'MF_FRAC_UP', 'MF_W_UP', 'MF_V_UP', 'MF_U_UP',
                ]
            
            vars_avail = [var for var in vars_mnh if var in ds.variables]
            
            # subset
            var3d = ds[vars_avail]
            # var3d_1km = ds_1km[vars_mnh]
            # var3d_200m = ds_200m[vars_mnh]
            var3d = tools.flux_pt_to_mass_pt(var3d, only_basic_vars=False)
            # var3d_1km_cen = tools.flux_pt_to_mass_pt(var3d_1km, only_basic_vars=False)
            # var3d_200m_cen = tools.flux_pt_to_mass_pt(var3d_200m, only_basic_vars=False)
            var1d = var3d.isel(nj=index_lat, ni=index_lon).squeeze()
            
            # Diag
            try:
                var1d['WS'], var1d['WD'] = tools.calc_ws_wd(var1d['UT'], var1d['VT'])
                var1d['RES_TKE'] = 0.5*(var1d['U2ME']+var1d['V2ME']+var1d['W2ME'])
                var1d['RES_TKE_VERTI'] = 0.5*(var1d['W2ME'])
                var1d['RES_TKE_HORIZ'] = 0.5*(var1d['U2ME']+var1d['V2ME'])
                var1d['SBG_TKE'] = var1d['TKEMME']
                var1d['TOT_TKE'] = var1d['SBG_TKE'] + var1d['RES_TKE']
                # var1d['TOT_TI'] = np.sqrt(1.5*var1d['TOT_TKE'])/var1d['WS']
                var1d['RES_TKE_FRAC'] = var1d['RES_TKE']/var1d['TOT_TKE']
                var1d['TOT_TKE_VERTI'] = var1d['RES_TKE_VERTI'] * (1/var1d['RES_TKE_FRAC'])
                var1d['TOT_TKE_HORIZ'] = var1d['RES_TKE_HORIZ'] * (1/var1d['RES_TKE_FRAC'])
                var1d['CepsOverLeps'] = -var1d['TKE_DISS'] / var1d['TKET']**(3/2)            
            except KeyError:
                print(f'NOT ALL TKE DIAGS AVAILABLE in {source}')
                pass
            
            try:
                var1d['TKE_MF_UP'] = \
                    0.5*(var1d['MF_FRAC_UP'] / (1 - var1d['MF_FRAC_UP'])) * \
                    ((var1d['MF_W_UP'] - var1d['WT'])**2 + \
                     (var1d['MF_U_UP'] - var1d['UT'])**2 + \
                     (var1d['MF_V_UP'] - var1d['VT'])**2)
                var1d['TKE_EQ'] = var1d['TKET'] + var1d['TKE_MF_UP']
            except KeyError:
                print(f'NOT ALL EDKF DIAGS AVAILABLE in {source}')
                pass
                
            verti_profiles[source] = var1d.to_dataframe()
            
            # keep only low layer of atmos (~ABL)
            # var3d_low = var3d.where(var3d.level<toplevel, drop=True)
            # var1d_column = var3d_low.isel(nj=index_lat, ni=index_lon).squeeze()
        
        # ---- Lidar
        elif source == 'lidar':
            varname_dict = sources_var[source]
            varname_list = [item for sublist in varname_dict.values() for item in sublist]
            
            whole_lidar = tools.extract_profile_lidar(
                wanted_date, site=site, 
                # lidar_vars=['tke', 'tke_var', 'edr'],
                lidar_vars=varname_list,
                )
            if site == 'salin-giraud':
                whole_lidar['tke'] = whole_lidar['tke_horiz']
            
            if whole_lidar is not None:
                 profile_lidar_dict = {
                    key: whole_lidar[key].to_dict() for key in whole_lidar.keys()}
                 verti_profiles[source] = pd.DataFrame(profile_lidar_dict)
                
            if ('CepsOverLeps' in varname_list):
                verti_profiles['lidar']['CepsOverLeps'] = \
                    verti_profiles['lidar']['edr']/ verti_profiles['lidar']['tke']**(3/2)
                    
        # ---- RS
        elif source in ['rs', 'radiosounding']:
            rs_varname_dict = {
                'ws': 'FF',
                'wd': 'DD',
                }
            whole_rs = tools.extract_profile_rs_mf(site, wanted_date)
            if whole_rs is not None:
                verti_profiles[source] = pd.DataFrame(dict(
                    zip(whole_rs['ALTI'], whole_rs[rs_varname_dict[varname]])))
               
    # Put in global profiles dict
    profiles_all_dates[wanted_date] = verti_profiles

    
    #%% COMPUTATION of errors
    if error_computation and (len(profile_sources) > 1):
    
        errors = {}
        diff, bias, rmse = {}, {}, {}
    
        for i, var in enumerate(varname_list):
            try:
                lidar_zrange = list(verti_profiles['lidar'][var].keys())
                lidar_values = list(verti_profiles['lidar'][var])
                arome_zrange = list(verti_profiles['arome'][var].keys())
                arome_values = list(verti_profiles['arome'][var])
            
                errors[var] = tools.compute_errors(lidar_values, arome_values, 
                                              lidar_zrange, arome_zrange,
                                              method='interp')
            except KeyError:
                errors[var] = {}
                errors[var]['bias'] = None
                errors[var]['rmse'] = None
    
        print(f'errors: {errors}')
    
#%% COMPUTE MEAN verti profiles
mean_profiles = {}  # per source
for source in profile_sources:
    mean_dict = {}  # per var
    
    varname_dict = sources_var[source]
    varname_list = [item for sublist in varname_dict.values() for item in sublist]

    for var in varname_list:
                
        prof_dict_temp = {}
        for date in datetimes:
            try:
                prof_dict_temp[date] = profiles_all_dates[date][source][var]
                # prof_list_temp.append(profiles_all_dates.values())
            except KeyError:
                print(f'Source {source} does not have data for {date}')
                pass
            
        if var == 'wd':
            mean_dict[var] = pd.DataFrame(prof_dict_temp).mean(axis=1)
            mean_dict[var][:] = circmean(pd.DataFrame(prof_dict_temp), 
                                             axis=1, high=360)
        else:
            mean_dict[var] = pd.DataFrame(prof_dict_temp).mean(axis=1)
            
    mean_profiles[source] = mean_dict

#%% PLOT verti profiles
if plot_figure:
    
    nb_subplots = pd.DataFrame(sources_var).shape[0]        
    fig, ax = plt.subplots(1, nb_subplots, 
                           figsize=(3 + 2*nb_subplots, 6),
                           sharey=True)
    title = f'mean_profile_{date_start}_to_{date_end}'
    fig.suptitle(title)

    for source in profile_sources:
        varname_dict = sources_var[source]
            
        try:
            linecolor = color_dict_source[source]
        except KeyError:
            linecolor = color_dict_source['default']
            
        for i in varname_dict.keys():
            
            for j, var in enumerate(varname_dict[i]):
                linestyles = ['-', '--', '-.', ':']
                
                try:
                    # determine label for the source
                    if 'mnh_' in source:
                        sourcelabel1 = source.split('_')[0] + '_' + \
                            source.split('_')[2]
                    else:
                        sourcelabel1 = source
                        
                    # add var name if multiple vars plotted
                    # if len(varname_dict[i]) > 1:
                    if var in ['TKET', 'TOT_TKE']:
                        sourcelabel = sourcelabel1 + '_TKE_TOT'
                    else:
                        sourcelabel = sourcelabel1 + f'_{var}'
                    # else: 
                    # sourcelabel = sourcelabel1
                     
                    ax[i].plot(mean_profiles[source][var], 
                               mean_profiles[source][var].keys(), 
                               marker='+', linestyle=linestyles[j%4],
                               color=linecolor,
                               label=sourcelabel)
                except KeyError:
                    print(f'{source} data does not contain this key: {var}')
            
    for i in range(nb_subplots):
        var = varname_dict[i][0]
        varlow = var.lower()
        
        # ax[i].set_xlabel = var
        ax[i].set_ylim(0, max_height)
        ax[i].set_title(varname_dict[i][0])
        
        if varlow in ['t', 'theta', 'theta_v']:
            ax[i].set_xlim(272, 286)
            # ax[i].set_xlim(290, 308)
        elif varlow == 'q':
            ax[i].set_xlim(0, 0.020)
        elif varlow in ['tke', 'tke_var', 'tot_tke', 'tke_tot', 'tket', 'tke_eq']:
            ax[i].set_xlim(0, 3)
        elif varlow in ['tke_horiz', 'tke_verti']:
            ax[i].set_xlim(0, 2)
        elif varlow == 'ws':
            ax[i].set_xlim(0, 20)
        elif varlow == 'wd':
            ax[i].set_xlim(0, 360)
            ax[i].set_xticks(ticks=[0,90,180,270,360], 
                             labels=['N','E','S','W','N'])
        
        ax[i].grid()
    
        if error_computation:
            # add errors values on graph
            ax[i].text(.01, .99, 'Bias: {0}'.format(errors[var]['bias']), 
                       ha='left', va='top', transform=ax[i].transAxes)
            ax[i].text(.01, .95, 'RMSE: {0}'.format(errors[var]['rmse']), 
                       ha='left', va='top', transform=ax[i].transAxes)
            ax[i].legend(loc='upper right')
        else:
            ax[i].legend(loc='best')
            
        plt.subplots_adjust(left=0.07, right=0.95)

    if save_figure:
        tools.save_figure(f'{wanted_date}-{profile_sources}-{varname_list}_mean',
                          save_folder)
