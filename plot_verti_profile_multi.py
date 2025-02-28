#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for plotting vertical profiles from AROME .fa files

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

import global_variables as gv
import tools


folder_path = '/home/lunelt/Data/analysis/'
arome_profiles_folder = '/home/lunelt/Data/data_NEMO/arome_profiles/'

site = 'planier'
varname = 'wind'     # tke, t, u, v, q, ws, wd
varname_list = [
    # 'theta', 
    # 'q', 
    'tke',
    'tke_verti',
    'tke_horiz', 
    # 'ws', 
    # 'wd',
    ]

max_height = 500  # [m]
date_start, date_end = '2023-01-05 08:15:00', '2023-01-05 09:15:00'

rs_varname_dict = {
    'ws': 'FF',
    'wd': 'DD',
    }

profile_sources =  [
    'lidar',
    # 'arome',
    'mnh_02_1km',
    # 'mnh_05_1km_noEDKF_3DTKE',
    # 'mnh_021_200m',
    'mnh_022_40m',
    ]
mnh_expe = '1_planier_0105'

error_computation = False
plot_figure = True
save_figure = True
add_res_tke_frac = False

last_folder_name = '-'.join(profile_sources)
save_folder = f'figures/verti_profiles_{site}/0-{max_height}/{last_folder_name}/'

# Regular expression to match filenames and extract date and hour
pattern = r"arome\.AN\.(\d{4})(\d{2})(\d{2})\.(\d{2})"  # to have all date
# pattern = r"arome\.AN\.(\d{4})(\d{2})(\d{2})\.12"  # only midday
# pattern = r"verti_profile_arome_planier_(\d{4})(\d{2})(\d{2})\.(\d{2}).pickle"  # to have all date


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

# datetimes = [pd.Timestamp('20230105T1400')]

color_dict_source = {
    'arome': 'b',
    'lidar': 'k',
    'mnh_02_1km': 'r',
    'mnh_021_200m': 'orange',
    'mnh_022_40m': 'olive',
    'rs': 'grey',
    'default': 'green',
    }

# pre-processing of varname_list:
# ensure loading of prognostic variables
varname_prog_list = ['t', 'q', 'tke', 'u', 'v']
varname_load_list = [var for var in varname_list if var in varname_prog_list]
if 'theta' in varname_list:
    varname_load_list.append('t')
if 'theta_v' in varname_list:
    varname_load_list.append('t')
    varname_load_list.append('theta')
if ('ws' in varname_list) or ('wd' in varname_list):
    varname_load_list.append('u')
    varname_load_list.append('v')

#%% LOAD the verti profiles
for wanted_date in datetimes:

    verti_profiles = {}
    # profile_sources_orig = profile_sources.deep
    
    for source in profile_sources:
        print(source)
        if source == 'arome':
            verti_profiles[source] = {}
            
            datetime_formatted = wanted_date.strftime('%Y%m%d.%H')
            filename_pickle = f"verti_profile_arome_{site}_{datetime_formatted}.pickle"
            
            if filename_pickle in os.listdir(f'{arome_profiles_folder}/{site}/'):
                verti_profiles[source] = pd.read_pickle(
                    f'{arome_profiles_folder}/{site}/{filename_pickle}')
            else:
                continue
                # for varname in varname_load_list:
                #     temp = tools.extract_profile_arome(
                #         site, wanted_date, varname, max_height=max_height,
                #         # save_to_pickle=True,
                #         verbose=False)
                #     verti_profiles[source][varname] = temp[varname]
        #
        # elif source in ['02_1km', '021_200m', '022_40m']:  # MNH
        elif source[:4] == 'mnh_':  # MNH
            try:
                filename = tools.get_simu_filepath(
                        f'{mnh_expe}/{source[4:]}/', 
                        wanted_date=wanted_date, output_type='backup')
            except KeyError:
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
                ]

            # subset
            var3d = ds[vars_mnh]
            # var3d_1km = ds_1km[vars_mnh]
            # var3d_200m = ds_200m[vars_mnh]
            var3d = tools.flux_pt_to_mass_pt(var3d, only_basic_vars=False)
            # var3d_1km_cen = tools.flux_pt_to_mass_pt(var3d_1km, only_basic_vars=False)
            # var3d_200m_cen = tools.flux_pt_to_mass_pt(var3d_200m, only_basic_vars=False)
            var1d = var3d.isel(nj=index_lat, ni=index_lon).squeeze()
            
            # Diag
            var1d['RES_TKE'] = 0.5*(var1d['U2ME']+var1d['V2ME']+var1d['W2ME'])
            var1d['RES_TKE_VERTI'] = 0.5*(var1d['W2ME'])
            var1d['RES_TKE_HORIZ'] = 0.5*(var1d['U2ME']+var1d['V2ME'])
            var1d['SBG_TKE'] = var1d['TKEMME']
            var1d['TOT_TKE'] = var1d['SBG_TKE'] + var1d['RES_TKE']
            var1d['RES_TKE_FRAC'] = var1d['RES_TKE']/var1d['TOT_TKE']
            var1d['TOT_TKE_VERTI'] = var1d['RES_TKE_VERTI'] * (1/var1d['RES_TKE_FRAC'])
            var1d['TOT_TKE_HORIZ'] = var1d['RES_TKE_HORIZ'] * (1/var1d['RES_TKE_FRAC'])
            # var1d['TOT_TI'] = np.sqrt(1.5*var1d['TOT_TKE'])/var1d['WS']
            
            variable_correspondance_mnh_arome = {  # key is mnh, value is arome
                # 'TEMP': 't',
                'THT': 'theta',  # diag in arome
                'RVT': 'q',
                # 'TKET': 'tke',
                'TOT_TKE': 'tke',
                'UMME': 'u',
                # 'UT': 'u',
                'VMME': 'v',
                # 'VT': 'v', 
                'WMME': 'w',
                'TOT_TKE_VERTI': 'tke_verti',
                'TOT_TKE_HORIZ': 'tke_horiz',
                # 'WT': 'w',
                # 'WS': 'ws', # diag in both
                # 'WD': 'wd', # diag in both
                }
            vars_corr_mnh_aro_list = variable_correspondance_mnh_arome.keys()
            
            # change MNH var names
            vars_mnh_dict = {var: variable_correspondance_mnh_arome[var] for var in vars_corr_mnh_aro_list}
            var1d = var1d.rename_vars(vars_mnh_dict)
                
            verti_profiles[source] = var1d.to_dataframe()
            # keep only low layer of atmos (~ABL)
            # var3d_low = var3d.where(var3d.level<toplevel, drop=True)
            # var1d_column = var3d_low.isel(nj=index_lat, ni=index_lon).squeeze()
        #
        elif source == 'lidar':
            whole_lidar = tools.extract_profile_lidar(wanted_date, site=site)
            if site == 'salin-giraud':
                whole_lidar['tke'] = whole_lidar['tke_horiz']
            if whole_lidar is not None:
                 profile_lidar_dict = {
                    key: whole_lidar[key].to_dict() for key in whole_lidar.keys()}
                 verti_profiles[source] = pd.DataFrame(profile_lidar_dict)

        #
        elif source in ['rs', 'radiosounding']:
            whole_rs = tools.extract_profile_rs_mf(site, wanted_date)
            if whole_rs is not None:
                verti_profiles[source] = pd.DataFrame(dict(
                    zip(whole_rs['ALTI'], whole_rs[rs_varname_dict[varname]])))
    
    #%% DIAG

    # turn to pd.DataFrame to simplify diags
    if 'arome' in profile_sources:
        try:
            df = pd.DataFrame(verti_profiles['arome'])
            df['p'] = tools.height_to_pressure_std(df.index)
            df['theta'] = tools.potential_temperature_from_temperature(df['p'], 
                                                                       df['t'])
        except:
            pass
    # df['ws'], df['wd'] = tools.calc_ws_wd(df['u'], df['v'])
    
    if ('ws' in varname_list) or ('wd' in varname_list):
        # for source in ['arome', '02_1km', '021_200m', '022_40m']:
        for source in profile_sources:
            try:
                df = verti_profiles[source]
                df['ws'], df['wd'] = tools.calc_ws_wd(df['u'], df['v'])
            except:
                pass

    
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
    
    
    #%% PLOT verti profiles
    if plot_figure:
        if add_res_tke_frac:
            nb_subplots = len(varname_list) + 1
        else:
            nb_subplots = len(varname_list)
        
        fig, ax = plt.subplots(1, nb_subplots, 
                               figsize=(3 + 2*len(varname_list), 6),
                               sharey=True)
        fig.suptitle(wanted_date)
    
        for i, var in enumerate(varname_list):
            for source in profile_sources:
                # if 'mnh' in source:
                #     var = gv.variable_correspondance_arome_mnh[var]
                try:
                    linecolor = color_dict_source[source]
                except KeyError:
                    linecolor = color_dict_source['default']
                    
                try:
                    if 'mnh_' in source:
                        if source == 'mnh_02_1km':
                            # sourcelabel = 'mnh_1km_EDKF_1DTKE'
                            sourcelabel = 'mnh_1km'
                        if source == 'mnh_022_40m':
                            # sourcelabel = 'mnh_1km_EDKF_1DTKE'
                            sourcelabel = 'mnh_40m'
                        else:
                            sourcelabel = 'mnh_' + source[7:]
                    else:
                        sourcelabel = source
                        
                    ax[i].plot(verti_profiles[source][var], 
                               verti_profiles[source][var].keys(), 
                               marker='+', linestyle='--',
                               color=linecolor,
                               label=sourcelabel)
                except KeyError:
                    print(f'{source} data does not contain this key: {var}')
                    
                if add_res_tke_frac and source[:4]=='mnh_':
                    ax[nb_subplots-1].plot(
                        verti_profiles[source]['RES_TKE_FRAC'], 
                        verti_profiles[source]['RES_TKE_FRAC'].keys(), 
                        marker='+', linestyle='--',
                        color=linecolor,
                        label=sourcelabel)
                
            # ax[i].set_xlabel = var
            ax[i].set_ylim(0, max_height)
            ax[i].set_title(var)
            
            if var in ['t', 'theta', 'theta_v']:
                ax[i].set_xlim(280, 292)
                # ax[i].set_xlim(290, 308)
            elif var == 'q':
                ax[i].set_xlim(0, 0.020)
            elif var == 'tke':
                ax[i].set_xlim(0, 3)
            elif var in ['tke_horiz', 'tke_verti']:
                ax[i].set_xlim(0, 2)
            elif var == 'ws':
                ax[i].set_xlim(0, 25)
            elif var == 'wd':
                ax[i].set_xlim(0, 360)
                ax[i].set_xticks(ticks=[0,90,180,270,360], labels=['N','E','S','W','N'])
            
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
        
        if add_res_tke_frac:
            # ax[nb_subplots-1].set_xlabel = 'fraction of resolved tke'
            ax[nb_subplots-1].set_ylim(0, max_height)
            ax[nb_subplots-1].set_title('fraction of resolved tke')
            ax[nb_subplots-1].set_xlim(0, 1)
            ax[nb_subplots-1].grid()
        
        if save_figure:
            tools.save_figure(f'{wanted_date}-{profile_sources}',
                              save_folder)
