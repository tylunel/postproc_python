#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021

Plot vertical profile from LIDAR and RADAR data
Specific modif for article1 on irrigation breeze.
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import global_variables as gv
import pandas as pd
import tools
from scipy.stats import circmean, circstd

##############################

site = 'cendrosa'

if site == 'elsplans':
    source_obs_list = ['uhf', 
#                       'lidar', 
#                       'mast', 
                       'radiosonde']
elif site in ['cendrosa', 'linyola']:
    source_obs_list = ['uhf', 
                       'windcube', 
#                       'mast', 
                       'radiosonde']
elif site in ['irta', 'irta-corn']:
    source_obs_list = ['windrass', 
                       'mast'
                       ]

wanted_date = '20210722-1200'
toplevel = 2500

# 'uhf', 'windcube', 'mast'
simu_list = ['irr_d2_old', 
             'std_d2_old',
             ]

# Path in simu is the direct 1d column
straight_profile = True
# Path in simu is average of neighbouring grid points
mean_profile = True
column_width = 8

figsize = [8, 7] #small for presentation: [6, 5], big: [9, 7]

save_plot = True
save_folder = './fig/'
plt.rcParams.update({'font.size': 11})

##############################
colordict = {'irr_d2': 'g', 
             'std_d2': 'r',
             'irr_d1': 'g', 
             'std_d1': 'r', 
             'irr_d2_old': 'g', 
             'std_d2_old': 'r', 
             'obs_uhf': 'k',
             'obs_mast': 'k',
             'obs_windcube': 'k',
             'obs_windrass': 'm',
             'obs_lidar': 'm',
             'obs_radiosonde': 'k',
             }

markerdict = { 
             'obs_uhf': '+',
             'obs_mast': '*',  # 1 = tri down
             'obs_windcube': 'x',
             'obs_lidar': 'x',
             'obs_windrass': 'x',
             'obs_radiosonde': '.',
             }

# for conversion of level asl to agl
alti_site = gv.whole[site]['alt']

wanted_month = str(pd.Timestamp(wanted_date).month).zfill(2)  # format with 2 figures
wanted_day = str(pd.Timestamp(wanted_date).day).zfill(2)

obs_dict = {}
# load dataset and set parameters
if site == 'elsplans':
    
    # -------- UHF ---------
    if 'uhf' in source_obs_list:
        ds_temp = xr.open_dataset(
            gv.global_data_liaise + '/elsplans/UHF_low/' + \
            f'LIAISE_ELS-PLANS_LAERO_UHFWindProfiler-LowMode-2MIN_L2_2021{wanted_month}_V1.nc')
        ds_temp['WS'], ds_temp['WD'] = tools.calc_ws_wd(ds_temp['UWE'], ds_temp['VSN'])
        
        # keep time of interest
        ds_temp['time_dist'] = np.abs(ds_temp.time - pd.Timestamp(wanted_date).to_datetime64())
        ds_t = ds_temp.where(ds_temp['time_dist'] == ds_temp['time_dist'].min(), 
                             drop=True).squeeze()
        # check that time dist is ok
        if ds_t['time_dist'] > pd.Timedelta(35, 'min'):
            ds_t = ds_t * np.nan
        # convert level asl to agl
        ds_t = ds_t.rename({'level': 'level_asl'})
        ds_t['level_agl'] = ds_t.level_asl-alti_site
        ds_t = ds_t.set_coords(['level_agl'])  
        # integration_time is time between two data pts [min]
        ds_t['integration_time'] = 2

        obs_dict['uhf'] = ds_t
        
        if obs_dict['uhf']['WS'].isnull().all():  # if only NaN values
            print('No UHF data available')
            source_obs_list.remove('uhf')
    
    # -------- LIDAR ---------
    if 'lidar' in source_obs_list:
        ds_lidar = tools.open_ukmo_lidar(
                gv.global_data_liaise + f'/elsplans/lidar/2021{wanted_month}/',
                filter_low_data=True, level_low_filter=60, create_netcdf=False,
                )
        ds_lidar['time_dist'] = np.abs(ds_lidar.time - pd.Timestamp(wanted_date).to_datetime64())
    #    ds_temp['time_dist'] = np.abs(np.array([pd.Timestamp(time) - pd.Timestamp(wanted_date) for time in ds_temp.time.values]))
        ds_t = ds_lidar.where(ds_lidar['time_dist'] == ds_lidar['time_dist'].min(), 
                              drop=True).squeeze()
        # check that time dist is ok
        if ds_t['time_dist'] > pd.Timedelta(35, 'min'):
            ds_t = ds_t * np.nan
        # integration_time is time between two data pts [min]
        ds_t['integration_time'] = 30
        
        obs_dict['lidar'] = ds_t
    
    # -------- MAST 50m ---------
    if 'mast' in source_obs_list:
        datafolder = gv.global_data_liaise + '/elsplans/mat_50m/5min_v4/'
        out_filename_obs = f'LIAISE_ELS-PLANS_UKMO_MTO-05MIN_L2_2021{wanted_month}{wanted_day}_V4.0.nc'
        ds_temp = xr.open_dataset(datafolder + out_filename_obs)
        # keep time of interest
        ds_temp['time_dist'] = np.abs(ds_temp.time - pd.Timestamp(wanted_date).to_datetime64())
        ds_t = ds_temp.where(ds_temp['time_dist'] == ds_temp['time_dist'].min(), 
                             drop=True).squeeze()
        # if two datetime are as close to required datetime, keep the first
        try:
            ds_t = ds_t.isel(time=0)
            print("""Warning: Multiple data found close to wanted_date -
                      only first is kept""")
        except ValueError:
            pass
        # check that time dist is ok
        if ds_t['time_dist'] > pd.Timedelta(35, 'min'):
            ds_t = ds_t * np.nan
            
        # keep verti wind profile only
        ds_verti = xr.Dataset()
        ds_verti['time'] = ds_t.time
        ds_verti['WD'] = xr.DataArray([ds_t['DIR_10m'].values, ds_t['DIR_25m'].values, ds_t['DIR_50m'].values], 
                             coords={'level_agl': [10, 25, 50]})
        ds_verti['WS'] = xr.DataArray([ds_t['UTOT_10m'].values, ds_t['UTOT_25m'].values, ds_t['UTOT_50m'].values], 
                             coords={'level_agl': [10, 25, 50]})
        # integration_time is time between two data pts [min]
        ds_t['integration_time'] = 30

        obs_dict['mast'] = ds_verti
    
    # -------- RADIOSOUNDINGS ---------------
    if 'radiosonde' in source_obs_list:
        datafolder = gv.global_data_liaise + site + '/radiosoundings/'
        try:
            filename = tools.get_obs_filename_from_date(
                    datafolder, wanted_date,
                    dt_threshold=pd.Timedelta('0 days 00:35:00'),
                    regex_date='202107\d\d.\d\d\d\d')
            
            obs = tools.open_ukmo_rs(datafolder, filename)
            obs = obs.rename({'height': 'level_agl', 'windSpeed': 'WS', 
                              'windDirection': 'WD', 'time': 'time_i'})
            obs_low = obs.where(xr.DataArray(obs['level_agl'].values<toplevel, dims='index'), 
                                     drop=True)
            delta_time = pd.Timedelta((obs_low.time_i.max() - obs_low.time_i.min()).values / 2)
            mean_time = obs_low.time_i.min().values[()] + delta_time
            obs_low['time'] = mean_time
            
            if wanted_date == '20210716-2000':  # issue in this particular radiosounding with level data
                obs_low['level_agl'][:40] = np.array(
                        [0,1,3,6,9,12,15,18,22,26,29,33,36,39,43,46,50,54,58,62,66,69,73,
                         78,81,85,88,92,96,100,104,108,112,116,120,124,127,131,135,139])  # this array comes from RS at 21pm
                obs_low['level_agl'][40:] = obs_low['level_agl'][40:] + obs_low['level_agl'][39]
            # integration_time is time between two data pts [min]
            obs_low['integration_time'] = 0.016  #=1s
            
            obs_dict['radiosonde'] = obs_low
        except ValueError:
            print('No radiosonde available')
            source_obs_list.remove('radiosonde')
    
    
elif site in ['cendrosa', 'linyola']:
    
    # -------- UHF -------------
    if 'uhf' in source_obs_list:
        ds_uhf = xr.open_dataset(
            gv.global_data_liaise + '/cendrosa/UHF_high/' + \
            f'MF-CNRM-Toulouse_UHF-RADAR_L2B-LM-Hourly-Mean_2021-{wanted_month}-{wanted_day}T00-00-00_1D_V2-10.nc'
            )
        ds_uhf['WS'], ds_uhf['WD'] = tools.calc_ws_wd(ds_uhf['UWE'], ds_uhf['VSN'])
        # keep time of interest
        ds_uhf['time_dist'] = np.abs(ds_uhf.time - pd.Timestamp(wanted_date).to_datetime64())
        ds_t = ds_uhf.where(ds_uhf['time_dist'] == ds_uhf['time_dist'].min(), 
                             drop=True).squeeze()
        # convert level asl to agl
        ds_t['level_agl'] = ds_t['level'] - 137  # correction of level, pers. comm. Alexandre Paci
        # if two datetime are as close to required datetime, keep the first
        try:
            ds_t = ds_t.isel(time=0)
            print("""Warning: Multiple data found close to wanted_date -
                      only first is kept""")
        except ValueError:
            pass
        # check that time dist is ok
        if ds_t['time_dist'] > pd.Timedelta(35, 'min'):
            ds_t = ds_t * np.nan
        # integration_time is time between two data pts [min]
        ds_t['integration_time'] = 60
        # Write result for this source
        obs_dict['uhf'] = ds_t
        
        if obs_dict['uhf']['WS'].isnull().all():  # if only NaN values
            print('No UHF data available')
            source_obs_list.remove('uhf')
        
    # ---------- WINDCUBE ---------
    if 'windcube' in source_obs_list:
        ds_wcube = xr.open_dataset(
            gv.global_data_liaise + '/cendrosa/lidar_windcube/' + \
            f'LIAISE_LA-CENDROSA_CNRM_LIDARwindcube-WIND_L2_2021{wanted_month}{wanted_day}_V1.nc')
        
        # dataorigin == 'windcube':
        ds_wcube = ds_wcube.drop_dims(['level'])
        ds_wcube['level'] = xr.DataArray(ds_wcube.ff_class.data, 
            coords={'level': ds_wcube.ff_class.data,})
        # unify dimension coordinate of all variable
        for var in ds_wcube:
            ds_wcube[var] = ds_wcube[var].swap_dims({f'{var}_class': 'level'})
        # drop old level coordinates
        ds_wcube = ds_wcube.drop_dims(['ff_class', 'dd_class', 'ffmin_class', 'ffmax_class', 
                           'ffstd_class', 'data_availabily_class', 
                           'CNR_class', 'CNRmin_class'])
        # rename variables
        ds_wcube = ds_wcube.rename({'ff': 'WS', 'dd': 'WD'})
        
        ds_wcube['time_dist'] = np.abs(ds_wcube.time - pd.Timestamp(wanted_date).to_datetime64())
        ds_t = ds_wcube.where(ds_wcube['time_dist'] == ds_wcube['time_dist'].min(), 
                             drop=True).squeeze()
        # check that time dist is ok
        if ds_t['time_dist'] > pd.Timedelta(35, 'min'):
            ds_t = ds_t * np.nan
        ds_t = ds_t.rename({'level': 'level_agl'})
        # integration_time is time between two data pts [min]
        ds_t['integration_time'] = 10
        
        obs_dict['windcube'] = ds_t
    
    # --------- MAST ---------
    if 'mast' in source_obs_list:
        freq = 30
        datafolder = gv.global_data_liaise + f'/cendrosa/{freq}min/'
        filename = f'LIAISE_LA-CENDROSA_CNRM_MTO-FLUX-{freq}MIN_L2_2021-{wanted_month}-{wanted_day}_V2.nc'
        ds_mast = xr.open_dataset(datafolder + filename)
        # keep time of interest
        ds_mast['time_dist'] = np.abs(ds_mast.time - pd.Timestamp(wanted_date).to_datetime64())
        ds_t = ds_mast.where(ds_mast['time_dist'] == ds_mast['time_dist'].min(), 
                             drop=True).squeeze()
        # check that time dist is ok
        if ds_t['time_dist'] > pd.Timedelta(35, 'min'):
            ds_t = ds_t * np.nan
        
        # keep verti wind profile only
        ds_verti = xr.Dataset()
        ds_verti['time'] = ds_t.time
        ds_verti['WD'] = xr.DataArray([ds_t['wd_1'].values, ds_t['wd_2'].values, 
                                       ds_t['wd_3'].values, ds_t['wd_4'].values,], 
                             coords={'level_agl': [3, 10, 25, 50]})
        ds_verti['WS'] = xr.DataArray([ds_t['ws_1'].values, ds_t['ws_2'].values, 
                                       ds_t['ws_3'].values, ds_t['ws_4'].values,], 
                             coords={'level_agl': [3, 10, 25, 50]})    
        # integration_time is time between two data pts [min]
        ds_t['integration_time'] = freq
        
        # Write result for this source
        obs_dict['mast'] = ds_verti
    
    # -------- RADIOSOUNDINGS ---------------
    if 'radiosonde' in source_obs_list:
        datafolder = gv.global_data_liaise + site + '/radiosoundings/'
        try:
            filename = tools.get_obs_filename_from_date(
                    datafolder, wanted_date,
                    dt_threshold=pd.Timedelta('0 days 00:35:00'),
                    regex_date='202107\d\d.\d\d\d\d')

            obs = xr.open_dataset(datafolder + filename)
            obs = obs.rename({'altitude': 'level_asl', 'windSpeed': 'WS', 
                              'windDirection': 'WD', 'time': 'time_i'})
            obs['level_agl'] = obs['level_asl'] - gv.sites[site]['alt']
            obs_low = obs.where(xr.DataArray(obs['level_agl'].values<toplevel, dims='time_i'), 
                                     drop=True)
            delta_time = pd.Timedelta((obs_low.time_i.max() - obs_low.time_i.min()).values / 2)
            mean_time = obs_low.time_i.min().values[()] + delta_time
            obs_low['time'] = mean_time
    
            # integration_time is time between two data pts [min]
            obs_low['integration_time'] = 0.016  #=1s
            
            obs_dict['radiosonde'] = obs_low
        except ValueError:
            print('No radiosonde available')
            source_obs_list.remove('radiosonde')
        
        
elif site in ['irta', 'irta-corn']:
    
    # ---------- WINDRASS -----------
    if 'windrass' in source_obs_list:
        datafolder = gv.global_data_liaise + f'/irta-corn/windrass/'
        filename = f'LIAISE_IRTA-ET0_SMC_WINDRASS_L0_2021_{wanted_month}{wanted_day}_V01.nc'
        ds_temp = xr.open_dataset(datafolder + filename)
        
        ds_temp = ds_temp.rename({'Z': 'level_agl'})
        
        ds_temp['time_dist'] = np.abs(ds_temp.time - pd.Timestamp(wanted_date).to_datetime64())
        ds_t = ds_temp.where(ds_temp['time_dist'] == ds_temp['time_dist'].min(), 
                             drop=True).squeeze()
        # check that time dist is ok
        if ds_t['time_dist'] > pd.Timedelta(35, 'min'):
            ds_t = ds_t * np.nan
        
        obs_dict['windrass'] = ds_t
    
    # ---------- SEB Station -----------
    if 'mast' in source_obs_list:
        datafolder = gv.global_data_liaise + '/irta-corn/seb/'
        filename = 'LIAISE_IRTA-CORN_UIB_SEB-10MIN_L2.nc'
        ds_temp = xr.open_dataset(datafolder + filename)
        
        ds_temp['level_agl'] = 2
        
        ds_temp = ds_temp.where(~ds_temp.time.isnull(), drop=True)
        
        ds_temp['time_dist'] = np.abs(ds_temp.time - pd.Timestamp(wanted_date).to_datetime64())
        ds_t = ds_temp.where(ds_temp['time_dist'] == ds_temp['time_dist'].min(), 
                             drop=True).squeeze()
        obs_dict['mast'] = ds_t
    
else:
    raise KeyError("No radar data for this site")



#%% PLOT
# --- OBS ---

fig, ax = plt.subplots(1, 2, sharey=True, figsize=figsize,)

for source in source_obs_list:
    # exact time of obs (may vary depending on sources)
    obstime = pd.Timestamp(obs_dict[source].time.values).strftime('%d_%H:%M')
    if source == 'radiosonde':
        markersize = 7
    else:
        markersize = 25
    
    ax[0].scatter(obs_dict[source]['WS'], obs_dict[source].level_agl, 
               label=f'obs_{source}',
               color=colordict[f'obs_{source}'],
               marker=markerdict[f'obs_{source}'],
#               linestyle=':',
               s=markersize,
               )
    
    ax[1].scatter(obs_dict[source]['WD'], obs_dict[source].level_agl, 
               label=f'obs_{source}',
               color=colordict[f'obs_{source}'],
               marker=markerdict[f'obs_{source}'],
#               linestyle=':',
               s=markersize,
               )


# --- SIMU ---

lat, lon = gv.whole[site]['lat'], gv.whole[site]['lon']

ws1d = {}
wd1d = {}
height = {}

for model in simu_list:     # model will be 'irr' or 'std'
    # retrieve and open file
    filename_simu = tools.get_simu_filepath(model, wanted_date)
    ds = xr.open_dataset(filename_simu)
    # put u, v, w in middle of grid
    ds = tools.center_uvw(ds)
    
    # find indices from lat,lon values 
    index_lat, index_lon = tools.indices_of_lat_lon(ds, lat, lon)
    # keep only variable of interest
#    var3d = ds[var_simu]
    var3d = ds[['UT','VT']]
    # keep only low layer of atmos (~ABL)
    var3d_low = var3d.where(var3d.level<toplevel, drop=True)
    
    if mean_profile:
        ut_3d_column = var3d_low['UT'][
            :, 
            int(index_lat-column_width/2)+1:int(index_lat+column_width/2)+1, 
            int(index_lon-column_width/2)+1:int(index_lon+column_width/2)+1]
        vt_3d_column = var3d_low['VT'][
            :, 
            int(index_lat-column_width/2)+1:int(index_lat+column_width/2)+1, 
            int(index_lon-column_width/2)+1:int(index_lon+column_width/2)+1]
        
#        wd_3d_column = mpcalc.wind_direction(ut_3d_column, vt_3d_column)
#        ws_3d_column = mpcalc.wind_speed(ut_3d_column, vt_3d_column)
        ws_3d_column, wd_3d_column = tools.calc_ws_wd(ut_3d_column, 
                                                      vt_3d_column)
        
        ws_1d = ws_3d_column.mean(dim=['nj', 'ni'])
        ws_1d_std = ws_3d_column.std(dim=['nj', 'ni'])
        
        # Averaging direction is not trivial, use of circular mean here (cf https://en.wikipedia.org/wiki/Circular_mean)
#        wd_1d = wd_3d_column.mean(dim=['nj', 'ni'])
#        wd_1d_std = wd_3d_column.std(dim=['nj', 'ni'])
        cmean_interm = circmean(np.array(wd_3d_column), high=360, axis=1)
        wd_1d = circmean(cmean_interm, high=360, axis=1)
        cstd_interm = circstd(np.array(wd_3d_column), high=360, axis=1)
        wd_1d_std = np.mean(cstd_interm, axis=1)
        
        
        # SIMU PLOT
        # Wind Speed
        simu_time = pd.Timestamp(var3d.time.values).strftime('%d_%H:%M')
        
        if 'irr' in model:
            label_legend = 'simu_IRR'
        elif 'std' in model:
            label_legend = 'simu_STD'
        
        ax[0].plot(ws_1d.data, ws_1d.level, 
                 ls='--', 
                 color=colordict[model], 
                 label=label_legend,
                 )
        
        ax[0].fill_betweenx(
                ws_1d.level,
                ws_1d.data + ws_1d_std.data,
                ws_1d.data - ws_1d_std.data,
                alpha=0.2,
                facecolor=colordict[model],
                )
        # Wind Direction
        ax[1].plot(wd_1d, ws_1d.level, 
                 ls='--', 
                 color=colordict[model], 
                 label=label_legend,
                 )
        ax[1].fill_betweenx(
                ws_1d.level,
                wd_1d + wd_1d_std,
                wd_1d - wd_1d_std,
                alpha=0.2,
                facecolor=colordict[model],
                )
        
ax[0].grid()
ax[0].set_xlim([0,9])
ax[0].set_xlabel('wind speed [m s$^{-1}$]')
ax[0].set_ylabel('height AGL [m]')

ax[1].set_xlim([0,360])
ax[1].set_xticks([0, 90, 180, 270, 360], ['N', 'E', 'S', 'W', 'N'])
ax[1].set_xlabel('wind direction')
ax[1].grid()
ax[1].legend(loc='upper left')

plt.ylim([0, toplevel])
plot_title = f"Wind profile over {gv.sites[site]['longname']}"
fig.suptitle(plot_title, y=0.93)

if save_plot:
    tools.save_figure(plot_title, save_folder)


