#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021

Cf also plot_verti_profile.py

"""
import numpy as np
import tools
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import global_variables as gv


########## Independant parameters ###############
wanted_date = '20210716-2100'   #1910 and 2110 relevant for windSpeed
site = 'cendrosa'  # 'cendrosa', 'elsplans', 'irta'

# variable name from MNH files: 'THT', THTV 'RVT', 'WS'
var_simu = 'THT'
# variable name from obs files: 'potentialTemperature', 'mixingRatio', virtualPotentialTemperature, windSpeed
var_obs = 'potentialTemperature'

xlabel = 'Wind speed [m s$^{-1}$]'  #Specific humidity [kg kg$^{-1}$], Potential temperature [K], Wind speed [m s$^{-1}$]
coeff_corr = 1  #to switch from obs to simu2

vmin, vmax = 297, 308  # for THT on 16th
#vmin, vmax = 304, 312  # for THT on 21st
# vmin, vmax = 0, 12  # for WS
# vmin, vmax = None, None  # for RVT

simu_list = [
            'irrswi1_d1_16_10min',
            'irrlagrip30_d1_16_10min',
            'std_d1', 
            ]

simu_names = {
        'irrswi1_d1': 'FC_IRR',
        'irrswi1_d1_16_10min': 'FC_IRR',
        'std_d1':  'NOIRR',
        'irrlagrip30_d1': 'THOLD_IRR',
        'irrlagrip30_d1_16_10min': 'THOLD_IRR',
        }

dict_marinada_arrival_timedelta = {
        'irrswi1_d1': -pd.Timedelta(30, 'm'),
        'irrswi1_d1_16_10min': -pd.Timedelta(30, 'm'),
        'irrlagrip30_d1': -pd.Timedelta(40, 'm'),
        'irrlagrip30_d1_16_10min': -pd.Timedelta(40, 'm'),
        'std_d1': -pd.Timedelta(70, 'm'),
        }

time_offset = False
simu_only = False

# Path in simu is average of neighbouring grid points
mean_profile = False
column_width = 3
# Path in simu follows real RS path  #issue: fix discontinuities
follow_rs_position = False

# highest level AGL plotted
toplevel = 2000

plot_title = ''
save_plot = True
save_folder = f'./fig/verti_profiles/{var_simu}/'
figsize=(4.5, 7)

##################################################

if site in ['cendrosa', 'elsplans', 'irta']:
    lat = gv.sites[site]['lat']
    lon = gv.sites[site]['lon']
else:
    raise ValueError('Site without radiosounding')

if var_obs == 'mixingRatio':
    coeff_corr = 0.001

fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111)

colordict = gv.colordict

#%% OBS PARAMETERS
#if site == 'elsplans_alt':
#    datafolder = \
#            '/cnrm/surface/lunelt/data_LIAISE/'+ 'elsplans' +'/radiosoundings/'
#else:
#    datafolder = \
#            gv.global_data_liaise + site + '/radiosoundings/'
#
#filename = tools.get_obs_filename_from_date(
#        datafolder, 
#        wanted_date,
#        dt_threshold=pd.Timedelta('0 days 00:45:00'),
#        regex_date='202107\d\d.\d\d\d\d')
        

#%% LOAD OBS DATASET
if simu_only == False:
    try:
        if site == 'cendrosa':
            datafolder = gv.global_data_liaise + site + '/radiosoundings/'
            filename = tools.get_obs_filename_from_date(datafolder,  wanted_date,
                                                        dt_threshold=pd.Timedelta('0 days 00:45:00'),
                                                        regex_date='202107\d\d.\d\d\d\d')
            obs = xr.open_dataset(datafolder + filename)
        elif site == 'elsplans':
            datafolder = gv.global_data_liaise + site + '/radiosoundings/'
            filename = tools.get_obs_filename_from_date(datafolder,  wanted_date,
                                                        dt_threshold=pd.Timedelta('0 days 00:45:00'),
                                                        regex_date='202107\d\d.\d\d\d\d')
            obs = tools.open_ukmo_rs(datafolder, filename)
        elif site == 'irta':
            datafolder = gv.global_data_liaise + '/irta-corn/windrass/'
            filename = f'LIAISE_IRTA-ET0_SMC_WINDRASS_L0_2021_{wanted_date[4:6]}{wanted_date[6:8]}_V01.nc'
            obs = xr.open_dataset(datafolder + filename)    
            obs['time_dist'] = np.abs(obs.time - pd.Timestamp(wanted_date).to_datetime64())
            ds_t = obs.where(obs['time_dist'] == obs['time_dist'].min(), drop=True).squeeze()
            # check that time dist is ok
            if ds_t['time_dist'] > pd.Timedelta(35, 'min'):
                ds_t = ds_t * np.nan
            obs = ds_t
        obs_available = True
    except FileNotFoundError:
        obs_available = False
else:
    obs_available = False
    
#%% OBS PLOT

#p_obs = obs.pressure.values * units.hPa
#
#if obs.temperature.mean().values > 200:
#    T_obs = (obs.temperature).values * units.kelvin
#else:
#    T_obs = (obs.temperature).values * units.degC
#
#if obs.dewPoint.mean().values > 200:
#    Td_obs = (obs.dewPoint).values * units.kelvin
#else:
#    Td_obs = (obs.dewPoint).values * units.degC
#
if obs_available:
    if site == 'cendrosa':
        obs = obs.rename({'altitude': 'level_asl'})
        obs['level_agl'] = obs.level_asl - gv.sites[site]['alt']
        obs['pressure'] = obs['pressure']* 100  # convert from hPa to Pa
        # keep only low layer of atmos (~ABL)
        obs_low = obs.where(xr.DataArray(obs.level_agl.values<toplevel, dims='time'), 
                            drop=True)
    elif site == 'elsplans':
        obs = obs.rename({'height': 'level_agl'})
        obs['pressure'] = obs['pressure']* 100  # convert from hPa to Pa
        obs['temperature'] = obs['temperature'] + 273.15 # convert from °C to K
        # keep only low layer of atmos (~ABL)
        obs_low = obs.where(xr.DataArray(obs.level_agl.values<toplevel, dims='index'), 
                            drop=True)
    elif site == 'irta':
        obs = obs.rename({'Z': 'level_agl', 'AIR_T': 'temperature'})
        obs['level_asl'] = obs['level_agl'] + gv.sites[site]['alt']
        obs['pressure'] = tools.height_to_pressure_std(obs['level_asl'])
        # keep only low layer of atmos (~ABL)
        obs_low = obs.where(obs['level_agl'] < toplevel, drop=True)
    
    # get time of radiosounding
    # __ release time
    release_time = pd.Timestamp(obs.time.values[0]).strftime('%H:%M')
    # __ mean time of plotted radiosounding
    # time_array = obs.where(obs.level_agl<toplevel, drop=True).time.data
    # mean_obs_time = pd.Timestamp(np.nanmean([tsp.value for tsp in time_array])).strftime('%H:%M')
    # time to use afterward:
    # obstime = mean_obs_time
    obstime = release_time
    
    obs_low['potentialTemperature'] = tools.potential_temperature_from_temperature(
            obs_low['pressure'], obs_low['temperature'])
    obs_low['virtualPotentialTemperature'] = \
        obs_low['potentialTemperature']*(1 + 0.61*obs_low['mixingRatio']/1000)
    
    obs_low[var_obs] = obs_low[var_obs]*coeff_corr
    
    ax.plot(obs_low[var_obs], obs_low['level_agl'],
            # label='obs_radiosondes',
            label=f'obs_radiosonde_{obstime}',
            color=colordict['obs'],
            )

    

## - add wind barbs
#wind_speed_obs = obs.windSpeed.values * units.meter_per_second
#wind_dir_obs = obs.windDirection.values * units.degrees
#u_obs, v_obs = mpcalc.wind_components(wind_speed_obs, wind_dir_obs)
#n = 30  #keep data every nth point
#skew.plot_barbs(p_obs[1::n], u_obs[1::n], v_obs[1::n])

#%% LOAD SIMU DATASET
var1d = {}
height = {}

for model in simu_list:
    print('model: ', model)
    
    if time_offset:
        wanted_date_model = pd.Timestamp(wanted_date) + dict_marinada_arrival_timedelta[model]
    else:
        wanted_date_model = pd.Timestamp(wanted_date)
        # label = f'simu_{simu_names[model]}'
    
    # retrieve and open file
    output_freq = gv.output_freq_dict[model]
    filename_simu = tools.get_simu_filepath(model, wanted_date_model, 
                                            file_suffix='',  #'dg' or ''
                                            out_suffix='.OUT',
                                            output_freq=output_freq)
    ds = xr.open_dataset(filename_simu)
    ds['THTV'] = ds['THT']*(1 + 0.61*ds['RVT'])
    if var_simu == 'WS':
        ds = tools.center_uvw(ds)
        ds['WS'], _ = tools.calc_ws_wd(ds['UT'], ds['VT'])
    
    # find indices from lat,lon values 
    index_lat, index_lon = tools.indices_of_lat_lon(ds, lat, lon, 
                                                    verbose=False)
    # keep only variable of interest
    var3d = ds[var_simu]
    # keep only low layer of atmos (~ABL)
    var3d_low = var3d.where(var3d.level<toplevel, drop=True)
    
    # get simulation file datetime
    try:
        simu_time = pd.Timestamp(var3d.time.values).strftime('%H:%M')
    except:
        simu_time = pd.Timestamp(var3d.time.values[0]).strftime('%H:%M')
        
    if time_offset:
        label = f'simu_{simu_names[model]}_{simu_time}'
    else:
        label = f'simu_{simu_names[model]}_{simu_time}'
    
    if mean_profile:
        var3d_column = var3d_low.isel(
            nj=np.arange(int(index_lat-column_width/2),int(index_lat+column_width/2)),
            ni=np.arange(int(index_lon-column_width/2),int(index_lon+column_width/2)).squeeze()
            )
        var1d_column = var3d_column.mean(dim=['nj', 'ni'])
        var1d_column_std = var3d_column.std(dim=['nj', 'ni'])
        ax.plot(var1d_column.data, var1d_column.level,
                label=label,
                c=colordict[model],
                )
        ax.fill_betweenx(var1d_column.level, 
                          var1d_column.data - var1d_column_std.data,
                          var1d_column.data + var1d_column_std.data,
                          alpha=0.3, 
                          facecolor=colordict[model],
                          )
    else:  # straight profile
        var1d_column = var3d_low.isel(nj=index_lat, ni=index_lon).squeeze()
        # SIMU PLOT
        ax.plot(var1d_column.data, var1d_column.level,
                 ls='-', 
                 color=colordict[model],
                 label=label,
                 )
        
    # Realistic path of radiosounding (with interpolation)
    if follow_rs_position:
        var1d[model] = []
        height[model] = []
        for i, h in enumerate(obs.height):
            if not pd.isna(h):
                lat_i = obs.latitude[i].data
                lon_i = obs.longitude[i].data
                index_lat, index_lon = tools.indices_of_lat_lon(ds, lat_i.data, lon_i.data)
                var1d_temp = var3d_low[0, :, index_lat, index_lon]
                height[model].append(float(h))
                var1d[model].append(float(var1d_temp.interp(level=h)))
    
        ax.plot(var1d[model], height[model], 
                 ls=':', 
                 color=colordict[model], 
                 label=model+'_interp'
                 )

#TODO: add wind barbs
#TODO: add CAPE and CIN ?


#%% GRAPH ESTHETIC

ax.set_title(plot_title)

ax.set_ylabel('Height a.g.l. [m]')
ax.set_ylim([0, toplevel])

ax.set_xlabel(xlabel)
#ax.set_xticks(ticks = np.arange(vmin, vmax, 2),)
ax.set_xlim([vmin, vmax])
    
ax.legend(loc='upper left')
ax.grid()
#ax.tight_layout()
plt.subplots_adjust(top=0.95,left=0.2)

#plt.show()


#add special lines
if mean_profile:
    figname = 'verti mean profile {0}-{1}-{2}-{3}pts'.format(
        var_simu, site, wanted_date, column_width)
else:
    figname = 'verti profile {0}-{1}-{2}'.format(
        var_simu, site, wanted_date)

#%% Save plot
if save_plot:
    tools.save_figure(figname, save_folder)


    