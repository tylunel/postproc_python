#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021

Cf also plot_verti_profile.py

example for skewT graph here:
    https://unidata.github.io/MetPy/latest/examples/plots/Skew-T_Layout.html#sphx-glr-examples-plots-skew-t-layout-py
"""
import numpy as np
import tools
import pandas as pd
import matplotlib.pyplot as plt
from metpy.units import units
import metpy.calc as mpcalc
import xarray as xr
import global_variables as gv


########## Independant parameters ###############
wanted_date = '20210722-1200'
site = 'elsplans'  # 'cendrosa', 'elsplans', 'irta'

# variable name from MNH files: 'THT', 'RVT'
var_simu = 'MRV'

if var_simu == 'THT':
    vmin, vmax = 306, 314.5
    var_simu_longname = 'Potential temperature'
    # variable name from obs files: 'potentialTemperature', 'mixingRatio'
    var_obs = 'potentialTemperature'
    coeff_corr = 1
if var_simu in ['RVT', 'MRV']:
    var_simu_longname = 'Specific humidity'
    # variable name from obs files: 'potentialTemperature', 'mixingRatio'
    var_obs = 'mixingRatio'
    if var_simu == 'RVT':  # in kg/kg
        coeff_corr = 0.001
        vmin, vmax = 0.04, 0.014
    elif var_simu == 'MRV':  # in g/kg
        coeff_corr = 1
        vmin, vmax = 4, 14

simu_list = [
            'std_d2_old', 
            'irr_d2_old'
            ]

# Vertical path to show in simu:
# Path in simu is the direct 1d column
straight_profile = True
# Path in simu is average of neighbouring grid points
mean_profile = True
column_width = 10
# Path in simu follows real RS path  #issue: fix discontinuities
follow_rs_position = False

# highest level AGL plotted
toplevel = 2500

save_plot = True
save_folder = '/home/lunelt/Documents/redaction/article1_irrigation_breeze/fig/'
figsize=(4.5, 7)
plt.rcParams.update({'font.size': 11})

##################################################

if site in ['cendrosa', 'elsplans', 'irta']:
    lat = gv.sites[site]['lat']
    lon = gv.sites[site]['lon']
else:
    raise ValueError('Site without radiosounding')

fig = plt.figure(figsize=figsize)

colordict = {'irr_d2': 'g', 
             'std_d2': 'r',
             'irr_d1': 'g', 
             'std_d1': 'r', 
             'irr_d2_old': 'g', 
             'std_d2_old': 'r', 
             'obs': 'k'}

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

if site == 'cendrosa':
    obs = obs.rename({'altitude': 'level_asl'})
    obs['level_agl'] = obs.level_asl - gv.sites[site]['alt']
    obs['pressure'] = obs['pressure']* 100  # convert from hPa to Pa
    # keep only low layer of atmos (~ABL)
    obs_low = obs.where(xr.DataArray(obs.level_agl.values<toplevel, dims='time'), 
                        drop=True)
    
    # --------- MAST ---------
#    freq = 30
#    datafolder = gv.global_data_liaise + f'/cendrosa/{freq}min/'
#    filename = f'LIAISE_LA-CENDROSA_CNRM_MTO-FLUX-{freq}MIN_L2_2021-{wanted_month}-{wanted_day}_V2.nc'
#    ds_mast = xr.open_dataset(datafolder + filename)
#    # keep time of interest
#    ds_mast['time_dist'] = np.abs(ds_mast.time - pd.Timestamp(wanted_date).to_datetime64())
#    ds_t = ds_mast.where(ds_mast['time_dist'] == ds_mast['time_dist'].min(), 
#                         drop=True).squeeze()
#    # check that time dist is ok
#    if ds_t['time_dist'] > pd.Timedelta(35, 'min'):
#        ds_t = ds_t * np.nan
#    
#    # keep verti wind profile only
#    ds_verti = xr.Dataset()
#    ds_verti['time'] = ds_t.time
#    ds_verti['ta'] = xr.DataArray([ds_t['ta_1'].values, ds_t['ta_2'].values, 
#                                   ds_t['ta_3'].values, ds_t['ta_4'].values,], 
#                         coords={'level_agl': [3, 10, 25, 50]})   
#    # integration_time is time between two data pts [min]
#    ds_t['integration_time'] = freq
#    
#    # Write result for this source
#    obs_dict['mast'] = ds_verti
    
elif site == 'elsplans':
    obs = obs.rename({'height': 'level_agl'})
    obs['pressure'] = obs['pressure']* 100  # convert from hPa to Pa
    obs['temperature'] = obs['temperature'] + 273.15  # from °C to K
    # keep only low layer of atmos (~ABL)
    obs_low = obs.where(xr.DataArray(obs.level_agl.values<toplevel, dims='index'), 
                        drop=True)
elif site == 'irta':
    obs = obs.rename({'Z': 'level_agl', 'AIR_T': 'temperature'})
    obs['level_asl'] = obs['level_agl'] + gv.sites[site]['alt']
    obs['pressure'] = tools.height_to_pressure_std(obs['level_asl'])
    # keep only low layer of atmos (~ABL)
    obs_low = obs.where(obs['level_agl'] < toplevel, drop=True)


obs_low['potentialTemperature'] = tools.potential_temperature_from_temperature(
        obs_low['pressure'], obs_low['temperature'])

obs_low[var_obs] = obs_low[var_obs]*coeff_corr

plt.plot(obs_low[var_obs], obs_low['level_agl'], 
         label='obs_radiosonde', 
         color=colordict['obs']
         )
plt.grid()
    

## - add wind barbs
#wind_speed_obs = obs.windSpeed.values * units.meter_per_second
#wind_dir_obs = obs.windDirection.values * units.degrees
#u_obs, v_obs = mpcalc.wind_components(wind_speed_obs, wind_dir_obs)
#n = 30  #keep data every nth point
#skew.plot_barbs(p_obs[1::n], u_obs[1::n], v_obs[1::n])

#%% LOAD SIMU DATASET
var1d = {}
height = {}

for model in simu_list:     # model will be 'irr' or 'std'
    # retrieve and open file
    filename_simu = tools.get_simu_filepath(model, wanted_date)
    ds = xr.open_dataset(filename_simu)
    
    # find indices from lat,lon values 
    index_lat, index_lon = tools.indices_of_lat_lon(ds, lat, lon)
    # keep only variable of interest
    var3d = ds[var_simu]
    # keep only low layer of atmos (~ABL)
    var3d_low = var3d.where(var3d.level<toplevel, drop=True)
    
    if 'irr' in model:
        label_legend = 'simu_IRR'
    elif 'std' in model:
        label_legend = 'simu_STD'
    
    if mean_profile:
        var3d_column = var3d_low[
            0, :, 
            int(index_lat-column_width/2):int(index_lat+column_width/2), 
            int(index_lon-column_width/2):int(index_lon+column_width/2)]
        var1d_column = var3d_column.mean(dim=['nj', 'ni'])
        var1d_column_std = var3d_column.std(dim=['nj', 'ni'])
        plt.plot(var1d_column.data, var1d_column.level, 
                 label=label_legend, 
                 ls='--', 
                 c=colordict[model],
                 )
        plt.fill_betweenx(var1d_column.level, 
                          var1d_column.data - var1d_column_std.data,
                          var1d_column.data + var1d_column_std.data,
                          alpha=0.3, 
                          facecolor=colordict[model],
                          )
    elif straight_profile:
        var1d_column = var3d_low[0, :, index_lat, index_lon] #1st index is time, 2nd is Z,..
        # SIMU PLOT
        plt.plot(var1d_column.data, var1d_column.level, 
                 ls='--', 
                 color=colordict[model], 
                 label=label_legend
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
    
        plt.plot(var1d[model], height[model], 
                 ls=':', 
                 color=colordict[model], 
                 label=model+'_interp'
                 )

#TODO: add wind barbs
#TODO: add CAPE and CIN ?


#%% GRAPH ESTHETIC
#add special lines
plot_title = f"{var_simu_longname} profile\n over {gv.sites[site]['longname']}"
figname = f"{var_simu_longname} profile over {gv.sites[site]['longname']}"

plt.title(plot_title)
plt.xlim([vmin, vmax])
plt.ylabel('height AGL [m]')
plt.xlabel(var_simu_longname + ' [' + var3d_low.units + ']')
plt.legend()

plt.subplots_adjust(left=0.2, right=0.9, bottom=0.1) 


#%% GET ABL HEIGHT
#obs_tht = mpcalc.potential_temperature(p_obs, T_obs)
#obs_u, obs_v = mpcalc.wind_components(obs.windSpeed, obs.windDirection)
##
##bulk_Ri = mpcalc.bulk_richardson_number(
##    obs.altitude*units.meter, 
##    obs_tht, 
##    obs_u.values*units.meter_per_second, 
##    obs_v.values*units.meter_per_second)
#
#bulk_Ri = mpcalc.bulk_richardson_number(
#    obs.altitude.values, 
#    obs_tht, 
#    obs_u.values, 
#    obs_v.values)
#
#bulk_Ri = bulk_Ri.m
#
#print('--- hbl in obs: ---')
#hbl_bulk_Ri = mpcalc.boundary_layer_height_from_bulk_richardson_number(
#        obs.altitude.values, bulk_Ri)
#print("hbl_bulk_Ri = " + str(hbl_bulk_Ri))

#hbl_tht = mpcalc.boundary_layer_height_from_potential_temperature(
#        obs.altitude*units.meter, obs_tht)
#print("hbl_tht = " + str(hbl_tht.values))
#
#hbl_temp = mpcalc.boundary_layer_height_from_temperature(
#        obs.altitude*units.meter, obs.temperature)
#print("hbl_temp = " + str(hbl_temp.values))
#
#hbl_parcel = mpcalc.boundary_layer_height_from_parcel(
#        obs.altitude*units.meter, obs_tht)
#print("hbl_parcel = " + str(hbl_parcel.values))
#
#hbl_spec_humid, dqdz = mpcalc.boundary_layer_height_from_specific_humidity(
#        obs.altitude*units.meter, obs.mixingRatio)
##obs_rv = moving_average(obs.mixingRatio.values, window_size=5)
##hbl_spec_humid_2, dqdz = mpcalc.boundary_layer_height_from_specific_humidity(
##        obs.altitude*units.meter, obs_rv)
#print("hbl_spec_humid = " + str(hbl_spec_humid.values))


#%% Save plot
if save_plot:
    tools.save_figure(figname, save_folder)


    