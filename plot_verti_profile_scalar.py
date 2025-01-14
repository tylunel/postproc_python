#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021

"""
import numpy as np
import tools
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import global_variables as gv


########## Independant parameters ###############
wanted_date = '20210722-1300'
site = 'cendrosa'  # 'cendrosa', 'elsplans', 'irta'

# variable name from MNH files: 'THT', THTV 'RVT', 'WS'
var_simu = 'RVT'
# variable name from obs files: 'potentialTemperature', 'mixingRatio', virtualPotentialTemperature, windSpeed
var_obs = 'mixingRatio'

coeff_corr = 1  #to switch from obs to simu2

#vmin, vmax = 288, 314  # for THT
vmin, vmax = None, None

simu_list = [
#            'irrswi1_d1_16_10min',
            'irrswi1_d1',
            'std_d1', 
#            'irrlagrip30_d1',
#            'irrlagrip30_d1_old',
            # 'irr_d2',
            ]

simu_only = False

# Path in simu is average of neighbouring grid points
mean_profile = True
column_width = 3
# Path in simu follows real RS path  #issue: fix discontinuities
follow_rs_position = False

# highest level AGL plotted
toplevel = 2500

save_plot = True
save_folder = f'figures/verti_profiles/{site}/{var_simu}/'
figsize=(5, 7)

##################################################

lat = gv.whole[site]['lat']
lon = gv.whole[site]['lon']

if site not in ['cendrosa', 'elsplans', 'irta'] and simu_only != True:
    raise ValueError('Site without radiosounding')

if var_obs == 'mixingRatio':
    coeff_corr = 0.001

fig = plt.figure(figsize=figsize)

colordict = {'irr_d2': 'g', 
             'std_d2': 'r',
             'irr_d1': 'g', 
             'std_d1': 'r', 
             'irrlagrip30_d1': 'orange',
             'irrlagrip30_d1_old': 'yellow',
             'irrswi1_d1': 'b',
             'irrswi1_d1_16_10min': 'b',
             'irrswi1_d1_old': 'c',
             'irr_d2_old': 'g', 
             'std_d2_old': 'r', 
             'obs': 'k'}
      
legend_dict = {
    'obs': 'obs',
    'std_d1': 'simu_STD',
    'irrswi1_d1': 'simu_IRR',
    'irrlagrip30_d1': 'simu_IRR_THLD',
    } 

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
    
    obs_low['potentialTemperature'] = tools.potential_temperature_from_temperature(
            obs_low['pressure'], obs_low['temperature'])
    obs_low['virtualPotentialTemperature'] = \
        obs_low['potentialTemperature']*(1 + 0.61*obs_low['mixingRatio']/1000)
    
    obs_low[var_obs] = obs_low[var_obs]*coeff_corr
    
    plt.plot(obs_low[var_obs], obs_low['level_agl'], 
             label='obs', 
             color=colordict['obs']
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
    
    # Retrieve and open file
    ds = tools.open_relevant_file(model, wanted_date, var_simu)
    
    ds = tools.flux_pt_to_mass_pt(ds)
    
    # find indices from lat,lon values 
    index_lat, index_lon = tools.indices_of_lat_lon(ds, lat, lon, 
                                                    verbose=False)
    # keep only variable of interest
    var3d = ds[var_simu]
    # keep only low layer of atmos (~ABL)
    var3d_low = var3d.where(var3d.level<toplevel, drop=True)
    
    # get simulation file datetime
    try:
        simu_time = pd.Timestamp(var3d.time.values).strftime('%d_%H:%M')
    except:
        simu_time = pd.Timestamp(var3d.time.values[0]).strftime('%d_%H:%M')
    
    if mean_profile:
        var3d_column = var3d_low.isel(
            nj=np.arange(int(index_lat-column_width/2),int(index_lat+column_width/2)),
            ni=np.arange(int(index_lon-column_width/2),int(index_lon+column_width/2))
            ).squeeze()
        var1d_column = var3d_column.mean(dim=['nj', 'ni'])
        var1d_column_std = var3d_column.std(dim=['nj', 'ni'])
        plt.plot(var1d_column.data, var1d_column.level,
                 # label=f'mean_&_stdev_{model}_{simu_time}',
                 label=legend_dict[model],
                 c=colordict[model],
                 )
        plt.fill_betweenx(var1d_column.level, 
                          var1d_column.data - var1d_column_std.data,
                          var1d_column.data + var1d_column_std.data,
                          alpha=0.3, 
                          facecolor=colordict[model],
                          )
    else:  # straight profile
        var1d_column = var3d_low.isel(nj=index_lat, ni=index_lon).squeeze()
        # SIMU PLOT
        plt.plot(var1d_column.data, var1d_column.level, 
                 ls='--', 
                 color=colordict[model], 
#                 label=model,
                 # label=f'{model}_{simu_time}',
                 label=legend_dict[model]
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


#%% GRAPH ESTHETIC
#add special lines
if mean_profile:
    plot_title = 'Vertical mean profile for {0} above {1} \n on {2} averaged over {3}x{3}pts'.format(
        var_simu, site, wanted_date, column_width)
    figname = 'verti mean profile {0}-{1}-{2}-{3}pts'.format(
        var_simu, site, wanted_date, column_width)
else:
    plot_title = 'Vertical profile for {0} at {1} on {2}'.format(
        var_simu, site, wanted_date)
    figname = 'verti profile {0}-{1}-{2}'.format(
        var_simu, site, wanted_date)

#plot_title = "Potential temperature profile\nabove La Cendrosa\non July 22 at 12:00"
plt.title(plot_title)
plt.xlim([vmin, vmax])
plt.ylim([0, toplevel])
plt.ylabel('height AGL (m)')
try:
    plt.xlabel(var3d_low.long_name + '_[' + var3d_low.units + ']')
except AttributeError:
    plt.xlabel(var_simu)
plt.legend(loc='upper left')
plt.grid()
plt.tight_layout()

#plt.show()

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


    