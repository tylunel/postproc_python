#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021

Cf also plot_verti_profile.py

example for skewT graph here:
    https://unidata.github.io/MetPy/latest/examples/plots/Skew-T_Layout.html#sphx-glr-examples-plots-skew-t-layout-py
"""
#import numpy as np
import tools
import pandas as pd
import matplotlib.pyplot as plt
from metpy.units import units
import metpy.calc as mpcalc
import xarray as xr
import global_variables as gv


########## Independant parameters ###############
wanted_date = '20210729-2300'
site = 'elsplans'

# variable name from MNH files: 'THT', 'RVT'
var_simu = 'THT'
# variable name from obs files: 'potentialTemperature', 'mixingRatio',
var_obs = 'potentialTemperature'
coeff_corr = 1  #to switch from obs to simu2

simu_list = [
            'std_d1', 
            'irr_d1'
            ]

# Vertical path to show in simu:
# Path in simu is the direct 1d column
straight_profile = False
# Path in simu is average of neighbouring grid points
mean_profile = True
column_width = 3
# Path in simu follows real RS path  #issue: fix discontinuities
follow_rs_position = False


# highest level AGL plotted
toplevel = 2500

save_plot = True
save_folder = 'figures/verti_profiles/{0}/{1}/'.format(site, var_simu)
figsize=(5, 7)

##################################################

if site in ['cendrosa', 'elsplans']:
    lat = gv.sites[site]['lat']
    lon = gv.sites[site]['lon']
else:
    raise ValueError('Site without radiosounding')

if var_obs == 'mixingRatio':
    coeff_corr = 0.001

fig = plt.figure(figsize=figsize)

colordict = {'irr_d2': 'g', 
             'std_d2': 'r',
             'irr_d1': 'g', 
             'std_d1': 'r', 
             'irr_d2_old': 'g', 
             'std_d2_old': 'r', 
             'obs': 'k'}

#%% OBS PARAMETERS
if site == 'elsplans_alt':
    datafolder = \
            '/cnrm/surface/lunelt/data_LIAISE/'+ 'elsplans' +'/radiosoundings/'
else:
    datafolder = \
            gv.global_data_liaise + site + '/radiosoundings/'

filename = tools.get_obs_filename_from_date(
        datafolder, 
        wanted_date,
        dt_threshold=pd.Timedelta('0 days 00:45:00'),
        regex_date='202107\d\d.\d\d\d\d')
        

#%% LOAD OBS DATASET
if site == 'cendrosa':
    obs = xr.open_dataset(datafolder + filename)
elif 'elsplans' in site:
    obs = tools.open_ukmo_rs(datafolder, filename)
    
#%% OBS PLOT

p_obs = obs.pressure.values * units.hPa

if obs.temperature.mean().values > 200:
    T_obs = (obs.temperature).values * units.kelvin
else:
    T_obs = (obs.temperature).values * units.degC

if obs.dewPoint.mean().values > 200:
    Td_obs = (obs.dewPoint).values * units.kelvin
else:
    Td_obs = (obs.dewPoint).values * units.degC

obs['potentialTemperature'] = mpcalc.potential_temperature(obs.pressure, 
                                                           obs.temperature)

obs[var_obs] = obs[var_obs]*coeff_corr

if site == 'cendrosa':
    obs['height'] = obs.altitude - 240
    # keep only low layer of atmos (~ABL)
    obs_low = obs.where(xr.DataArray(obs.height.values<toplevel, dims='time'), 
                        drop=True)
else:
    obs_low = obs.where(xr.DataArray(obs.height.values<toplevel, dims='index'), 
                        drop=True)

plt.plot(obs_low[var_obs], obs_low.height, 
         label='obs', 
         color=colordict['obs']
         )
plt.grid()
    

## - add wind barbs
#wind_speed_obs = obs.windSpeed.values * units.meter_per_second
#wind_dir_obs = obs.windDirection.values * units.degrees
#u_obs, v_obs = mpcalc.wind_components(wind_speed_obs, wind_dir_obs)
#n = 30  #keep data every nth point
#skew.plot_barbs(p_obs[1::n], u_obs[1::n], v_obs[1::n])

## - theoretical lifted air parcel (for CAPE and CIN)
##data processing - remove NaN values needed
#T_obs = T_obs[~np.isnan(p)]
#Td_obs = Td_obs[~np.isnan(p)]
#p_obs = p_obs[~np.isnan(p)]
##profile of air parcel
#Tparcel = mpcalc.parcel_profile(p_obs, T_obs[0], Td_obs[0])
#skew.plot(p, Tparcel, 'k', linewidth=1)
#skew.shade_cape(p_obs, T_obs, Tparcel)
#skew.shade_cin(p_obs, T_obs, Tparcel)

#%% LOAD SIMU DATASET
var1d = {}
height = {}

for model in simu_list:     # model will be 'irr' or 'std'
    # retrieve and open file
    filename_simu = tools.get_simu_filename(model, wanted_date)
    ds = xr.open_dataset(filename_simu)
    
    # find indices from lat,lon values 
    index_lat, index_lon = tools.indices_of_lat_lon(ds, lat, lon)
    # keep only variable of interest
    var3d = ds[var_simu]
    # keep only low layer of atmos (~ABL)
    var3d_low = var3d.where(var3d.level<toplevel, drop=True)
    
    if mean_profile:
        var3d_column = var3d_low[
            0, :, 
            int(index_lat-column_width/2):int(index_lat+column_width/2), 
            int(index_lon-column_width/2):int(index_lon+column_width/2)]
        var1d_column = var3d_column.mean(dim=['nj', 'ni'])
        var1d_column_std = var3d_column.std(dim=['nj', 'ni'])
        plt.plot(var1d_column.data, var1d_column.level, 
                 label='mean_&_stdev_{0}'.format(model), 
                 c=colordict[model],
                 )
        plt.fill_betweenx(var1d_column.level, 
                          var1d_column.data - var1d_column_std.data,
                          var1d_column.data + var1d_column_std.data,
                          alpha=0.3, 
                          facecolor=colordict[model],
                          )
    if straight_profile:
        var1d_column = var3d_low[0, :, index_lat, index_lon] #1st index is time, 2nd is Z,..
        # SIMU PLOT
        plt.plot(var1d_column.data, var1d_column.level, 
                 ls='--', 
                 color=colordict[model], 
                 label=model
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
plt.ylabel('height AGL (m)')
plt.xlabel(var3d_low.standard_name + '_[' + var3d_low.units + ']')
plt.legend()
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


    