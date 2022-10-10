#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021

Cf also plot_verti_profile.py

example for skewT graph here:
    https://unidata.github.io/MetPy/latest/examples/plots/Skew-T_Layout.html#sphx-glr-examples-plots-skew-t-layout-py
"""
#import os
#import numpy as np
import pandas as pd
#import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from tools import indices_of_lat_lon, get_obs_filename_from_date, get_simu_filename, open_ukmo_rs
from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
import xarray as xr


########## Independant parameters ###############
wanted_date = pd.Timestamp('20210722-1500')
site = 'elsplans'
save_plot = False

simu_list = ['irr', 'std']

save_plot = True
##################################################

if site == 'cendrosa':
    lat = 41.6925905
    lon = 0.9285671
elif site == 'elsplans':
    lat = 41.590111 
    lon = 1.029363  
else:
    raise ValueError('Site without radiosounding')


fig = plt.figure(figsize=(9, 5))
skew = SkewT(fig, rotation=45, aspect=120)    #aspect default is 80.5 / rotation of 59deg gives nearly vertical line for dry adiabats (vertical temp = same potential temp)
colormap = ['r', 'b', 'g', 'c', 'm', 'y', 'k']

#%% OBS PARAMETERS
for wanted_date in [wanted_date,]:
    datafolder = \
                '/cnrm/surface/lunelt/data_LIAISE/'+ site +'/radiosoundings/'
    # IF FILENAME KNOWN:
    #filename = 'LIAISE_LA-CENDROSA_CNRM_RS-ascent_L2_20210722-1200_V1.nc'
    # IF ONLY DATE KNOWN
    filename = None    
    if filename is None:
        filename = get_obs_filename_from_date(
                datafolder, 
                wanted_date,
                dt_threshold=pd.Timedelta('0 days 00:45:00'),
                regex_date='202107\d\d.\d\d\d\d')
            
    
    #%% LOAD OBS DATASET
    if site == 'cendrosa':
        obs = xr.open_dataset(datafolder + filename)
    elif site == 'elsplans':
        obs = open_ukmo_rs(datafolder, filename)
        
    datetime_beg = pd.Timestamp(obs.time.values[0])
    
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
        
    skew.plot(p_obs, T_obs, 'k--', label='obs_T')
    skew.plot(p_obs, Td_obs, 'k-.', label='obs_Td')
    
    ## - add wind barbs
    wind_speed_obs = obs.windSpeed.values * units.meter_per_second
    wind_dir_obs = obs.windDirection.values * units.degrees
    u_obs, v_obs = mpcalc.wind_components(wind_speed_obs, wind_dir_obs)
    n = 30  #keep data every nth point
    skew.plot_barbs(p_obs[1::n], u_obs[1::n], v_obs[1::n])
    
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

for i, model in enumerate(simu_list):     # model will be 'irr' or 'std'
    filename_simu = get_simu_filename(model, wanted_date)
    ds = xr.open_dataset(filename_simu)

    datetime = pd.Timestamp(ds.time.values[0])
    
    T_3d = ds.TEMP
    p_3d = ds.PABST
    rh_3d = ds.REHU    
    
    # find indices from lat,lon values 
    index_lat, index_lon = indices_of_lat_lon(ds, lat, lon)
    
    #%% SIMU PLOT
    index_high = 66     #70=>3500m, 67=>3000m, 60=>2000m
    
    #1st index is time, 2nd is Z,..
    T = T_3d.data[0, 0:index_high, index_lat, index_lon] * units.degC
    p = p_3d.data[0, 0:index_high, index_lat, index_lon]/100 * units.hPa
    rh = rh_3d.data[0, 0:index_high, index_lat, index_lon]/100
    Td = mpcalc.dewpoint_from_relative_humidity(T, rh)
    
    skew.plot(p, T, ls='--', color=colormap[i], label=model+'_T')
    skew.plot(p, Td, ls='-.', color=colormap[i], label=model+'_Td')

#TODO: add wind barbs
#TODO: add CAPE and CIN ?



#%% GRAPH ESTHETIC
#add special lines
skew.plot_dry_adiabats(linewidth=1) #linewidth default is 1.5
#skew.plot_moist_adiabats(linewidth=1)
skew.plot_mixing_lines(linewidth=1, color='b')

skew.ax.set_ylim(1000, 650)
skew.ax.set_xlim(0, 50)
plot_title = 'Skew-T profile at {0}\n at {1}'.format(site, datetime_beg)
plt.title(plot_title)
plt.legend()

# Add a secondary axis that automatically converts between pressure and height
# assuming a standard atmosphere. The value of -0.12 puts the secondary axis
# 0.12 normalized (0 to 1) coordinates left of the original axis.
secax = skew.ax.secondary_yaxis("right",                              
    functions=(lambda p: mpcalc.pressure_to_height_std(units.Quantity(p, 'mbar')).m_as('km'),
               lambda h: mpcalc.height_to_pressure_std(units.Quantity(h, 'km')).m))
secax.yaxis.set_major_locator(plt.FixedLocator([0, 0.5, 1, 1.5, 2, 2.5, 3]))
secax.yaxis.set_minor_locator(plt.NullLocator())
secax.yaxis.set_major_formatter(plt.ScalarFormatter())
secax.set_ylabel('height (km)')

plt.show()

#%% GET ABL HEIGHT
obs_tht = mpcalc.potential_temperature(p_obs, T_obs)
obs_u, obs_v = mpcalc.wind_components(obs.windSpeed, obs.windDirection)
#
#bulk_Ri = mpcalc.bulk_richardson_number(
#    obs.altitude*units.meter, 
#    obs_tht, 
#    obs_u.values*units.meter_per_second, 
#    obs_v.values*units.meter_per_second)

bulk_Ri = mpcalc.bulk_richardson_number(
    obs.altitude.values, 
    obs_tht, 
    obs_u.values, 
    obs_v.values)

bulk_Ri = bulk_Ri.m

print('--- hbl in obs: ---')
hbl_bulk_Ri = mpcalc.boundary_layer_height_from_bulk_richardson_number(
        obs.altitude.values, bulk_Ri)
print("hbl_bulk_Ri = " + str(hbl_bulk_Ri))

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
    filename = (plot_title)
    filename = filename.replace('=', '').replace('(', '').replace(')', '')
    filename = filename.replace(' ', '_').replace(',', '').replace('.', '_')
    plt.savefig('figures/skewt/'+filename)

    