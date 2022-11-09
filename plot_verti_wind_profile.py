#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021

Cf also plot_verti_profile.py

"""
#import os
import numpy as np
import pandas as pd
#import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import tools
#from metpy.plots import SkewT
from metpy.units import units
import metpy.calc as mpcalc
import xarray as xr


########## Independant parameters ###############
#wanted_date = '20210722-2000'
wanted_date = '20210722-2000'

site = 'cendrosa'

# Variables considered:
# variable names from MNH files: 'UT' and 'VT'
# variable names from obs files: 'windSpeed' and 'windDirection'

#Barbs
add_barbs = True
interpolated_barbs = True
barb_length = 5

simu_list = ['irr', 
             'std'
             ]

# highest level AGL plotted
toplevel = 2500

save_plot = True
save_folder = 'figures/verti_profiles/winds/'
##################################################

if site == 'cendrosa':
    lat = 41.6925905
    lon = 0.9285671
elif site == 'elsplans':
    lat = 41.590111 
    lon = 1.029363  
else:
    raise ValueError('Site without radiosounding')


fig, ax = plt.subplots(1, 3, sharey=True, figsize=[10, 7],
                       gridspec_kw={'width_ratios': [2, 2, 1]})

colordict = {'obs': 'k', 'irr': 'g', 'std': 'orange'}

#parameter for placing the barbs from left to right
barb_pos = 0

#%% OBS PARAMETERS

datafolder = \
            '/cnrm/surface/lunelt/data_LIAISE/'+ site +'/radiosoundings/'

filename = tools.get_obs_filename_from_date(
        datafolder, 
        wanted_date,
        dt_threshold=pd.Timedelta('0 days 00:45:00'),
        regex_date='202107\d\d.\d\d\d\d')
        

#%% LOAD OBS DATASET
if site == 'cendrosa':
    obs = xr.open_dataset(datafolder + filename)
elif site == 'elsplans':
    obs = tools.open_ukmo_rs(datafolder, filename)
    
#%% OBS PLOT

#coeff_corr = 1  #to switch from obs to simu
#obs[var_obs] = obs[var_obs]*coeff_corr

#Data selection
if site == 'cendrosa':
    obs['height'] = obs.altitude - 240  # 240 if altitude ASL of cendrosa
    # keep only low layer of atmos (~ABL)
    obs_low = obs.where(xr.DataArray(obs.height.values<toplevel, dims='time'), 
                        drop=True)
else:
    # for elsplans the field 'height' already exists
    obs_low = obs.where(xr.DataArray(obs.height.values<toplevel, dims='index'), 
                        drop=True)

#Profile plot
ax[0].plot(obs_low['windSpeed'], obs_low.height, 
           label='obs', 
           color=colordict['obs']
           )

ax[1].scatter(obs_low['windDirection'], obs_low.height, 
           label='obs', 
           color=colordict['obs'],
           s=1  #size of marker
           )

#Barbs plot
if add_barbs:
    if interpolated_barbs:
        #Interpolated barbs
        level_range = np.arange(0, toplevel, step=50)
        wd_interp = np.interp(level_range, obs.height, obs.windDirection)
        ws_interp = np.interp(level_range, obs.height, obs.windSpeed)
        #first compute u and v components
        u, v = mpcalc.wind_components(ws_interp * units['m/s'],
                                      wd_interp * units('degree'))
        #add wind barbs on the side for wind direction
        x = np.empty_like(level_range)
        x.fill(barb_pos)
        ax[2].barbs(x, 
                    level_range, 
                    u, 
                    v,
                    barbcolor=colordict['obs'],
                    length=barb_length)
        barb_pos += 1  #increment barb_position for next barbs
    else:
        #first compute u and v components
        u, v = mpcalc.wind_components(obs_low['windSpeed'], 
                                      obs_low['windDirection'])
        #add wind barbs on the side for wind direction
        x = np.empty_like(obs_low['windSpeed'])
        x.fill(barb_pos)
        ax[2].barbs(x, 
                    obs_low.height, 
                    u, 
                    v,
                    barbcolor=colordict['obs'],
                    length=barb_length)
        barb_pos += 1  #increment barb_position for next barbs


#%% LOAD SIMU DATASET

for model in simu_list:     # model will be 'irr' or 'std'
    # retrieve and open file
    filename_simu = tools.get_simu_filename(model, wanted_date)
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
    ut_1d = var3d_low['UT'][:, index_lat, index_lon] #1st index is z
    vt_1d = var3d_low['VT'][:, index_lat, index_lon] #1st index is z
    
    wd_1d = mpcalc.wind_direction(ut_1d, vt_1d)
    ws_1d = mpcalc.wind_speed(ut_1d, vt_1d)
    
    # SIMU PLOT
    # Wind Speed
    ax[0].plot(ws_1d.data, ws_1d.level, 
             ls='--', 
             color=colordict[model], 
             label=model
             )
    # Wind Direction
    ax[1].scatter(wd_1d.data, wd_1d.level, 
             ls='--', 
             color=colordict[model], 
             label=model,
             s=1  #size of marker
             )
    # Barbs
    if add_barbs:
        if interpolated_barbs:
            #Interpolated barbs
            level_range = np.arange(0, toplevel, step=50)
            ut_interp = np.interp(level_range, ut_1d.level, ut_1d)
            vt_interp = np.interp(level_range, vt_1d.level, vt_1d)
            #add wind barbs on the side for wind direction
            x = np.empty_like(level_range)
            x.fill(barb_pos)
            ax[2].barbs(x, 
                        level_range, 
                        ut_interp, 
                        vt_interp,
                        barbcolor=colordict[model],
                        length=barb_length)
            barb_pos += 1  #increment barb_position for next barbs
        else:
            #add wind barbs on the side for wind direction
            x = np.empty_like(ut_1d.data)
            x.fill(barb_pos)
            ax[2].barbs(x, ut_1d.level.data, ut_1d.data, vt_1d.data,
                        barbcolor=colordict[model],
                        length=barb_length)
            barb_pos += 1  #increment barb_position for next barbs

#%% GRAPH ESTHETIC
#add special lines

ax[0].set_ylabel('height AGL (m)')
ax[0].set_xlabel('wind speed')
ax[0].grid()
ax[0].legend()

ax[1].set_xlabel('wind direction')
ax[1].set_xticks([0, 90, 180, 270, 360])
ax[1].set_xticklabels(['N', 'E', 'S', 'W', 'N'], 
                      rotation=0, fontsize=12)
ax[1].grid()

if add_barbs:
    ax[2].set_xlim([-0.9, barb_pos+0.9])
    ax[2].spines[['top', 'right', 'left']].set_visible(False)
    ax[2].set_xticks(range(barb_pos))
    ax[2].set_xticklabels(colordict.keys(), 
                          rotation=0, fontsize=12)

plot_title = 'Vertical profile for wind at {0} on {1}'.format(
        site, wanted_date)
fig.suptitle(plot_title)

#%% Save plot
if save_plot:
    tools.save_figure(plot_title, save_folder)

    