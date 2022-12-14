#!/usr/bin/env python3
"""
@author: tylunel
Creation : 07/01/2021

Multiple attempts to use MetPy BarbPlot(), but multiple issues.
Best option seems to be the simplest as follows:
"""

#import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
import pandas as pd 
import tools


#########################################"""
model = 'irr'
ilevel = 3  #0 is Halo, 1:2m, 2:6.1m, 3:10.5m, 10:49.3m, 20:141m, 30:304m, 40:600m, 50:1126m, 60:2070
skip_barbs = 10 # 1/skip_barbs will be printed
barb_length = 4
# Datetime
wanted_date = '20210724-1200'

speed_plane = 'horiz'  # 'horiz': horizontal 'normal' wind, 'verti' for W

if speed_plane == 'verti':
    vmax_cbar = 4
    vmin_cbar = -vmax_cbar
    cmap_name = 'seismic'
elif speed_plane == 'horiz':
    vmax_cbar = 10
    vmin_cbar = -10
    cmap_name = 'seismic'

zoom_on = None  #None for no zoom, 'liaise' or 'urgell'

if zoom_on == 'liaise':
    skip_barbs = 3 # 1/skip_barbs will be printed
    barb_length = 5.5
    lat_range = [41.45, 41.8]
    lon_range = [0.7, 1.2]
elif zoom_on == 'urgell':
    skip_barbs = 6 # 1/skip_barbs will be printed
    barb_length = 4.5
    lat_range = [41.1, 42.1]
    lon_range = [0.2, 1.7]
    
    
save_plot = True
save_folder = './figures/winds/diff/'

###########################################

simu_folders = {
        'irr': '2.13_irr_2021_22-27/', 
        'std': '1.11_ECOII_2021_ecmwf_22-27/'
         }
father_folder = '/cnrm/surface/lunelt/NO_SAVE/nc_out/'


#%%
ws_layer = {}

for model in simu_folders:
    datafolder = father_folder + simu_folders[model]
    
    filename = tools.get_simu_filename(model, wanted_date)
    
    # load file, dataset and set parameters
    ds1 = xr.open_dataset(filename,
    #        datafolder + 'LIAIS.2.SEG36.001.nc', 
                          decode_coords="coordinates",
    #                      coordinates=['latitude_u', 'longitude_u'],
    #                      grid_mapping=latitude
                          )    
    
    if speed_plane == 'horiz':
        ws = mpcalc.wind_speed(ds1['UT'] * units.meter_per_second, 
                               ds1['VT'] * units.meter_per_second)
    #wd = mpcalc.wind_direction(ds1['UT'] * units.meter_per_second, 
    #                           ds1['VT'] * units.meter_per_second)
    elif speed_plane == 'verti':
        ws = ds1['WT']
    
    # keep only layer of interest
    ws_layer[model] = ws[0, ilevel, :, :]
    
ws_diff = ws_layer['irr'] - ws_layer['std']


fig1 = plt.figure(figsize=(13,9))
plt.pcolormesh(ds1.longitude, ds1.latitude, ws_diff,
#               cbar_kwargs={"orientation": "horizontal", "shrink": 0.7}
               cmap=cmap_name,
               vmin=vmin_cbar,
               vmax=vmax_cbar               
              )

cbar = plt.colorbar()
cbar.set_label('Wind speed [m/s]')


#%% IRRIGATED COVERS

pgd = xr.open_dataset(
    '/cnrm/surface/lunelt/NO_SAVE/nc_out/2.01_pgds_irr/' + \
    'PGD_400M_CovCor_v26_ivars.nc')

#from scipy.ndimage.filters import gaussian_filter
#sigma = 0.1     #default is 0.1
#irr_covers = gaussian_filter(pgd.COVER369.data, sigma)
irr_covers = pgd.COVER369.data

plt.contour(pgd.longitude.data, 
          pgd.latitude.data, 
          irr_covers,
          levels=0,   #+1 -> number of contour to plot 
          linestyles='solid',
          linewidths=1.,
          colors='g'
#          colors=['None'],
#          hatches='-'
          )

#%% POINTS SITES
sites = {'cendrosa': {'lat': 41.6925905,
                      'lon': 0.9285671},
         'preixana': {'lat': 41.59373,
                      'lon': 1.07250},
         'elsplans': {'lat': 41.590111,
                      'lon': 1.029363},
        }

for site in sites:
    plt.scatter(sites[site]['lon'],
                sites[site]['lat'],
                color='r',
                s=15        #size of markers
                )
    if site == 'elsplans':
        plt.text(sites[site]['lon']-0.1,
                 sites[site]['lat']-0.03, 
                 site, 
                 fontsize=12)
    else:
        plt.text(sites[site]['lon']+0.01,
                 sites[site]['lat']+0.01, 
                 site, 
                 fontsize=12)


#%% FIGURE OPTIONS
if speed_plane == 'horiz':
    level_agl = ws_layer['std'].level
if speed_plane == 'verti':
    level_agl = ws_layer['std'].level_w
    
        
plot_title = '{4} winds at {0}m on {1} for simu {2} zoomed on {3}'.format(
        np.round(level_agl, decimals=1), 
        pd.to_datetime(ws_layer['std'].time.values).strftime('%Y-%m-%dT%H%M'),
        model,
        zoom_on,
        speed_plane)
plt.title(plot_title)

if zoom_on is not None:
    plt.ylim(lat_range)
    plt.xlim(lon_range)

if save_plot:
    tools.save_figure(plot_title, save_folder)
