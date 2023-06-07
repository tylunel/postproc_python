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
import global_variables as gv
import shapefile


#########################################"""
simu_folders = ['irr_d1', 'std_d1']
folder_res = 'diff_std_irr/d2'
domain_nb = int(simu_folders[0][-1])

ilevel = 10  #0 is Halo, 1:2m, 2:6.1m, 3:10.5m, 10:49.3m, 20:141m, 30:304m, 40:600m, 50:1126m, 60:2070
skip_barbs = 10 # 1/skip_barbs will be printed
barb_length = 4.5
# Datetime
wanted_date = '20210729-2300'

speed_plane = 'horiz'  # 'horiz': horizontal 'normal' wind, 'verti' for W

if speed_plane == 'verti':
    vmax_cbar = 2
    vmin_cbar = -vmax_cbar
    cmap_name = 'seismic'
elif speed_plane == 'horiz':
    vmax_cbar = 10
    vmin_cbar = -10
    cmap_name = 'seismic'

zoom_on = 'd2'  #None for no zoom, 'liaise' or 'urgell'

save_plot = True
save_folder = './figures/winds/{0}/{1}/'.format(folder_res, ilevel)

barb_size_option = 'very_weak_winds'  # 'weak_winds' or 'standard'


###########################################
if zoom_on == 'liaise':
    skip_barbs = 2 # 1/skip_barbs will be printed
    barb_length = 5.5
    lat_range = [41.45, 41.8]
    lon_range = [0.7, 1.2]
    figsize=(9,7)
elif zoom_on == 'urgell':
    skip_barbs = 2 # 1/skip_barbs will be printed
    barb_length = 4.5
    lat_range = [41.1, 42.1]
    lon_range = [0.2, 1.7]
    figsize=(11,9)
elif zoom_on == 'urgell-paper':
    skip_barbs = 6 # 1/skip_barbs will be printed
    barb_length = 4.5
    lat_range = [41.37, 41.92]
    lon_range = [0.6, 1.4]
    figsize=(9,7)
elif zoom_on == 'd2':
    skip_barbs = 3 # 1/skip_barbs will be printed
    barb_length = 4.5
    lat_range = [40.8106, 42.4328]
    lon_range = [-0.6666, 1.9364]
    figsize=(11,9)
elif zoom_on == None:
    skip_barbs = 8 # 1/skip_barbs will be printed
    barb_length = 4.5
    if domain_nb == 1:
        figsize=(13,7)
    elif domain_nb == 2:
        figsize=(10,7)

barb_size_increments = gv.barb_size_increments
barb_size_description = gv.barb_size_description

    
#%%
ws_layer = {}
ut_layer = {}
vt_layer = {}
wt_layer = {}

for model in simu_folders:
#    datafolder = father_folder + simu_folders[model]
    
    filename = tools.get_simu_filename(model, wanted_date,
                                   global_simu_folder=gv.global_simu_folder)
    
    # load file, dataset and set parameters
    ds1 = xr.open_dataset(filename,
    #        datafolder + 'LIAIS.2.SEG36.001.nc', 
                          decode_coords="coordinates",
    #                      coordinates=['latitude_u', 'longitude_u'],
    #                      grid_mapping=latitude
                          )
    ds_centered = tools.center_uvw(ds1)
    
    ut_layer[model] = ds_centered['UT'][ilevel, :, :]
    vt_layer[model] = ds_centered['VT'][ilevel, :, :]
    wt_layer[model] = ds_centered['WT'][ilevel, :, :]
    
    if speed_plane == 'horiz':
#        ws = mpcalc.wind_speed(ds1['UT'] * units.meter_per_second, 
#                               ds1['VT'] * units.meter_per_second)
        ws_layer[model] = mpcalc.wind_speed(ut_layer[model], vt_layer[model])
    #wd = mpcalc.wind_direction(ds1['UT'] * units.meter_per_second, 
    #                           ds1['VT'] * units.meter_per_second)
#    elif speed_plane == 'verti':
#        ws_layer[model] = wt_layer[model]
    
    # keep only layer of interest
#    ws_layer[model] = ws[0, ilevel, :, :]
    
ws_diff = ws_layer[simu_folders[0]] - ws_layer[simu_folders[1]]
ut_diff = ut_layer[simu_folders[0]] - ut_layer[simu_folders[1]]
vt_diff = vt_layer[simu_folders[0]] - vt_layer[simu_folders[1]]
wt_diff = wt_layer[simu_folders[0]] - wt_layer[simu_folders[1]]

if domain_nb == 1:
    fig1 = plt.figure(figsize=figsize)
elif domain_nb == 2:
    fig1 = plt.figure(figsize=figsize)

#%% PLOT
    
## 1. Only wind speed
#plt.pcolormesh(ds1.longitude, ds1.latitude, ws_diff,
##               cbar_kwargs={"orientation": "horizontal", "shrink": 0.7}
#               cmap=cmap_name,
#               vmin=vmin_cbar,
#               vmax=vmax_cbar               
#              )
#
#cbar = plt.colorbar()
#cbar.set_label('Wind speed [m/s]')


# 2. Complete
# WIND SPEED COLORMAP
if speed_plane == 'horiz':
    ws = ws_diff
#    ws = mpcalc.wind_speed(ut_diff * units.meter_per_second, 
#                           vt_diff * units.meter_per_second)
#wd = mpcalc.wind_direction(ds1['UT'] * units.meter_per_second, 
#                           ds1['VT'] * units.meter_per_second)
elif speed_plane == 'verti':
    ws = wt_diff

## keep only layer of interest
#ws_layer = ws[0, ilevel, :, :]

plt.pcolormesh(ds1.longitude, ds1.latitude, ws,
#               cbar_kwargs={"orientation": "horizontal", "shrink": 0.7}
               cmap=cmap_name,
               vmin=vmin_cbar,
               vmax=vmax_cbar               
              )
cbar = plt.colorbar()
cbar.set_label('Wind speed [m/s]')

# WIND BARBS

X = ds1.longitude
Y = ds1.latitude
U = ut_diff
V = vt_diff

plt.barbs(X[::skip_barbs, ::skip_barbs], Y[::skip_barbs, ::skip_barbs], 
          U[::skip_barbs, ::skip_barbs], V[::skip_barbs, ::skip_barbs],
          pivot='middle',
          length=barb_length,     #length of barbs
          sizes={
#                 'spacing':1, 
#                 'height':1,
#                 'width':1,
                 'emptybarb':0.01},
          barb_increments=barb_size_increments[barb_size_option]
          )
plt.annotate(barb_size_description[barb_size_option],
             xy=(0.1, 0.05),
             xycoords='subfigure fraction'
             )



#%% IRRIGATED, SEA and COUNTRIES BORDERS

if domain_nb == 2:
    pgd = xr.open_dataset(
        gv.global_simu_folder + \
        '2.01_pgds_irr/PGD_400M_CovCor_v26_ivars.nc')
elif domain_nb == 1:
    pgd = xr.open_dataset(
        gv.global_simu_folder + \
        '2.01_pgds_irr/PGD_2KM_CovCor_v26_ivars.nc')

#Irrigation borders
#from scipy.ndimage.filters import gaussian_filter
#sigma = 0.1     #default is 0.1
#irr_covers = gaussian_filter(pgd.COVER369.data, sigma)
irr_covers = pgd.COVER369.data
plt.contour(pgd.longitude.data, 
            pgd.latitude.data, 
            irr_covers,
            levels=0,   #+1 -> number of contour to plot 
            linestyles='solid',
            linewidths=1.5,
            colors='g'
#            colors=['None'],
#            hatches='-'
            )

#Sea borders
sea_covers = pgd.COVER001.data
plt.contour(pgd.longitude.data, 
            pgd.latitude.data, 
            sea_covers,
            levels=0,   #+1 -> number of contour to plot 
            linestyles='solid',
            linewidths=1.,
            colors='k'
    #        colors=['None'],
    #        hatches='-'
            )

#France borders
sf = shapefile.Reader("TM-WORLD-BORDERS/TM_WORLD_BORDERS-0.3.sph")
shapes=sf.shapes()
france = shapes[64].points
france_df = pd.DataFrame(france, columns=['lon', 'lat'])
france_S = france_df[france_df.lat < 43.35]
france_SW = france_S[france_S.lon < 2.95]
plt.plot(france_SW.lon, france_SW.lat,
         color='k',
         linewidth=1)

#%% POINTS SITES
points = ['cendrosa', 'elsplans', 
#          'puig formigosa', 'tossal baltasana', 
          'tossal gros', 
'tossal torretes', 
#'moncayo', 'tres mojones', 
#          'guara', 'caro', 'montserrat', 'joar',
          ]
sites = {key:gv.whole[key] for key in points}

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
    level_agl = ws_diff.level
if speed_plane == 'verti':
    level_agl = ws_diff.level_w
#level_agl = ws_diff.level
    
plot_title = '{4} wind diff at {0}m on {1} for simu {2} zoomed on {3}'.format(
        np.round(level_agl, decimals=1), 
        pd.to_datetime(ws_diff.time.values).strftime('%Y-%m-%dT%H%M'),
        model,
        zoom_on,
        speed_plane)
plt.title(plot_title)

if zoom_on is None:
    plt.ylim([ws_diff.latitude.min(), ws_diff.latitude.max()])
    plt.xlim([ws_diff.longitude.min(), ws_diff.longitude.max()])
else:
    plt.ylim(lat_range)
    plt.xlim(lon_range)

if save_plot:
    tools.save_figure(plot_title, save_folder)
