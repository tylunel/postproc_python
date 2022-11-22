#!/usr/bin/env python3
"""
@author: tylunel
Creation : 07/01/2021

Multiple attempts to use MetPy BarbPlot(), but multiple issues.
Best option seems to be the simplest as follows:
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
import pandas as pd 
import tools
import shapefile

#########################################"""
model = 'irr_d1'
ilevel = 50  #0 is Halo, 1:2m, 2:6.1m, 3:10.5m, 10:49.3m, 20:141m, 30:304m, 40:600m, 50:1126m, 60:2070
skip_barbs = 5 # 1/skip_barbs will be printed
barb_length = 4.5
# Datetime
wanted_date = '20210729-2300'

domain_nb = 1

speed_plane = 'verti'  # 'horiz': horizontal 'normal' wind, 'verti' for W

if speed_plane == 'verti':
    vmax_cbar = 5
    vmin_cbar = -vmax_cbar
    cmap_name = 'seismic'
elif speed_plane == 'horiz':
    vmax_cbar = 15
    vmin_cbar = 0
    cmap_name = 'BuPu'

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
    
    
save_plot = False
save_folder = './figures/winds/domain{0}/{1}/{2}/'.format(
        domain_nb, speed_plane, ilevel)

barb_size_option = 'weak_winds'  # 'weak_winds' or 'standard'


###########################################

barb_size_increments = {
        'weak_winds': {'half':1.94, 'full':3.88, 'flag':19.4},
        'standard': {'half':5, 'full':10, 'flag':50},
        }
barb_size_description = {
        'weak_winds': "barb increments: half=1m/s=1.94kt, full=2m/s=3.88kt, flag=10m/s=19.4kt",
        'standard': "barb increments: half=5kt=2.57m/s, full=10kt=5.14m/s, flag=50kt=25.7m/s",
        }

filename = tools.get_simu_filename_d1(model, wanted_date)

# load file, dataset and set parameters
ds1 = xr.open_dataset(filename,
#        datafolder + 'LIAIS.2.SEG36.001.nc', 
                      decode_coords="coordinates",
#                      coordinates=['latitude_u', 'longitude_u'],
#                      grid_mapping=latitude
                      )

if domain_nb == 1:
    fig1 = plt.figure(figsize=(17,9))
elif domain_nb == 2:
    fig1 = plt.figure(figsize=(13,9))

#%% WIND SPEED COLORMAP
if speed_plane == 'horiz':
    ws = mpcalc.wind_speed(ds1['UT'] * units.meter_per_second, 
                           ds1['VT'] * units.meter_per_second)
#wd = mpcalc.wind_direction(ds1['UT'] * units.meter_per_second, 
#                           ds1['VT'] * units.meter_per_second)
elif speed_plane == 'verti':
    ws = ds1['WT']

# keep only layer of interest
ws_layer = ws[0, ilevel, :, :]

plt.pcolormesh(ds1.longitude, ds1.latitude, ws_layer,
#               cbar_kwargs={"orientation": "horizontal", "shrink": 0.7}
               cmap=cmap_name,
               vmin=vmin_cbar,
               vmax=vmax_cbar               
              )
cbar = plt.colorbar()
cbar.set_label('Wind speed [m/s]')
#%% WIND BARBS

X = ds1.longitude
Y = ds1.latitude
U = ds1.UT[0, ilevel, :,:]
V = ds1.VT[0, ilevel, :,:]

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
        '/cnrm/surface/lunelt/NO_SAVE/nc_out/2.01_pgds_irr/' + \
        'PGD_400M_CovCor_v26_ivars.nc')
elif domain_nb == 1:
    pgd = xr.open_dataset(
        '/cnrm/surface/lunelt/NO_SAVE/nc_out/2.01_pgds_irr/' + \
        'PGD_2KM_CovCor_v26_ivars.nc')

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
            linewidths=1.,
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
#sites = {'cendrosa': {'lat': 41.6925905,
#                      'lon': 0.9285671},
##         'preixana': {'lat': 41.59373,
##                      'lon': 1.07250},
#         'elsplans': {'lat': 41.590111,
#                      'lon': 1.029363},
#         'tossal baltasana': {'lat': 41.3275,
#                              'lon': 1.00336},
#         'puig formigosa': {'lat': 41.42179,
#                            'lon': 1.44177},
#         'tossal gros': {'lat': 41.47857,
#                         'lon': 1.12942},
#         'tossal torretes': {'lat': 42.02244,
#                             'lon': 0.93800}
#        }
sites = tools.sites

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
                 fontsize=9)
    else:
        plt.text(sites[site]['lon']+0.01,
                 sites[site]['lat']+0.01, 
                 site, 
                 fontsize=9)


#%% FIGURE OPTIONS
if speed_plane == 'horiz':
    level_agl = ws_layer.level
if speed_plane == 'verti':
    level_agl = ws_layer.level_w
        
plot_title = '{4} winds at {0}m on {1} for simu {2} zoomed on {3}'.format(
        np.round(level_agl, decimals=1), 
        pd.to_datetime(ws_layer.time.values).strftime('%Y-%m-%dT%H%M'),
        model,
        zoom_on,
        speed_plane)
plt.title(plot_title)

if zoom_on is not None:
    plt.ylim(lat_range)
    plt.xlim(lon_range)

if save_plot:
    tools.save_figure(plot_title, save_folder)
