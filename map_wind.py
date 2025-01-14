#!/usr/bin/env python3
"""
@author: tylunel
Creation : 07/01/2021

"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import metpy.calc as mcalc
from metpy.units import units
import pandas as pd 
import tools
import shapefile
import global_variables as gv
from metpy.plots import StationPlot
import os

#########################################"""
model = 'irrswi1_d1'
domain_nb = 1

ilevel = 3  #0 is Halo, 1:2m, 2:6.1m, 3:10.5m, 10:49.3m, 20:141m, 30:304m, 40:600m, 50:1126m, 60:2070, 66:2930

# Datetime
wanted_date = '20210722-1200'

speed_plane = 'horiz'  # 'horiz': horizontal normal wind, 'verti' for W

if speed_plane == 'verti':
    vmax_cbar = 1
    vmin_cbar = -vmax_cbar
    cmap_name = 'seismic'
elif speed_plane == 'horiz':
    vmax_cbar = 15
    vmin_cbar = 0
    cmap_name = 'BuPu'

zoom_on = 'urgell-paper'  #None for no zoom, 'urgell', 'liaise'

add_smc_obs = True

save_plot = True
save_folder = f'./figures/winds/{model}/{speed_plane}/{ilevel}/zoom_{zoom_on}/'

barb_size_option = 'standard'  # 'weak_winds' or 'standard'
isoalti_list = [600]
###########################################

if add_smc_obs:
    alpha = 0.4
    if zoom_on == 'marinada':
        barb_length_coeff = 1.2
    else:
        barb_length_coeff = 1.1
else:
    alpha = 0.9

if add_smc_obs and ilevel > 6:
    raise ValueError(f"""Height of model level and of observation stations
                     are significantly different:
                     - SMC stations: 2-10m
                     - model: {gv.layers_height_MNH_LIAISE[ilevel]}m""")

barb_size_increments = gv.barb_size_increments
barb_size_description = gv.barb_size_description

prop = gv.zoom_domain_prop[zoom_on]
skip_barbs = prop['skip_barbs']
skip_barbs = 2
barb_length = prop['barb_length']
lat_range = prop['lat_range']
lon_range = prop['lon_range']
figsize = prop['figsize']
# OR: #locals().update(gv.zoom_domain_prop[zoom_on])

# for paper: hard-coded:
# skip_barbs = 3
#barb_length = 5.5
#barb_size_option = 'weak_winds'

plt.rcParams.update({'font.size': 11})

filename = tools.get_simu_filepath(model, wanted_date,
                                   global_simu_folder=gv.global_simu_folder)

# load file, dataset and set parameters
ds1 = xr.open_dataset(filename,
                      decode_coords="coordinates",
                      )

# other calculations
ds1['DIV'] = mcalc.divergence(ds1['UT'], ds1['VT'])

fig1 = plt.figure(figsize=figsize)

#%% WIND SPEED COLORMAP
if speed_plane == 'horiz':
    ws = mcalc.wind_speed(ds1['UT'] * units.meter_per_second, 
                          ds1['VT'] * units.meter_per_second)
elif speed_plane == 'verti':
    ws = ds1['WT']

# keep only layer of interest
ws_layer = ws.squeeze()[ilevel, :, :]
#ws_layer = ds1['ZS'].squeeze()[:, :]

plt.pcolormesh(ds1.longitude, ds1.latitude, ws_layer,
#               cbar_kwargs={"orientation": "horizontal", "shrink": 0.7}
               cmap=cmap_name,
               vmin=vmin_cbar,
               vmax=vmax_cbar,
               alpha=0.8,
#               vmin=0,
#               vmax=1000  
              )
cbar = plt.colorbar()
cbar.set_label('Wind speed [m/s]')
#%% WIND BARBS

X = ds1.longitude
Y = ds1.latitude

U = ds1['UT'].squeeze()[ilevel, :,:]
V = ds1['VT'].squeeze()[ilevel, :,:]

plt.barbs(X[::skip_barbs, ::skip_barbs], Y[::skip_barbs, ::skip_barbs], 
          U[::skip_barbs, ::skip_barbs], V[::skip_barbs, ::skip_barbs],
          pivot='middle',
          length=barb_length,     #length of barbs
          sizes={
#                 'spacing':1, 
#                 'height':1,
#                 'width':1,
                 'emptybarb':0.01},
          barb_increments=barb_size_increments[barb_size_option],
          alpha=alpha,
          )
plt.annotate(barb_size_description[barb_size_option],
             fontsize=9,
             xy=(0.1, 0.015),
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


# --- ISOLINES FOR ALTI
# contour line of isoaltitude
isoalti = ds1['ZS']
plt.contour(isoalti.longitude.data, 
            isoalti.latitude.data, 
            isoalti,
            levels=isoalti_list,   #+1 -> number of contour to plot 
            linestyles=':',
            linewidths=1.,
            colors='k',
            )

#%% POINTS SITES

points = ['cendrosa', 'elsplans', 
#          'irta-corn', 
#          'border_irrig_noirr',
#          'puig formigosa', 
#          'tossal_baltasana', 
#          'lleida',
#          'tossal_gros', 
#          'tossal_torretes', 
#          'coll_lilla',
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
    # print site name on fig:
    try:
        sitename = sites[site]['longname']
    except KeyError:
        sitename = site
        
    plt.text(sites[site]['lon']+0.01,
             sites[site]['lat']+0.01, 
             sitename, 
             fontsize=15)

#%% STATION PLOT
    
if add_smc_obs:
    ax = plt.gca()
    
    # --- with SMC data of liaise database ---
#    stations_2m = ['VH', 'WK', 'V1', 'WB', 'VM', 'WX', 'WA', 'WC', 'V8', 'XI',
#                   'XM', 'WL', 'UM', 'WI', 'VE']
#    stations_10m = ['VK', 'C6', 'C7', 'C8', 'D1', 'XD', 'XR', 'XA', 'VP', 'VB','VQ']
#    stations_unk = ['YJ', 'CW', 'MR', 'VM', 'WV', 'VD', 'YD', 'XX', 'YJ', ]
#    stations_all = stations_2m + stations_10m + stations_unk
#    
#    for station in stations_all:
#        # get data
#        datafolder = gv.global_data_liaise + '/SMC/22_stations_liaise/'
#        filename = f'LIAISE_{station}_SMC_MTO-1MN_L0_{wanted_date[:8]}_V01.nc'
#        try:
#            obs = xr.open_dataset(datafolder + filename)
#            # find closest time
#            obs['time_dist'] = np.abs(obs.time - pd.Timestamp(wanted_date).to_datetime64())
#            obs_t = obs.where(obs['time_dist'] == obs['time_dist'].min(), 
#                              drop=True).squeeze()
#        except (FileNotFoundError, ValueError):
#            continue
#        
#        obs_t['UT'], obs_t['VT'] = tools.calc_u_v(obs_t['WS'], obs_t['WD'])
#    
#        # plot station
#        location = StationPlot(ax, obs['lon'], obs['lat'])
#        location.plot_barb(obs_t['UT'], obs_t['VT'],
#                           pivot='middle',
#                           length=barb_length*barb_length_coeff,     #length of barbs
#                           sizes={'emptybarb':0.1},
#                           barb_increments=barb_size_increments[barb_size_option]
#                           )
        
    # --- with global SMC data ---
    datafolder = gv.global_data_liaise + '/SMC/ALL_stations_july/'
    for filename in os.listdir(datafolder):
        # filename = 'C6.nc'
        file_path = os.path.join(datafolder, filename)
        if os.path.isfile(file_path):
            try:
                obs = xr.open_dataset(file_path)
                # struggling with the datetime formats in the 3 next lines...
                obs['datetime'] = [pd.Timestamp(str((elt.data))) for elt in obs['datetime']]
                obs['datetime64'] = [elt.data.astype('int64') for elt in obs['datetime']]
                obs['datetime64'] = obs['datetime64'].swap_dims({'datetime64' :'datetime'})
                # compute distance to wanted datetime
                obs['time_dist'] = np.abs(obs['datetime64'] - pd.Timestamp(wanted_date).to_datetime64().astype('int64'))
                # keep the data closest to wanted datetime
                obs_t = obs.where(obs['time_dist'] == obs['time_dist'].min(), 
                                  drop=True).squeeze()
                # get height of wind measurement
                wind_height = int((obs_t['obs_wind_height'].data))
                obs_t['UT'], obs_t['VT'] = tools.calc_u_v(
                    obs_t[f'VV{wind_height}'], obs_t[f'DV{wind_height}'])
            
                # --- plot station data ---
                if (lon_range[0] < obs['lon'] < lon_range[1] and \
                    lat_range[0] < obs['lat'] < lat_range[1]):
                    
                    # Create the station object and set the location of it
                    location = StationPlot(ax, obs['lon'], obs['lat'])
                    # plot the wind
                    location.plot_barb(
                            obs_t['UT'].data, obs_t['VT'].data,
                           pivot='tip',  # 'tip' or 'middle'
                           length=barb_length*barb_length_coeff,     #length of barbs
                           sizes={'emptybarb':0.1},
                           barb_increments=barb_size_increments[barb_size_option]
                           )
                    
                    if wind_height != 10:
                        ax.text(obs['lon']+0.008, obs['lat']-0.016, 
                                 wind_height, 
                                 fontsize=7)
                    print(f'{filename} plotted')
                    
            except (FileNotFoundError, ValueError, IndexError, TypeError) as e:
                print(f"Error for {obs['station_name']}:")
                if hasattr(e, 'message'):
                    print(e.message)
                else:
                    print(e)
                continue
                
                

#%% FIGURE OPTIONS
if speed_plane == 'horiz':
    level_agl = ws_layer.level
if speed_plane == 'verti':
    level_agl = ws_layer.level_w
    
plt.xlabel('longitude')
plt.ylabel('latitude')
        

plot_title = '{0}m wind - {1} - {2}'.format(
        np.round(level_agl, decimals=1), 
        pd.to_datetime(ws_layer.time.values).strftime('%Y-%m-%dT%H%M'),
        model)
plt.title(plot_title)

if zoom_on is not None:
    # plt.ylim([ws_layer.latitude.min(), ws_layer.latitude.max()])
    # plt.xlim([ws_layer.longitude.min(), ws_layer.longitude.max()])
# else:
    plt.ylim(lat_range)
    plt.xlim(lon_range)

save_title = '{4} wind at {0}m on {1} for simu {2} zoomed on {3}'.format(
        np.round(level_agl, decimals=1), 
        pd.to_datetime(ws_layer.time.values).strftime('%Y-%m-%dT%H%M'),
        model,
        zoom_on,
        speed_plane)
if save_plot:
    tools.save_figure(save_title, save_folder)
