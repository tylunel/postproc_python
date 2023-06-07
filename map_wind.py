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

#########################################"""
model = 'irr_d1'
ilevel = 5  #0 is Halo, 1:2m, 2:6.1m, 3:10.5m, 10:49.3m, 20:141m, 30:304m, 40:600m, 50:1126m, 60:2070, 66:2930

var = 'PGF'  #'WIND' or 'PGF'

# Datetime
wanted_date = '20210729-2300'

#domain_nb = int(model[-1])
domain_nb = 1

speed_plane = 'horiz'  # 'horiz': horizontal normal wind, 'verti' for W

if var == 'WIND':
    if speed_plane == 'verti':
        vmax_cbar = 1
        vmin_cbar = -vmax_cbar
        cmap_name = 'seismic'
    elif speed_plane == 'horiz':
        vmax_cbar = 15
        vmin_cbar = 0
        cmap_name = 'BuPu'
elif var == 'PGF':
    if speed_plane == 'verti':
        vmax_cbar = 0.1
        vmin_cbar = -vmax_cbar
        cmap_name = 'seismic'
    elif speed_plane == 'horiz':
        vmax_cbar = 0.0003
        vmin_cbar = 0
        cmap_name = 'BuPu'

zoom_on = 'urgell'  #None for no zoom, 'urgell', 'liaise'

save_plot = True
save_folder = f'./figures/winds/{model}/{speed_plane}/{ilevel}/zoom_{zoom_on}/'

if var == 'WIND':
    barb_size_option = 'weak_winds'  # 'weak_winds' or 'standard'
elif var == 'PGF':
    barb_size_option = 'pgf_weak'


###########################################

barb_size_increments = gv.barb_size_increments
barb_size_description = gv.barb_size_description

prop = gv.zoom_domain_prop[zoom_on]
skip_barbs = prop['skip_barbs']
barb_length = prop['barb_length']
lat_range = prop['lat_range']
lon_range = prop['lon_range']
figsize = prop['figsize']
# OR: #locals().update(gv.zoom_domain_prop[zoom_on])

filename = tools.get_simu_filename(model, wanted_date,
                                   global_simu_folder=gv.global_simu_folder)
#filename = tools.get_simu_filename(
#        model, 
#        wanted_date,
#        file_suffix='',  #'dg' or ''
#        out_suffix='.OUT',
#        global_simu_folder=gv.global_temp_folder)

# load file, dataset and set parameters
ds1 = xr.open_dataset(filename,
                      decode_coords="coordinates",
                      )

# other calculations

ds1['DIV'] = mcalc.divergence(ds1['UT'], ds1['VT'])

ds1['DENS'] = mcalc.density(
    ds1['PRES']*units.hectopascal,
    ds1['TEMP']*units.celsius, 
    ds1['RVT']*units.gram/units.gram)

ds1['MSLP3D'] = tools.calc_mslp(ds1, ilevel=None)

ds1['PRES_GRAD_W'], ds1['PRES_GRAD_U'], ds1['PRES_GRAD_V'] = \
    mcalc.gradient(ds1['MSLP3D'].squeeze()[:, :, :], axes=['level', 'ni', 'nj'])
ds1['PGF_U'] = -(1/ds1['DENS'])*ds1['PRES_GRAD_U']
ds1['PGF_V'] = -(1/ds1['DENS'])*ds1['PRES_GRAD_V']
ds1['PGF_W'] = -(1/ds1['DENS'])*ds1['PRES_GRAD_W']
ds1['PGF'], ds1['PGF_dir'] = tools.calc_ws_wd(ds1['PGF_U'], ds1['PGF_V'])


fig1 = plt.figure(figsize=figsize)

#%% WIND SPEED COLORMAP
if speed_plane == 'horiz':
    if var == 'WIND':
        ws = mcalc.wind_speed(ds1['UT'] * units.meter_per_second, 
                              ds1['VT'] * units.meter_per_second)
    elif var == 'PGF':
        ws, _ = tools.calc_ws_wd(ds1['PGF_U'], ds1['PGF_V'])
#wd = mpcalc.wind_direction(ds1['UT'] * units.meter_per_second, 
#                           ds1['VT'] * units.meter_per_second)
elif speed_plane == 'verti':
    if var == 'WIND':
        ws = ds1['WT']
    elif var == 'PGF':
        ws = ds1['PGF_W']

# keep only layer of interest
ws_layer = ws.squeeze()[ilevel, :, :]
#ws_layer = ds1['ZS'].squeeze()[:, :]

plt.pcolormesh(ds1.longitude, ds1.latitude, ws_layer,
#               cbar_kwargs={"orientation": "horizontal", "shrink": 0.7}
               cmap=cmap_name,
               vmin=vmin_cbar,
               vmax=vmax_cbar
#               vmin=0,
#               vmax=1000  
              )
cbar = plt.colorbar()
cbar.set_label('Wind speed [m/s]')
#%% WIND BARBS

X = ds1.longitude
Y = ds1.latitude
if var == 'WIND':
    U = ds1['UT'].squeeze()[ilevel, :,:]
    V = ds1['VT'].squeeze()[ilevel, :,:]
elif var == 'PGF':
    U = ds1['PGF_U'].squeeze()[ilevel, :,:]
    V = ds1['PGF_V'].squeeze()[ilevel, :,:]

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
             xy=(0.1, 0.02),
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

points = ['cendrosa', 'elsplans', 
#          'irta-corn', 
#          'border_irrig_noirr',
#          'puig formigosa', 
          'tossal_baltasana', 
          'tossal_gros', 
#          'tossal_torretes', 
          'coll_lilla',
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
#    if site == 'elsplans':
#        plt.text(sites[site]['lon']-0.1,
#                 sites[site]['lat']-0.03, 
#                 site, 
#                 fontsize=9)
#    else:
    plt.text(sites[site]['lon']+0.01,
             sites[site]['lat']+0.01, 
             site, 
             fontsize=12)

#%% STATION PLOT
#from metpy.plots import StationPlot
#
#ax = plt.gca()
#Cendro = StationPlot(ax, sites['cendrosa']['lon'], sites['cendrosa']['lat'])
#Cendro.plot_barb(4, 5, 
#                 pivot='middle',
#                 length=barb_length*4,     #length of barbs
#                  sizes={
#        #                 'spacing':1, 
#        #                 'height':1,
#        #                 'width':1,
#                         'emptybarb':0.01},
#                  barb_increments=barb_size_increments[barb_size_option]
#                  )

#%% FIGURE OPTIONS
if speed_plane == 'horiz':
    level_agl = ws_layer.level
if speed_plane == 'verti':
    level_agl = ws_layer.level_w
    
plt.xlabel('longitude')
plt.ylabel('latitude')
        
plot_title = '{4} {5} at {0}m on {1} for simu {2} zoomed on {3}'.format(
        np.round(level_agl, decimals=1), 
        pd.to_datetime(ws_layer.time.values).strftime('%Y-%m-%dT%H%M'),
        model,
        zoom_on,
        speed_plane,
        var)
plt.title(plot_title)


if zoom_on is None:
    plt.ylim([ws_layer.latitude.min(), ws_layer.latitude.max()])
    plt.xlim([ws_layer.longitude.min(), ws_layer.longitude.max()])
else:
    plt.ylim(lat_range)
    plt.xlim(lon_range)

if save_plot:
    tools.save_figure(plot_title, save_folder)
