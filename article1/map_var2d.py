#!/usr/bin/env python3
"""
@author: tylunel
Creation : 07/01/2021

Script for plotting simple colormaps 
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import tools
import shapefile
import pandas as pd
import global_variables as gv
import metpy.calc as mcalc
from metpy.units import units
from shapely.geometry import Polygon
import os
from metpy.plots import StationPlot

###############################################
model = 'irr_d2'  # std_d2 or irr_d2

domain_nb = int(model[-1])

wanted_date = '20210722-2300'

color_map = 'YlGnBu'  # BuPu, coolwarm, viridis, RdYlGn, jet,... (add _r to reverse)
                    # YlOrBr for orography
                    # YlGn for LAI
                    # YlGnBu for SWI
                    # binary_r for albedo

var_name = 'TSWI_T_ISBA'   #LAI_ISBA, TALB_ISBA, TSWI_T_ISBA
vmin = -0.2
vmax = 1.4
cbar_title = 'Soil Water Index'  # 'Soil Water Index', 'Leaf Area Index'
plot_title = 'SWI for simu IRR'

# level, only useful if var 3D
ilevel = 1  #0 is Halo, 1:2m, 2:6.12m, 3:10.49m, 10:49.3m, 20:141m, 30:304m, 40:600m, 50:1126m, 60:2070m, 66:2930m
ilevel_wind = 3

zoom_on = 'urgell-paper'  #None for no zoom, 'liaise' or 'urgell'

add_winds = False
add_smc_obs = False
add_pgf = False
marinada_areas = False
barb_size_option = 'weak_winds'  # 'weak_winds' or 'standard'

save_plot = True

if add_smc_obs:
    save_folder = f'./fig/'
    if ilevel > 6:
        raise ValueError(f"""Height of model level and of observation stations
                         are significantly different:
                         - SMC stations: 2-10m
                         - model: {gv.layers_height_MNH_LIAISE[ilevel]}m""")
else:
    save_folder = f'./fig/'
##############################################

prop = gv.zoom_domain_prop[zoom_on]
skip_barbs = prop['skip_barbs']
barb_length = prop['barb_length']
lat_range = prop['lat_range']
lon_range = prop['lon_range']
figsize = prop['figsize']
# OR: #locals().update(gv.zoom_domain_prop[zoom_on])

if add_smc_obs:
    alpha = 0.4
    if zoom_on == 'marinada':
        barb_length_coeff = 1.2
    else:
        barb_length_coeff = 1.1
else:
    alpha = 0.9

# size of font on figure
plt.rcParams.update({'font.size': 11})

filename = tools.get_simu_filepath(model, wanted_date,
                                   global_simu_folder=gv.global_simu_folder)

# load dataset, default datetime okay as pgd vars are all the same along time
ds = xr.open_dataset(filename)
#ds1 = xr.open_dataset(
#        gv.global_simu_folder + \
#        '2.01_pgds_irr/PGD_400M_CovCor_v26_ivars.nc')


# DIAGs
ds1 = tools.subset_ds(ds, 
                      zoom_on=zoom_on,
                      nb_indices_exterior=5,
#                      lat_range=[lat_range[0],lat_range[1]+0.1], 
#                      lon_range=[lon_range[0],lon_range[1]+0.1],
                      )

ds1 = tools.center_uvw(ds1)

ds1['WS'], ds1['WD'] = tools.calc_ws_wd(ds1['UT'], ds1['VT'])

#ds_diag = tools.diag_lowleveljet_height_5percent(ds1[['WS', 'ZS']])

#%%
subset_ds = ds1[['WS', 'ZS', 'TKET', 'HBLTOP']]

#ds1['DIV'] = mcalc.divergence(ds1['UT'], ds1['VT'])
#
#ds1['DENS'] = mcalc.density(
#    ds1['PRES']*units.hectopascal,
#    ds1['TEMP']*units.celsius, 
#    ds1['RVT']*units.gram/units.gram)
#
#ds1['MSLP3D'] = tools.calc_mslp(ds1, ilevel=None)


#%% DATA SELECTION and ZOOM

#varNd = ds_diag[var_name]
varNd = ds1[var_name]
#remove single dimensions
varNd = varNd.squeeze()

if len(varNd.shape) == 2:
    var2d = varNd
elif len(varNd.shape) == 3:
    var2d = varNd[ilevel,:,:]
    
# remove 999 values, and replace by nan
var2d = var2d.where(~(var2d == 999))
# filter the outliers
#var2d = var2d.where(var2d <= vmax)


#%% PLOT OF VAR_NAME
fig1 = plt.figure(figsize=figsize)

cmap = plt.cm.get_cmap(color_map).copy()
#cmap.set_under('c')

# --- COLORMAP
#plt.contourf(var2d.longitude, var2d.latitude, var2d,
#             levels=20,
#               levels=np.linspace(vmin, vmax, (vmax-vmin)*2+1), # fixed colorbar
plt.pcolormesh(var2d.longitude, var2d.latitude, var2d,
#               cbar_kwargs={"orientation": "horizontal", "shrink": 0.7}
               cmap=cmap,
#               extend = 'both',  #highlights the min and max in edges values
               vmin=vmin, vmax=vmax,
               )

#plt.imshow(var2d.longitude, var2d.latitude, var2d,
##               cbar_kwargs={"orientation": "horizontal", "shrink": 0.7}
#               cmap=color_map,
##               levels=np.linspace(vmin, vmax, (vmax-vmin)*4+1), # fixed colorbar
##               extend = 'both',  #highlights the min and max in edges values
#               vmin=vmin, vmax=vmax,
#               levels=20
#               )

#cbar = plt.colorbar(boundaries=[vmin, vmax])
cbar = plt.colorbar()

cbar.set_label(cbar_title)

#cbar.set_label('LAI [$m^2 m^{-2}$]')
    
#cbar.set_clim(vmin, vmax)

# --- WIND BARBS

barb_size_increments = gv.barb_size_increments
barb_size_description = gv.barb_size_description

if (add_winds & add_pgf):
    raise ValueError('Only add_winds or add pgf can be true')

if add_winds or add_pgf:
    X = ds1.longitude
    Y = ds1.latitude
    if add_winds:
        U = ds1['UT'].squeeze()[ilevel_wind, :,:]
        V = ds1['VT'].squeeze()[ilevel_wind, :,:]
    elif add_pgf:
        U = ds1['PGF_U'].squeeze()[ilevel_wind, :,:]
        V = ds1['PGF_V'].squeeze()[ilevel_wind, :,:]
    
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
                 xy=(0.1, 0.01),
                 xycoords='subfigure fraction',
                 
                 )

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
#        filename = 'C6.nc'
        file_path = os.path.join(datafolder, filename)
        if os.path.isfile(file_path):
            try:
                # --- open and pre-process the station data ---
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
                obs_t['T_kelvin'] = obs_t['T'] + 273.15

                obs_t['P_pa'] = tools.height_to_pressure_std(
                        obs_t['altitude'], p0=ds1['MSLP'].mean()*100)
                obs_t['THT'] = tools.potential_temperature_from_temperature(
                    obs_t['P_pa'], obs_t['T_kelvin'])

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
                    # plot a scalar variable in the circle
                    ax.scatter(obs['lon'], obs['lat'],
                                color=cmap((obs_t['THT']-vmin)/(vmax-vmin)),
                                edgecolors='k')
                    # add wind measurement height if different from 10m
                    if wind_height != 10:
                        ax.text(obs['lon']+0.008, obs['lat']+0.008, 
                                 wind_height, 
                                 fontsize=7)
                    # plot a scalar variable in the circle
                    ax.scatter(obs['lon'], obs['lat'],
                                color=cmap((obs_t['THT']-vmin)/(vmax-vmin)),
                                edgecolors='k')
                    # state that this station was plotted
                    print(f'{filename} plotted')
            except (FileNotFoundError, ValueError, IndexError, TypeError) as e:
                print(f"Error for {obs['station_name']}:")
                if hasattr(e, 'message'):
                    print(e.message)
                else:
                    print(e)
                continue


# --- IRRIGATED, SEA and COUNTRIES BORDERS

if domain_nb == 2:
    pgd = xr.open_dataset(
        gv.global_simu_folder + \
        '2.01_pgds_irr/PGD_400M_CovCor_v26_ivars.nc')
elif domain_nb == 1:
    pgd = xr.open_dataset(
        gv.global_simu_folder + \
        '2.01_pgds_irr/PGD_2KM_CovCor_v26_ivars.nc')

# Irrigation borders
##from scipy.ndimage.filters import gaussian_filter
##sigma = 0.1     #default is 0.1
##irr_covers = gaussian_filter(pgd.COVER369.data, sigma)
#irr_covers = pgd.COVER369.data
#plt.contour(pgd.longitude.data, 
#            pgd.latitude.data, 
#            irr_covers,
#            levels=0,   #+1 -> number of contour to plot 
#            linestyles='solid',
#            linewidths=1.5,
#            colors='g'
##            colors=['None'],
##            hatches='-'
#            )

# Sea borders
sea_covers = pgd.COVER001.data
plt.contour(pgd.longitude.data, 
            pgd.latitude.data, 
            sea_covers,
            levels=0,   #+1 -> number of contour to plot 
            linestyles='solid',
            linewidths=1.,
            colors='w'
    #        colors=['None'],
    #        hatches='-'
            )

#France borders
sf = shapefile.Reader("../TM-WORLD-BORDERS/TM_WORLD_BORDERS-0.3.sph")
shapes=sf.shapes()
france = shapes[64].points
france_df = pd.DataFrame(france, columns=['lon', 'lat'])
france_S = france_df[france_df.lat < 43.35]
france_SW = france_S[france_S.lon < 2.95]
plt.plot(france_SW.lon, france_SW.lat,
         color='k',
         linewidth=1)

# --- POINTS SITES

points = [
        'cendrosa',
#        'ponts',
          'elsplans', 
#          'irta-corn',
          'lleida', 
#          'zaragoza',
#          'puig formigosa', 
#          'tossal_baltasana', 
#          'tossal_gros', 
#          'coll_lilla',
#          'serra_tallat',
#          'torredembarra',
#          'tossal_torretes', 
#       'moncayo', 'tres_mojones', 
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
             fontsize=12)

# --- AREAS FOR MARINADA STUDY
if marinada_areas:
    areas_corners = gv.areas_corners
    polygon_dict = {}
    
    for it_area, area in enumerate(areas_corners):
        print(area)
        corners = areas_corners[area]
        corners_coordinates = []
        for corner in corners:
            corners_coordinates.append(
                (gv.whole[corner]['lon'], gv.whole[corner]['lat']))
    
        polygon = Polygon(corners_coordinates)
        polygon_dict[area] = polygon
        plt.plot(*polygon.exterior.xy)
    #    
    #    # add name in the middle of the shapes:
    #    plt.text(polygon.centroid.xy[0][0]-0.05,
    #             polygon.centroid.xy[1][0]-0.05, 
    #             area, 
    #             fontsize=12,
    #             fontstyle='italic')



# --- FIGURE OPTIONS and ZOOM
    
#if len(varNd.shape) == 2:
#    plot_title = '{0} - {1} for simu {2}'.format(
#        wanted_date, var_name, model)
#elif len(varNd.shape) == 3:
#    plot_title = '{0} - {1} for simu {2} at {3}m'.format(
#        wanted_date, var_name, model, var2d.level.round())



plt.title(plot_title)
plt.xlabel('longitude', fontsize=12)
plt.ylabel('latitude', fontsize=12)

plt.ylim(lat_range)
plt.xlim(lon_range)

if save_plot:
    tools.save_figure(plot_title, save_folder)

#%%
#    
#plt.figure()
#p9_filt = ds1.PATCHP9.where(ds1.PATCHP9 < 2, other=0)
#plt.pcolormesh(p9_filt.longitude.data,p9_filt.latitude.data,p9_filt, vmin=0, vmax=1)
#
#sea_covers = pgd.COVER001.data
#p9_filt['sea'] = p9_filt*0 + sea_covers
#p9_sea = p9_filt.where(p9_filt.sea == 1, np.nan)
#plt.pcolormesh(p9_sea.longitude.data,p9_sea.latitude.data,p9_sea, vmin=0, vmax=1, cmap='binary')
