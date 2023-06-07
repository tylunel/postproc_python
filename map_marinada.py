#!/usr/bin/env python3
"""
@author: tylunel
Creation : 07/01/2021

Script for plotting simple colormaps 
"""

import matplotlib.pyplot as plt
import xarray as xr
import tools
import shapefile
import pandas as pd
import global_variables as gv
import metpy.calc as mcalc
from metpy.units import units
from shapely.geometry import Polygon
import time
import numpy as np


###############################################
model = 'irr_d1'

domain_nb = int(model[-1])

wanted_date = '20210715-1900'

color_map = 'YlOrBr'    # BuPu, coolwarm, viridis, RdYlGn, jet,... (add _r to reverse)

var_name = 'ZS'   #LAI_ISBA, ZO_ISBA, PATCHP7, ALBNIR_S, MSLP, TG1_ISBA, RAINF_ISBA, CLDFR
vmin = 0
vmax = 1200

# level, only useful if var 3D
ilevel = 10  #0 is Halo, 1:2m, 2:6.12m, 3:10.49m, 10:49.3m, 20:141m, 30:304m, 40:600m, 50:1126m, 60:2070m, 66:2930m

zoom_on = 'urgell'  #None for no zoom, 'liaise' or 'urgell'

save_plot = True
#save_folder = './figures/scalar_maps/pgd/'
save_folder = './figures/zonal_maps/{1}/{2}/'.format(
        domain_nb, model, var_name)

add_winds = True
barb_size_option = 'weak_winds'  # 'weak_winds' or 'standard'

##############################################

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



filename1 = tools.get_simu_filename(model, wanted_date,
                                   global_simu_folder=gv.global_simu_folder)

prev_date = (pd.Timestamp(wanted_date)- pd.Timedelta(1, 'h')).strftime('%Y%m%d-%H%M')
filename0 = tools.get_simu_filename(model, prev_date)

# load dataset, default datetime okay as pgd vars are all the same along time
ds0 = xr.open_dataset(filename0)
ds1 = xr.open_dataset(filename1)
#ds1 = xr.open_dataset(
#        gv.global_simu_folder + \
#        '2.01_pgds_irr/PGD_400M_CovCor_v26_ivars.nc')


#%% DIAG CALCULATION
ds1 = tools.center_uvw(ds1)
ds0 = tools.center_uvw(ds0)

#ds1['DIV'] = mcalc.divergence(ds1['UT'], ds1['VT'])
#ds1['PRES_GRAD_W'], ds1['PRES_GRAD_U'], ds1['PRES_GRAD_V'] = \
#    mcalc.gradient(ds1['PRES'].squeeze()[:, :, :])
#ds1['DENS'] = mcalc.density(
#    ds1['PRES']*units.hectopascal,
#    ds1['TEMP']*units.celsius, 
#    ds1['RVT']*units.gram/units.gram)
#
#ds1['PGF_U'] = -(1/ds1['DENS'])*ds1['PRES_GRAD_U']
#ds1['PGF_V'] = -(1/ds1['DENS'])*ds1['PRES_GRAD_V']
#ds1['PGF_W'] = -(1/ds1['DENS'])*ds1['PRES_GRAD_W']
#
#ds1['PGF'], ds1['PGF_dir'] = tools.calc_ws_wd(ds1['PGF_U'], ds1['PGF_V'])
ds1['WS'], ds1['WD'] = tools.calc_ws_wd(ds1['UT'], ds1['VT'])

ds0['WS'], ds0['WD'] = tools.calc_ws_wd(ds0['UT'], ds0['VT'])

#%% CONDITIONS OF MARINADA
var_list=['RVT', 'WS', 'WD', 'ZS']
ds1_red = ds1[var_list].isel(level=ilevel)
ds0_red = ds0[var_list].isel(level=ilevel)

cond1 = (90 < ds1_red['WD'])
cond2 = (ds1_red['WD'] < 200)
winddir_cond = np.logical_and(cond1, cond2)
humidity_cond = (ds1_red['RVT'] > ds0_red['RVT']*1.05)

marinada = np.logical_and(winddir_cond, humidity_cond)

ds_filt = ds1.where(marinada)

#%% DATA SELECTION and ZOOM

varNd = ds_filt[var_name]
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

plt.contourf(var2d.longitude, var2d.latitude, var2d,
             levels=20,
#plt.pcolormesh(var2d.longitude, var2d.latitude, var2d,
#               cbar_kwargs={"orientation": "horizontal", "shrink": 0.7}
               cmap=color_map,
               alpha=0.5,
#               levels=np.linspace(vmin, vmax, (vmax-vmin)*4+1), # fixed colorbar
#               extend = 'both',  #highlights the min and max in edges values
               vmin=vmin, vmax=vmax,
#               levels=20
               )

cbar = plt.colorbar(boundaries=[vmin, vmax])
try:
    cbar.set_label(var2d.long_name)
except AttributeError:
    cbar.set_label(var_name)
#cbar.set_clim(vmin, vmax)

#%% WIND BARBS

barb_size_increments = gv.barb_size_increments
barb_size_description = gv.barb_size_description

if add_winds:
    X = ds1.longitude
    Y = ds1.latitude
    U = ds1.UT.squeeze()[ilevel, :,:]
    V = ds1.VT.squeeze()[ilevel, :,:]
    
    plt.barbs(X[::skip_barbs, ::skip_barbs], Y[::skip_barbs, ::skip_barbs], 
              U[::skip_barbs, ::skip_barbs], V[::skip_barbs, ::skip_barbs],
              pivot='middle',
              length=barb_length,     #length of barbs
              sizes={
    #                 'spacing':1, 'height':1,'width':1,
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

# --- Irrigation borders
    
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

# --- Sea borders
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

# --- France borders
sf = shapefile.Reader("TM-WORLD-BORDERS/TM_WORLD_BORDERS-0.3.sph")
shapes=sf.shapes()
france = shapes[64].points
france_df = pd.DataFrame(france, columns=['lon', 'lat'])
france_S = france_df[france_df.lat < 43.35]
france_SW = france_S[france_S.lon < 2.95]
plt.plot(france_SW.lon, france_SW.lat,
         color='k',
         linewidth=1)


#%% AREAS

areas_corners = {
    'irrig': ['lleida', 'balaguer', 
              'claravalls', 'borges_blanques'],
    'dry': ['claravalls', 'borges_blanques', 
            'els_omellons', 'sant_marti', 'fonolleres'],
    'slope_west': ['els_omellons', 'sant_marti', 'fonolleres',
                   'santa_coloma', 'tossal_gros', 'villobi'],
    'slope_east': ['santa_coloma', 'tossal_gros', 'villobi',
                   'tossal_purunyo', 'puig_cabdells', 'puig_formigosa'],
    'coast': ['tossal_purunyo', 'puig_cabdells', 'puig_formigosa',
              'calafell', 'tarragona',],
              }

polygon_dict = {}

for area in areas_corners:
    print(area)
    corners = areas_corners[area]
    corners_coordinates = []
    for corner in corners:
        corners_coordinates.append(
            (gv.whole[corner]['lon'], gv.whole[corner]['lat']))

    polygon = Polygon(corners_coordinates)
    polygon_dict[area] = polygon
    
    
    data_in = ds_filt[['WS', 'WD']].isel(level=ilevel)
    # Classify points within the polygon
    t0 = time.time()
    classified_points = tools.get_points_in_polygon(data_in, polygon)
    print('time get_pts_.. : ', time.time()-t0)
    # concatenate data
    extracted_ds = xr.concat(classified_points, 'ind')
    # keep layer of interest
#    extracted_da = extracted_ds['WS'][:, ilevel]
    extracted_layer = extracted_ds
    
    # filter:
#    filtered_da = extracted_da.where(extracted_da > 2)
    filtered_ds = extracted_layer.where(90 < extracted_layer['WD']).where(extracted_layer['WD'] < 200)
    layer_for_fig = filtered_ds['WS']

    # plot    
    # extrated area plot
    plt.scatter(layer_for_fig.longitude, layer_for_fig.latitude, layer_for_fig.values,
                color='b',
#                c=layer_for_fig.values, 
#                cmap='BuPu', s=10,
                )
    plt.plot(*polygon.exterior.xy)

#%% POINTS SITES

points = [
        'cendrosa',
        'ponts',
#          'elsplans', 
#          'irta-corn',
          'lleida', 
          'zaragoza',
#          'puig formigosa', 
          'tossal_baltasana', 
          'tossal_gros', 
          'tossal_torretes', 
#       'moncayo', 'tres_mojones', 
#          'guara', 'caro', 'montserrat', 'joar',
          ]

sites = {key:gv.whole[key] for key in points}

for site in sites:
    plt.scatter(sites[site]['lon'],
                sites[site]['lat'],
                color='r',
                s=12        #size of markers
                )
    plt.text(sites[site]['lon']+0.01,
             sites[site]['lat']+0.01, 
             site.capitalize(), 
             fontsize=14)


#%% FIGURE OPTIONS and ZOOM
if len(varNd.shape) == 2:
    plot_title = '{0} - {1} for simu {2}'.format(
        wanted_date, var_name, model)
elif len(varNd.shape) == 3:
    plot_title = '{0} - {1} for simu {2} at {3}m'.format(
        wanted_date, var_name, model, var2d.level.round())

plt.title(plot_title)
plt.xlabel('longitude')
plt.ylabel('latitude')

if zoom_on is None:
    plt.ylim([var2d.latitude.min(), var2d.latitude.max()])
    plt.xlim([var2d.longitude.min(), var2d.longitude.max()])
else:
    plt.ylim(lat_range)
    plt.xlim(lon_range)

if save_plot:
    tools.save_figure(plot_title, save_folder)
