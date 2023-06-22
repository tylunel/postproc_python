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

wanted_date = '20210729-2300'

color_map = 'YlOrBr'    # BuPu, coolwarm, viridis, RdYlGn, jet,... (add _r to reverse)

var_name = 'ZS'   #LAI_ISBA, ZO_ISBA, PATCHP7, ALBNIR_S, MSLP, TG1_ISBA, RAINF_ISBA, CLDFR
vmin = 0
vmax = 1500

# level, only useful if var 3D
ilevel = 10  #0 is Halo, 1:2m, 2:6.12m, 3:10.49m, 10:49.3m, 20:141m, 30:304m, 40:600m, 50:1126m, 60:2070m, 66:2930m

zoom_on = 'marinada'  #None for no zoom, 'liaise' or 'urgell'

save_plot = True
#save_folder = './figures/scalar_maps/pgd/'
save_folder = f'./figures/zonal_maps/{model}/{var_name}/'

add_winds = True
barb_size_option = 'weak_winds'  # 'weak_winds' or 'standard'

##############################################

prop = gv.zoom_domain_prop[zoom_on]
skip_barbs = prop['skip_barbs']
barb_length = prop['barb_length']
lat_range = prop['lat_range']
lon_range = prop['lon_range']
figsize = prop['figsize']


filename1 = tools.get_simu_filename(model, wanted_date,
                                   global_simu_folder=gv.global_simu_folder)

prev_date = (pd.Timestamp(wanted_date)- pd.Timedelta(1, 'h')).strftime('%Y%m%d-%H%M')
filename0 = tools.get_simu_filename(model, prev_date)

# load dataset, default datetime okay as pgd vars are all the same along time
#ds0 = xr.open_dataset(filename0)
ds1 = xr.open_dataset(filename1)
#ds1 = xr.open_dataset(
#        gv.global_simu_folder + \
#        '2.01_pgds_irr/PGD_400M_CovCor_v26_ivars.nc')


#%% DIAG CALCULATION
ds1 = tools.center_uvw(ds1)
#ds0 = tools.center_uvw(ds0)

ds1['WS'], ds1['WD'] = tools.calc_ws_wd(ds1['UT'], ds1['VT'])
#ds0['WS'], ds0['WD'] = tools.calc_ws_wd(ds0['UT'], ds0['VT'])


#%% DATA SELECTION and ZOOM

#varNd = ds_filt[var_name]
varNd = ds1['ZS']

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


#%% CONDITIONS OF MARINADA
var_list=['RVT', 'WS', 'WD', 'ZS']
#ds1_red = ds1[var_list].isel(level=ilevel)
ds1_red = ds1[var_list]
#ds0_red = ds0[var_list].isel(level=ilevel)

cond1 = (90 < ds1_red['WD'])
cond2 = (ds1_red['WD'] < 200)
winddir_cond = np.logical_and(cond1, cond2)
#humidity_cond = (ds1_red['RVT'] > ds0_red['RVT']*1.05)

#marinada = np.logical_and(winddir_cond, humidity_cond)
marinada = winddir_cond

#ds_filt = ds1_red.where(marinada)
ds_filt = ds1_red

#%% AREAS

areas_corners = gv.areas_corners

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
    
    
#    data_in = ds_filt[['WS', 'WD']].isel(level=ilevel)
#    # Classify points within the polygon    
#    t0 = time.time()
#    lon_list = polygon.exterior.xy[0]
#    lat_list = polygon.exterior.xy[1]
#    data_in_red = tools.subset_ds(data_in, 
#                    lat_range=[np.min(lat_list), np.max(lat_list)], 
#                    lon_range=[np.min(lon_list), np.max(lon_list)])
#    classified_points = tools.get_points_in_polygon(data_in_red, polygon)
#    print('time get_pts_.. : ', time.time()-t0)
#    
#    # concatenate data
#    extracted_ds = xr.concat(classified_points, 'ind')
#    # keep layer of interest
##    extracted_da = extracted_ds['WS'][:, ilevel]
#    extracted_layer = extracted_ds
#    
#    # filter:
##    filtered_da = extracted_da.where(extracted_da > 2)
#    filtered_ds = extracted_layer.where(90 < extracted_layer['WD']).where(extracted_layer['WD'] < 200)
#    layer_for_fig = filtered_ds['WS']
#
#    # plot    
#    # extrated area plot
#    plt.scatter(layer_for_fig.longitude, layer_for_fig.latitude, layer_for_fig.values,
#                color='b',
##                c=layer_for_fig.values, 
##                cmap='BuPu', s=10,
#                )
    
    plt.plot(*polygon.exterior.xy)

#%% POINTS SITES

points = [
        'cendrosa',
#        'ponts',
          'elsplans', 
#          'irta-corn',
          'coll_lilla',
#          'lleida', 
#          'zaragoza',
#          'puig formigosa', 
#          'tossal_baltasana', 
          'tossal_gros', 
#          'tossal_torretes',
          'torredembarra',
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


#%% TEST
#site = 'cendrosa'
#lat, lon, _ = gv.whole[site].values()
#indlat, indlon = tools.indices_of_lat_lon(ds1, lat, lon)
#column = ds1[['WS', 'WD', 'TKET', 'RVT']].isel(nj=indlat, ni=indlon)
#
#ds_subset = tools.subset_ds(ds1[['WS', 'WD', 'TKET', 'RVT', 'ZS']], 
#                            zoom_on='marinada')
#
##ds_new = tools.diag_lowleveljet_height_5percent(ds_subset)
#
#ds_subset['H_LOWJET_TKE'] = ds_subset['ZS']*0
#
#t0 = time.time()
##length = len(ds_subset.ni)
##for i, ni in enumerate(ds_subset.ni):
##    print(f'i = {i}/{length}')
##    for j, nj in enumerate(ds_subset.nj):
#j=40
#i=30
#        
#column = ds_subset.isel(nj=j, ni=i)
#column['dTKETdz'] = xr.DataArray(coords={'level':ds_subset.level}, data=np.gradient(column['TKET']))
#column['d2TKETd2z'] = xr.DataArray(coords={'level':ds_subset.level}, data=np.gradient(column['dTKETdz']))
#
## 1er diag de H_MARI sur ws
#top_layer_agl = 1000
#column = column.where(column.level<top_layer_agl, drop=True)

#jet_level_indices = []
#jet_top_indices = []
#research_jet_top = False
#for ilevel, level_agl in enumerate(column.level):
#    # first look at the jet height
#    if not research_jet_top:
#        if ilevel<3:
#            pass
#        else:
#            sign_temp = column['dWSdz'][ilevel] * column['dWSdz'][ilevel-1]
#            if sign_temp < 0:  # sign change
#                if column['d2WSd2z'][ilevel] < 0:  # is a maximum
#                    jet_level_indices.append(ilevel)
#                    jet_speed = column.isel(level=ilevel)['WS']
#                    jet_speed_95 = jet_speed*0.95
#                    research_jet_top = True
#    # second step where we look at the height at which 5% threshold is found
#    elif research_jet_top:
#        if column.isel(level=ilevel)['WS'] < jet_speed_95:
#            jet_top_indices.append(ilevel)
#            research_jet_top = False
#
#try:
#    H_LOWJET = float(column.isel(level=jet_top_indices[0]).level)
#except IndexError:
#    H_LOWJET = np.nan
            
#        ds_subset['H_LOWJET'][j, i] = H_LOWJET
        
#print(time.time()-t0)

#look for JETS: with fitted functions
#from scipy.interpolate import UnivariateSpline
#from scipy.optimize import fsolve
## first derivative must be 0
#dWSdz_fitted = UnivariateSpline(column.level, column['dWSdz'], 
#                            s=0,  # important in order to have a fit really close to data
#                            k=4)
#fd2WSd2z_fitted = UnivariateSpline(column.level, column['d2WSd2z'], 
#                            s=0,  # important in order to have a fit really close to data
#                            k=4)
#height_jets = fsolve(dWSdz_fitted, np.arange(10, 500, 100))
## second derivative must be negative
##for i, height in enumerate(height_jets):
#[height for height in height_jets if fd2WSd2z_fitted(height) < 0]

#fig, ax = plt.subplots(1, 3)
#ax[0].plot(column['TKET'], column.level)
#ax[1].plot(column['dTKETdz'], column.level)
##ax[1].plot(dWSdz_fitted(column.level), column.level)
#ax[2].plot(column['d2TKETd2z'], column.level)
##ax[2].scatter(heights_jets['d2WSd2z'], heights_jets.level)
#for axe in ax:
#    axe.set_ylim([0,top_layer_agl])
#    axe.grid()

# 1er diag de H_MARI sur ws
#low_column = column.where(column.level<800, drop=True)

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
