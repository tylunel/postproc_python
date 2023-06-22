#!/usr/bin/env python3
"""
@author: tylunel
Creation : 07/01/2021

Script for plotting simple colormaps 
"""

import matplotlib.pyplot as plt
import xarray as xr
import tools
#import shapefile
#import pandas as pd
import global_variables as gv
#import metpy.calc as mcalc
#from metpy.units import units
from shapely.geometry import Polygon
import time
import numpy as np


###############################################
model = 'irr_d1'

wanted_date = '20210717-2300'

var_name = 'TKET'   #LAI_ISBA, ZO_ISBA, PATCHP7, ALBNIR_S, MSLP, TG1_ISBA, RAINF_ISBA, CLDFR

# level, only useful if var 3D
ilevel = 10  #0 is Halo, 1:2m, 2:6.12m, 3:10.49m, 10:49.3m, 20:141m, 30:304m, 40:600m, 50:1126m, 60:2070m, 66:2930m

save_plot = True
#save_folder = './figures/scalar_maps/pgd/'
save_folder = f'./figures/zonal_barplot/{model}/{var_name}/'

barb_size_option = 'weak_winds'  # 'weak_winds' or 'standard'

##############################################

barb_length = 6

filename1 = tools.get_simu_filename(model, wanted_date,
                                   global_simu_folder=gv.global_simu_folder)

# load dataset, default datetime okay as pgd vars are all the same along time
ds1 = xr.open_dataset(filename1)


## if need of previous timestep:

#prev_date = (pd.Timestamp(wanted_date)- pd.Timedelta(1, 'h')).strftime('%Y%m%d-%H%M')
#filename0 = tools.get_simu_filename(model, prev_date)
#ds0 = xr.open_dataset(filename0)
#ds0['WS'], ds0['WD'] = tools.calc_ws_wd(ds0['UT'], ds0['VT'])
#ds0 = tools.center_uvw(ds0)

#%% DIAG CALCULATION
ds1 = tools.center_uvw(ds1)

ds1['WS'], ds1['WD'] = tools.calc_ws_wd(ds1['UT'], ds1['VT'])

#%% DATA SELECTION and ZOOM

#varNd = ds_filt[var_name]
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


#%% CONDITIONS OF MARINADA

#var_list=['WS', 'WD', 'ZS', 'UT', 'VT', var_name]
#
##ds1_red = ds1[var_list].isel(level=ilevel)
#ds1_red = ds1[var_list]
##ds0_red = ds0[var_list].isel(level=ilevel)
#
#cond1 = (90 < ds1_red['WD'])
#cond2 = (ds1_red['WD'] < 200)
#winddir_cond = np.logical_and(cond1, cond2)
##humidity_cond = (ds1_red['RVT'] > ds0_red['RVT']*1.05)
#
##marinada = np.logical_and(winddir_cond, humidity_cond)
#marinada = winddir_cond
#
##ds_filt = ds1_red.where(marinada)
#ds_filt = ds1_red

#%% AREAS

areas_corners = gv.areas_corners
polygon_dict = {}
distance_to_sea_dict = {}

var_list=['WS', 'WD', 'ZS', 'UT', 'VT', var_name]
data_in = ds1[var_list].isel(level=ilevel)

zonal_average_dict = {}

for area in areas_corners:
    print(area)
    corners = areas_corners[area]
    corners_coordinates = []
    for corner in corners:
        corners_coordinates.append(
            (gv.whole[corner]['lon'], gv.whole[corner]['lat']))

    polygon = Polygon(corners_coordinates)
    polygon_dict[area] = polygon
    
    distance_to_sea_dict[area] = tools.distance_from_lat_lon(
            [polygon.centroid.xy[1][0], polygon.centroid.xy[0][0]],
            list(gv.towns['torredembarra'].values()))
    
    # Classify points within the polygon --  V1  
    t0 = time.time()
    lon_list = polygon.exterior.xy[0]
    lat_list = polygon.exterior.xy[1]
    data_in_red = tools.subset_ds(data_in, 
                    lat_range=[np.min(lat_list), np.max(lat_list)], 
                    lon_range=[np.min(lon_list), np.max(lon_list)])
    classified_points = tools.get_points_in_polygon(data_in_red, polygon)
    print('time get_pts_.. : ', time.time()-t0)
    
    
    # concatenate data
    extracted_ds = xr.concat(classified_points, 'ind')
    # keep layer of interest
#    extracted_da = extracted_ds['WS'][:, ilevel]
    extracted_layer = extracted_ds
    
    zonal_average_dict[area] = extracted_layer
    
    # FILTER:
#    filtered_da = extracted_da.where(extracted_da > 2)
#    filtered_ds = extracted_layer.where(90 < extracted_layer['WD']).where(extracted_layer['WD'] < 200)
#    layer_for_fig = filtered_ds['WS']

    # MAP PLOT  
    # extrated area plot
#    plt.scatter(layer_for_fig.longitude, layer_for_fig.latitude, layer_for_fig.values,
#                color='b',
##                c=layer_for_fig.values, 
##                cmap='BuPu', s=10,
#                )
#    plt.plot(*polygon.exterior.xy)


#%% PLOT
mean_var = [float(zonal_average_dict[key][var_name].mean()) for key in zonal_average_dict]
mean_ut = [float(zonal_average_dict[key]['UT'].mean()) for key in zonal_average_dict]
mean_vt = [float(zonal_average_dict[key]['VT'].mean()) for key in zonal_average_dict]
zone_area = np.array([polygon_dict[key].area for key in polygon_dict])
distance_to_sea = list(distance_to_sea_dict.values())
zone_area_norm = zone_area/zone_area.max() * 15

xticks = list(zonal_average_dict.keys())

plt.figure()
plt.gca().invert_xaxis()

if var_name == 'RVT':
    barb_height = 0.015
    yaxislabel = 'specific humidity [kg/kg]'
    colorvar = 'b'
if var_name == 'THT':
    mean_var = np.array(mean_var) - 273.15
    barb_height = 45
    yaxislabel = 'potential temperature [°C]'
    colorvar = 'r'
if var_name == 'TKET':
    barb_height = 2
    yaxislabel = 'TKE [m2/s2]'
    colorvar = 'y'
else:
    barb_height = np.max(mean_var) + np.max(mean_var)/5
    yaxislabel = var_name

plt.bar(distance_to_sea, mean_var, zone_area_norm,
        color=colorvar)
plt.barbs(distance_to_sea, [barb_height]*6, 
          mean_ut, mean_vt,
          pivot='middle', length=barb_length,     #length of barbs
          sizes={'emptybarb':0.01},
          barb_increments=gv.barb_size_increments[barb_size_option]
          )
plt.annotate(gv.barb_size_description[barb_size_option],
             xy=(np.max(distance_to_sea), barb_height*1.1),
             xycoords='data'
             )


# FIGURE OPTIONS
if len(varNd.shape) == 2:
    plot_title = f'{wanted_date} - {var_name} for simu {model}'
elif len(varNd.shape) == 3:
    level_agl = var2d.level.round()
    plot_title = f'{wanted_date} - {var_name} for simu {model} at {level_agl}m'
plt.title(plot_title)

plt.ylabel(yaxislabel)
plt.xlabel('zones')
plt.xticks(distance_to_sea, xticks, rotation=20)
plt.ylim([0, barb_height*1.15])

if save_plot:
    tools.save_figure(plot_title, save_folder)
