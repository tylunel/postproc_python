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

###############################################
#models = ['irr_d1', 'irr_d2']
folder_res = 'diff_d1.11_d1.16'
domain_nb = 2

wanted_date = '20210724-2300'

color_map = 'seismic'    # jet, seismic, BuPu, coolwarm, viridis, RdYlGn, 

var_name = 'FRAC_NATURE'   #LAI_ISBA, ZO_ISBA, PATCHP7, ALBNIR_S, MSLP, TG1_ISBA, RAINF_ISBA, CLDFR
vmin=-300
vmax=300

# level, only useful if var 3D
ilevel = 1  #0 is Halo, 1:2m, 2:6.12m, 3:10.49m, 10:49.3m, 20:141m, 30:304m, 40:600m, 50:1126m, 60:2070m

zoom_on = None  #None for no zoom, 'liaise' or 'urgell'

save_plot = False
save_folder = './figures/scalar_maps/{0}/{1}/'.format(folder_res, var_name)

##############################################

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

#%% LOAD DATA

var_layer = {}

dict_pgd = {
    'pgd2': '/cnrm/surface/lunelt/NO_SAVE/nc_out/1.16_std_d2_21-22/PGD_400M2.nc',
    'pgd1': '/cnrm/surface/lunelt/NO_SAVE/nc_out/1.16_std_d2_21-22/PGD_400M.neste1.nc'}


for key in dict_pgd:
#for var_name in ['T2M_ISBA', 'TEMP']:
#    model='irr_d2'
    filename = dict_pgd[key]
    
    # load dataset, default datetime okay as pgd vars are all the same along time
    ds1 = xr.open_dataset(filename)
    
    #%% DATA SELECTION and ZOOM
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

    var_layer[key] = var2d
#    var_layer[var_name] = var2d


var_diff = var_layer[list(dict_pgd.keys())[0]].data - var_layer[list(dict_pgd.keys())[1]].data
#var_diff = var_layer['T2M_ISBA'] - var_layer['TEMP'] -273.15

#%% PLOT OF VAR_NAME
if domain_nb == 1:
    fig1 = plt.figure(figsize=(13,7))
elif domain_nb == 2:
    fig1 = plt.figure(figsize=(10,7))

plt.contourf(var_diff,
#               cbar_kwargs={"orientation": "horizontal", "shrink": 0.7}
               cmap=color_map,
               vmin=vmin, vmax=vmax,
               levels=20
               )

cbar = plt.colorbar(boundaries=[vmin, vmax])
cbar.set_label(var2d.long_name)
#cbar.set_clim(vmin, vmax)

##%% IRRIGATED, SEA and COUNTRIES BORDERS
#
#if domain_nb == 2:
#    pgd = xr.open_dataset(
#        '/cnrm/surface/lunelt/NO_SAVE/nc_out/2.01_pgds_irr/' + \
#        'PGD_400M_CovCor_v26_ivars.nc')
#elif domain_nb == 1:
#    pgd = xr.open_dataset(
#        '/cnrm/surface/lunelt/NO_SAVE/nc_out/2.01_pgds_irr/' + \
#        'PGD_2KM_CovCor_v26_ivars.nc')
#
##Irrigation borders
##from scipy.ndimage.filters import gaussian_filter
##sigma = 0.1     #default is 0.1
##irr_covers = gaussian_filter(pgd.COVER369.data, sigma)
#irr_covers = pgd.COVER369.data
#plt.contour(pgd.longitude.data, 
#            pgd.latitude.data, 
#            irr_covers,
#            levels=0,   #+1 -> number of contour to plot 
#            linestyles='solid',
#            linewidths=1.,
#            colors='g'
##            colors=['None'],
##            hatches='-'
#            )
#
##Sea borders
#sea_covers = pgd.COVER001.data
#plt.contour(pgd.longitude.data, 
#            pgd.latitude.data, 
#            sea_covers,
#            levels=0,   #+1 -> number of contour to plot 
#            linestyles='solid',
#            linewidths=1.,
#            colors='k'
#    #        colors=['None'],
#    #        hatches='-'
#            )
#
##France borders
#sf = shapefile.Reader("TM-WORLD-BORDERS/TM_WORLD_BORDERS-0.3.sph")
#shapes=sf.shapes()
#france = shapes[64].points
#france_df = pd.DataFrame(france, columns=['lon', 'lat'])
#france_S = france_df[france_df.lat < 43.35]
#france_SW = france_S[france_S.lon < 2.95]
#plt.plot(france_SW.lon, france_SW.lat,
#         color='k',
#         linewidth=1)
#
##%% POINTS SITES
#
#points = ['cendrosa', 'elsplans', 'puig formigosa', 'tossal baltasana', 
#          'tossal gros', 'tossal torretes', 'moncayo', 'tres mojones', 
#          'guara', 'caro', 'montserrat', 'joar',]
#sites = {key:gv.whole[key] for key in points}
#
#for site in sites:
#    plt.scatter(sites[site]['lon'],
#                sites[site]['lat'],
#                color='k',
#                s=10        #size of markers
#                )
#    if site == 'elsplans':
#        plt.text(sites[site]['lon']-0.1,
#                 sites[site]['lat']-0.03, 
#                 site, 
#                 fontsize=9)
#    else:
#        plt.text(sites[site]['lon']+0.01,
#                 sites[site]['lat']+0.01, 
#                 site, 
#                 fontsize=9)


#%% FIGURE OPTIONS and ZOOM
#if len(varNd.shape) == 2:
#    plot_title = '{0} - {1} diff between {2} and {3}'.format(
#        wanted_date, var_name, models[0], models[1])
#elif len(varNd.shape) == 3:
#    plot_title = '{0} - {1} diff between {2} and {3} at {4}m'.format(
#        wanted_date, var_name, models[0], models[1], var2d.level.round())

plot_title = 'diff between pgd400m1.11 and pgd400m1.15'

plt.title(plot_title)

if zoom_on is not None:
    plt.ylim(lat_range)
    plt.xlim(lon_range)


if save_plot:
    tools.save_figure(plot_title, save_folder)