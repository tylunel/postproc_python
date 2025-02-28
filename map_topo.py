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
# model = '1_planier_0105/02_1km'
# model = '1_planier_0105/021_200m'
model = '1_planier_0105/022_40m'

color_map = 'YlOrBr'    # jet, seismic, BuPu, coolwarm, viridis, RdYlGn, 
                        # BuPu, coolwarm, viridis, RdYlGn, jet,... (add _r to reverse)
                        # YlOrBr for orography

var_name = 'ZS'   #LAI_ISBA, ZO_ISBA, PATCHP7, ALBNIR_S, MSLP, TG1_ISBA, RAINF_ISBA, CLDFR
vmin=0
vmax=800

# level, only useful if var 3D
ilevel = 1  #0 is Halo, 1:2m, 2:6.12m, 3:10.49m, 10:49.3m, 20:141m, 30:304m, 40:600m, 50:1126m, 60:2070m

sea_in_cyan = True
france_borders = False
points_to_plot = [
    'planier',
    'marseille',
    'carry',
    ]

zoom_on = 'planier_40m'
prop = gv.zoom_domain_prop[zoom_on]
lat_range = prop['lat_range']
lon_range = prop['lon_range']
figsize = prop['figsize']  # if None automatic scaling

save_plot = True
save_folder = f'./figures/topo_map/{model}'

plot_title = 'Topography'
##############################################


#%% LOAD DATA
pgd_filename = f'{gv.global_simu_folder}/{model}/{gv.pgd_filename_dict[model]}'
pgd = xr.open_dataset(pgd_filename)
# pgd = xr.open_dataset(
    # '/home/lunelt/Data/mnh_run/' + \
    # 'PGD_1KM.nc')

    
#%% DATA SELECTION and ZOOM
varNd = pgd[var_name]
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

if figsize is None:  # not working well /!\
    xaxis_dist = float(pgd.ni.max() - pgd.ni.min())  # in km
    yaxis_dist = float(pgd.nj.max() - pgd.nj.min())  # in km
    scale = 1/20000
    figsize = (xaxis_dist*scale*1.2, yaxis_dist*scale)


#%%
fig = plt.figure(figsize=figsize)

#%% Sea borders
if sea_in_cyan:
    cmap = plt.cm.get_cmap(color_map).copy()
    if var_name == 'ZS':
        vmin=0.1
        cmap.set_under('c')  # for plotting the sea in cyan
else:                   # plot the contours
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


pgd1 = var2d

plt.pcolormesh(pgd1.longitude.data, 
               pgd1.latitude.data, 
               pgd1,
               cmap=cmap,
               vmin=vmin, vmax=vmax,
               )

cbar = plt.colorbar()
cbar.set_label('altitude [m]')

#plt.title(plot_title)

plt.xlabel('longitude', fontsize=12)
plt.ylabel('latitude', fontsize=12)

# plt.ylim([pgd1.latitude.min(), pgd1.latitude.max()])
# plt.xlim([pgd1.longitude.min(), pgd1.longitude.max()])


#%% TOPO



#%% France borders
if france_borders:
    sf = shapefile.Reader("../TM-WORLD-BORDERS/TM_WORLD_BORDERS-0.3.sph")
    shapes = sf.shapes()
    france = shapes[64].points
    france_df = pd.DataFrame(france, columns=['lon', 'lat'])
    france_S = france_df[france_df.lat < 43.8]
    france_SE = france_S[france_S.lon > 4]
    plt.plot(france_SE.lon, france_SE.lat,
             color='k',
             linewidth=1)

# Rivers
# rivers = shapefile.Reader("/home/lunelt/postproc_python/rivers_europe_37253/rivers_europe_37253.shp")
# # Check here to find ID of river: https://data.apps.fao.org/catalog/dataset/e0243940-e5d9-487c-8102-45180cf1a99f/resource/59557e5b-c852-4974-8576-8d954587b102
# ebro = rivers.shapes()[39555].points
# ebro_df = pd.DataFrame(ebro, columns=['lon', 'lat'])
# plt.plot(ebro_df.lon, ebro_df.lat,
#          color='b',
#          linewidth=1)

# segre = rivers.shapes()[38572].points
# segre_df = pd.DataFrame(segre, columns=['lon', 'lat'])
# plt.plot(segre_df.lon, segre_df.lat,
#          color='b',
#          linewidth=1)

#%% POINTS SITES

sites = {key:gv.whole[key] for key in points_to_plot}

for site in sites:
    plt.scatter(sites[site]['lon'],
                sites[site]['lat'],
                color='r',
                s=10        #size of markers
                )
    # print site name on fig:
    try:
        sitename = sites[site]['longname']
    except KeyError:
        sitename = site
        
    if site == 'lleida':
        plt.text(sites[site]['lon']-0.45,
                 sites[site]['lat']+0.02, 
                 sitename, 
                 fontsize=10)
    else:
        plt.text(sites[site]['lon']+0.02,
                 sites[site]['lat']+0.02, 
                 sitename, 
                 fontsize=10)


#%% FIGURE OPTIONS and ZOOM
#if len(varNd.shape) == 2:
#    plot_title = '{0} - {1} diff between {2} and {3}'.format(
#        wanted_date, var_name, models[0], models[1])
#elif len(varNd.shape) == 3:
#    plot_title = '{0} - {1} diff between {2} and {3} at {4}m'.format(
#        wanted_date, var_name, models[0], models[1], var2d.level.round())

#plot_title = 'diff between pgd400m1.11 and pgd400m1.15'
#
#plt.title(plot_title)
#
if zoom_on is not None:
   plt.ylim(lat_range)
   plt.xlim(lon_range)


if save_plot:
    tools.save_figure(plot_title, save_folder)