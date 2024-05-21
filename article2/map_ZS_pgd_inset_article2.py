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
import matplotlib as mpl

###############################################

wanted_date = '20210724-2300'

var_name = 'ZS'   #LAI_ISBA, ZO_ISBA, PATCHP7, ALBNIR_S, MSLP, TG1_ISBA, RAINF_ISBA, CLDFR
vmin, vmax = 0, 1500

color_map = 'YlOrBr'    # BuPu, coolwarm, viridis, RdYlGn, jet,... (add _r to reverse)
                        # YlOrBr for orography
norm_cm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)
#norm_cm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax)  # for TKE
#cmap = plt.cm.get_cmap(color_map)
#cmap.set_over("white")
#cmap.set_under("blue")

# level, only useful if var 3D
ilevel = 1  #0 is Halo, 1:2m, 2:6.12m, 3:10.49m, 10:49.3m, 20:141m, 30:304m, 40:600m, 50:1126m, 60:2070m

zoom_on = None  #None for no zoom, 'liaise' or 'urgell'

save_plot = True
save_folder = './fig/'

add_inset = True

if add_inset:
    plot_title = 'Domains of model with inset'
else:
    plot_title = 'Domains of model'

plt.rcParams.update({'font.size': 11})
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
#    'pgd2': '/cnrm/surface/lunelt/NO_SAVE/nc_out/1.16_std_d2_21-22/PGD_400M2.nc',
#    'pgd1': '/cnrm/surface/lunelt/NO_SAVE/nc_out/1.16_std_d2_21-22/PGD_400M.neste1.nc',
    'pgd1': '/cnrm/surface/lunelt/NO_SAVE/nc_out/2.01_pgds_irr/PGD_2KM.nc',
    'pgd2': '/cnrm/surface/lunelt/NO_SAVE/nc_out/2.01_pgds_irr/PGD_400M.nc'}


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

#%%
#var_diff = var_layer[list(dict_pgd.keys())[0]].data - var_layer[list(dict_pgd.keys())[1]].data
##var_diff = var_layer['T2M_ISBA'] - var_layer['TEMP'] -273.15
#
##%% PLOT OF VAR_NAME
#if domain_nb == 1:
#    fig1 = plt.figure(figsize=(13,7))
#elif domain_nb == 2:
#    fig1 = plt.figure(figsize=(10,7))
#
#plt.contourf(var_diff,
##               cbar_kwargs={"orientation": "horizontal", "shrink": 0.7}
#               cmap=color_map,
#               vmin=vmin, vmax=vmax,
#               levels=20
#               )
#
#cbar = plt.colorbar(boundaries=[vmin, vmax])
#cbar.set_label(var2d.long_name)
#cbar.set_clim(vmin, vmax)


#%%

fig = plt.figure(figsize=(12, 6))
pgd1 = var_layer['pgd1']
pgd2 = var_layer['pgd2']

# to mark the edge of domains, the strategy is to overlap 2 maps of the same size, 
# with the one below with edges colored 
plt.pcolormesh(pgd1.longitude.data, 
               pgd1.latitude.data, 
               pgd1,
               cmap=color_map,
               edgecolor='r',
               linewidths=6,
               )
plt.pcolormesh(pgd1.longitude.data+0.002, 
               pgd1.latitude.data-0.002, 
               pgd1,
               cmap=color_map,
               norm=norm_cm,
               )

#plt.pcolormesh(pgd2.longitude.data, 
#               pgd2.latitude.data, 
#               pgd2,
#               cmap=color_map,
#               edgecolor='r',
#               linewidths=6
#               )
#plt.pcolormesh(pgd2.longitude.data+0.002, 
#               pgd2.latitude.data-0.002, 
#               pgd2,
#               cmap=color_map,
##               edgecolor='k',
#               )

cbar = plt.colorbar(location='left',  # 'left', 'right', 'top', 'bottom'
#                    fraction=0.01
                    )
cbar.set_label('altitude [m]')

plt.xlabel('longitude', 
#           fontsize=12
           )
plt.ylabel('latitude', 
#           fontsize=12
           )

plt.ylim([pgd1.latitude.min()-0.05, pgd1.latitude.max()+0.05])
plt.xlim([pgd1.longitude.min()-0.05, pgd1.longitude.max()+0.05])

#%% INSET ZOOM
### add 'zoom_on' domain

ax = plt.gca()

if add_inset:
    x0 = 1.1
    y0 = 0.1
    width = 0.36
    height = 0.8
    axins = ax.inset_axes(
        [x0, y0, width, height],
        xlim=(gv.zoom_domain_prop['marinada']['lon_range'][0], 
              gv.zoom_domain_prop['marinada']['lon_range'][1]), 
        ylim=(gv.zoom_domain_prop['marinada']['lat_range'][0], 
              gv.zoom_domain_prop['marinada']['lat_range'][1]), 
        xticklabels=[], yticklabels=[])
    #axins.imshow(Z2, extent=extent, origin="lower")
    axins.pcolormesh(pgd1.longitude.data+0.002, 
                     pgd1.latitude.data-0.002, 
                     pgd1,
                     cmap=color_map,
                     norm=norm_cm,
                     )
    ax.indicate_inset_zoom(axins, edgecolor="black")
    # inset axes
    lon_ticks = [0.8, 1.0, 1.2, 1.4]
    axins.set_xticks(lon_ticks)
    axins.set_xticklabels(lon_ticks)
    lat_ticks = [41.2, 41.4, 41.6, 41.8]
    axins.set_yticks(lat_ticks)
    axins.set_yticklabels(lat_ticks)
    
    plt.subplots_adjust(left=0, right=0.72, bottom=0.2)
else:
    domain_lon = [
            gv.zoom_domain_prop['marinada']['lon_range'][1],
            gv.zoom_domain_prop['marinada']['lon_range'][0],
            gv.zoom_domain_prop['marinada']['lon_range'][0],
            gv.zoom_domain_prop['marinada']['lon_range'][1],
            gv.zoom_domain_prop['marinada']['lon_range'][1],]
    domain_lat = [
            gv.zoom_domain_prop['marinada']['lat_range'][1],
            gv.zoom_domain_prop['marinada']['lat_range'][1],
            gv.zoom_domain_prop['marinada']['lat_range'][0],
            gv.zoom_domain_prop['marinada']['lat_range'][0],
            gv.zoom_domain_prop['marinada']['lat_range'][1],]
    ax.plot(domain_lon, domain_lat,
            color='purple',
            linewidth=2)
    plt.subplots_adjust(left=0., right=0.9, bottom=0.1)
    
#%% IRRIGATED, SEA and COUNTRIES BORDERS
domain_nb = 1

if domain_nb == 2:
    pgd = xr.open_dataset(
        '/cnrm/surface/lunelt/NO_SAVE/nc_out/2.01_pgds_irr/' + \
        'PGD_400M_CovCor_v26_ivars.nc')
elif domain_nb == 1:
    pgd = xr.open_dataset(
        '/cnrm/surface/lunelt/NO_SAVE/nc_out/2.01_pgds_irr/' + \
        'PGD_2KM_CovCor_v26_ivars.nc')

if add_inset:
    sbplt_list = [ax, axins]
else:
    sbplt_list = [ax,]
    
for sbplt in sbplt_list:
    #Irrigation borders
    #from scipy.ndimage.filters import gaussian_filter
    #sigma = 0.1     #default is 0.1
    #irr_covers = gaussian_filter(pgd.COVER369.data, sigma)
    irr_covers = pgd.COVER369.data
    sbplt.contour(pgd.longitude.data, 
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
    sbplt.contour(pgd.longitude.data, 
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
    sf = shapefile.Reader("../TM-WORLD-BORDERS/TM_WORLD_BORDERS-0.3.sph")
    shapes=sf.shapes()
    france = shapes[64].points
    france_df = pd.DataFrame(france, columns=['lon', 'lat'])
    france_S = france_df[france_df.lat < 43.35]
    france_SW = france_S[france_S.lon < 2.95]
    sbplt.plot(france_SW.lon, france_SW.lat,
             color='k',
             linewidth=1)




#%% POINTS SITES

### --- FOR MAIN MAP

points = [
#        'cendrosa',
#        'ponts',
#          'elsplans', 
#          'irta-corn',
        'barcelona',
          'lleida', 
          'zaragoza',
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
    ax.scatter(sites[site]['lon'],
                sites[site]['lat'],
                color='r',
                s=10        #size of markers
                )
    # print site name on fig:
    try:
        sitename = sites[site]['longname']
    except KeyError:
        sitename = site

    ax.text(sites[site]['lon']+0.02,
         sites[site]['lat']+0.02, 
         sitename, 
         fontsize=10)
    

### --- FOR INSET
if add_inset:
    points_inset = [
              'cendrosa',
              'elsplans', 
              'lleida',
              'coll_lilla',
              'serra_tallat',
              'torredembarra',
              ]
    sites = {key:gv.whole[key] for key in points_inset}
    
    for site in sites:
        axins.scatter(sites[site]['lon'],
                    sites[site]['lat'],
                    color='r',
                    s=10        #size of markers
                    )
        # print site name on fig:
        try:
            sitename = sites[site]['longname']
        except KeyError:
            sitename = site
        
        if site == 'torredembarra':
            axins.text(sites[site]['lon']-0.45,
                 sites[site]['lat']+0.02, 
                 sitename, 
                 fontsize=10)
        else:
            axins.text(sites[site]['lon']+0.02,
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
#if zoom_on is not None:
#    plt.ylim(lat_range)
#    plt.xlim(lon_range)


if save_plot:
    tools.save_figure(plot_title, save_folder)
