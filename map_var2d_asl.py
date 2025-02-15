#!/usr/bin/env python3
"""
@author: tylunel
Creation : 07/01/2021

Script for plotting colormaps of scalar variables at constant height ASL
"""

import matplotlib.pyplot as plt
import xarray as xr
import tools
import shapefile
import pandas as pd
import global_variables as gv
import os

###############################################
model = 'irr_d1'

wanted_date = '20210721-1200'

color_map = 'jet'    # BuPu, coolwarm, viridis, RdYlGn, jet

var_name = 'PRES'   #LAI_ISBA, ZO_ISBA, PATCHP7, ALBNIR_S, MSLP, TG1_ISBA, RAINF_ISBA, CLDFR
vmin=None  # None if unknown
vmax=None

# level or altitude, only useful if var 3D
alti_type = 'asl'

if alti_type == 'agl':
    ilevel = 3  #0 is Halo, 1:2m, 2:6.12m, 3:10.49m, 10:49.3m, 20:141m, 30:304m, 40:600m, 50:1126m, 60:2070m
if alti_type == 'asl':  # needs costly interpolation
    h_alti = 2000  # in m

zoom_on = None  #None for no zoom, 'liaise' or 'urgell'

save_file_asl = True
save_plot = False
save_folder = f'./figures/scalar_maps/{model}/{alti_type}/{var_name}/'

##############################################

prop = gv.zoom_domain_prop[zoom_on]
skip_barbs = prop['skip_barbs']
barb_length = prop['barb_length']
lat_range = prop['lat_range']
lon_range = prop['lon_range']
figsize = prop['figsize']


filename = tools.get_simu_filepath(model, wanted_date)

# load dataset, default datetime okay as pgd vars are all the same along time
ds = xr.open_dataset(filename)
ds = tools.subset_ds(ds, zoom_on=zoom_on)
#ds = ds[['PRES', 'ZS']]

#%% DATA SELECTION and ZOOM

varNd = ds[var_name]
#remove single dimensions
varNd = varNd.squeeze()

if len(varNd.shape) == 2:
    var2d = varNd
elif len(varNd.shape) == 3:
    if alti_type == 'agl':
        var2d = varNd[ilevel, :, :]
        # remove 999 values, and replace by nan
        var2d = var2d.where(~(var2d == 999))
        # filter the outliers
        #var2d = var2d.where(var2d <= vmax)
    elif alti_type == 'asl':
        new_var_name = '{0}_{1}m_asl'.format(var_name, h_alti)
        nc_filename_out = filename + '_' + new_var_name
        
        if os.path.exists(nc_filename_out):    
            var2d = xr.open_dataset(nc_filename_out)[new_var_name]
        else:
            print("Interpolation at altitude (asl) ", h_alti)
            var2darray = tools.interp_iso_asl(h_alti, ds, var_name)
            
            var2d = var2darray.rename(new_var_name).assign_attrs(
                {'long_name': new_var_name, 'comment': new_var_name})

            var2d = var2d.squeeze()
            
            if save_file_asl:
                print("Creation of file: ", nc_filename_out)
                # Save new field as netCDF to load it faster next time
                nc_filename_out = filename + '_' + new_var_name
                var2d.to_netcdf(nc_filename_out)
        
    


#%% PLOT OF VAR_NAME

fig1 = plt.figure(figsize=figsize)

plt.contourf(var2d.longitude, var2d.latitude, var2d,
#               cbar_kwargs={"orientation": "horizontal", "shrink": 0.7}
               cmap=color_map,
               vmin=vmin, vmax=vmax,
               levels=20
               )

cbar = plt.colorbar(boundaries=[vmin, vmax])
cbar.set_label(var2d.long_name)
#cbar.set_clim(vmin, vmax)

#%% IRRIGATED, SEA and COUNTRIES BORDERS

domain_nb = int(model[-1])

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

points = ['cendrosa', 'elsplans', 'zaragoza', 'lleida',
          # 'puig_formigosa', 'tossal_baltasana', 
          # 'tossal_torretes', 'moncayo', 'tres_mojones', 
          # 'guara', 'caro', 'montserrat', 'joar',
          ]
sites = {key:gv.whole[key] for key in points}

for site in sites:
    plt.scatter(sites[site]['lon'],
                sites[site]['lat'],
                color='k',
                s=10        #size of markers
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


#%% FIGURE OPTIONS and ZOOM
if len(varNd.shape) == 2:
    plot_title = '{0} - {1} for simu {2}'.format(
        wanted_date, var_name, model)
elif len(varNd.shape) == 3:
    if alti_type == 'agl':
        plot_title = '{0} - {1} for simu {2} at {3}m AGL'.format(
            wanted_date, var_name, model, var2d.level.round())
    elif alti_type == 'asl':
        plot_title = '{0} - {1} for simu {2} at {3}m ASL'.format(
            wanted_date, var_name, model, h_alti)

plt.title(plot_title)

if zoom_on is None:
    plt.ylim([var2d.latitude.min(), var2d.latitude.max()])
    plt.xlim([var2d.longitude.min(), var2d.longitude.max()])
else:
    plt.ylim(lat_range)
    plt.xlim(lon_range)


if save_plot:
    tools.save_figure(plot_title, save_folder)
