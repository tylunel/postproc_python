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
model = 'irr_d1'

domain_nb = int(model[-1])

wanted_date = '20210722-1500'

color_map = 'jet'    # BuPu, coolwarm, viridis, RdYlGn, jet,... (add _r to reverse)

var_name = 'MSLP'   #LAI_ISBA, ZO_ISBA, PATCHP7, ALBNIR_S, MSLP, TG1_ISBA, RAINF_ISBA, CLDFR
vmin = 1005
vmax = 1025

# level, only useful if var 3D
ilevel = 30  #0 is Halo, 1:2m, 2:6.12m, 3:10.49m, 10:49.3m, 20:141m, 30:304m, 40:600m, 50:1126m, 60:2070m, 66:2930m

zoom_on = None  #None for no zoom, 'liaise' or 'urgell'

save_plot = False
save_folder = './figures/scalar_maps/pgd/'
#save_folder = './figures/scalar_maps/domain{0}/{1}/{2}/'.format(
#        domain_nb, model, var_name)

add_winds = True
barb_size_option = 'standard'  # 'weak_winds' or 'standard'
skip_barbs = 8 # 1/skip_barbs will be printed
barb_length = 4.5

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

filename = tools.get_simu_filename(model, wanted_date)

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


#%% PLOT OF VAR_NAME
if domain_nb == 1:
    fig1 = plt.figure(figsize=(13,7))
elif domain_nb == 2:
    fig1 = plt.figure(figsize=(10,7))

plt.contourf(var2d.longitude, var2d.latitude, var2d,
#               cbar_kwargs={"orientation": "horizontal", "shrink": 0.7}
               cmap=color_map,
               vmin=vmin, vmax=vmax,
               levels=20
               )

cbar = plt.colorbar(boundaries=[vmin, vmax])
cbar.set_label(var2d.long_name)
#cbar.set_clim(vmin, vmax)

#%% WIND BARBS

barb_size_increments = {
        'weak_winds': {'half':1.94, 'full':3.88, 'flag':19.4},
        'standard': {'half':5, 'full':10, 'flag':50},
        }
barb_size_description = {
        'weak_winds': "barb increments: half=1m/s=1.94kt, full=2m/s=3.88kt, flag=10m/s=19.4kt",
        'standard': "barb increments: half=5kt=2.57m/s, full=10kt=5.14m/s, flag=50kt=25.7m/s",
        }

if add_winds:
    X = ds1.longitude
    Y = ds1.latitude
    U = ds1.UT[0, ilevel, :,:]
    V = ds1.VT[0, ilevel, :,:]
    
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
                 xy=(0.1, 0.05),
                 xycoords='subfigure fraction'
                 )


#%% IRRIGATED, SEA and COUNTRIES BORDERS

if domain_nb == 2:
    pgd = xr.open_dataset(
        '/cnrm/surface/lunelt/NO_SAVE/nc_out/2.01_pgds_irr/' + \
        'PGD_400M_CovCor_v26_ivars.nc')
elif domain_nb == 1:
    pgd = xr.open_dataset(
        '/cnrm/surface/lunelt/NO_SAVE/nc_out/2.01_pgds_irr/' + \
        'PGD_2KM_CovCor_v26_ivars.nc')

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

points = ['cendrosa', 
#          'elsplans', 'irta-corn',
          'lleida', 'zaragoza']
#          'puig formigosa', 'tossal baltasana', 
#          'tossal gros', 'tossal torretes', 'moncayo', 'tres mojones', 
#          'guara', 'caro', 'montserrat', 'joar',]

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
    plot_title = '{0} - {1} for simu {2} at {3}m'.format(
        wanted_date, var_name, model, var2d.level.round())

plt.title(plot_title)

if zoom_on is None:
    plt.ylim([var2d.latitude.min(), var2d.latitude.max()])
    plt.xlim([var2d.longitude.min(), var2d.longitude.max()])
else:
    plt.ylim(lat_range)
    plt.xlim(lon_range)

if save_plot:
    tools.save_figure(plot_title, save_folder)
