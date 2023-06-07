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

###############################################
model = 'irr_d1'

domain_nb = int(model[-1])

wanted_date = '20210729-2300'

color_map = 'YlOrBr'    # BuPu, coolwarm, viridis, RdYlGn, YlOrBr, jet,... (add _r to reverse)

var_name = 'ZS'   #LAI_ISBA, ZO_ISBA, PATCHP7, ALBNIR_S, MSLP, TG1_ISBA, RAINF_ISBA, CLDFR
vmin = 0
vmax = 1200

# level, only useful if var 3D
ilevel = 16  #0 is Halo, 1:2m, 2:6.12m, 3:10.49m, 10:49.3m, 20:141m, 30:304m, 40:600m, 50:1126m, 60:2070m, 66:2930m

zoom_on = 'urgell'  #None for no zoom, 'liaise' or 'urgell'

save_plot = True
#save_folder = './figures/scalar_maps/pgd/'
#save_folder = './figures/scalar_maps/domain{0}/{1}/{2}/'.format(
#        domain_nb, model, var_name)
save_folder = f'./figures/scalar_maps/{model}/{var_name}/{ilevel}/'

add_winds = False
add_pgf = True
barb_size_option = 'pgf_weak'  # 'weak_winds' or 'standard'

##############################################

prop = gv.zoom_domain_prop[zoom_on]
skip_barbs = prop['skip_barbs']
barb_length = prop['barb_length']
lat_range = prop['lat_range']
lon_range = prop['lon_range']
figsize = prop['figsize']
# OR: #locals().update(gv.zoom_domain_prop[zoom_on])

filename = tools.get_simu_filename(model, wanted_date,
                                   global_simu_folder=gv.global_simu_folder)

# load dataset, default datetime okay as pgd vars are all the same along time
ds1 = xr.open_dataset(filename)
#ds1 = xr.open_dataset(
#        gv.global_simu_folder + \
#        '2.01_pgds_irr/PGD_400M_CovCor_v26_ivars.nc')


# DIAGs
ds1 = tools.center_uvw(ds1)

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
fig1 = plt.figure(figsize=figsize)

#plt.contourf(var2d.longitude, var2d.latitude, var2d,
#             levels=20,
#               levels=np.linspace(vmin, vmax, (vmax-vmin)*2+1), # fixed colorbar
plt.pcolormesh(var2d.longitude, var2d.latitude, var2d,
#               cbar_kwargs={"orientation": "horizontal", "shrink": 0.7}
               cmap=color_map,
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

try:
    cbar.set_label(var2d.long_name)
except AttributeError:
    cbar.set_label(var_name)
#cbar.set_clim(vmin, vmax)

#%% WIND BARBS

barb_size_increments = gv.barb_size_increments
barb_size_description = gv.barb_size_description

if (add_winds & add_pgf):
    raise ValueError('Only add_winds or add pgf can be true')

if add_winds or add_pgf:
    X = ds1.longitude
    Y = ds1.latitude
    if add_winds:
        U = ds1['UT'].squeeze()[ilevel, :,:]
        V = ds1['VT'].squeeze()[ilevel, :,:]
    elif add_pgf:
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
            linewidths=1.5,
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
            colors='w'
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

points = [
        'cendrosa',
#        'ponts',
          'elsplans', 
#          'irta-corn',
#          'lleida', 
#          'zaragoza',
#          'puig formigosa', 
#          'tossal_baltasana', 
          'tossal_gros', 
          'coll_lilla', 
#          'tossal_torretes', 
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
#    if site == 'elsplans':
#        plt.text(sites[site]['lon']-0.1,
#                 sites[site]['lat']-0.03, 
#                 site, 
#                 fontsize=9)
#    else:
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
