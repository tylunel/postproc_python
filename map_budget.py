#!/usr/bin/env python3
"""
@author: tylunel
Creation : 07/01/2021

Script for plotting maps of budget from MNH 000 files
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import tools
import shapefile
import pandas as pd
import global_variables as gv

###############################################
model = 'irr_d1'

domain_nb = int(model[-1])

wanted_date = '20210717-2300'

budget_type = 'WW'
nb_var = 5

var_name_bu_list_dict = {  # includes only physical and most significant terms 
        'TK': ['DISS', 'TR', 'ADV', 'DP', 'TP', ],
        'TH': ['VTURB', 'MAFL', 'ADV', 'RAD', 'DISSH'],
        'RV': ['VTURB', 'MAFL', 'ADV',],
        'VV': ['COR', 'VTURB', 'MAFL', 'PRES', 'ADV'],
        'UU': ['COR', 'VTURB', 'MAFL', 'PRES', 'ADV'],
        'WW': ['COR', 'VTURB', 'MAFL', 'PRES', 'ADV'],
        }

var_name_bu_list = var_name_bu_list_dict[budget_type]
var_name = var_name_bu_list[nb_var]

#Common to all budgets:
#INIF: initial value
#ENDF: final value
#AVEF: average value
#ASSE time filter (Asselin)
#ADV total advection
#FRC forcing
#DIF numerical diffusion
#REL relaxation

# TK:
#DRAG drag force (DRAGB and DRAGEOL for building and wind turbines)
#DP dynamic production
#TP thermal production
#DISS dissipation of TKE
#TR turbulent transport

# UU or VV:
#COR coriolis
#VTURB vertical turbulent diffusion (HTURB available if turb 3D)
#MAFL mass flux
#PRES pressure gradient force



if budget_type == 'RV':
    coef_visu = 100000
    scale_val = 0.0000005
    unit = 'kg.kg-1.s-1'
elif budget_type == 'TH':
    coef_visu = 100
    scale_val = 0.002
    unit = 'K.s-1'
elif budget_type == 'TK':
    coef_visu = 10
    scale_val = 0.01
    unit = 'm2.s-3'
elif budget_type in ['UU', 'VV', 'WW']:
    coef_visu = 20
    scale_val = 0.005
    unit = 'm.s-2'
    
vmax = scale_val * 2
color_map = 'coolwarm'    

if var_name == 'PGF':
    vmin = 0
    color_map = 'BuPu'
    budget_type_list = ['UU', 'VV']
else:
    vmin = -vmax
    color_map = 'coolwarm'    # BuPu, coolwarm, viridis, RdYlGn, jet,... (add _r to reverse)
    budget_type_list = [budget_type,]
#else:
#    vmin = None
#    color_map = 'jet'    # BuPu, coolwarm, viridis, RdYlGn, jet,... (add _r to reverse)
    
# level, only useful if var 3D
ilevel = 10  #0 is Halo, 1:2m, 2:6.12m, 3:10.49m, 10:49.3m, 20:141m, 30:304m, 40:600m, 50:1126m, 60:2070m, 66:2930m

zoom_on = 'marinada'  #None for no zoom, 'liaise' or 'urgell'

save_plot = True
save_folder = f'./figures/budget_maps/{model}/{budget_type}/{var_name}/{ilevel}/'

add_barbs = 'wind'   # 'pgf' or 'wind'

barb_size_option = 'standard'  # 'weak_winds' or 'standard'

##############################################

prop = gv.zoom_domain_prop[zoom_on]
skip_barbs = prop['skip_barbs']*2
barb_length = prop['barb_length']
lat_range = prop['lat_range']
lon_range = prop['lon_range']
figsize = prop['figsize']


day = pd.Timestamp(wanted_date).day
hour = pd.Timestamp(wanted_date).hour

filename_bu = gv.global_simu_folder + gv.simu_folders[model] + f'LIAIS.1.SEG{day}.000.nc'

ds_dict = {}
for budget_type in budget_type_list:
    ds_dict[budget_type] = tools.open_budget_file(filename_bu, budget_type)


if var_name == 'PGF':
    var_name_effect = 'PRES'

    pgf = {}
    pgf['UU'] = ds_dict['UU'][var_name_effect].swap_dims(
            {'cart_ni_u': 'ni_u', 'cart_nj_u': 'nj_u'})
    pgf['VV'] = ds_dict['VV'][var_name_effect].swap_dims(
            {'cart_ni_v': 'ni_v', 'cart_nj_v': 'nj_v'})
    
    pgf['VV'] = pgf['VV'].interp(ni_v=pgf['UU'].ni_u.values, nj_v=pgf['UU'].nj_u.values).rename(
            {'ni_v': 'ni_u', 'nj_v': 'nj_u'})
    
    pgf['PGF'], pgf['DIR'] = tools.calc_ws_wd(pgf['UU'], pgf['VV'])

#%% DATA SELECTION and ZOOM
if var_name == 'PGF':
    ds = pgf
    budget_type = 'WW'
    latitude_X = 'latitude_u'
    longitude_X = 'longitude_u'
else: 
    budget_type = 'WW'
    ds = ds_dict[budget_type]
    latitude_X = [key for key in list(ds.coords) if 'latitude' in key][0]
    longitude_X = [key for key in list(ds.coords) if 'longitude' in key][0]
    
varNd = ds[var_name]
var2d = varNd.isel(time_budget=hour, level=ilevel)

#ds1 = tools.center_uvw(ds1)

# remove 999 values, and replace by nan
var2d = var2d.where(~(var2d == 999))
# filter the outliers
#var2d = var2d.where(var2d <= vmax)


#%% test for getting proportion of actoin by each parameter
#test = ds.isel(cart_ni_v=20, cart_nj_v=40, level=10, time_budget=12).drop(['INIF', 'ENDF', 'AVEF'])
#total_budget = 0
#for key in test:
#    total_budget += abs(float(test[key]))
#test_norm = test/total_budget

#%% PLOT OF VAR_NAME
fig1 = plt.figure(figsize=figsize)

#plt.contourf(var2d.longitude, var2d.latitude, var2d,
#             levels=20,
#               levels=np.linspace(vmin, vmax, (vmax-vmin)*2+1), # fixed colorbar
#               extend = 'both',  #highlights the min and max in edges values
plt.pcolormesh(ds[longitude_X], 
               ds[latitude_X], 
               var2d,
#               cbar_kwargs={"orientation": "horizontal", "shrink": 0.7}
               cmap=color_map,
               vmin=vmin, vmax=vmax,
               )

#%%
#cbar = plt.colorbar(boundaries=[vmin, vmax])
cbar = plt.colorbar()

try:
    cbar.set_label(var2d.long_name)
except AttributeError:
    cbar.set_label(var_name)
#cbar.set_clim(vmin, vmax)

# --- WIND BARBS

barb_size_increments = gv.barb_size_increments
barb_size_description = gv.barb_size_description

if add_barbs == 'pgf':
    X = pgf['UU'].longitude
    Y = pgf['UU'].latitude
    U = pgf['UU'].isel(time_budget=hour, level=ilevel)
    V = pgf['VV'].isel(time_budget=hour, level=ilevel)
    
    plt.barbs(X[::skip_barbs, ::skip_barbs], Y[::skip_barbs, ::skip_barbs], 
              U[::skip_barbs, ::skip_barbs], V[::skip_barbs, ::skip_barbs],
              pivot='middle',
              length=barb_length,     #length of barbs
              sizes={'emptybarb':0.01},
              barb_increments=barb_size_increments[barb_size_option]
              )
    plt.annotate(barb_size_description[barb_size_option],
                 xy=(0.1, 0.05),
                 xycoords='subfigure fraction'
                 )
elif add_barbs == 'wind':
    filename1 = tools.get_simu_filename(model, wanted_date,
                                   global_simu_folder=gv.global_simu_folder)
    ds1 = xr.open_dataset(filename1)
    
    X = ds1.longitude
    Y = ds1.latitude
    U = ds1.UT.squeeze()[ilevel, :,:]
    V = ds1.VT.squeeze()[ilevel, :,:]
    
    plt.barbs(X[::skip_barbs, ::skip_barbs], Y[::skip_barbs, ::skip_barbs], 
              U[::skip_barbs, ::skip_barbs], V[::skip_barbs, ::skip_barbs],
              pivot='middle',
              length=barb_length,     #length of barbs
              sizes={
    #                 'spacing':1, 'height':1, 'width':1,
                     'emptybarb':0.01},
              barb_increments=barb_size_increments[barb_size_option],
              alpha=0.2,
              )
    plt.annotate(barb_size_description[barb_size_option],
                 xy=(0.1, 0.05),
                 xycoords='subfigure fraction'
                 )

# --- IRRIGATED, SEA and COUNTRIES BORDERS

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

# --- POINTS SITES

points = [
        'cendrosa',
          'elsplans',
          'torredembarra',
#          'lleida',
        'coll_lilla',
#          'tossal_baltasana', 
          'tossal_gros', 
#          'tossal_torretes',
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


# --- FIGURE OPTIONS and ZOOM
if len(varNd.shape) == 3:
    plot_title = f'{wanted_date} - {var_name}-{budget_type} for simu {model}'
elif len(varNd.shape) == 4:
    plot_title = f'{wanted_date} - {var_name}-{budget_type} for simu {model} at level {ilevel}'

plt.title(plot_title)
plt.xlabel('longitude')
plt.ylabel('latitude')

plt.ylim(lat_range)
plt.xlim(lon_range)

if save_plot:
    tools.save_figure(plot_title, save_folder)
