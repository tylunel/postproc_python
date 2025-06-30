#!/usr/bin/env python3
"""
@author: tylunel
Creation : 07/01/2021

Script for plotting simple colormaps 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import tools
import shapefile
import pandas as pd
import global_variables as gv
# import metpy.calc as mcalc
# from metpy.units import units
# from metpy.plots import StationPlot
from shapely.geometry import Polygon
# import os

###############################################

# model = '1_planier_0105/02_1km'
# model = '1_planier_0105/04_1km_COARE'
# model = '1_planier_0105/05_1km_noEDKF_3DTKE'
# model = '1_planier_0105/06_1km_noEDKF'
# model = '1_planier_0105/07_1km_3DTKE'
# model = '1_planier_0105/022_200m_EDKF'
# model = '1_planier_0105/0221_40m'
# model = '1_planier_0105/0211_40m'
# model = '2_planier_0122/04_1km_COARE'
# model = '2_planier_0122/05_1km_RM17'
# model = '2_planier_0122/0412_40m_bigD3'
model = '2_planier_0122/0411_40m'
# model = '3_planier_0127/04_1km_COARE'
# model = '4_planier_0515/04_1km_COARE'
# model = '5_planier_1020/04_1km_COARE'
# model = '5_planier_1020/041_200m'
# model = '7_planier_1017/04_1km'
# model = '7_planier_1017/041_200m'
output_type = 'backup'

wanted_date = '20230122-0230'

# color_map = 'magma_r'   # BuPu, coolwarm, viridis, RdYlGn, jet,... (add _r to reverse)
# /!\ NOW given in gv.minmax_dict

var_name = 'MEAN_DISS'   #CepsOverLeps, ZO_ISBA, PATCHP7, ALBNIR_S, MSLP, TG1_ISBA, RAINF_ISBA, CLDFR, TSWI_T_ISBA, SWI3_ISBA
background_var_name = 'ZS'   #LAI_ISBA, ZO_ISBA, PATCHP7, ALBNIR_S, MSLP, TG1_ISBA, RAINF_ISBA, CLDFR, TSWI_T_ISBA, SWI3_ISBA

# var_name_long = 'Turbulence Intensity [%]'
var_name_long = var_name

# level, only useful if var 3D
# ilevel = 20  #0 is Halo, 1:2m, 2:6.12m, 3:10.49m, 10:49.3m, 20:141m, 30:304m, 40:600m, 50:1126m, 60:2070m, 66:2930m
# ilevel_wind = ilevel
wanted_alti = 140  # alti agl

zoom_on = 'planier_40m'  #None for no zoom, 'eastern_gol'

add_winds = True
if output_type == 'diag':
    add_winds = False
add_pgf = False
marinada_areas = False
sea_borders = True
france_borders = False
barb_size_option = 'weak_winds'  # 'weak_winds' or 'standard'
sites_to_plot = [
    'planier',
    'marseille',
    # 'carry',
    # 'salin-giraud',
    ]

isoalti_list = [100, 200, 300]
obs_circle_size = 50
cbar_loc = 'right'  # 'left', 'right', 'top', 'bottom'

save_plot = False
#save_folder = './figures/scalar_maps/pgd/'

##############################################

wanted_month = str(pd.Timestamp(wanted_date).month).zfill(2)  # format with 2 figures
wanted_day = str(pd.Timestamp(wanted_date).day).zfill(2)

try:
    vmin = gv.minmax_dict[var_name]['vmin']
    vmax = gv.minmax_dict[var_name]['vmax']
    # vstep = gv.minmax_dict[var_name]['vstep']  # not needed because pcolormesh, not contourf
    color_map = gv.minmax_dict[var_name]['colormap']
    # levels_cmap = np.arange(vmin, vmax, vstep)
except KeyError:  # if no value in minmax_dict for this variable
    vmin = gv.minmax_dict[None]['vmin']
    vmax = gv.minmax_dict[None]['vmax']
    color_map = gv.minmax_dict[None]['colormap']
    
prop = gv.zoom_domain_prop[zoom_on]
skip_barbs = prop['skip_barbs']
barb_length = prop['barb_length']
lat_range = prop['lat_range']
lon_range = prop['lon_range']
figsize = prop['figsize']
# skip_barbs = 10
# figsize = (6,5)
# OR: #locals().update(gv.zoom_domain_prop[zoom_on])

# skip_barbs = 50
# barb_length = 7
alpha = 0.4  # transparency of barbs

# size of font on figure
plt.rcParams.update({'font.size': 11})

filename = tools.get_simu_filepath(f'{model}', wanted_date,
                                   output_type = output_type,
                                   global_simu_folder=gv.global_simu_folder,
                                   verbose=True)

try:
    if '_1km' in model:
        pgd_filename = f'{gv.global_simu_folder}/{model}/PGD_1KM.nc'
    elif '_200m' in model:
            pgd_filename = f'{gv.global_simu_folder}/{model}/PGD_200M.nc'
    elif '_40m' in model:
        pgd_filename = f'{gv.global_simu_folder}/{model}/PGD_40M.nc'
    else:
        raise KeyError
except KeyError:
    pgd_filename = f'{gv.global_simu_folder}/{model}/{gv.pgd_filename_dict[model]}'
    
# load dataset, default datetime okay as pgd vars are all the same along time
ds = xr.open_dataset(filename)

filename_bu = tools.get_simu_filepath(f'{model}', wanted_date,
                                    output_type='diachronic',
                                    global_simu_folder=gv.global_simu_folder)

budget_families = [
    'LES_budgets', 'Budgets', 'Profilers'
    ]

#Options are:
# LES_budgets_types = [
#     'BU_KE', 'BU_THL2', 'BU_WTHL', 'BU_RT2', 'BU_WRT', 'BU_THLR',
#     'Miscellaneous', 'Mean', 'Resolved', 'Radiation', 'Surface',
#     ]
# LES_budgets_averaging = [
#     'Time_averaged', 'Not_time_averaged',
#     ]
LES_budgets_type = ['Mean', 'Resolved', ]
LES_budgets_averaging = 'Time_averaged'
# groupname_les_budget = f"LES_budgets/{LES_budgets_type}/Cartesian/{LES_budgets_averaging}/Not_normalized/cart"
# ds_lesbu = xr.open_dataset(filename_bu, group=groupname_les_budget)
ds_lesbu = tools.open_les_budget_file(filename_bu, 
                                      LES_budgets_type=LES_budgets_type,
                                      LES_budgets_averaging=LES_budgets_averaging)

nles_iinf, nles_isup = 100, 370
nles_jinf, nles_jsup = 10, 300

# Budgets_types = [
#     'UU', 'VV', 'WW', 'TH', 'TK', 'RV',
#     ]
# groupname_budget = f"Budgets/{Budgets_types[0]}/"

# Profilers_types = [
#     'planier',
#     ]
# groupname_profiler = f"Profilers/{Profilers_types[0]}/Vertical_profile"
# ds_profiler = xr.open_dataset(filename_bu, group=groupname_profiler)

#%% DIAG
if var_name in ['MEAN_TOT_PROD', 'CepsOverLeps']:
    ds_lesbu['MEAN_TOT_PROD'] = ds_lesbu['MEAN_DP'] + ds_lesbu['MEAN_TP']
    # ds_lesbu['CepsOverLeps_subgrid'] = ds_lesbu['MEAN_DISS'] / ds_lesbu['RES_KE']**(3/2)
    ds_lesbu['CepsOverLeps'] = ds_lesbu['MEAN_DISS'] / ds_lesbu['RES_KE']**(3/2)


#%% DATA SELECTION and ZOOM

#varNd = ds_diag[var_name]
varNd = ds[background_var_name]
#remove single dimensions
varNd = varNd.squeeze()

if len(varNd.shape) == 2:
    var2d = varNd
elif len(varNd.shape) == 3:
    # var2d = varNd[ilevel,:,:]
    var2d = varNd.interp(level=wanted_alti)

# remove 999 values, and replace by nan
var2d = var2d.where(~(var2d == 999))
# filter the outliers
#var2d = var2d.where(var2d <= vmax)


#%% PLOT OF BACKGROUND VAR_NAME

if figsize is None:
    xaxis_dist = float(ds.ni.max() - ds.ni.min())  # in km
    yaxis_dist = float(ds.nj.max() - ds.nj.min())  # in km
    scale = 1/20000
    figsize = (xaxis_dist*scale*1.2, yaxis_dist*scale)
    
if cbar_loc == 'bottom':
    figsize = (figsize[0]-2, figsize[1]+1)
    cbar_frac = 0.05
else:
    cbar_frac = 0.15  # default value


fig1 = plt.figure(figsize=figsize)

cmap = plt.cm.get_cmap(gv.minmax_dict[background_var_name]['colormap']).copy()

# if var_name == 'ZS':
#     vmin=0.1
#     cmap.set_under('c')  # for plotting the sea in cyan

# norm_cm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)  # default normalization

# --- COLORMAP
plt.pcolormesh(var2d.longitude, var2d.latitude, var2d,
#               cbar_kwargs={"orientation": "horizontal", "shrink": 0.7}
               cmap=cmap,
               # levels=levels_cmap,
               # norm=norm_cm,
#               extend = 'both',  #highlights the min and max in edges values
               # vmin=vmin, vmax=vmax,
               )

#%% PLOT OF LES VAR_NAME

timevalues_lesbu = ds_lesbu.interp(level_les=wanted_alti)
homovalue_lesbu = timevalues_lesbu.sel(time_les_avg=pd.Timestamp(wanted_date), 
                                       method='nearest')
var2d_lesbu = var2d * np.nan
var2d_lesbu[nles_jinf:nles_jsup, nles_iinf:nles_isup] = homovalue_lesbu[var_name]

cmap = plt.cm.get_cmap(color_map).copy()
norm_cm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)  # default normalization

plt.pcolormesh(var2d_lesbu.longitude, var2d_lesbu.latitude, var2d_lesbu,
#               cbar_kwargs={"orientation": "horizontal", "shrink": 0.7}
               cmap=cmap,
               # levels=levels_cmap,
               norm=norm_cm,
#               extend = 'both',  #highlights the min and max in edges values
               # vmin=vmin, vmax=vmax,
               )

# cbar = plt.colorbar(boundaries=[vmin, vmax])
cbar = plt.colorbar(location=cbar_loc,
#                    orientation=cbar_orientation,
                    fraction=cbar_frac
                    )
try:
    cbar.set_label(var_name_long)
except (AttributeError, NameError):
    cbar.set_label(var_name)
#cbar.set_clim(vmin, vmax)

#%% PLOT TOPO FEATURES:

# --- IRRIGATED, SEA and COUNTRIES BORDERS

pgd = xr.open_dataset(pgd_filename)

# --- Sea borders
if sea_borders:
    sea_covers = pgd.COVER001.data
    plt.contour(pgd.longitude.data, 
                pgd.latitude.data, 
                sea_covers,
                levels=0,   #+1 -> number of contour to plot 
                linestyles='solid',
                linewidths=2,
                colors='w'
        #        colors=['None'],
        #        hatches='-'
                )

# --- France borders
if france_borders:
    sf = shapefile.Reader("TM-WORLD-BORDERS/TM_WORLD_BORDERS-0.3.sph")
    shapes=sf.shapes()
    france = shapes[64].points
    france_df = pd.DataFrame(france, columns=['lon', 'lat'])
    # france_temp = france_df[france_df.lat < 43.35]
    # france_subset = france_temp[france_S.lon < 2.95]
    france_subset = france_df
    plt.plot(france_subset.lon, france_subset.lat,
             color='k',
             linewidth=1)


# --- ISOLINES FOR ALTI
# contour line of isoaltitude
try:
    isoalti = ds['ZS']
    plt.contour(isoalti.longitude.data, 
                isoalti.latitude.data, 
                isoalti,
                levels=isoalti_list,   #+1 -> number of contour to plot 
                linestyles=':',
                linewidths=1.,
                colors='k',
                )
except:
    print('NO iso alti lines plotted')


# --- POINTS SITES
sites = {key:gv.whole[key] for key in sites_to_plot}

for site in sites:
    plt.scatter(sites[site]['lon'],
                sites[site]['lat'],
                color='k',
                s=12        #size of markers
                )
    # print site name on fig:
    try:
        sitename = sites[site]['longname']  # 'acronym', 'longname'
    except KeyError:
        sitename = site
        
    plt.text(sites[site]['lon']+0.01,
             sites[site]['lat']+0.01, 
             sitename, 
             fontsize=14)

# --- FIGURE OPTIONS and ZOOM
    
if len(varNd.shape) == 2:
    save_title = '{0} - {1} for simu {2}'.format(
        wanted_date, var_name, model)
    plot_title = '{0} - {1}'.format(
        wanted_date, var_name)
    save_folder = f'./figures/scalar_maps/{model}/{var_name}/{zoom_on}/'
elif len(varNd.shape) == 3:
    save_title = '{0} - {1} for simu {2} at {3}m'.format(
        wanted_date, var_name, model, int(var2d.level))
    plot_title = '{0} - {1} at {2}m'.format(
        wanted_date, var_name, int(var2d.level))
    save_folder = f'./figures/scalar_maps/{model}/{var_name}/{wanted_alti}m/{zoom_on}/'
    
# plot_title = f'{var_name} at {np.round(float(var2d.level))}m'
# plot_title = 'Wind and TI at 140m during a Mistral event'

plt.title(plot_title)
plt.xlabel('longitude', fontsize=12)
plt.ylabel('latitude', fontsize=12)

plt.ylim(lat_range)
plt.xlim(lon_range)

plt.subplots_adjust(left=0.11, right=1, top=0.90, bottom=0.1)

if save_plot:
    tools.save_figure(plot_title, save_folder)

#%% Misc
