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
# model = '2_planier_0122/06_1km_UPDRAFTDIAG'
# model = '2_planier_0122/05_1km_RM17'
# model = '2_planier_0122/041_200m'
# model = '2_planier_0122/0411_40m'
# model = '3_planier_0127/04_1km_COARE'
model = '3_planier_0127/0411_40m'
# model = '4_planier_0515/04_1km_COARE'
# model = '5_planier_1020/04_1km_COARE'
# model = '5_planier_1020/041_200m'
# model = '7_planier_1017/04_1km'
# model = '7_planier_1017/041_200m'
# model = '7_planier_1017/0414_40m_eastD3'

output_type = 'backup'
budget_case = False

wanted_date = '20230127T0300'

# color_map = 'magma_r'   # BuPu, coolwarm, viridis, RdYlGn, jet,... (add _r to reverse)
# /!\ NOW given in gv.minmax_dict

var_name = 'WS'   #LAI_ISBA, ZO_ISBA, PATCHP7, ALBNIR_S, MSLP, TG1_ISBA, RAINF_ISBA, CLDFR, TSWI_T_ISBA, SWI3_ISBA

# var_name_long = 'Turbulence Intensity [%]'
var_name_long = var_name

# level, only useful if var 3D
# ilevel = 20  #0 is Halo, 1:2m, 2:6.12m, 3:10.49m, 10:49.3m, 20:141m, 30:304m, 40:600m, 50:1126m, 60:2070m, 66:2930m
# ilevel_wind = ilevel
wanted_alti = 140  # alti agl

zoom_on = 'any_40m'  #None for no zoom, 'eastern_gol'
# zoom_on = None  #None for no zoom, 'eastern_gol'

logscale = False
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
    # 'marseille',
    # 'carry',
    # 'salin-giraud',
    ]

isoalti_list = [100, 200, 300]
obs_circle_size = 50
cbar_loc = 'right'  # 'left', 'right', 'top', 'bottom'

save_plot = True
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

#%% Budget files handling

if budget_case:
    filename_bu = tools.get_simu_filepath(f'{model}', wanted_date,
                                        output_type='diachronic',
                                        global_simu_folder=gv.global_simu_folder)
    
    # budget_families = [
    #     'LES_budgets', 'Budgets', 'Profilers'
    #     ]
    
    # -- Budget of LES --
    #Options are:
    # LES_budgets_types = [
    #     'BU_KE', 'BU_THL2', 'BU_WTHL', 'BU_RT2', 'BU_WRT', 'BU_THLR',
    #     'Miscellaneous', 'Mean', 'Resolved', 'Radiation', 'Surface',
    #     ]
    # LES_budgets_averaging = [
    #     'Time_averaged', 'Not_time_averaged',
    #     ]
    # LES_budgets_type = 'BU_TKE'
    # LES_budgets_averaging = 'Time_averaged'
    # groupname_les_budget = f"LES_budgets/{LES_budgets_type}/Cartesian/{LES_budgets_averaging}/Not_normalized/cart"
    # ds_lesbu = xr.open_dataset(filename_bu, group=groupname_les_budget)
    
    # -- Budget of param --
    Budgets_types = [
        'UU', 'VV', 'WW', 'TH', 'TK', 'RV',
        ]
    # groupname_budget = f"Budgets/{Budgets_types[0]}/"
    ds_timeseries = tools.open_budget_file(filename_bu, 'TH')
    # ds_timeseries = tools.open_multiple_budget_file(filename_bu, 
    #                                                 Budgets_types)
    
    ds_bu = ds_timeseries.sel(time_budget=pd.Timestamp(wanted_date), 
                           method='nearest')

# # -- Profilers --
# Profilers_types = [
#     'planier',
#     ]
# groupname_profiler = f"Profilers/{Profilers_types[0]}/Vertical_profile"
# ds_profiler = xr.open_dataset(filename_bu, group=groupname_profiler)


#%% DIAGs
try:
    ds_sub = tools.subset_ds(ds, 
                          zoom_on=zoom_on,
                          nb_indices_exterior=5,
                          )
except IndexError:
    ds_sub = ds

if (var_name in ['WS', 'WD', 'RES_TKE', 'TOT_TKE', 'U_STAR']) or add_winds:
    # Centering all vars in mass pt ref
    ds1 = tools.flux_pt_to_mass_pt(
        ds_sub, only_basic_vars=True,
        basic_vars=['UT', 'VT', 'WT', 
                    'UMME', 'VMME', 'WMME',
                    'U2ME', 'V2ME', 'W2ME',
                    'FMU', 'FMV',
                    ],
        )
    # Computations
    ds1['WS'], ds1['WD'] = tools.calc_ws_wd(ds1['UT'], ds1['VT'])
    try:
        ds1['U_STAR'] = tools.calc_u_star_sim(ds1['FMU'], ds1['FMV'])
        ds1['WS'], ds1['WD'] = tools.calc_ws_wd(ds1['UT'], ds1['VT'])
        ds1['RES_TKE'] = 0.5* (ds1['U2ME'] + ds1['V2ME'] + ds1['W2ME'])
        ds1['SBG_TKE'] = ds1['TKEMME']
        ds1['TOT_TKE'] = ds1['SBG_TKE'] + ds1['RES_TKE']
        ds1['RES_TKE_FRAC'] = ds1['RES_TKE']/ds1['TOT_TKE']
        ds1['TOT_TI'] = np.sqrt(1.5*ds1['TOT_TKE'])/ds1['WS']
        if var_name == 'ABL_TOP':
            ds1['ABL_TOP'] = tools.diag_abl_top(ds1)
    except:
        print('--- No diagnostic computed ---')
elif (var_name in ['CepsOverLeps', ]) :
    ds['CepsOverLeps'] = ds['TKE_DISS'] / ds['TKET']**(3/2)
    ds1 = ds
elif (var_name in ['TOT_PROD', ]) :
    ds['TOT_PROD'] = ds['DP'] + ds['TP']
    ds1 = ds
elif var_name in ['TKE_EQ', 'TKE_MF_UP']:
    ds1 = tools.flux_pt_to_mass_pt(
        ds_sub, only_basic_vars=True,
        basic_vars=[
            'UT', 'VT', 'WT', 
            'MF_FRAC_UP', 'MF_W_UP',
            ],
        )
    ds1['TKE_MF_UP'] = \
        ds1['MF_FRAC_UP'] / (1 - ds1['MF_FRAC_UP']) * \
        (ds1['MF_W_UP'] - ds1['WT'])**2
    # ds1['TKE_MF_UP'] = \
    #     ds1['MF_W_UP']**2 * (ds1['MF_FRAC_UP']**2 - ds1['MF_FRAC_UP']) - \
    #     2*ds1['WT']*ds1['MF_W_UP'] * (1 - ds1['MF_FRAC_UP'])**2 + \
    #     ds1['WT']**2 * ((1-ds1['MF_FRAC_UP'])**2 - ds1['MF_FRAC_UP'])
    ds1['TKE_EQ'] = ds1['TKET'] + ds1['TKE_MF_UP']
else:
    ds1 = ds

#ds_diag = tools.diag_lowleveljet_height_5percent(ds1[['WS', 'ZS']])

# subset_ds = ds1[['WS', 'ZS', 'TKET', 'HBLTOP']]

#ds1['DIV'] = mcalc.divergence(ds1['UT'], ds1['VT'])
#
#ds1['DENS'] = mcalc.density(
#    ds1['PRES']*units.hectopascal,
#    ds1['TEMP']*units.celsius, 
#    ds1['RVT']*units.gram/units.gram)
#
#ds1['MSLP3D'] = tools.calc_mslp(ds1, ilevel=None)


#%% DATA SELECTION and ZOOM

#varNd = ds_diag[var_name]
varNd = ds1[var_name]
#remove single dimensions
varNd = varNd.squeeze()

if len(varNd.shape) == 2:
    var2d = varNd
elif len(varNd.shape) == 3:
    # var2d = varNd[ilevel,:,:]
    try:
        var2d = varNd.interp(level=wanted_alti)
    except ValueError:
        var2d = varNd.interp(level_w=wanted_alti)
    
# remove 999 values, and replace by nan
var2d = var2d.where(~(var2d == 999))
# filter the outliers
#var2d = var2d.where(var2d <= vmax)


#%% PLOT OF VAR_NAME

if figsize is None:
    xaxis_dist = float(ds.ni.max() - ds.ni.min())  # in km
    yaxis_dist = float(ds.nj.max() - ds.nj.min())  # in km
    
    scale=10/yaxis_dist # scale so as y dim of figure = 10
    
    figsize = (xaxis_dist*scale*1.2, yaxis_dist*scale)
    
if cbar_loc == 'bottom':
    figsize = (figsize[0]-2, figsize[1]+1)
    cbar_frac = 0.05
else:
    cbar_frac = 0.15  # default value


fig1 = plt.figure(figsize=figsize)

cmap = plt.cm.get_cmap(color_map).copy()

if var_name == 'ZS':
    vmin=0.1
    cmap.set_under('c')  # for plotting the sea in cyan

if logscale:
    norm_cm=mpl.colors.LogNorm(vmin=0.01, vmax=vmax)  # for TKE
else:
    norm_cm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)  # default normalization



# --- COLORMAP
#plt.contourf(var2d.longitude, var2d.latitude, var2d,
#             levels=20,
#               levels=np.linspace(vmin, vmax, (vmax-vmin)*2+1), # fixed colorbar
plt.pcolormesh(var2d.longitude, var2d.latitude, var2d,
#               cbar_kwargs={"orientation": "horizontal", "shrink": 0.7}
               cmap=cmap,
               # levels=levels_cmap,
               norm=norm_cm,
#               extend = 'both',  #highlights the min and max in edges values
               # vmin=vmin, vmax=vmax,
               )

#cbar = plt.colorbar(boundaries=[vmin, vmax])
cbar = plt.colorbar(location=cbar_loc,
#                    orientation=cbar_orientation,
                    fraction=cbar_frac
                    )

try:
    cbar.set_label(var_name_long)
except (AttributeError, NameError):
    cbar.set_label(var_name)

#cbar.set_label('LAI [$m^2 m^{-2}$]')
    
#cbar.set_clim(vmin, vmax)

# --- WIND BARBS

barb_size_increments = gv.barb_size_increments
barb_size_description = gv.barb_size_description

if (add_winds & add_pgf):
    raise ValueError('Only add_winds or add pgf can be true')

if add_winds or add_pgf:
    X = ds1.longitude
    Y = ds1.latitude
    if add_winds:
        # U = ds1['UT'].squeeze()[ilevel_wind, :,:]
        # V = ds1['VT'].squeeze()[ilevel_wind, :,:]
        # U = ds1['UMME'].squeeze()[ilevel_wind, :,:]
        # V = ds1['VMME'].squeeze()[ilevel_wind, :,:]
        U = ds1['UT'].squeeze().interp(level=wanted_alti)
        V = ds1['VT'].squeeze().interp(level=wanted_alti)
    elif add_pgf:
        # U = ds1['PGF_U'].squeeze()[ilevel_wind, :,:]
        # V = ds1['PGF_V'].squeeze()[ilevel_wind, :,:]
        U = ds1['PGF_U'].squeeze().interp(level=wanted_alti)
        V = ds1['PGF_V'].squeeze().interp(level=wanted_alti)
    
    plt.barbs(X[::skip_barbs, ::skip_barbs], Y[::skip_barbs, ::skip_barbs], 
              U[::skip_barbs, ::skip_barbs], V[::skip_barbs, ::skip_barbs],
              pivot='middle',
              length=barb_length,     #length of barbs
              sizes={
    #                 'spacing':1, 
    #                 'height':1,
    #                 'width':1,
                     'emptybarb':0.01},
              barb_increments=barb_size_increments[barb_size_option],
              alpha=alpha,
              )
    # plt.annotate(barb_size_description[barb_size_option],
    #              xy=(0.1, 0.01),
    #              xycoords='subfigure fraction',
    #              )

# --- TOPO FEATURES:

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
    isoalti = ds1['ZS']
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

# --- AREAS FOR MARINADA STUDY
if marinada_areas:
    areas_corners = gv.areas_corners
    polygon_dict = {}
    
    for it_area, area in enumerate(areas_corners):
        print(area)
        corners = areas_corners[area]
        corners_coordinates = []
        for corner in corners:
            corners_coordinates.append(
                (gv.whole[corner]['lon'], gv.whole[corner]['lat']))
    
        polygon = Polygon(corners_coordinates)
        polygon_dict[area] = polygon
        plt.plot(*polygon.exterior.xy)
    #    
    #    # add name in the middle of the shapes:
    #    plt.text(polygon.centroid.xy[0][0]-0.05,
    #             polygon.centroid.xy[1][0]-0.05, 
    #             area, 
    #             fontsize=12,
    #             fontstyle='italic')


# --- FIGURE OPTIONS and ZOOM
    
if len(varNd.shape) == 2:
    save_title = '{0} - {1} for simu {2}'.format(
        wanted_date, var_name, model)
    plot_title = '{0} - {1}'.format(
        wanted_date, var_name)
    save_folder = f'./figures/scalar_maps/{model}/{var_name}/{zoom_on}/'
elif len(varNd.shape) == 3:
    save_title = '{0} - {1} for simu {2} at {3}m'.format(
        wanted_date, var_name, model, wanted_alti)
    plot_title = '{0} - {1} at {2}m'.format(
        wanted_date, var_name, wanted_alti)
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
