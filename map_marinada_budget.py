#!/usr/bin/env python3
"""
@author: tylunel
Creation : 07/01/2021

Script for mapping winds and aggregated budgets per zone
"""

import matplotlib.pyplot as plt
import xarray as xr
import tools
#import shapefile
import pandas as pd
import global_variables as gv
import metpy.calc as mcalc
from metpy.units import units
from shapely.geometry import Polygon
import numpy as np
import matplotlib as mpl

###############################################
model = 'irrswi1_d1'

domain_nb = int(model[-1])

wanted_date = '20210716-1300'

color_map = 'YlOrBr'    # BuPu, coolwarm, viridis, RdYlGn, jet,... (add _r to reverse)

var_name = 'ZS'   #LAI_ISBA, ZO_ISBA, PATCHP7, ALBNIR_S, MSLP, TG1_ISBA, RAINF_ISBA, CLDFR
vmin = 0
vmax = 1500

# level, only useful if var 3D
#ilevel = 24  #0 is Halo, 1:2m, 2:6.12m, 3:10.49m, 10:49.3m, 20:141m, 30:304m, 40:600m, 50:1126m, 60:2070m, 66:2930m
ilevel_low = 10
ilevel_high = 30


zoom_on = 'marinada'  #None for no zoom, 'liaise' or 'urgell'

add_winds = True
barb_size_option = 'standard'  # 'weak_winds' or 'standard'
arrow_width = 0.003  # 0.004 default

# for BUDGET part
budget_type = 'UV'

var_name_bu_list_dict = {  # includes only physical and most significant terms 
        'TK': ['TOT', 'ADV', 'DISS', 'TR', 'DP', 'TP',],
        'TH': ['TOT', 'ADV', 'VTURB', 'MAFL','RAD', 'DISSH',],
        'RV': ['TOT', 'ADV', 'VTURB', 'MAFL',],
        'VV': ['TOT', 'ADV', 'COR', 'VTURB', 'MAFL', 'PRES',],
        'UU': ['TOT', 'ADV', 'COR', 'VTURB', 'MAFL', 'PRES',],
#        'UV': ['TOT_UU', 'COR_UU', 'VTURB_UU', 'MAFL_UU', 'PRES_UU', 'ADV_UU',
#               'TOT_VV', 'COR_VV', 'VTURB_VV', 'MAFL_VV', 'PRES_VV', 'ADV_VV',],
        'UV': ['TOT', 'ADV', 'COR', 'VTURB', 'MAFL', 'PRES',],
        'WW': ['TOT', 'ADV', 'VTURB', 'GRAV', 'PRES',],
        }
var_name_bu_list = var_name_bu_list_dict[budget_type]

save_plot = True
#save_folder = './figures/scalar_maps/pgd/'
save_folder = f'./figures/zonal_maps_budget/{model}/{ilevel_low}-{ilevel_high}/{budget_type}/'

##############################################

colordict_bu = {'ADV': 'm',
                'TOT': 'grey',
                #TK
                'TP': 'r', 
                'DP': 'b', 
                'DISS': 'g', 
                'TR': 'y',
                #TH
                'RAD': 'r', 
                'DISSH': 'c',
                # UU, VV
                'COR': 'g',
                'VTURB': 'y',
                'MAFL': 'b',
                'PRES': 'r',
                # WW
                'GRAV': 'b',
                }

prop = gv.zoom_domain_prop[zoom_on]
skip_barbs = prop['skip_barbs']*2
barb_length = prop['barb_length']
lat_range = prop['lat_range']
lon_range = prop['lon_range']
figsize = prop['figsize']

filename1 = tools.get_simu_filepath(model, wanted_date,
                                   global_simu_folder=gv.global_simu_folder)
ds1 = xr.open_dataset(filename1)

#prev_date = (pd.Timestamp(wanted_date)- pd.Timedelta(1, 'h')).strftime('%Y%m%d-%H%M')
#filename0 = tools.get_simu_filename(model, prev_date)
#ds0 = xr.open_dataset(filename0)

day = pd.Timestamp(wanted_date).day
hour = pd.Timestamp(wanted_date).hour

filename_bu = gv.global_simu_folder + gv.simu_folders[model] + f'LIAIS.1.SEG{day}.000.nc'

if budget_type in ['PROJ', 'UV']:
    ds_bu = tools.compound_budget_file(filename_bu).isel(time_budget=hour)
    ds_bu['TOT_UU'] = (ds_bu['ENDF_UU'] - ds_bu['INIF_UU'])/3600
    ds_bu['TOT_VV'] = (ds_bu['ENDF_VV'] - ds_bu['INIF_VV'])/3600
else:
    ds_bu = tools.open_budget_file(filename_bu, budget_type).isel(time_budget=hour)
    ds_bu['TOT'] = (ds_bu['ENDF'] - ds_bu['INIF'])/3600


#%% DIAG CALCULATION
ds1 = tools.center_uvw(ds1)
#ds0 = tools.center_uvw(ds0)

ds1['WS'], ds1['WD'] = tools.calc_ws_wd(ds1['UT'], ds1['VT'])
#ds0['WS'], ds0['WD'] = tools.calc_ws_wd(ds0['UT'], ds0['VT'])


#%% DATA SELECTION and ZOOM

#varNd = ds_filt[var_name]
varNd = ds1['ZS']

#remove single dimensions
varNd = varNd.squeeze()

if len(varNd.shape) == 2:
    var2d = varNd
elif len(varNd.shape) == 3:
    var2d = varNd[ilevel_low:ilevel_high,:,:]
    
# remove 999 values, and replace by nan
var2d = var2d.where(~(var2d == 999))
# filter the outliers
#var2d = var2d.where(var2d <= vmax)


#%% PLOT OF VAR_NAME
#fig1 = plt.figure(figsize=figsize)
fig1, ax = plt.subplots(figsize=figsize)

pcm = ax.contourf(var2d.longitude, var2d.latitude, var2d,
             levels=20,
#ax.pcolormesh(var2d.longitude, var2d.latitude, var2d,
#               cbar_kwargs={"orientation": "horizontal", "shrink": 0.7}
               cmap=color_map,
               alpha=0.5,
#               levels=np.linspace(vmin, vmax, (vmax-vmin)*4+1), # fixed colorbar
#               extend = 'both',  #highlights the min and max in edges values
               vmin=vmin, vmax=vmax,
#               levels=20
               )

cbar = fig1.colorbar(pcm, ax=ax)
try:
    cbar.set_label(var2d.long_name)
except AttributeError:
    cbar.set_label(var_name)

#%% WIND BARBS

barb_size_increments = gv.barb_size_increments
barb_size_description = gv.barb_size_description

if add_winds:
    X = ds1.longitude
    Y = ds1.latitude
    U = ds1.UT.squeeze()[ilevel_low:ilevel_high, :, :]
    V = ds1.VT.squeeze()[ilevel_low:ilevel_high, :, :]
    
    U = U.mean(dim='level')
    V = V.mean(dim='level')
    
    ax.barbs(X[::skip_barbs, ::skip_barbs], Y[::skip_barbs, ::skip_barbs], 
              U[::skip_barbs, ::skip_barbs], V[::skip_barbs, ::skip_barbs],
              pivot='middle',
              length=barb_length,     #length of barbs
              sizes={
    #                 'spacing':1, 'height':1, 'width':1,
                     'emptybarb':0.01},
              barb_increments=barb_size_increments[barb_size_option],
              alpha=0.2,
              )
    ax.annotate(barb_size_description[barb_size_option],
                xy=(0.01, 0.97),
                xycoords='axes fraction',  # for putting it inside of figure     
#                xy=(0.1, 0.05),
#                xycoords='subfigure fraction',  # for putting it out of figure
                fontsize=9,
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

# --- Irrigation borders
    
##from scipy.ndimage.filters import gaussian_filter
##sigma = 0.1     #default is 0.1
##irr_covers = gaussian_filter(pgd.COVER369.data, sigma)
#irr_covers = pgd.COVER369.data
#ax.contour(pgd.longitude.data, 
#            pgd.latitude.data, 
#            irr_covers,
#            levels=0,   #+1 -> number of contour to plot 
#            linestyles='solid',
#            linewidths=1.5,
#            colors='g'
##            colors=['None'],
##            hatches='-'
#            )

# --- Sea borders
sea_covers = pgd.COVER001.data
ax.contour(pgd.longitude.data, 
            pgd.latitude.data, 
            sea_covers,
            levels=0,   #+1 -> number of contour to plot 
            linestyles='solid',
            linewidths=1.,
            colors='k'
    #        colors=['None'],
    #        hatches='-'
            )


#%% CONDITIONS OF MARINADA
#var_list=['RVT', 'WS', 'WD', 'ZS']
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

#%% BUDGETS PER AREAS
#import matplotlib as mpl
#norm_cm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
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
    
    data_in = ds_bu.isel(level=np.arange(ilevel_low, ilevel_high))
    
    # Classify points within the polygon
    lon_list = polygon.exterior.xy[0]
    lat_list = polygon.exterior.xy[1]
    data_in_red = tools.subset_ds(data_in, 
                    lat_range=[np.min(lat_list), np.max(lat_list)], 
                    lon_range=[np.min(lon_list), np.max(lon_list)],
                    nb_indices_exterior=2)
    classified_points = tools.get_points_in_polygon(data_in_red, polygon)
    
    # concatenate data
    extracted_ds = xr.concat(classified_points, 'ind')
    # keep layer of interest
#    extracted_da = extracted_ds['WS'][:, ilevel]
    
    # filter:
#    filtered_da = extracted_da.where(extracted_da > 2)
#    filtered_ds = extracted_layer.where(90 < extracted_layer['WD']).where(extracted_layer['WD'] < 200)
    filtered_ds = extracted_ds

    for it_var, var_name_bu in enumerate(var_name_bu_list):
    
        # PLOT arrows
        
        # OPTIONS2: mean
        if budget_type == 'RV':
            coef_visu = 100000
            scale_val = 0.0000002
            unit = 'kg.kg-1.s-1'
        elif budget_type == 'TH':
            coef_visu = 100
            scale_val = 0.001
            unit = 'K.s-1'
        elif budget_type == 'TK':
            coef_visu = 10
            scale_val = 0.01
            unit = 'm2.s-3'
        elif budget_type == 'WW':
            coef_visu = 1
            scale_val = 0.1
            unit = 'm.s-2'
        elif budget_type in ['UU', 'VV']:
            coef_visu = 25
            scale_val = 0.005
            unit = 'm.s-2'
        elif budget_type in ['UV', 'PROJ']:
            coef_visu = 25
            scale_val = 0.005
            unit = 'm.s$^{-2}$'
        
        if budget_type in ['UV', 'PROJ']:
            horiz_compo = float(filtered_ds[f'{var_name_bu}_UU'].mean()) * coef_visu
            verti_compo = float(filtered_ds[f'{var_name_bu}_VV'].mean()) * coef_visu
        else:
            horiz_compo = 0
            verti_compo = float(filtered_ds[var_name_bu].mean()) * coef_visu
#            verti_compo=abs(float(layer_for_fig.mean()))

        lon = polygon.centroid.xy[0][0]
        lat = polygon.centroid.xy[1][0]

        lon_offset = {  # offset on longitude axis to avoid overlapping of arrows
                'irrig': -0.02, 'dry': -0.04, 'slope_west':-0.05, 
                'conca_barbera':-0.04, 'alt_camp':-0.04, 'coast':-0., 'sea':0.06}
        
        if budget_type in ['UV', 'PROJ']:
            lon_arrow = lon + lon_offset[area] + 0.01*np.cos(np.arctan2(verti_compo, horiz_compo))
            lat_arrow = lat + verti_compo*0.1
            if var_name_bu == 'TOT':  # place the total evolution on the side
                lon_arrow += 0.05
                lat_arrow += 0.02
        else:
            lon_arrow = lon + lon_offset[area] + it_var*0.012 + 0.01*np.sin(np.arctan2(verti_compo, horiz_compo))
            lat_arrow = lat + verti_compo*0.1
            
        ax.arrow(lon_arrow, lat_arrow,                    # arrow location
                 horiz_compo, verti_compo,          # arrow size
                 width=arrow_width, color=colordict_bu[var_name_bu],  # esthetics
                 )
            
        # add legend in the bottom left corner:
        if it_area == 0:
            lon_bottom_left = lon_range[0] + 0.07
            lat_bottom_left = lat_range[0] + 0.07
            lon_arrow = lon_bottom_left + it_var*0.035
            lon_legend = lon_bottom_left + it_var*0.035 - 0.025 
            
            scale_val_arrow = scale_val*coef_visu
            
            ax.arrow(lon_arrow, lat_bottom_left,    # arrow location
                     0, scale_val_arrow,      # arrow size
                     width=arrow_width, color=colordict_bu[var_name_bu], # esthetics
                     )
            ax.text(lon_legend, lat_bottom_left, var_name_bu, 
                    rotation=90)
            
            if it_var == 0:  # for first value plotted, add a title and scale
                # add title of legend
                ax.text(lon_bottom_left, lat_bottom_left-0.04, 
                        'Legend', fontweight='bold',
                        rotation=0)
                
                # add scale of legend:
                #plot arrow equivalent to y axis
                ax.arrow(lon_bottom_left - 0.035, lat_bottom_left, 
                         0, scale_val_arrow,
                         width=arrow_width, color='k',
                         )
                ax.text(lon_bottom_left - 0.065, lat_bottom_left, 
                        f'scale: {scale_val} {unit}', 
                        rotation=90)
    
    ax.plot(*polygon.exterior.xy)

    
#%% POINTS SITES

points = [
#        'cendrosa',
##        'ponts',
#          'elsplans', 
##          'irta-corn',
#          'coll_lilla',
##          'lleida', 
##          'zaragoza',
##          'puig formigosa', 
##          'tossal_baltasana', 
#          'tossal_gros', 
##          'tossal_torretes',
#          'torredembarra',
          ]

sites = {key:gv.whole[key] for key in points}

for site in sites:
    ax.scatter(sites[site]['lon'],
                sites[site]['lat'],
                color='r',
                s=12        #size of markers
                )
#    ax.text(sites[site]['lon']+0.01,
#             sites[site]['lat']+0.01, 
#             site.capitalize(), 
#             fontsize=14)


#%% FIGURE OPTIONS and ZOOM

level_low = int(float(ds_bu.level[ilevel_low]))
level_high = int(float(ds_bu.level[ilevel_high]))
plot_title = f'{wanted_date} - {budget_type} for simu {model} between {level_low}-{level_high}m agl'

ax.set_title(plot_title)
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')

if zoom_on is None:
    ax.set_ylim([var2d.latitude.min(), var2d.latitude.max()])
    ax.set_xlim([var2d.longitude.min(), var2d.longitude.max()])
else:
    ax.set_ylim(lat_range)
    ax.set_xlim(lon_range)

if save_plot:
    tools.save_figure(plot_title, save_folder)
