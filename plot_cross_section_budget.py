#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 09:50:51 2022

@author: lunelt
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import xarray as xr
import tools
#import metpy.calc as mcalc
#from metpy.units import units
import global_variables as gv
import matplotlib as mpl
import pandas as pd
from shapely.geometry import Point, LineString


########## Independant parameters ###############

# Simulation to show: 'irr' or 'std'
model = 'irrswi1_d1'

# Datetime
wanted_date = '20210716-2300'

budget_type = 'UV'

var_name_bu_list_dict = {  # includes only physical and most significant terms 
        'TK': ['DISS', 'TR', 'ADV', 'DP', 'TP', ],
        'TH': ['VTURB', 'MAFL', 'ADV', 'RAD', 'DISSH'],
        'RV': ['VTURB', 'MAFL', 'ADV',],
        'VV': ['COR', 'VTURB', 'MAFL', 'PRES', 'ADV'],
        'UU': ['COR', 'VTURB', 'MAFL', 'PRES', 'ADV'],
        'WW': ['VTURB', 'GRAV', 'PRES', 'ADV',],
        'PROJ': ['COR', 'VTURB', 'MAFL', 'PRES', 'ADV'],  #is combination of UU an VV
        'UV': ['COR', 'VTURB', 'MAFL', 'PRES', 'ADV'],  #eq. to PROJ
        }

var_name_bu_list = var_name_bu_list_dict[budget_type]

var_name_bu = 'PRES'
# --- nb_var is used by run_day_multi_var.sh:
#nb_var = 5
#var_name_bu = var_name_bu_list[nb_var]

varname_contourmap = 'THTV'

colormap='coolwarm'

# values color for contourf plot
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
elif budget_type == 'WW':
    coef_visu = 1
    scale_val = 0.1
    unit = 'm.s-2'
elif budget_type in ['UU', 'VV', 'UV', 'PROJ']:
    coef_visu = 20
    scale_val = 0.002
    unit = 'm.s$^{-2}$'
else:
    vmin = None
    vmax = None

vmax = scale_val * 2
vmin = -vmax

norm_cm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)
#norm_cm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax)  # for TKE

# for contours
minmax_dict = {
    'DIV': {'vmin': -0.0015, 'vmax': 0.0015, 'colormap':'coolwarm'},
    'WS': {'vmin': 1, 'vmax': 10, 'colormap':'BuPu'},
    'THT': {'vmin': 306, 'vmax': 314,'colormap':'OrRd'},
    'THTV': {'vmin': 290, 'vmax': 312, 'colormap':'OrRd'},
    'TKET': {'vmin': 0.05, 'vmax': 3, 'colormap':'OrRd'},
    'RVT': {'vmin': 0, 'vmax': 0.02, 'colormap':'OrRd'},
    'WT': {'vmin': -1.5, 'vmax': 1.5, 'colormap':'coolwarm'},
    'MSLP3D': {'vmin': None, 'vmax': None, 'colormap':'coolwarm'},
    None: {'vmin': None, 'vmax': None, 'colormap':'coolwarm'},
    }

vmin_contour = minmax_dict[varname_contourmap]['vmin']
vmax_contour = minmax_dict[varname_contourmap]['vmax']



# Surface variable to show below the section
surf_var = 'LE_ISBA'
surf_var_label = surf_var

# Set type of wind representation: 'verti_proj' or 'horiz'
vector_visu = 'verti_proj'

# altitude ASL or height AGL: 'asl' or 'agl'
alti_type = 'asl'
# maximum level (height AGL) to plot
toplevel = 1500

# where to place the cross section
nb_points_beyond = 0
site_start = 'cendrosa'
site_end = 'torredembarra'

sites_to_project = ['elsplans', 'serra_tallat', 'coll_lilla']

# Arrow/barbs esthetics:
skip_barbs_x = 2
skip_barbs_y = 5    #if 1: 1barb/10m, if 5: 1barb/50m, etc
arrow_size = 1.2  #works for arrow and barbs
barb_size_option = 'weak_winds'  # 'weak_winds' or 'standard'


# Save the figure
figsize = (12,7)
save_plot = True
save_folder = f'./figures/cross_sections/{model}/section_{site_start}_{site_end}/{vector_visu}/{budget_type}_{var_name_bu}/'

###########################################
#domain to consider: 1 or 2
domain_nb = int(model[-1])

barb_size_increments = gv.barb_size_increments
barb_size_description = gv.barb_size_description


end = (gv.whole[site_end]['lat'], gv.whole[site_end]['lon'])
start = (gv.whole[site_start]['lat'], gv.whole[site_start]['lon'])

if gv.whole[site_start]['lon'] > gv.whole[site_end]['lon']:
    raise ValueError("site_start must be west of site_end")


filepath = tools.get_simu_filepath(model, wanted_date, file_suffix='dg',
                                   out_suffix='')
ds = xr.open_dataset(filepath)

day = pd.Timestamp(wanted_date).day
hour = pd.Timestamp(wanted_date).hour

filename_bu = gv.global_simu_folder + gv.simu_folders[model] + f'LIAIS.1.SEG{day}.000.nc'

if budget_type in ['PROJ', 'UV']:
    ds_bu = tools.open_multiple_budget_file(filename_bu).isel(time_budget=hour)
    ds_bu[f'{var_name_bu}_VAL'], ds_bu[f'{var_name_bu}_DIR'] = tools.calc_ws_wd(
            ds_bu[f'{var_name_bu}_UU'], ds_bu[f'{var_name_bu}_VV'])
else:
    ds_bu = tools.open_budget_file(filename_bu, budget_type).isel(time_budget=hour)

# Computation of other diagnostic variable
ds = tools.center_uvw(ds)
ds['WS'], ds['WD'] = tools.calc_ws_wd(ds['UT'], ds['VT'])
ds['THTV'] = ds['THT']*(1 + 0.61*ds['RVT'] - (ds['MRR']+ds['MRC'])/1000)

#try:
#    data_reduced = ds[['UT', 'VT', 'WT', 'ZS',
#                   'TEMP', 'PRES', 'HBLTOP',
#                   'DENS', 'DIV', 'WS', 'WD',
#                   varname_colormap, varname_contourmap, surf_var]]
#except:
data_reduced = ds[['UT', 'VT', 'WT', 'WS', 'ZS', 'THT',
                   'HBLTOP',
                   surf_var, 
                   varname_contourmap
           ]]
data_redsub = tools.subset_ds(data_reduced, 
                              lat_range = [start[0], end[0]], 
                              lon_range = [start[1], end[1]],
                              nb_indices_exterior=nb_points_beyond+2)

data = data_redsub


if budget_type in ['PROJ', 'UV']:
    data_bu = ds_bu[[f'{var_name_bu}_VAL', f'{var_name_bu}_DIR',
                     f'{var_name_bu}_VV', f'{var_name_bu}_UU']]
else:
    data_bu = ds_bu[[var_name_bu,]]
    



#get total maximum height of relief on domain
max_ZS = data['ZS'].max()
if alti_type == 'asl':
    level_range = np.arange(10, toplevel+max_ZS, 10)
else:
    level_range = np.arange(10, toplevel, 10)
    
    
#%% BULK RI and FROUDE
g = 9.81  #m/s2
data['RI_BULK'] = ((g/data['THTV'])*(data['THTV'] - data['THTV'].isel(level=1))*data['level']) / \
    (data['UT']**2 + data['VT']**2)
data['FROUDE'] = 1/np.sqrt(data['RI_BULK'])

site = 'elsplans'
ilat, ilon = tools.indices_of_lat_lon(
        data, gv.sites[site]['lat'], gv.sites[site]['lon'], verbose=True)
elsplans = data.isel(nj=ilat, ni=ilon)  # elsplans
# Fr=1 around ilevel=38 - 530m - for date='20210716-1500' / Fr lower higher
elsplans.isel(level=38)
    

#%% STANDARD DATA
### -- create section line

line = tools.line_coords(data, start, end, 
                         nb_indices_exterior=nb_points_beyond)
ni_range = line['ni_range']
nj_range = line['nj_range']
slope = line['slope']

section = []
abscisse_coords = []
abscisse_sites = {}

if slope == 'vertical':
    angle = np.pi/2
else:
    angle = np.arctan(slope)

### -- compute compoound of UU and VV
data['WPROJ'] = tools.windvec_verti_proj(data['UT'], data['VT'], 
                                        data.level, angle)
#data['WPROJ_OPPOSITE'] = - data['WPROJ']
#
#data = tools.diag_lowleveljet_height(data, 
#                                     wind_var='WPROJ_OPPOSITE', 
#                                     new_height_var='HLOWJET_WPROJ',
#                                     upper_bound=0.9)
#data = tools.diag_lowleveljet_height(data,
#                                     wind_var='WS', 
#                                     new_height_var='HLOWJET_WS',
#                                     upper_bound=0.70)
#data = tools.diag_lowleveljet_height(data,
#                                     wind_var='WS', 
#                                     new_height_var='HLOWJET_MAX',
#                                     upper_bound=1)

### -- interpolate on line
print('section interpolation on {0} points (~1sec/pt)'.format(len(ni_range)))
for i, ni in enumerate(ni_range):
    nj=nj_range[i]
    #interpolation of all variables on ni_range
    profile = data.interp(ni=ni, 
                          nj=nj, 
                          level=level_range).expand_dims({'i_sect':[i]})
    section.append(profile)
    
    #store values of lat-lon for the horiz axis
    lat = np.round(profile.latitude.values, decimals=3)
    lon = np.round(profile.longitude.values, decimals=3)
    latlon = str(lat) + '\n' + str(lon)
    abscisse_coords.append(latlon)
    
    #Store values of i and name of site in dict for horiz axis
    if slope == 'vertical':
        if nj == line['nj_start']:
            abscisse_sites[i] = site_start
        elif nj == line['nj_end']:
            abscisse_sites[i] = site_end
    else:
        if ni == line['ni_start']:
            abscisse_sites[i] = site_start
        elif ni == line['ni_end']:
            abscisse_sites[i] = site_end

#concatenation of all profile in order to create the 2D section dataset
section_ds = xr.concat(section, dim="i_sect")

#%% BUDGET DATA
### -- create section line

line_bu = tools.line_coords(data_bu, start, end, 
                         nb_indices_exterior=nb_points_beyond)
ni_range_bu = line_bu['ni_range']
nj_range_bu = line_bu['nj_range']

### -- compute compoound of UU and VV
if budget_type in ['PROJ', 'UV']:
    data_bu[f'{var_name_bu}_PROJ'] = tools.windvec_verti_proj(
        data_bu[f'{var_name_bu}_UU'], data_bu[f'{var_name_bu}_VV'], 
        data_bu.level, angle)

### -- interpolate on line
section = []
abscisse_coords = []
abscisse_sites = {}
    
print('section interpolation on {0} points (~10 ms/pt) for budget'.format(len(ni_range_bu)))
for i, ni in enumerate(ni_range_bu):
    nj=nj_range_bu[i]
    #interpolation of all variables on ni_range
    profile = data_bu.interp(ni=ni,
                             nj=nj,
                             level=level_range).expand_dims({'i_sect':[i]})
    section.append(profile)
    
    #store values of lat-lon for the horiz axis
    lat = np.round(profile.latitude.values, decimals=3)
    lon = np.round(profile.longitude.values, decimals=3)
    latlon = str(lat) + '\n' + str(lon)
    abscisse_coords.append(latlon)
    
    #Store values of i and name of site in dict for horiz axis
    if slope == 'vertical':
        if nj == line_bu['nj_start']:
            abscisse_sites[i] = site_start
        elif nj == line_bu['nj_end']:
            abscisse_sites[i] = site_end
    else:
        if ni == line_bu['ni_start']:
            abscisse_sites[i] = site_start
        elif ni == line_bu['ni_end']:
            abscisse_sites[i] = site_end

#concatenation of all profile in order to create the 2D section dataset
section_ds_bu = xr.concat(section, dim="i_sect")


#%% PLOT
# create figure
fig, ax = plt.subplots(2, figsize=figsize,
                       gridspec_kw={'height_ratios': [20, 1]})

## --- Subplot of section, i.e. the main plot ----
#get maximum height of relief in cross-section
max_ZS = section_ds['ZS'].max()

# remove top layers of troposphere
section_ds = section_ds.where(section_ds.level<(level_range.max()), drop=True)
section_ds_bu = section_ds_bu.where(section_ds_bu.level<(level_range.max()), drop=True)


## --- Adapt to alti_type ------
#create grid mesh (eq. to X)
X = np.meshgrid(section_ds.i_sect, section_ds.level)[0]
Xmesh = xr.DataArray(X, dims=['level', 'i_sect'])
#create alti mesh (eq. to Y)
if alti_type == 'asl': 
    #compute altitude ASL from height AGL, and transpose (eq. Y)
    alti = section_ds.ZS[:, 0] + section_ds.level
    alti = alti.T
    #for plot
    ylabel = 'altitude ASL [m]'
elif alti_type == 'agl':
    #create grid mesh (eq. Y)
    alti = np.meshgrid(section_ds.i_sect, section_ds.level)[1]
    alti = xr.DataArray(alti, dims=['level', 'i_sect'])
    #for plot
    ylabel = 'height AGL [m]'
    

### 1.1. Color map (pcolor or contourf)
#data1 = section_ds[varname_colormap]
if budget_type in ['PROJ', 'UV']:
    data1 = section_ds_bu[f'{var_name_bu}_PROJ']
else:
    data1 = section_ds_bu[var_name_bu]

cm = ax[0].pcolormesh(Xmesh,
                      alti,
                      data1.T,
                      cmap=colormap,
                      norm=norm_cm,  # for logscale of colormap
#                      vmin=vmin, vmax=vmax
                      )

#cm = ax[0].contourf(Xmesh,
#                    alti,
#                    data1.T, 
#                    cmap=colormap,  # 'OrRd', 'coolwarm'
##                    levels=np.linspace(298, 315, 18),  # to keep always same colorbar limits
##                    levels=np.linspace(vmin, vmax, vmax-vmin+1),  # to keep always 1K per color variation
##                    levels=np.linspace(vmin, vmax, 20),
##                    levels=20,
#                    extend = 'both',  #highlights the min and max in different color
#                    vmin=vmin, vmax=vmax,  # for adaptative colormap
##                    vmin=800, vmax=1000,  # for PRES
#                    )
#manage colorbar
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='2%', pad=0.05)
cbar = fig.colorbar(cm, cax=cax, orientation='vertical')
try:
    cbar.set_label(f'{data1.long_name} [{data1.units}]')
except:
    cbar.set_label(f'{var_name_bu} [{unit}]')

### 1.2. Contour map
if varname_contourmap in ['HBLTOP', 'HLOWJET', 'HLOWJET_WS']:  #1D
    ax[0].plot(section_ds['HBLTOP'] + section_ds['ZS'],
              linestyle='--', color='r')
    ax[0].plot(section_ds['HLOWJET_WS'] + section_ds['ZS'],
              linestyle='-.', color='g')
#    ax[0].plot(section_ds['HLOWJET_MAX'] + section_ds['ZS'],
#              linestyle='-.', color='y')
else:
    data2 = section_ds[varname_contourmap]  # x1000 to get it in g/kg if RVT
    cont = ax[0].contour(Xmesh,
                         alti,
                         data2.T,
                         levels=np.linspace(vmin_contour, vmax_contour, vmax_contour-vmin_contour+1),
                         cmap='copper_r'  #viridis_r, copper_r
                         )
    ax[0].clabel(cont, cont.levels, inline=True, fontsize=10)

### 1.3. Winds or acceleration Vector projected: 
if vector_visu == 'horiz':            # 2.1 winds - flat direction and force
    
    # If you wish to have the arrows representing the acceleration:
#    if budget_type == 'PROJ':  # acceleration data
#        data_u = section_ds_bu['PRES_UU']
#        data_v = section_ds_bu['PRES_VV']
#    else:  # wind data
    data_u = section_ds['UT']
    data_v = section_ds['VT']
        
    ax[0].barbs(
            #Note that X & alti have dimensions reversed
            Xmesh[::skip_barbs_y, ::skip_barbs_x], 
            alti[::skip_barbs_y, ::skip_barbs_x], 
            #Here dimensions are in the proper order
            data_u[::skip_barbs_x, ::skip_barbs_y].T, 
            data_v[::skip_barbs_x, ::skip_barbs_y].T, 
            pivot='middle',
            length=5*arrow_size,     #length of barbs
            sizes={
    #              'spacing':1, 'height':1, 'width':1,
                    'emptybarb':0.01},
            barb_increments=barb_size_increments[barb_size_option] # [kts], 1.94kt = 1m/s
            )
    ax[0].annotate(barb_size_description[barb_size_option],
                   xy=(0.1, 0.9),
                   xycoords='subfigure fraction'
                   )
    
elif vector_visu == 'verti_proj':     # 2.2  winds - verti and projected wind
    
    # If you wish to have the arrows representing the acceleration:
#    if budget_type == 'PROJ':  # acceleration data
#        compo_horiz = section_ds_bu[f'{var_name_bu}_PROJ'] * 1000
#        compo_verti = section_ds_bu[f'{var_name_bu}_PROJ'] * 0
#        unit = ' 10$^{-3}$ m.s$^{-2}$}'
#    else:  # wind data
    compo_horiz = section_ds['WPROJ']  # horizontal component
    compo_verti = section_ds['WT']  # vertical component
    unit = ' m.s$^{-1}$'
    
    Q = ax[0].quiver(
            #Note that X & alti have dimensions reversed
            Xmesh[::skip_barbs_y, ::skip_barbs_x],
            alti[::skip_barbs_y, ::skip_barbs_x],
            #Here dimensions are in the proper order
            compo_horiz[::skip_barbs_x, ::skip_barbs_y].T,
            compo_verti[::skip_barbs_x, ::skip_barbs_y].T,
            pivot='middle',
            scale=150/arrow_size,  # arrows scale, if higher, smaller arrows
            alpha=0.4,
            )
    #add arrow scale in top-right corner
    vector_max = abs(compo_horiz[::skip_barbs_x, ::skip_barbs_y]).max()
    ax[0].quiverkey(Q, 0.8, 0.9, 
                    U=vector_max, 
                    label=str((np.round(vector_max, decimals=1)).data) + unit, 
                    labelpos='E',
                    coordinates='figure')

### Plot aesthetics

### 1. Main plot - cross-section ---
    
# projection  of other sites between sites start and end
for site_inter in sites_to_project:
    coords_site_inter = (gv.whole[site_inter]['lat'], gv.whole[site_inter]['lon'])
    
    point_site_inter = Point(coords_site_inter)
    line_cross_section = LineString([start, end])
    dist = line_cross_section.project(point_site_inter)
    coords_site_inter_proj = list(line_cross_section.interpolate(dist).coords)[0]
    
    fraction_lon_point_inter = (coords_site_inter_proj[1] - start[1]) / (end[1] - start[1])
    # in term of abscisse
    list_abscisses_sites = list(abscisse_sites.keys())
    diff_abscisses = list_abscisses_sites[1] - list_abscisses_sites[0]
    abscisse_inter = fraction_lon_point_inter * diff_abscisses + list_abscisses_sites[0]
    # add to the dict
    abscisse_sites[abscisse_inter] = site_inter 

# x-axis with sites names
ax[0].set_xticks(list(abscisse_sites.keys()))
ax[0].set_xticklabels(list(abscisse_sites.values()), 
                   rotation=0, fontsize=12)
# x-axis with lat-lon values
#ax.set_xticks(data1.i_sect[::10])
#ax.set_xticklabels(abscisse_coords[::10], rotation=0, fontsize=9)

# set y limits (height ASL)
if alti_type == 'asl':
    min_ZS = section_ds['ZS'].min()
    ax[0].set_ylim([min_ZS, max_ZS + toplevel])
ax[0].set_ylabel(ylabel)


### 2. Subplot of surface characteristic ---

#data_soil = section_ds[surf_var][:, :2]  #keep 2 equivalent levels for plot
#p9 = ax[1].pcolor(data_soil.i_sect, 
#                  data_soil.level, 
#                  data_soil.transpose(), 
#                  cmap='YlGn',
#                  vmin=0, vmax=0.4
#                  )
## create colorbar dedicated to the subplot
#divider = make_axes_locatable(ax[1])
#cax = divider.append_axes('right', size='2%', pad=0.05)
#cbar2 = fig.colorbar(p9, cax=cax, orientation='vertical')
##cbar2.set_label(surf_var_label)
#cbar2.set_label('[m³/m³]')

#ax[1].set_xticks(ticks = data_soil.i_sect.values[::9],
#                 labels = (data_soil.i_sect.values * \
#                           line['nij_step']/1000)[::9].round(decimals=1)
#                 )

subplot_type = 'distance'

if subplot_type == 'surface_var':
    labels_arr = np.arange(0,100,10)
    tick_pos = labels_arr/ (line['nij_step']/1000)
    ax[1].set_xticks(ticks = tick_pos,
                     labels = labels_arr
                     )
    ax[1].set_xlabel('distance [km]')
    
    ax[1].set_yticks([])
    #ax[1].set_ylabel(surf_var)
    ax[1].set_ylabel('soil moisture')
    
if subplot_type == 'distance':
    # remove the surface subplot
    fig.delaxes(ax[1])
    
    # get index of torredembarra in abscisse_sites
#    torredembarra_ind = list(abscisse_sites.values()).index('coma_ruga')
    torredembarra_ind = list(abscisse_sites.values()).index('torredembarra')

    # get corresponding abscisse for torredembarra
    torredembarra_xval = list(abscisse_sites.keys())[torredembarra_ind]
    
    def ftest(x):
        return -(x - torredembarra_xval) * (line['nij_step']/1000)
    def ftest_recip(x):
        return -(x/(line['nij_step']/1000) + torredembarra_xval)
    
    # add secondary axis
    secax = ax[0].secondary_xaxis(-0.1, functions=(ftest, ftest_recip))
    secax.set_xlabel('distance to the sea [km]')

### Global options
plot_title = f'{wanted_date}-{model}-{budget_type}-{var_name_bu}-{vector_visu}'
#plot_title = 'Cross section of ABL between irrigated and rainfed areas on July 22 at 12:00 - {0}'.format(
#        model)
#plot_title = 'Cross section on July 22 at 12:00 - {0}'.format(model)
fig.suptitle(plot_title)

if save_plot:
    tools.save_figure(plot_title, save_folder)

    