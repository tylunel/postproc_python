#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 09:50:51 2022

@author: lunelt
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import tools
#import metpy.calc as mcalc
#from metpy.units import units
import global_variables as gv
import pandas as pd
from shapely.geometry import Point, LineString


########## Independant parameters ###############

# Simulation to show: 'irr' or 'std'
models = [
          'irrlagrip30_d1',
#          'irrlagrip30_d1_old', 
          'std_d1',
          'irrswi1_d1', 
          ]

# Datetime
wanted_date = '20210716-1200'

budget_type = 'UV'

var_name_bu_list_dict = {  # includes only physical and most significant terms 
        'TK': ['DISS', 'TR', 'ADV', 'DP', 'TP', ],
        'TH': ['VTURB', 'MAFL', 'ADV', 'RAD', 'DISSH'],
        'RV': ['VTURB', 'MAFL', 'ADV',],
        'VV': ['COR', 'VTURB', 'MAFL', 'PRES', 'ADV'],
        'UU': ['COR', 'VTURB', 'MAFL', 'PRES', 'ADV'],
        'WW': ['VTURB', 'GRAV', 'PRES', 'ADV',],
        'PROJ': ['COR', 'VTURB', 'MAFL', 'PRES', 'ADV'],  #is projection of UU an VV in transect
        'UV': ['COR', 'VTURB', 'MAFL', 'PRES', 'ADV'],  #eq. to PROJ
        }

var_name_bu_list = var_name_bu_list_dict[budget_type]

var_to_plot_bu = ['PRES',]
# --- nb_var is used by run_day_multi_var.sh:
#nb_var = 5
#var_to_plot_bu = [var_name_bu_list[nb_var],]

var_to_plot = ['THTV',]
#vmin, vmax = 0, 1000

add_wind_barbs = True

# level (height AGL) to plot, or list of levels (will be averaged in this case)
line_level = [50, 60, 70, 80, 90, 100,]  #[m]
line_level = np.arange(50,300,25)

# where to place the cross section
nb_points_beyond = 4
site_start = 'cendrosa'
site_end = 'torredembarra'

sites_to_project = ['elsplans', 'serra_tallat', 'coll_lilla']

# min and max for budget plots
if budget_type == 'RV':
    scale_val = 0.0000005
    unit = 'kg.kg-1.s-1'
elif budget_type == 'TH':
    scale_val = 0.003
    unit = 'K.s-1'
elif budget_type == 'TK':
    scale_val = 0.01
    unit = 'm2.s-3'
elif budget_type == 'WW':
    scale_val = 0.1
    unit = 'm.s-2'
elif budget_type in ['UU', 'VV', 'UV', 'PROJ']:
    scale_val = 0.006
    unit = 'm.s$^{-2}$'
else:
    vmin_bu = None
    vmax_bu = None

vmax_bu = scale_val * 2
vmin_bu = -vmax_bu

# Save the figure
figsize = (12,7)
save_plot = True
save_folder = f'./figures/linear_cross_sections/section_{site_start}_{site_end}/{budget_type}-{var_to_plot_bu}_{var_to_plot}/'

###########################################


colordict = {'irr_d2': 'g', 
             'std_d2': 'r',
             'irr_d1': 'g', 
             'std_d1': 'r', 
             'irrlagrip30_d1': 'orange',
             'irrlagrip30_d1_old': 'yellow',
             'irrswi1_d1': 'b',
             'irr_d2_old': 'g', 
             'std_d2_old': 'r', 
             'obs': 'k'}

barb_size_increments = gv.barb_size_increments
barb_size_description = gv.barb_size_description


end = (gv.whole[site_end]['lat'], gv.whole[site_end]['lon'])
start = (gv.whole[site_start]['lat'], gv.whole[site_start]['lon'])

if gv.whole[site_start]['lon'] > gv.whole[site_end]['lon']:
    raise ValueError("site_start must be west of site_end")

#%% LOAD DATA (STANDARD and BUDGET)
section_ds_dict = {}
section_ds_bu_dict = {}

for model in models:
    # get day and month of wanted_datetime
    day = pd.Timestamp(wanted_date).day
    hour = pd.Timestamp(wanted_date).hour
    
    # Standard data
    filepath = tools.get_simu_filepath(model, wanted_date, 
                                       file_suffix='',
                                       out_suffix='.OUT',
                                       verbose=True)
    ds = xr.open_dataset(filepath)
    
    # Computation of wind speed diag variable
    ds = tools.center_uvw(ds)
    ds['WS'], ds['WD'] = tools.calc_ws_wd(ds['UT'], ds['VT'])
    ds['THTV'] = ds['THT']*(1 + 0.61*ds['RVT'])
    
    data_reduced = ds[['UT', 'VT', 'WT', 'WS', 'THT',
                       *var_to_plot
                       ]]
    data_redsub = tools.subset_ds(data_reduced, 
                                  lat_range = [start[0], end[0]], 
                                  lon_range = [start[1], end[1]],
                                  nb_indices_exterior=nb_points_beyond+2)
    data = data_redsub


    # Budget data
    filename_bu = gv.global_simu_folder + gv.simu_folders[model] + f'LIAIS.1.SEG{day}.000.nc'
    
    for i, var_name_bu in enumerate(var_to_plot_bu):
        if budget_type in ['PROJ', 'UV']:
            ds_bu = tools.compound_budget_file(filename_bu).isel(time_budget=hour)
            ds_bu[f'{var_name_bu}_VAL'], ds_bu[f'{var_name_bu}_DIR'] = tools.calc_ws_wd(
                    ds_bu[f'{var_name_bu}_UU'], ds_bu[f'{var_name_bu}_VV'])
            if i==0:
                data_bu = ds_bu[[f'{var_name_bu}_VAL', f'{var_name_bu}_DIR',
                                 f'{var_name_bu}_VV', f'{var_name_bu}_UU']]
            else:
                data_bu[[f'{var_name_bu}_VAL', f'{var_name_bu}_DIR',
                         f'{var_name_bu}_VV', f'{var_name_bu}_UU']] = \
                         ds_bu[[f'{var_name_bu}_VAL', f'{var_name_bu}_DIR',
                                f'{var_name_bu}_VV', f'{var_name_bu}_UU']]
        else:
            ds_bu = tools.open_budget_file(filename_bu, budget_type).isel(time_budget=hour)
            if i==0:
                data_bu = ds_bu[[var_name_bu,]]
            else:
                data_bu[[var_name_bu,]] = ds_bu[[var_name_bu,]]

    
    #get total maximum height of relief on domain
    diag_list = ['HLOWJET', 'HLOWJET_07', 'FROUDE',]
    var_in_diag_list = any(elt in diag_list for elt in var_to_plot)
    
    if var_in_diag_list:
        toplevel = 2500
        level_range = np.arange(10, toplevel, 20)
    else:
        level_range = line_level

    
    
#%% STANDARD DATA
    ### -- create section line
    if var_to_plot != []:
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
        
        ### -- interpolate on line
        print('section interpolation on {0} points for standard data (~0.1sec/pt)'.format(len(ni_range)))
        for i, ni in enumerate(ni_range):
            nj=nj_range[i]
            #interpolation of all variables on ni_range
            profile = data.interp(ni=ni, 
                                  nj=nj, 
                                  level=level_range).expand_dims({'i_sect':[i]})
            if type(level_range) in [list, np.ndarray]:
                profile = profile.mean(dim='level')
                
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
        
    
        #%% DIAGs on Standard Data: BULK RI and FROUDE
        if var_in_diag_list:
            g = 9.81  #m/s2
            data['RI_BULK'] = ((g/data['THVREF'])*(data['THVREF'] - data['THVREF'].isel(level=1))*data['level']) / \
                (data['UT']**2 + data['VT']**2)
            data['FROUDE_RI'] = 1/np.sqrt(data['RI_BULK'])
            
            section_ds = tools.diag_lowleveljet_height(section_ds,
                                                 wind_var='WS', 
                                                 new_height_var='HLOWJET_07',
                                                 upper_bound=0.70)
            
            section_at_hjet = section_ds.sel(
                level = section_ds['HLOWJET_07'], method='nearest')
            section_below_hjet = section_ds.where(
                    section_ds.level < section_ds['HLOWJET_07'], drop=True)
            section_ds['FROUDE_HJET_07'] = section_at_hjet['THT']*section_below_hjet['WS'].mean(dim='level')**2 / \
                (g * section_at_hjet['HLOWJET_07'] * (section_at_hjet['THT'] - section_below_hjet['THT'].mean(dim='level')))
                
    
            section_ds = section_ds.isel(level=line_level)
       
        section_ds_dict[model] = section_ds
    
#%% BUDGET DATA
    ### -- create section line
    if var_to_plot_bu != []:
        line_bu = tools.line_coords(data_bu, start, end, 
                                 nb_indices_exterior=nb_points_beyond)
        ni_range_bu = line_bu['ni_range']
        nj_range_bu = line_bu['nj_range']
        
        ### -- compute compoound of UU and VV
        for var_name_bu in var_to_plot_bu:
            if budget_type in ['PROJ', 'UV']:
                data_bu[f'{var_name_bu}_PROJ'] = tools.windvec_verti_proj(
                    data_bu[f'{var_name_bu}_UU'], data_bu[f'{var_name_bu}_VV'], 
                    data_bu.level, angle)
        
        ### -- interpolate on line
        section = []
        abscisse_coords = []
        abscisse_sites = {}
            
        print('section interpolation on {0} points for budget (~10 ms/pt)'.format(len(ni_range_bu)))
        for i, ni in enumerate(ni_range_bu):
            nj=nj_range_bu[i]
            #interpolation of all variables on ni_range
            profile = data_bu.interp(ni=ni,
                                     nj=nj,
                                     level=line_level).expand_dims({'i_sect':[i]})
            if type(level_range) in [np.ndarray, list]:
                profile = profile.mean(dim='level')
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
        
        section_ds_bu_dict[model] = section_ds_bu


#%% PLOT

# nb of subplot:
if add_wind_barbs:
    nb_subplot = len(var_to_plot_bu) + len(var_to_plot) + 1
    figsize = (11, 2+3*nb_subplot)
    gridspec_kw = {'height_ratios': [4]*(nb_subplot-1) + [2]}
    wind_barbs_yticks = []
else:
    nb_subplot = len(var_to_plot_bu) + len(var_to_plot)
    figsize = (11, 1+3*nb_subplot)
    gridspec_kw = {}

# create figure
fig, ax = plt.subplots(nb_subplot, figsize=figsize,
                       gridspec_kw=gridspec_kw
                       )
if nb_subplot == 1:
    ax = [ax,]  # to make ax subscriptable and have flexible code afterward

for i_model, model in enumerate(models):
    
    # budget subplots
    for i, var_name_bu in enumerate(var_to_plot_bu):
        # loading here only allows not to load if empty dict
        section_ds_bu = section_ds_bu_dict[model]
        
        var_plot = f'{var_name_bu}_PROJ'
        ax[i].plot(section_ds_bu['i_sect'], section_ds_bu[var_plot],
                   label=model,
                   color=colordict[model])
        ax[i].set_ylabel(var_plot)
        ax[i].set_xticklabels([])  # remove x axis ticks labels
        ax[i].hlines(0, 
                     section_ds_bu['i_sect'].min(), section_ds_bu['i_sect'].max(),
                     color='k')
        ax[i].set_xlim(section_ds_bu['i_sect'].min(), section_ds_bu['i_sect'].max())
        ax[i].grid()
        ax[i].legend()
    
    # normal subplot
    for i, var_name in enumerate(var_to_plot):
        # loading here only allows not to load if empty dict
        section_ds = section_ds_dict[model]

        i_ax = len(var_to_plot_bu) + i
        ax[i_ax].plot(section_ds['i_sect'], section_ds[var_name],
                   label=model,
                   color=colordict[model])
        ax[i_ax].set_ylabel(var_name)
        ax[i_ax].set_xticklabels([])  # remove x axis ticks labels
        ax[i_ax].set_xlim(section_ds['i_sect'].min(), section_ds['i_sect'].max())
        ax[i_ax].grid()
        ax[i_ax].legend()
        
    if add_wind_barbs:
        barb_size_option = 'weak_winds'
        ax[nb_subplot-1].barbs(
            #Note that X & alti have dimensions reversed
            section_ds['i_sect'], 
            section_ds['i_sect']*0 + i_model, 
            #Here dimensions are in the proper order
            section_ds['UT'], 
            section_ds['VT'], 
            pivot='middle',
            color=colordict[model],
            length=6,     #length of barbs
            sizes={
    #              'spacing':1, 'height':1, 'width':1,
                    'emptybarb':0.01},
            barb_increments=barb_size_increments[barb_size_option],
            )
        ax[nb_subplot-1].set_xlim(section_ds['i_sect'].min(), section_ds['i_sect'].max())
        wind_barbs_yticks.append(model)


### Plot aesthetics

### set labels of sites ---
    
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
for i in range(nb_subplot):
    ax[i].set_xticks(list(abscisse_sites.keys()))
#    ax.set_xticks(list(abscisse_sites.keys()))
#    ax.set_xticks(list(abscisse_sites.keys()))

if add_wind_barbs:
    # put sites names in antepenultimate subplot
    ax[-2].set_xticklabels(list(abscisse_sites.values()), 
                   rotation=0, fontsize=12)
    # wind barbs subplot
    ax[-1].set_yticks(range(len(wind_barbs_yticks)))
    ax[-1].set_yticklabels(wind_barbs_yticks)
    ax[-1].set_ylim(-1, len(wind_barbs_yticks))
    ax[-1].set_xticklabels([])
    ax[-1].set_xticks([])
    ax[-1].annotate(barb_size_description[barb_size_option],
                     xy=(0, 0),
                     xycoords='axes fraction',
                     fontsize=9
                     )
    
    ax[-1].spines[[
        'top', 'bottom', 
#        'right', 'left'
        ]].set_visible(False)
    
else:
    ax[-1].set_xticklabels(list(abscisse_sites.values()), 
                   rotation=0, fontsize=12)
# x-axis with lat-lon values
#ax.set_xticks(data1.i_sect[::10])
#ax.set_xticklabels(abscisse_coords[::10], rotation=0, fontsize=9)


### add secondary axe ---
# create place for secondary axe
plt.subplots_adjust(bottom=0.25)

# get index of torredembarra in abscisse_sites
torredembarra_ind = list(abscisse_sites.values()).index('torredembarra')
# get corresponding abscisse for torredembarra
torredembarra_xval = list(abscisse_sites.keys())[torredembarra_ind]

def ftest(x):
    return -(x - torredembarra_xval) * (line['nij_step']/1000)
def ftest_recip(x):
    return -(x/(line['nij_step']/1000) + torredembarra_xval)

# add secondary axis
secax = ax[-1].secondary_xaxis(-0.25, functions=(ftest, ftest_recip))
secax.set_xlabel('distance to the sea [km]')


### Global options ---
plot_title = f'{wanted_date}-{model}-{line_level}m agl'
fig.suptitle(plot_title)


if save_plot:
    tools.save_figure(plot_title, save_folder)

    