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
          'std_d1',
          'irrswi1_d1', 
          ]

# Datetime
wanted_date = '20210717-0000'

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

var_name_bu = 'PRES'
# --- nb_var is used by run_day_multi_var.sh:
#nb_var = 5
#var_name_bu = var_name_bu_list[nb_var]

var_name = 'HLOWJET_07'
vmin, vmax = 0, 1000

# level (height AGL) to plot
line_level = 10  #[m]

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
save_folder = f'./figures/linear_cross_sections/section_{site_start}_{site_end}/{budget_type}-{var_name_bu}_{var_name}/'

###########################################


colordict = {'irr_d2': 'g', 
             'std_d2': 'r',
             'irr_d1': 'g', 
             'std_d1': 'r', 
             'irrlagrip30_d1': 'orange',
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
    filepath = tools.get_simu_filepath(model, wanted_date, file_suffix='dg',
                                       out_suffix='')
    ds = xr.open_dataset(filepath)
    
    day = pd.Timestamp(wanted_date).day
    hour = pd.Timestamp(wanted_date).hour
    
    filename_bu = gv.global_simu_folder + gv.simu_folders[model] + f'LIAIS.1.SEG{day}.000.nc'
    
    if budget_type in ['PROJ', 'UV']:
        ds_bu = tools.compound_budget_file(filename_bu).isel(time_budget=hour)
        ds_bu[f'{var_name_bu}_VAL'], ds_bu[f'{var_name_bu}_DIR'] = tools.calc_ws_wd(
                ds_bu[f'{var_name_bu}_UU'], ds_bu[f'{var_name_bu}_VV'])
    else:
        ds_bu = tools.open_budget_file(filename_bu, budget_type).isel(time_budget=hour)
    
    # Computation of other diagnostic variable
    ds = tools.center_uvw(ds)
    ds['WS'], ds['WD'] = tools.calc_ws_wd(ds['UT'], ds['VT'])
    

    data_reduced = ds[['UT', 'VT', 'WT', 'ZS', 'WS', 'THT',
               'HBLTOP', 'THVREF',
#               var_name
               ]]
    data_redsub = tools.subset_ds(data_reduced, 
                                  lat_range = [start[0], end[0]], 
                                  lon_range = [start[1], end[1]],
                                  nb_indices_exterior=nb_points_beyond+2)
    
    data = data_redsub
#    data = ds
    
    
    if budget_type in ['PROJ', 'UV']:
        data_bu = ds_bu[[f'{var_name_bu}_VAL', f'{var_name_bu}_DIR',
                         f'{var_name_bu}_VV', f'{var_name_bu}_UU']]
    else:
        data_bu = ds_bu[[var_name_bu,]]
        

    
    #get total maximum height of relief on domain
    if var_name in ['HLOWJET', 'HLOWJET_07', 'FROUDE',]:
        toplevel=2500
        level_range = np.arange(10, toplevel, 20)
    else:
        level_range = line_level

    
    
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
    
    ### -- interpolate on line
    print('section interpolation on {0} points for standard data (~0.1sec/pt)'.format(len(ni_range)))
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
    

#%% OTHER DIAGs: BULK RI and FROUDE
    if var_name in ['HLOWJET_07', 'HLOWJET', 'FROUDE',]:
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
            
#%%
        section_ds = section_ds.isel(level=line_level)
#%%   
    section_ds_dict[model] = section_ds
    
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
        
    print('section interpolation on {0} points for budget (~10 ms/pt)'.format(len(ni_range_bu)))
    for i, ni in enumerate(ni_range_bu):
        nj=nj_range_bu[i]
        #interpolation of all variables on ni_range
        profile = data_bu.interp(ni=ni,
                                 nj=nj,
                                 level=line_level).expand_dims({'i_sect':[i]})
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

# create figure
fig, ax = plt.subplots(3, figsize=figsize,
#                       gridspec_kw={'height_ratios': [5, 5, 1]}
                       )

for model in models:
    section_ds_bu = section_ds_bu_dict[model]
    section_ds = section_ds_dict[model]
    
    # first subplot
    var_plot = f'{var_name_bu}_PROJ'
    ax[0].plot(section_ds_bu['i_sect'], section_ds_bu[var_plot],
               label=model,
               color=colordict[model])
    ax[0].set_ylabel(var_plot)
    ax[0].set_xticklabels([])  # remove x axis ticks labels
    ax[0].hlines(0, 
                 section_ds_bu['i_sect'].min(), section_ds_bu['i_sect'].max(),
                 color='k')
    ax[0].set_xlim(section_ds_bu['i_sect'].min(), section_ds_bu['i_sect'].max())
    
    # second subplot
    var_plot = var_name
    ax[1].plot(section_ds['i_sect'], section_ds[var_plot],
               label=model,
               color=colordict[model])
    ax[1].set_ylabel(var_plot)
    ax[1].set_xticklabels([])  # remove x axis ticks labels
    ax[1].set_xlim(section_ds_bu['i_sect'].min(), section_ds_bu['i_sect'].max())
    
    # second subplot
    var_plot = 'FROUDE_HJET_07'
    ax[2].plot(section_ds['i_sect'], section_ds[var_plot],
               label=model,
               color=colordict[model])
    ax[2].hlines(1, 
                 section_ds_bu['i_sect'].min(), section_ds_bu['i_sect'].max(),
                 color='k', linestyle='--')
    ax[2].set_ylabel(var_plot)
    ax[2].set_xticklabels([])  # remove x axis ticks labels
    ax[2].set_xlim(section_ds_bu['i_sect'].min(), section_ds_bu['i_sect'].max())

ax[0].set_ylim(vmin_bu, vmax_bu)
ax[0].grid()
ax[0].legend()

ax[1].set_ylim(vmin, vmax)
ax[1].grid()
ax[1].legend()

ax[2].set_ylim(0, 6)
ax[2].grid()
ax[2].legend()

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
ax[0].set_xticks(list(abscisse_sites.keys()))
ax[1].set_xticks(list(abscisse_sites.keys()))
ax[2].set_xticks(list(abscisse_sites.keys()))
ax[-1].set_xticklabels(list(abscisse_sites.values()), 
                   rotation=0, fontsize=12)
# x-axis with lat-lon values
#ax.set_xticks(data1.i_sect[::10])
#ax.set_xticklabels(abscisse_coords[::10], rotation=0, fontsize=9)


### add secondary axe ---
# create place for secondary axe
plt.subplots_adjust(bottom=0.2)

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
plot_title = f'{wanted_date}-{model}-{budget_type}-{var_name_bu}-{line_level}m agl'
fig.suptitle(plot_title)


if save_plot:
    tools.save_figure(plot_title, save_folder)

    