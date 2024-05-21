#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 09:50:51 2022

@author: lunelt
"""
import sys
sys.path.insert(0, '/home/lunelt/postproc_python/')  # needed to run with run_day.sh

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import xarray as xr
import tools
import metpy.calc as mcalc
import global_variables as gv
import matplotlib as mpl
from shapely.geometry import Point, LineString

########## Independant parameters ###############

# Simulation to show: 'irr' or 'std'
model = 'irrswi1_d1'

# Datetime
wanted_date = '20210721-2300'

varname_colormap = 'WT'
plot_title = 'Wind'
cbar_label = 'Wind speed [m s$^{-1}$]'

varname_contourmap = 'THTV'

minmax_dict = {
    'DIV': {'vmin': -0.0015, 'vmax': 0.0015, 'colormap':'coolwarm'},
    'WS': {'vmin': 1, 'vmax': 12, 'colormap':'BuPu'},
    'THT': {'vmin': 306, 'vmax': 314,'colormap':'OrRd'},
    'THTV': {'vmin': 292, 'vmax': 312, 'colormap':'OrRd'},
    'TKET': {'vmin': 0.05, 'vmax': 3, 'colormap':'OrRd'},
    'RVT': {'vmin': 0, 'vmax': 0.02, 'colormap':'OrRd'},
    'WT': {'vmin': None, 'vmax': None, 'colormap':'coolwarm'},
    'MSLP3D': {'vmin': None, 'vmax': None, 'colormap':'coolwarm'},
    None: {'vmin': None, 'vmax': None, 'colormap':'coolwarm'},
    }


vmin = minmax_dict[varname_colormap]['vmin']
vmax = minmax_dict[varname_colormap]['vmax']
colormap = minmax_dict[varname_colormap]['colormap']
norm_cm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)
#norm_cm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax)  # for TKE

vmin_contour = minmax_dict[varname_contourmap]['vmin']
vmax_contour = minmax_dict[varname_contourmap]['vmax']
#vmin_contour, vmax_contour = None, None

# Surface variable to show below the section
subplot_type = 'distance'
surf_var = 'LE_ISBA'   # WG2_ISBA, H, LE
surf_var_label = 'LE_ISBA'
vmin_surf_var, vmax_surf_var = -100, 500

# Set type of wind representation: 'verti_proj' or 'horiz'
wind_visu = 'verti_proj'

# altitude ASL or height AGL: 'asl' or 'agl'
alti_type = 'asl'
# maximum level (height AGL) to plot
toplevel = 1500

# where to place the cross section
nb_points_beyond = 5
site_start = 'cendrosa'
site_end = 'torredembarra'

sites_to_project = ['elsplans', 'serra_tallat', 'serra_miramar']

# Arrow/barbs esthetics:
skip_barbs_x = 2
skip_barbs_y = 10    #if 1: 1barb/10m, if 5: 1barb/50m, etc
arrow_size = 1.2  #works for arrow and barbs
barb_size_option = 'weak_winds'  # 'weak_winds' or 'standard'

uib_adapt = False

# Save the figure
figsize = (12,6)
save_plot = True
save_folder = f'/home/lunelt/postproc_python/article2/fig/cross_sections/{varname_colormap}_{varname_contourmap}/'

plt.rcParams.update({'font.size': 11})
###########################################
#domain to consider: 1 or 2
domain_nb = int(model[-1])

barb_size_increments = gv.barb_size_increments
barb_size_description = gv.barb_size_description


# ----------- from MARIA ANTONIA ------------
if uib_adapt:
#    filename = '/cnrm/surface/lunelt/FROM_MARIAANTONIA/MSB21.3.12H18.002_BIS.cdf'
    filename = '/home/lunelt/Data/MSB21.3.12H18.002_BIS.cdf'
    
    #load file
    ds = xr.open_dataset(filename)
    
    # ------- Modify the dataset to have same structure than direct netcdf MNH output
    # remove the '--PROC1' part of variable names
    vardict = {}
    for var in ds.variables:
        print(var)
        if '--PROC1' in str(var):
            newvar = var.replace('--PROC1', '')
            vardict[var] = newvar
        
    ds = ds.rename(vardict)
    ds = ds.rename({'DIMX': 'ni', 'DIMY': 'nj', 'DIMZ': 'level'})    
    
    # IMPORTANT:
    # Put the real values here
    lon_min, lon_max = 2.5, 3.5   # min and max longitude values of the domain
    lat_min, lat_max = 39, 40.15
    level_min, level_max = 0, 10000
    # create the level AGL range. This is just an example with a square stretching.
    level_array = (np.linspace(np.sqrt(level_min), 
                              np.sqrt(level_max), 
                              len(ds.level)))**2
   # ---------------------------------------
    ds['level'] = level_array
                               
    lon_grid, lat_grid = np.meshgrid(np.linspace(lon_min, lon_max, len(ds.ni)), 
                                     np.linspace(lat_min, lat_max, len(ds.nj)))
    
#    ds = ds.assign_coords({'longitude': np.linspace(lon_min, lon_max, len(ds.ni))})
    ds = ds.assign_coords({
        'longitude': xr.DataArray(
            lon_grid,
            dims=['nj', 'ni']),
        'latitude': xr.DataArray(
            lon_grid,
            dims=['nj', 'ni']),
    })
    
    # ------- Parameters for choosing where to do the cross-section -----
#    end = (39.8, 3.25)
#    start = (39.65, 3.0)
    start = (39.1, 2.6)  # starting point for cross section
    end = (40, 3.4)   # ending point for cross section
    site_start = str(start)
    site_end = str(end)
    
    vmin, vmax = None, None
    vmin_contour, vmax_contour = None, None
    vmin_surf_var, vmax_surf_var = None, None
    
    # Check order of start and end sites
    if start[1] > end[1]:
        raise ValueError("site_start must be west of site_end")
    
else:
    filename = tools.get_simu_filepath(model, wanted_date,
                                       file_suffix='dg', 
                                       out_suffix='')
#    filename = '/home/lunelt/Data/temp_outputs/LIAIS.1.SEG03.009.nc'
    ds = xr.open_dataset(filename)
    
    end = (gv.whole[site_end]['lat'], gv.whole[site_end]['lon'])
    start = (gv.whole[site_start]['lat'], gv.whole[site_start]['lon'])

    # Check order of start and end sites
    if gv.whole[site_start]['lon'] > gv.whole[site_end]['lon']:
        raise ValueError("site_start must be west of site_end")



# pre-processing of data
ds_sub = tools.subset_ds(ds, 
                  lat_range = [start[0], end[0]], 
                  lon_range = [start[1], end[1]],
                  nb_indices_exterior=nb_points_beyond)    
ds_subcen = tools.center_uvw(ds_sub)

# Computation of other diagnostic variable
ds_subcen['DIV'] = mcalc.divergence(ds_subcen['UT'], ds_subcen['VT'])
ds_subcen['WS'], ds_subcen['WD'] = tools.calc_ws_wd(ds_subcen['UT'], ds_subcen['VT'])

ds_subcen['THTV'] = ds_subcen['THT']*(1 + 0.61*ds_subcen['RVT'] - (ds_subcen['MRR']+ds_subcen['MRC'])/1000)
ds_subcen['MSLP3D'] = tools.calc_mslp(ds)


data_reduced = ds_subcen[['UT', 'VT', 'WT', 'ZS',
#                   'TEMP', 'PRES', 'HBLTOP',
#                   'DENS', 'DIV', 'WS', 'WD',
               varname_colormap, varname_contourmap, surf_var]]


data = data_reduced
#data = data_redsub


#%% CREATE SECTION LINE

line = tools.line_coords(data, start, end, 
                         nb_indices_exterior=nb_points_beyond)
ni_range = line['ni_range']
nj_range = line['nj_range']
slope = line['slope']

if slope == 'vertical':
    angle = np.pi/2
else:
    angle = np.arctan(slope)
    
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
#                                     new_height_var='HLOWJET_NOSE',
#                                     upper_bound=0.70)
#data = tools.diag_lowleveljet_height(data,
#                                     wind_var='WS', 
#                                     new_height_var='HLOWJET_MAX',
#                                     upper_bound=1)

#%% INTERPOLATION

section = []
abscisse_coords = []
abscisse_sites = {}

#get total maximum height of relief on domain
max_ZS = data['ZS'].max()
if alti_type == 'asl':
    level_range = np.arange(10, toplevel+max_ZS, 10)
else:
    level_range = np.arange(10, toplevel, 10)
    
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


#%% PLOT
# create figure
fig, ax = plt.subplots(2, figsize=figsize,
                       gridspec_kw={'height_ratios': [20, 1]})

## --- Subplot of section, i.e. the main plot ----
#get maximum height of relief in cross-section
max_ZS = section_ds['ZS'].max()

# remove top layers of troposphere
section_ds = section_ds.where(section_ds.level<(level_range.max()), drop=True)

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
    ylabel = 'Altitude a.s.l. [m]'
elif alti_type == 'agl':
    #create grid mesh (eq. Y)
    alti = np.meshgrid(section_ds.i_sect, section_ds.level)[1]
    alti = xr.DataArray(alti, dims=['level', 'i_sect'])
    #for plot
    ylabel = 'Height a.g.l. [m]'
    

### 1.1. Color map (pcolor or contourf)
data1 = section_ds[varname_colormap]
#cm = ax[0].pcolormesh(Xmesh,
#                      alti,
#                      data1.T,
#                      cmap=colormap,
#                      norm=norm_cm,  # for logscale of colormap
##                      vmin=vmin, vmax=vmax
#                      )
if None in [vmin, vmax]:
    levels = 10
else:
    levels = np.linspace(vmin, vmax, 10)
    # levels = np.linspace(vmin, vmax, vmax-vmin+1)  # to have 1 unit per color variation
    
cm = ax[0].contourf(Xmesh,
                    alti,
                    data1.T, 
                    cmap=colormap,  # 'OrRd', 'coolwarm'
                    levels=levels,
                    extend = 'both',  #highlights the min and max in different color
                    )
#manage colorbar
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='2%', pad=0.05)
cbar = fig.colorbar(cm, cax=cax, orientation='vertical')
cbar.set_label(cbar_label)

### 1.2. Contour map
if varname_contourmap in ['HBLTOP', 'HLOWJET', 'HLOWJET_WS']:  #1D
#    ax[0].plot(section_ds['HBLTOP'] + section_ds['ZS'],
#              linestyle='--', color='r')
    ax[0].plot(section_ds['HLOWJET_NOSE'] + section_ds['ZS'],
              linestyle='-.', color='g')
    ax[0].plot(section_ds['HLOWJET_MAX'] + section_ds['ZS'],
              linestyle='-.', color='y')
else:  
    data2 = section_ds[varname_contourmap]  # x1000 to get it in g/kg if RVT
    cont = ax[0].contour(Xmesh,
                         alti,
                         data2.T,
                         cmap='copper_r',  #viridis_r, copper_r
#                         colors='r',
#                         levels=np.arange(vmin_contour, vmax_contour),
                         levels=np.linspace(vmin_contour, 
                                            vmax_contour, 
                                            vmax_contour-vmin_contour+1),
#                         vmin=vmin, vmax=vmax,  # for adaptative colormap
                         )
    ax[0].clabel(cont, cont.levels, inline=True, fontsize=13)

### 1.3. Winds
if wind_visu == 'horiz':            # 2.1 winds - flat direction and force
    ax[0].barbs(
            #Note that X & alti have dimensions reversed
            Xmesh[::skip_barbs_y, ::skip_barbs_x], 
            alti[::skip_barbs_y, ::skip_barbs_x], 
            #Here dimensions are in the proper order
            section_ds['UT'][::skip_barbs_x, ::skip_barbs_y].T, 
            section_ds['VT'][::skip_barbs_x, ::skip_barbs_y].T, 
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
elif wind_visu == 'verti_proj':     # 2.2  winds - verti and projected wind
    Q = ax[0].quiver(
            # Note that X & alti have dimensions reversed
            Xmesh[::skip_barbs_y, ::skip_barbs_x], 
            alti[::skip_barbs_y, ::skip_barbs_x], 
            # Here dimensions are in the proper order
            section_ds['WPROJ'][::skip_barbs_x, ::skip_barbs_y].T, 
            section_ds['WT'][::skip_barbs_x, ::skip_barbs_y].T, 
            pivot='middle',
            scale=150/arrow_size,  # arrows scale, if higher, smaller arrows
            alpha=0.4,
            )
    # Add arrow scale in top-right corner
    u_max = abs(section_ds['WPROJ'][::skip_barbs_x, ::skip_barbs_y]).max()
    ax[0].quiverkey(Q, 0.8, 0.9, 
                    U=u_max, 
                    label=str((np.round(u_max, decimals=1)).data) + ' m s$^{-1}$', 
                    labelpos='E',
                    coordinates='figure')

# add sites that are between the two defining sites 'start' and 'end'
for site_inter in sites_to_project:
    coords_site_inter = (gv.whole[site_inter]['lat'], gv.whole[site_inter]['lon'])
    
    point_site_inter = Point(coords_site_inter)
    line_cross_section = LineString([start, end])
    dist = line_cross_section.project(point_site_inter)
    coords_site_inter_proj = list(line_cross_section.interpolate(dist).coords)[0]
    
    fraction_lon_point_inter = (coords_site_inter_proj[1] - start[1]) / (end[1] - start[1])
    # in terms of abscisse
    list_abscisses_sites = list(abscisse_sites.keys())
    diff_abscisses = list_abscisses_sites[1] - list_abscisses_sites[0]
    abscisse_inter = fraction_lon_point_inter * diff_abscisses + list_abscisses_sites[0]
    # add to the dict
    abscisse_sites[abscisse_inter] = site_inter

# x-axis with sites names
# change shortnames to longnames
abscisse_sites_xaxis = {key:gv.whole[val]['longname'] for (key,val) in zip(abscisse_sites.keys(), abscisse_sites.values())}

# add names on xaxis
ax[0].set_xticks(list(abscisse_sites_xaxis.keys()))
ax[0].set_xticklabels(list(abscisse_sites_xaxis.values()), 
                      rotation=0, fontsize=12)
#ax[0].set_xticklabels(['La Cendrosa', 'Els Plans'], 
#                       rotation=0, fontsize=12)
# x-axis with lat-lon values
#ax.set_xticks(data1.i_sect[::10])
#ax.set_xticklabels(abscisse_coords[::10], rotation=0, fontsize=9)

# set y limits (height ASL)
if alti_type == 'asl':
    min_ZS = section_ds['ZS'].min()
    ax[0].set_ylim([min_ZS, max_ZS + toplevel])
ax[0].set_ylabel(ylabel)


### 2. Subplot of surface characteristic ---
#
#data_soil = section_ds[surf_var][:, :2]  #keep 2 equivalent levels for plot
#p9 = ax[1].pcolor(data_soil.i_sect, 
#                  data_soil.level, 
#                  data_soil.transpose(), 
#                  cmap='YlGn',
#                  vmin=vmin_surf_var, vmax=vmax_surf_var
#                  )
## create colorbar dedicated to the subplot
#divider = make_axes_locatable(ax[1])
#cax = divider.append_axes('right', size='2%', pad=0.05)
#cbar2 = fig.colorbar(p9, cax=cax, orientation='vertical')
##cbar2.set_label(surf_var_label)
#cbar2.set_label('[m³/m³]')
#
##ax[1].set_xticks(ticks = data_soil.i_sect.values[::9],
##                 labels = (data_soil.i_sect.values * \
##                           line['nij_step']/1000)[::9].round(decimals=1)
##                 )
#transect_length = (line['index_distance'] + 2*nb_points_beyond)*line['nij_step']/1000  # length in km
#labels_arr = np.arange(0, transect_length, 10, dtype=int)
#tick_pos = labels_arr / (line['nij_step']/1000)
#ax[1].set_xticks(ticks = tick_pos,
#                 labels = labels_arr
#                 )
#ax[1].set_xlabel('distance [km]')
#
#ax[1].set_yticks([])
##ax[1].set_ylabel(surf_var)
#ax[1].set_ylabel('soil moisture')

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
save_title = 'Cross section on {0}-{1}-{2}'.format(
        wanted_date, model, wind_visu)
#plot_title = 'Cross section of ABL between irrigated and rainfed areas on July 22 at 12:00 - {0}'.format(
#        model)
#plot_title = 'Cross section on July 22 at 12:00 - {0}'.format(model)
ax[0].set_title(plot_title)
#fig.suptitle(plot_title)

if save_plot:
    tools.save_figure(save_title, save_folder)
