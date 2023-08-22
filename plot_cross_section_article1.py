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
import metpy.calc as mcalc
from metpy.units import units
import global_variables as gv
import matplotlib as mpl

########## Independant parameters ###############

# Simulation to show: 'irr' or 'std'
model = 'irr_d2_old'

# Datetime
wanted_date = '20210722-1200'

varname_colormap = 'THT'
colormap='OrRd'  #coolwarm, 

# values color for contourf plot
if varname_colormap == 'DIV':
    vmin = -0.0015
    vmax = -vmin
elif varname_colormap == 'WS':
    vmin = 0
    vmax = 10
elif varname_colormap == 'THT':
    vmin = 306
    vmax = 314
elif varname_colormap == 'THTV':
    vmin = 300
    vmax = 312
elif varname_colormap == 'THVREF':
    vmin = 290
    vmax = 300
elif varname_colormap == 'TKET':
    vmin = 0.05
    vmax = 3
elif varname_colormap == 'RVT':
    vmin = 0
    vmax = 0.020
elif varname_colormap == 'WT':
    vmin = -1.5
    vmax = -vmin
else:
    vmin = None
    vmax = None

norm_cm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)
#norm_cm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax)  # for TKE

varname_contourmap = 'RVT'
vmin_contour, vmax_contour = 5, 15

# Surface variable to show below the section
surf_var = 'WG2_ISBA'
surf_var_label = 'Q_vol_soil'
vmin_surf_var, vmax_surf_var = 0, 0.4

# Set type of wind representation: 'verti_proj' or 'horiz'
wind_visu = 'verti_proj'

# altitude ASL or height AGL: 'asl' or 'agl'
alti_type = 'asl'
# maximum level (height AGL) to plot
toplevel = 2500

# where to place the cross section
nb_points_beyond = 10
site_start = 'cendrosa'
site_end = 'elsplans'


# Arrow/barbs esthetics:
skip_barbs_x = 2
skip_barbs_y = 10    #if 1: 1barb/10m, if 5: 1barb/50m, etc
arrow_size = 1.2  #works for arrow and barbs
barb_size_option = 'weak_winds'  # 'weak_winds' or 'standard'

uib_adapt = False

# Save the figure
figsize = (12,7)
save_plot = True
save_folder = '/home/lunelt/Documents/redaction/article1_irrigation_breeze/fig/'

plt.rcParams.update({'font.size': 11})
###########################################

barb_size_increments = gv.barb_size_increments
barb_size_description = gv.barb_size_description


end = (gv.whole[site_end]['lat'], gv.whole[site_end]['lon'])
start = (gv.whole[site_start]['lat'], gv.whole[site_start]['lon'])


if gv.whole[site_start]['lon'] > gv.whole[site_end]['lon']:
    raise ValueError("site_start must be west of site_end")

# Dependant parameters
filename = tools.get_simu_filepath(model, wanted_date)
ds = xr.open_dataset(filename)

# Computation of other diagnostic variable
ds['DENS'] = mcalc.density(
    ds['PRES']*units.hectopascal,
    ds['TEMP']*units.celsius, 
    ds['RVT']*units.gram/units.gram)

ds = tools.center_uvw(ds)
ds['DIV'] = mcalc.divergence(ds['UT'], ds['VT'])
ds['WS'], ds['WD'] = tools.calc_ws_wd(ds['UT'], ds['VT'])

ds['THTV'] = ds['THT']*(1 + 0.61*ds['RVT'] - (ds['MRR']+ds['MRC'])/1000)

try:
    data_reduced = ds[['UT', 'VT', 'WT', 'ZS',
#                   'TEMP', 'PRES', 'HBLTOP',
#                   'DENS', 'DIV', 'WS', 'WD',
                   varname_colormap, varname_contourmap, surf_var]]
except:
    data_reduced = ds[['UT', 'VT', 'WT', 'ZS',
#                   'TEMP', 'PRES', 'HBLTOP',
#                   'DENS', 'DIV', 'WS', 'WD',
                   varname_colormap, surf_var]]

data_redsub = tools.subset_ds(data_reduced, 
                              lat_range = [start[0], end[0]], 
                              lon_range = [start[1], end[1]],
                              nb_indices_exterior=nb_points_beyond+2)

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
    ylabel = 'altitude ASL [m]'
elif alti_type == 'agl':
    #create grid mesh (eq. Y)
    alti = np.meshgrid(section_ds.i_sect, section_ds.level)[1]
    alti = xr.DataArray(alti, dims=['level', 'i_sect'])
    #for plot
    ylabel = 'height AGL [m]'
    

### 1.1. Color map (pcolor or contourf)
data1 = section_ds[varname_colormap]
#cm = ax[0].pcolormesh(Xmesh,
#                      alti,
#                      data1.T,
#                      cmap=colormap,
#                      norm=norm_cm,  # for logscale of colormap
##                      vmin=vmin, vmax=vmax
#                      )

cm = ax[0].contourf(Xmesh,
                    alti,
                    data1.T, 
                    cmap=colormap,  # 'OrRd', 'coolwarm'
#                    levels=np.linspace(298, 315, 18),  # to keep always same colorbar limits
                    levels=np.linspace(vmin, vmax, vmax-vmin+1),  # to keep always 1K per color variation
#                    levels=np.linspace(vmin, vmax, 20),
#                    levels=20,
                    extend = 'both',  #highlights the min and max in different color
#                    vmin=vmin, vmax=vmax,  # for adaptative colormap
                    )
#manage colorbar
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='2%', pad=0.05)
cbar = fig.colorbar(cm, cax=cax, orientation='vertical')
try:
    cbar.set_label(f'{data1.long_name} [{data1.units}]')
except:
    cbar.set_label(varname_colormap)

### 1.2. Contour map
if varname_contourmap in ['HBLTOP', 'HLOWJET', 'HLOWJET_WS']:  #1D
#    ax[0].plot(section_ds['HBLTOP'] + section_ds['ZS'],
#              linestyle='--', color='r')
    ax[0].plot(section_ds['HLOWJET_NOSE'] + section_ds['ZS'],
              linestyle='-.', color='g')
    ax[0].plot(section_ds['HLOWJET_MAX'] + section_ds['ZS'],
              linestyle='-.', color='y')
else:  
    data2 = section_ds[varname_contourmap]*1000  # x1000 to get it in g/kg if RVT
    cont = ax[0].contour(Xmesh,
                         alti,
                         data2.T,
                         cmap='viridis_r',  #viridis is default,
#                         levels=np.arange(vmin_contour, vmax_contour),
                         levels=np.linspace(vmin_contour, vmax_contour, vmax_contour-vmin_contour+1),
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
            #Note that X & alti have dimensions reversed
            Xmesh[::skip_barbs_y, ::skip_barbs_x], 
            alti[::skip_barbs_y, ::skip_barbs_x], 
            #Here dimensions are in the proper order
            section_ds['WPROJ'][::skip_barbs_x, ::skip_barbs_y].T, 
            section_ds['WT'][::skip_barbs_x, ::skip_barbs_y].T, 
            pivot='middle',
            scale=150/arrow_size,  # arrows scale, if higher, smaller arrows
            )
    #add arrow scale in top-right corner
    u_max = abs(section_ds['WPROJ'][::skip_barbs_x, ::skip_barbs_y]).max()
    ax[0].quiverkey(Q, 0.8, 0.9, 
                    U=u_max, 
                    label=str((np.round(u_max, decimals=1)).data) + ' $m.s^{-1}$', 
                    labelpos='E',
                    coordinates='figure')


# x-axis with sites names
ax[0].set_xticks(list(abscisse_sites.keys()))
#ax[0].set_xticklabels(list(abscisse_sites.values()), 
#                       rotation=0, fontsize=12)
ax[0].set_xticklabels(['La Cendrosa', 'Els Plans'], 
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

data_soil = section_ds[surf_var][:, :2]  #keep 2 equivalent levels for plot
p9 = ax[1].pcolor(data_soil.i_sect, 
                  data_soil.level, 
                  data_soil.transpose(), 
                  cmap='YlGn',
                  vmin=vmin_surf_var, vmax=vmax_surf_var
                  )
# create colorbar dedicated to the subplot
divider = make_axes_locatable(ax[1])
cax = divider.append_axes('right', size='2%', pad=0.05)
cbar2 = fig.colorbar(p9, cax=cax, orientation='vertical')
#cbar2.set_label(surf_var_label)
cbar2.set_label('[m³/m³]')

#ax[1].set_xticks(ticks = data_soil.i_sect.values[::9],
#                 labels = (data_soil.i_sect.values * \
#                           line['nij_step']/1000)[::9].round(decimals=1)
#                 )
transect_length = (line['index_distance'] + 2*nb_points_beyond)*line['nij_step']/1000  # length in km
labels_arr = np.arange(0, transect_length, 5)
tick_pos = labels_arr / (line['nij_step']/1000)
ax[1].set_xticks(ticks = tick_pos,
                 labels = labels_arr
                 )
ax[1].set_xlabel('distance [km]')

ax[1].set_yticks([])
#ax[1].set_ylabel(surf_var)
ax[1].set_ylabel('soil moisture')

## Global options

if 'irr' in model:
    plot_title =  'IRR'
    figname = 'Cross_section_on_20210722-1200-irr_d2-verti_proj-domain2'
elif 'std' in model:
    plot_title =  'STD'
    figname = 'Cross_section_on_20210722-1200-std_d2-verti_proj-domain2'
            
ax[0].set_title(plot_title)
#fig.suptitle(plot_title)

if save_plot:
    tools.save_figure(figname, save_folder)
