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
import MNHPy.misc_functions as misc
import tools
import metpy.calc as mcalc
from metpy.units import units
import global_variables as gv

########## Independant parameters ###############

# Simulation to show: 'irr' or 'std'
model = 'irr_d1'
#domain to consider: 1 or 2
domain_nb = int(model[-1])

varname_colormap = 'DIV'
# values color for contourf plot
vmin = -0.0015
vmax = -vmin
colormap='coolwarm'

varname_contourmap = 'RVT'

# Surface variable to show below the section
surf_var = 'WG2_ISBA'
surf_var_label = 'Q_vol_soil'
# Set type of wind representation: 'verti_proj' or 'horiz'
wind_visu = 'verti_proj'
# Datetime
wanted_date = '20210729-2300'
# altitude ASL or height AGL: 'asl' or 'agl'
alti_type = 'asl'
# maximum level (height AGL) to plot
toplevel = 1500

# where to place the cross section
nb_points_beyond = 10
site_start = 'cendrosa'
site_end = 'torredembarra'


# Arrow/barbs esthetics:
skip_barbs_x = 2
skip_barbs_y = 5    #if 1: 1barb/10m, if 5: 1barb/50m, etc
arrow_size = 1.2  #works for arrow and barbs
barb_size_option = 'weak_winds'  # 'weak_winds' or 'standard'


# Save the figure
figsize = (12,7)
save_plot = True
save_folder = './figures/cross_sections/{0}/section_{1}_{2}/{3}/{4}/'.format(
        model, site_start, site_end, wind_visu, varname_colormap)


###########################################

barb_size_increments = gv.barb_size_increments
barb_size_description = gv.barb_size_description


end = (gv.whole[site_end]['lat'], gv.whole[site_end]['lon'])
start = (gv.whole[site_start]['lat'], gv.whole[site_start]['lon'])

if gv.whole[site_start]['lon'] > gv.whole[site_end]['lon']:
    raise ValueError("site_start must be west of site_end")

# Dependant parameters
filename = tools.get_simu_filename(model, wanted_date)

#load file
data_perso = xr.open_dataset(filename)

# Computation of other diagnostic variable
data_perso['DENS'] = mcalc.density(
    data_perso['PRES']*units.hectopascal,
    data_perso['TEMP']*units.celsius, 
    data_perso['RVT']*units.gram/units.gram)

data_perso = tools.center_uvw(data_perso)
data_perso['DIV'] = mcalc.divergence(data_perso['UT'], data_perso['VT'])


data_reduced = data_perso[['THT', 'RVT', 'UT', 'VT', 'WT', 'ZS',
                           'TEMP', 'PRES', 
                           'DENS', 'DIV',
                           surf_var]]
data = data_reduced


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
    
data['PROJ'] = misc.windvec_verti_proj(data['UT'], data['VT'], 
                                       data.level, angle)

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

# get location of minimum for DIV:
section_ds_ilevel = section_ds.isel(level=4)
location_min = section_ds_ilevel.where(
        section_ds_ilevel['DIV'] == section_ds_ilevel['DIV'].min(), 
        drop=True)


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
#                      cmap='rainbow',
#                      vmin=305, vmax=315)
cm = ax[0].contourf(Xmesh,
                    alti,
                    data1.T, 
                    cmap=colormap,  # 'OrRd', 'coolwarm'
#                    levels=np.linspace(298, 315, 18),  # to keep always same colorbar limits
#                    levels=np.linspace(vmin, vmax, vmax-vmin+1),  # to keep always 1K per color variation
                    levels=np.linspace(vmin, vmax, 20),
#                    levels=20,
                    extend = 'both',  #highlights the min and max in different color
                    vmin=vmin, vmax=vmax,  # for THT
#                    vmin=None, vmax=None,  # for adaptative colormap
#                    vmin=800, vmax=1000,  # for PRES
                    )
#manage colorbar
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='2%', pad=0.05)
cbar = fig.colorbar(cm, cax=cax, orientation='vertical')
cbar.set_label('theta [K]')


### 1.2. Contour map
data2 = section_ds[varname_contourmap]  # x1000 to get it in g/kg
cont = ax[0].contour(Xmesh,
                     alti,
                     data2.T,
                     cmap='viridis'  #viridis is default
                     )
ax[0].clabel(cont, cont.levels, inline=True, fontsize=10)
#labels = ['l1','l2','l3','l4','l5','l6','l7','l8','l9']
#for i in range(len(labels)):
#    cont.collections[i].set_label(labels[i])
#ax[0].legend()

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
            section_ds['PROJ'][::skip_barbs_x, ::skip_barbs_y].T, 
            section_ds['WT'][::skip_barbs_x, ::skip_barbs_y].T, 
            pivot='middle',
            scale=150/arrow_size,  # arrows scale, if higher, smaller arrows
            )
    #add arrow scale in top-right corner
    u_max = abs(section_ds['PROJ'][::skip_barbs_x, ::skip_barbs_y]).max()
    ax[0].quiverkey(Q, 0.8, 0.9, 
                    U=u_max, 
                    label=str((np.round(u_max, decimals=1)).data) + 'm/s', 
                    labelpos='E',
                    coordinates='figure')


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

data_soil = section_ds[surf_var][:, :2]  #keep 2 equivalent levels for plot
p9 = ax[1].pcolor(data_soil.i_sect, 
                  data_soil.level, 
                  data_soil.transpose(), 
                  cmap='YlGn',
                  vmin=0, vmax=0.4
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
labels_arr = np.arange(0,30,5)
tick_pos = labels_arr/ (line['nij_step']/1000)
ax[1].set_xticks(ticks = tick_pos,
                 labels = labels_arr
                 )
ax[1].set_xlabel('distance [km]')

ax[1].set_yticks([])
#ax[1].set_ylabel(surf_var)
ax[1].set_ylabel('soil moisture')

## Global options
plot_title = 'Cross section on {0}-{1}-{2}'.format(
        wanted_date, model, wind_visu)
#plot_title = 'Cross section of ABL between irrigated and rainfed areas on July 22 at 12:00 - {0}'.format(
#        model)
#plot_title = 'Cross section on July 22 at 12:00 - {0}'.format(model)
fig.suptitle(plot_title)

if save_plot:
    tools.save_figure(plot_title, save_folder)
