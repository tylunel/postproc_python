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
from tools import indices_of_lat_lon, get_simu_filename, line_coords

########## Independant parameters ###############
# Simulation to show: 'irr' or 'std'
model = 'irr'
# Surface variable to show below the section
surf_var = 'WG2_ISBA'
surf_var_label = 'Q_vol_soil'
# Set type of wind representation: 'verti_proj' or 'horiz'
wind_visu = 'verti_proj'
# Datetime
wanted_date = '20210722-1400'
# altitude ASL or height AGL: 'asl' or 'agl'
alti_type = 'asl'
# maximum level (height AGL) to plot
toplevel = 2500
# Save the figure
save_plot = True
save_folder = './figures/cross_sections/'
#################################################


# Dependant parameters
filename = get_simu_filename(model, wanted_date)

#load file
data_perso = xr.open_dataset(filename)
data_reduced = data_perso[['THT', 'RVT', 'UT', 'VT', 'WT', 'ZS',
                           surf_var]]
data = data_reduced



#%% Put all variable in the middle of the grid:

# Interpolate in middle of grid and rename the coords
data_UT = data['UT'].interp(ni_u=data.ni.values, nj_u=data.nj.values).rename({'ni_u': 'ni', 'nj_u': 'nj'})
data_VT = data['VT'].interp(ni_v=data.ni.values, nj_v=data.nj.values).rename({'ni_v': 'ni', 'nj_v': 'nj'})
data_WT = data['WT'].interp(level_w=data.level.values).rename({'level_w': 'level'})

# Create new datarray with everything centered in the middle
data = xr.merge([data_UT, data_VT, data_WT, 
                 data['THT'], data['RVT'], data['ZS'],
                 data[surf_var],
                 ])
# remove useless coordinates
data = data.drop(['latitude_u', 'longitude_u', 'latitude_v', 'longitude_v', ])
# consider time no longer as a dimension but just as a single coordinate
data = data.squeeze()

#%% CREATE SECTION LINE

site_start = 'cendrosa'
start = (41.6925905, 0.9285671) # CENDROSA
site_end = 'tarragona'
end = (41.1188, 1.2456)
#site_end = 'elsplans'
#end = (41.590111, 1.029363) # ELSPLANS

line = line_coords(data, start, end, nb_indices_exterior=10)
ni_range = line['ni_range']
nj_range = line['nj_range']
slope = line['slope']

angle = np.arctan(slope)
data['PROJ'] = misc.windvec_verti_proj(data['UT'], data['VT'], 
                                       data.level, angle)

#%% INTERPOLATION

section = []
abscisse_coords = []
abscisse_sites = {}

for i, ni in enumerate(ni_range):
    print(i)
    #interpolation of all variables on ni_range
    profile = data.interp(ni=ni, nj=nj_range[i]).expand_dims({'i_sect':[i]})
    section.append(profile)
    
    #store values of lat-lon for the horiz axis
    lat = np.round(profile.latitude.values, decimals=3)
    lon = np.round(profile.longitude.values, decimals=3)
    latlon = str(lat) + '\n' + str(lon)
    abscisse_coords.append(latlon)
    
    #Store values of i and name of site in dict for horiz axis
    if ni == line['ni_start']:
        abscisse_sites[i] = site_start
    elif ni == line['ni_end']:
        abscisse_sites[i] = site_end

#concatenation of all profile in order to create the 2D section dataset
section_ds = xr.concat(section, dim="i_sect")

#%% PLOT
# create figure
fig, ax = plt.subplots(2, figsize=(12,6),
                       gridspec_kw={'height_ratios': [20, 1]})

## --- Subplot of section, i.e. the main plot ----
# remove top layers of troposphere
section_ds = section_ds.where(section_ds.level<toplevel, drop=True)

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
    

### 1. Color map (pcolor or contourf)
data1 = section_ds['THT']
#cm = ax[0].pcolormesh(Xmesh,
#                      alti,
#                      data1.T, 
#                      cmap='rainbow',
#                      vmin=305, vmax=315)
cm = ax[0].contourf(Xmesh,
                    alti,
                    data1.T, 
                    cmap='rainbow',
                    levels=20,
                    vmin=305, vmax=315)
#manage colorbar
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='2%', pad=0.05)
cbar = fig.colorbar(cm, cax=cax, orientation='vertical')
cbar.set_label('theta [K]')


### 2. Contour map
data2 = section_ds['RVT']
cont = ax[0].contour(Xmesh,
                     alti,
                     data2.T)
ax[0].clabel(cont, cont.levels, inline=True, fontsize=8)

### 3. Winds
if wind_visu == 'horiz':            # 2.1 winds - flat direction and force
    skip_barbs_x = 5
    skip_barbs_y = 4
    ax[0].barbs(
            #Note that X & alti have dimensions reversed
            Xmesh[::skip_barbs_y, ::skip_barbs_x], 
            alti[::skip_barbs_y, ::skip_barbs_x], 
            #Here dimensions are in the proper order
            section_ds['UT'][::skip_barbs_x, ::skip_barbs_y].T, 
            section_ds['VT'][::skip_barbs_x, ::skip_barbs_y].T, 
            pivot='middle',
            length=5,     #length of barbs
            sizes={
    #              'spacing':1, 'height':1, 'width':1,
                 'emptybarb':0.01}
              )
elif wind_visu == 'verti_proj':     # 2.2  winds - verti and projected wind
    skip_barbs_x = 5
    skip_barbs_y = 4
    Q = ax[0].quiver(
            #Note that X & alti have dimensions reversed
            Xmesh[::skip_barbs_y, ::skip_barbs_x], 
            alti[::skip_barbs_y, ::skip_barbs_x], 
            #Here dimensions are in the proper order
            section_ds['PROJ'][::skip_barbs_x, ::skip_barbs_y].T, 
            section_ds['WT'][::skip_barbs_x, ::skip_barbs_y].T, 
            pivot='middle',
            scale=150,     #scale of arrows - if higher, arrows are smaller
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

ax[0].set_ylabel(ylabel)


### --- Subplot of surface characteristic ---

data_soil = section_ds[surf_var][:, :2]  #keep 2 equivalent levels for plot
p9 = ax[1].pcolor(data_soil.i_sect, 
                  data_soil.level, 
                  data_soil.transpose(), cmap='RdYlGn',
                  vmin=0, vmax=0.4
                  )
# create colorbar dedicated to the subplot
divider = make_axes_locatable(ax[1])
cax = divider.append_axes('right', size='2%', pad=0.05)
cbar2 = fig.colorbar(p9, cax=cax, orientation='vertical')
cbar2.set_label(surf_var_label)

ax[1].set_xticks(ticks = data_soil.i_sect.values[::9],
                 labels = (data_soil.i_sect.values * line['ni_step']/1000)[::9].round(decimals=1)
                 )
ax[1].set_xlabel('distance [km]')

ax[1].set_yticks([])
ax[1].set_ylabel(surf_var)

# Global options
plot_title = 'Cross section on '+ str(wanted_date) +'-'+ model +'-'+ wind_visu
fig.suptitle(plot_title)

if save_plot:
    filename = (plot_title)
    filename = filename.replace('=', '').replace('(', '').replace(')', '')
    filename = filename.replace(' ', '_').replace(',', '').replace('.', '_')
    plt.savefig(save_folder +str(filename))
