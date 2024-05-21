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
import metpy.calc as calc
from metpy.units import units
import global_variables as gv

########## Independant parameters ###############

# Simulation to show: 'irr' or 'std'
models_list = ['irr_d2_old', 'std_d2_old']  # put IRR first, STD second


# Set type of wind representation: 'verti_proj' or 'horiz'
wind_visu = 'verti_proj'

# Datetime
wanted_date = '20210722-1200'

# altitude ASL or height AGL: 'asl' or 'agl'
alti_type = 'asl'
# maximum level (height AGL) to plot
toplevel = 2500

# where to place the cross section
nb_points_beyond = 10
site_start = 'cendrosa'
site_end = 'elsplans'

varname_contourmap = 'RVT'
vmin_contour, vmax_contour = -4, 4

# Surface variable to show below the section
surf_var = 'WG2_ISBA'
surf_var_label = 'Q_vol_soil'
vmin_surf_var, vmax_surf_var = 0, 0.3

# Arrow/barbs esthetics:
skip_barbs_x = 2
skip_barbs_y = 10    #if 1: 1barb/10m, if 5: 1barb/50m, etc
arrow_size = 1.2  #works for arrow and barbs
barb_size_option = 'weak_winds'  # 'weak_winds' or 'standard'

# Save the figure
figsize = (12,5.5)
save_plot = True
save_folder = './fig/'

plt.rcParams.update({'font.size': 12})

###########################################

barb_size_increments = {
        'weak_winds': {'half':1.94, 'full':3.88, 'flag':19.4},
        'standard': {'half':5, 'full':10, 'flag':50},
        }
barb_size_description = {
        'weak_winds': "barb increments: half=1m/s=1.94kt, full=2m/s=3.88kt, flag=10m/s=19.4kt",
        'standard': "barb increments: half=5kt=2.57m/s, full=10kt=5.14m/s, flag=50kt=25.7m/s",
        }


end = (gv.whole[site_end]['lat'], gv.whole[site_end]['lon'])
start = (gv.whole[site_start]['lat'], gv.whole[site_start]['lon'])


section_ds = {}

for model in models_list:
    # Dependant parameters
    filename = tools.get_simu_filepath(model, wanted_date, 
    #                                      domain=domain_nb,
    #                                      file_suffix='001dg'
                                          )
    
    #load file
    data_perso = xr.open_dataset(filename)
    data_reduced = data_perso[['THT', 'RVT', 'UT', 'VT', 'WT', 'ZS',
                               'TEMP', 'PRES',
                               surf_var]]
    data = data_reduced
    
    
    #%% Put variables U, V, W in the middle of the grid:
    
    # with external function:
    data = tools.center_uvw(data)
    
    
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
        
    data['PROJ'] = tools.windvec_verti_proj(data['UT'], data['VT'], 
                                           data.level, angle)
    
    #%% INTERPOLATION
    
    section = []
    abscisse_coords = []
    abscisse_sites = {}
    
    #get total maximum height of relief on domain
    max_ZS = data['ZS'].max()
    level_range = np.arange(10, toplevel+max_ZS, 10)
    
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
    section_ds[model] = xr.concat(section, dim="i_sect")

section_diff = section_ds[models_list[0]] - section_ds[models_list[1]]
section_diff['ZS'] = section_ds[models_list[0]]['ZS']

#%% PLOT
# create figure
fig, ax = plt.subplots(2, figsize=figsize,
                       gridspec_kw={'height_ratios': [20, 1]})
plt.subplots_adjust(top=0.95)
plt.subplots_adjust(bottom=0.09)

## --- Subplot of section, i.e. the main plot ----
#get maximum height of relief in cross-section
max_ZS = section_diff['ZS'].max()

# remove top layers of troposphere
section_diff = section_diff.where(section_diff.level<(toplevel+max_ZS), drop=True)

## --- Adapt to alti_type ------
#create grid mesh (eq. to X)
X = np.meshgrid(section_diff.i_sect, section_diff.level)[0]
Xmesh = xr.DataArray(X, dims=['level', 'i_sect'])
#create alti mesh (eq. to Y)
if alti_type == 'asl': 
    #compute altitude ASL from height AGL, and transpose (eq. Y)
    alti = section_diff.ZS[:, 0] + section_diff.level
    alti = alti.T
    #for plot
    ylabel = 'altitude a.s.l. [m]'
elif alti_type == 'agl':
    #create grid mesh (eq. Y)
    alti = np.meshgrid(section_diff.i_sect, section_diff.level)[1]
    alti = xr.DataArray(alti, dims=['level', 'i_sect'])
    #for plot
    ylabel = 'height a.g.l. [m]'
    

# Computation of diagnostic variable
# section_diff['DENS'] = calc.density(
#     section_diff['PRES']*units.hectopascal,
#     section_diff['THT']*units.K, 
#     section_diff['RVT']*units.gram/units.gram)

### 1. Color map (pcolor or contourf)
data1 = section_diff['THT']
#cm = ax[0].pcolormesh(Xmesh,
#                      alti,
#                      data1.T, 
#                      cmap='rainbow',
#                      vmin=305, vmax=315)
cm = ax[0].contourf(Xmesh,
                    alti,
                    data1.T, 
                    cmap='coolwarm',  # seismic, coolwarm
                    levels=np.linspace(-5, 5, 11),  # to keep always same colorbar limits
#                    levels=20,
#                    extend = 'both',  #highlights the min and max in different color
                    vmin=-5, vmax=5,  # for THT
#                    vmin=None, vmax=None,  # for adaptative colormap
#                    vmin=800, vmax=1000,  # for PRES
                    )
#manage colorbar
divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='2%', pad=0.05)
cbar = fig.colorbar(cm, cax=cax, orientation='vertical')
cbar.set_label('potential temperature [K]')


### 2. Contour map
data2 = section_diff[varname_contourmap]*1000  # x1000 to get it in g/kg if RVT
cont = ax[0].contour(Xmesh,
                     alti,
                     data2.T,
                     cmap='seismic_r',  #viridis_r, seismic_r
                     # levels=np.linspace(vmin_contour, vmax_contour, (vmax_contour-vmin_contour+1)*1),
                     # levels= [-4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4],
                     levels= [-4, -3, -2, -1, 0, 1, 2, 3, 4],
#                     vmin=vmin, vmax=vmax,  # for adaptative colormap
                     )
ax[0].clabel(cont, cont.levels, inline=True, fontsize=13)


### 3. Winds
if wind_visu == 'horiz':            # 2.1 winds - flat direction and force
    ax[0].barbs(
            #Note that X & alti have dimensions reversed
            Xmesh[::skip_barbs_y, ::skip_barbs_x], 
            alti[::skip_barbs_y, ::skip_barbs_x], 
            #Here dimensions are in the proper order
            section_diff['UT'][::skip_barbs_x, ::skip_barbs_y].T, 
            section_diff['VT'][::skip_barbs_x, ::skip_barbs_y].T, 
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
            section_diff['PROJ'][::skip_barbs_x, ::skip_barbs_y].T, 
            section_diff['WT'][::skip_barbs_x, ::skip_barbs_y].T, 
            pivot='middle',
            scale=150/arrow_size,  # arrows scale, if higher, smaller arrows
            alpha=0.6,
            )
    #add arrow scale in top-right corner
    u_max = abs(section_diff['PROJ'][::skip_barbs_x, ::skip_barbs_y]).max()
    ax[0].quiverkey(Q, 0.8, 0.96, 
                    U=u_max,
                    label=str((np.round(u_max, decimals=1)).data) + 'm/s', 
                    labelpos='E',
                    coordinates='figure')


# x-axis with sites names
ax[0].set_xticks(list(abscisse_sites.keys()))
ax[0].set_xticklabels(['La Cendrosa', 'Els Plans'], 
                       rotation=0, fontsize=12)
# ax[0].set_xticklabels(list(abscisse_sites.values()), 
#                    rotation=0, fontsize=12)
# x-axis with lat-lon values
#ax.set_xticks(data1.i_sect[::10])
#ax.set_xticklabels(abscisse_coords[::10], rotation=0, fontsize=9)

# set y limits (height ASL)
min_ZS = section_diff['ZS'].min()
ax[0].set_ylim([min_ZS, max_ZS + toplevel])
ax[0].set_ylabel(ylabel)


### --- Subplot of surface characteristic ---

# data_soil = section_diff[surf_var][:, :2]  #keep 2 equivalent levels for plot
# p9 = ax[1].pcolor(data_soil.i_sect, 
#                   data_soil.level, 
#                   data_soil.transpose(), cmap='RdYlGn',
#                   vmin=-0.4, vmax=0.4
#                   )
# # create colorbar dedicated to the subplot
# divider = make_axes_locatable(ax[1])
# cax = divider.append_axes('right', size='2%', pad=0.05)
# cbar2 = fig.colorbar(p9, cax=cax, orientation='vertical')
# # cbar2.set_label(surf_var_label)

# ax[1].set_xticks(ticks = data_soil.i_sect.values[::9],
#                  labels = (data_soil.i_sect.values * \
#                            line['nij_step']/1000)[::9].round(decimals=1)
#                  )
# ax[1].set_xlabel('distance [km]')

# ax[1].set_yticks([])
# ax[1].set_ylabel(surf_var)



### 2. Subplot of surface characteristic ---

data_soil = section_diff[surf_var][:, :2]  #keep 2 equivalent levels for plot
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
cbar2.set_label('soil moisture\n [m³ m$^{-3}$]')

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


# Global options
plot_title =  'IRR - STD'
figname = 'Cross_section_on_20210722-1200_{0}-{1}'.format(
    models_list[0], models_list[1])
            
ax[0].set_title(plot_title)
#fig.suptitle(plot_title)

if save_plot:
    tools.save_figure(figname, save_folder)
