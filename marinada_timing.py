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
import pandas as pd

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
# Set type of wind representation: 'verti_proj' or 'horiz'
wind_visu = 'verti_proj'
# Datetime
init_res_dict = True
date_list = [
#        '20210715', 
        '20210716', 
#        '20210717', 
#        '20210718', 
#        '20210719', 
#        '20210720', 
        '20210721',
#        '20210722',
        ]
# altitude ASL or height AGL: 'asl' or 'agl'
alti_type = 'asl'
# maximum level (height AGL) to plot
toplevel = 1000

# where to place the cross section
nb_points_beyond = 10
site_end = 'torredembarra'
site_start = 'cendrosa'

var_list = ['i_sect', 'RVT', 'RVT_GRAD_HORIZ', 'THT', 'DIV', 'WPROJ', 'WT',
            'WPROJ_GRAD_HORIZ']

#debug = True

# INIT of RES DICT
#if not debug:
if init_res_dict:
    res_dict = {}
    res_filt_dict = {}
    
for date in date_list:
    res_dict[date] = {}
    for var in var_list:
        res_dict[date][var] = {}
        res_dict[date]['section'] = {}

    

###########################################

end = (gv.whole[site_end]['lat'], gv.whole[site_end]['lon'])
start = (gv.whole[site_start]['lat'], gv.whole[site_start]['lon'])

if gv.whole[site_start]['lon'] > gv.whole[site_end]['lon']:
    raise ValueError("site_start must be west of site_end")

for date in date_list:
    for hour in np.arange(7, 24):
        # compute date and time
        datime = pd.Timestamp(date) + pd.Timedelta(hour, 'h')
        print(datime)
        
        # FIND and LOAD corresponding file
        filename = tools.get_simu_filename(model, datime)
        data_perso = xr.open_dataset(filename)
        
        # Computation of other diagnostic variable
        data_perso['DENS'] = mcalc.density(
            data_perso['PRES']*units.hectopascal,
            data_perso['TEMP']*units.celsius, 
            data_perso['RVT']*units.gram/units.gram)
        
        data_perso = tools.center_uvw(data_perso)
        data_perso['DIV'] = mcalc.divergence(data_perso['UT'], data_perso['VT'])
        data_perso['WS'], data_perso['WD'] = tools.calc_ws_wd(data_perso['UT'], data_perso['VT'])
        
        data_reduced = data_perso[['THT', 'RVT', 'UT', 'VT', 'WT', 'ZS',
                                   'TEMP', 'PRES', 
                                   'DENS', 'DIV',
                                   'WS', 'WD',
                                   surf_var]]
        data = data_reduced
        
        
        #%% CREATE SECTION LINE
        
        line = tools.line_coords(data, start, end, 
                                 nb_indices_exterior=nb_points_beyond,
                                 verbose=False)
        ni_range = line['ni_range']
        nj_range = line['nj_range']
        slope = line['slope']
        
        # Compute projection of horiz winds into the line
        if slope == 'vertical':
            angle = np.pi/2
        else:
            angle = np.arctan(slope)  
        data['WPROJ'] = tools.windvec_verti_proj(data['UT'], data['VT'], 
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
            
    #    print('section interpolation on {0} points (~1sec/pt)'.format(len(ni_range)))
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
        
        # DIAG in section:
        section_ds['RVT_GRAD_HORIZ'] = (section_ds['RVT'].dims,  # dims
                                        np.gradient(section_ds['RVT'], axis=0)  # data
                                        )
        section_ds['WPROJ_GRAD_HORIZ'] = (section_ds['WPROJ'].dims,  # dims
                                        np.gradient(section_ds['WPROJ'], axis=0)  # data
                                        )
        
        # get location of minimum for DIV - Marinada front:
        # look at given level (4 => 50m agl)
        section_ds_ilevel = section_ds.isel(level=4)
        res_dict[date]['section'][datime] = section_ds
        
        # keep pts where wind dir is toward inland
        eastwind_pts = section_ds_ilevel.where(section_ds_ilevel['WPROJ']<0,
                                               drop=False)
        # keep the point where convergence is the biggest
        location_min = eastwind_pts.where(
                section_ds_ilevel['DIV'] == eastwind_pts['DIV'].min(), 
                drop=True)
        
        # retrieve data
        for var in var_list:
            if len(location_min[var]) == 0:
                res_dict[date][var][datime] = np.nan
            else:
                res_dict[date][var][datime] = float(location_min[var])
    

    df = pd.DataFrame(res_dict[date])
    
    # add some DIAGs
    coast_index = len(section_ds.i_sect) - nb_points_beyond
    df['dist_coast']=abs((df['i_sect']-coast_index)*line['nij_step'])/1000  # in km
    df['time'] = [datime.time() for datime in df.index]
    
    # FILTERING
    df_filt = df[df['DIV'] < -0.0005]
    
#    while not df_filt.i_sect.is_monotonic_decreasing:
    df_filt_monotonic = pd.DataFrame()
    for i in range(len(df_filt.index)):
        if i == 0:
            pass
        else:
            i_sect_diff = df_filt.iloc[i-1].i_sect - df_filt.iloc[i].i_sect
#            print(i_sect_diff)
            if i_sect_diff > 0:
                df_filt_monotonic = df_filt_monotonic.append(df_filt.iloc[i])
#                df_filt_monotonic = pd.concat([df_filt_monotonic, df_filt.iloc[i]])
#                df_filt.iloc[i]['DIV'] = 0
#                print(df_filt_monotonic)
    # remove values
#    df_filt = df_filt[df_filt['DIV'] == 0]
    
    # store filtered df
    res_filt_dict[date] = df_filt_monotonic

#%%

for date in res_filt_dict:
    res_filt_dict[date]['hour'] = [datime.hour for datime in res_filt_dict[date].index]
#    # convert from m to km
#    df_filt['dist_coast'] = df_filt['dist_coast']/1000

plt.figure(figsize=(7,7))
# plot progression of marinada
for date in date_list:
    df_filt = res_filt_dict[date]

    plt.plot(df_filt.dist_coast, df_filt.hour,
             label=date)


# add line of main sites:
distances_dict = {'torredembarra': 0.5,
                  'coll_lilla': 28,
                  'tossal_gros': 43.8,
                  'els_plans': 58.75,
                  'cendrosa': 72.81,
                  }
for site in distances_dict:
    plt.vlines(distances_dict[site], 6, 24,
               colors='k', linestyle='--')  # coll_lilla
    plt.text(distances_dict[site]+1, 6, site, rotation=90)

#plt.xlim(0, df_filt.dist_coast.max())
plt.xlim(0, 100)

plt.xlabel('distance to coast [km]')
plt.ylabel('hour UTC')
plt.legend()
plt.grid(axis='y')

#%% POST-PROCESSING

dict_mean_wind = {}

for date in res_dict:
    dict_mean_wind[date] = {}
    
    for datime in res_dict[date]['section']:
        hour = datime.hour
    #datime = pd.Timestamp('20210716-1800')
        section = res_dict[date]['section'][datime]
        mean_wind_synop = section['WPROJ'][1:10, 200:210].mean()
    #    res_dict[date]['section'][datime]['mean_wind_synop'] = mean_wind_synop
        dict_mean_wind[date][hour] = float(mean_wind_synop)
    
df_mean_wind = pd.DataFrame(dict_mean_wind)
# get wind mean synop at 12UTC
hour=12
df_mean_wind.loc[hour]

#%% ARRIVAL at TOSSAL GROS
arrival_dict = {}

for date in res_filt_dict:
    arrival_dict[date] = {}
    front_dict = res_filt_dict[date]
    for site in distances_dict:
        arrival_hour = np.interp(distances_dict[site], 
                                      front_dict['dist_coast'], 
                                      front_dict['hour'],
                                      left=np.nan,
                                      right=np.nan)
        arrival_dict[date][site] = arrival_hour
        
df_arrival = pd.DataFrame(arrival_dict)


#%% DATAFRAME for CORRELATION ANALYSIS


#df_mean_wind.loc[hour]
df_summary = pd.concat([
        df_mean_wind.loc[hour],  
        df_arrival.loc['tossal_gros']
        ], 
        axis=1,
        keys=['mean_wind_synop','arrival_tossal_gros'])
            
df_corr = df_summary.corr()


#%% Calculate wind speed

df_filt = res_filt_dict['20210716']

ws_dict = {}
for i in range(len(df_filt.hour)-1):
    ws = (df_filt.iloc[i+1].dist_coast - df_filt.iloc[i].dist_coast) / \
            (df_filt.iloc[i+1].hour - df_filt.iloc[i].hour)   # in km/h
    ws_dict[(df_filt.iloc[i+1].hour + df_filt.iloc[i].hour)/2] = ws/3.6
    print(ws/3.6)

#%% PLOT
## create figure
#fig, ax = plt.subplots(2, figsize=figsize,
#                       gridspec_kw={'height_ratios': [20, 1]})
#
### --- Subplot of section, i.e. the main plot ----
##get maximum height of relief in cross-section
#max_ZS = section_ds['ZS'].max()
#
## remove top layers of troposphere
#section_ds = section_ds.where(section_ds.level<(level_range.max()), drop=True)
#
### --- Adapt to alti_type ------
##create grid mesh (eq. to X)
#X = np.meshgrid(section_ds.i_sect, section_ds.level)[0]
#Xmesh = xr.DataArray(X, dims=['level', 'i_sect'])
##create alti mesh (eq. to Y)
#if alti_type == 'asl': 
#    #compute altitude ASL from height AGL, and transpose (eq. Y)
#    alti = section_ds.ZS[:, 0] + section_ds.level
#    alti = alti.T
#    #for plot
#    ylabel = 'altitude ASL [m]'
#elif alti_type == 'agl':
#    #create grid mesh (eq. Y)
#    alti = np.meshgrid(section_ds.i_sect, section_ds.level)[1]
#    alti = xr.DataArray(alti, dims=['level', 'i_sect'])
#    #for plot
#    ylabel = 'height AGL [m]'
#    
#
#### 1.1. Color map (pcolor or contourf)
#data1 = section_ds[varname_colormap]
##cm = ax[0].pcolormesh(Xmesh,
##                      alti,
##                      data1.T, 
##                      cmap='rainbow',
##                      vmin=305, vmax=315)
#cm = ax[0].contourf(Xmesh,
#                    alti,
#                    data1.T, 
#                    cmap=colormap,  # 'OrRd', 'coolwarm'
##                    levels=np.linspace(298, 315, 18),  # to keep always same colorbar limits
##                    levels=np.linspace(vmin, vmax, vmax-vmin+1),  # to keep always 1K per color variation
#                    levels=np.linspace(vmin, vmax, 20),
##                    levels=20,
#                    extend = 'both',  #highlights the min and max in different color
#                    vmin=vmin, vmax=vmax,  # for THT
##                    vmin=None, vmax=None,  # for adaptative colormap
##                    vmin=800, vmax=1000,  # for PRES
#                    )
##manage colorbar
#divider = make_axes_locatable(ax[0])
#cax = divider.append_axes('right', size='2%', pad=0.05)
#cbar = fig.colorbar(cm, cax=cax, orientation='vertical')
#cbar.set_label('theta [K]')
#
#
#### 1.2. Contour map
#data2 = section_ds[varname_contourmap]  # x1000 to get it in g/kg
#cont = ax[0].contour(Xmesh,
#                     alti,
#                     data2.T,
#                     cmap='viridis'  #viridis is default
#                     )
#ax[0].clabel(cont, cont.levels, inline=True, fontsize=10)
##labels = ['l1','l2','l3','l4','l5','l6','l7','l8','l9']
##for i in range(len(labels)):
##    cont.collections[i].set_label(labels[i])
##ax[0].legend()
#
#### 1.3. Winds
#if wind_visu == 'horiz':            # 2.1 winds - flat direction and force
#    ax[0].barbs(
#            #Note that X & alti have dimensions reversed
#            Xmesh[::skip_barbs_y, ::skip_barbs_x], 
#            alti[::skip_barbs_y, ::skip_barbs_x], 
#            #Here dimensions are in the proper order
#            section_ds['UT'][::skip_barbs_x, ::skip_barbs_y].T, 
#            section_ds['VT'][::skip_barbs_x, ::skip_barbs_y].T, 
#            pivot='middle',
#            length=5*arrow_size,     #length of barbs
#            sizes={
#    #              'spacing':1, 'height':1, 'width':1,
#                    'emptybarb':0.01},
#            barb_increments=barb_size_increments[barb_size_option] # [kts], 1.94kt = 1m/s
#            )
#    ax[0].annotate(barb_size_description[barb_size_option],
#                   xy=(0.1, 0.9),
#                   xycoords='subfigure fraction'
#                   )
#elif wind_visu == 'verti_proj':     # 2.2  winds - verti and projected wind
#    Q = ax[0].quiver(
#            #Note that X & alti have dimensions reversed
#            Xmesh[::skip_barbs_y, ::skip_barbs_x], 
#            alti[::skip_barbs_y, ::skip_barbs_x], 
#            #Here dimensions are in the proper order
#            section_ds['PROJ'][::skip_barbs_x, ::skip_barbs_y].T, 
#            section_ds['WT'][::skip_barbs_x, ::skip_barbs_y].T, 
#            pivot='middle',
#            scale=150/arrow_size,  # arrows scale, if higher, smaller arrows
#            )
#    #add arrow scale in top-right corner
#    u_max = abs(section_ds['PROJ'][::skip_barbs_x, ::skip_barbs_y]).max()
#    ax[0].quiverkey(Q, 0.8, 0.9, 
#                    U=u_max, 
#                    label=str((np.round(u_max, decimals=1)).data) + 'm/s', 
#                    labelpos='E',
#                    coordinates='figure')
#
#
## x-axis with sites names
#ax[0].set_xticks(list(abscisse_sites.keys()))
#ax[0].set_xticklabels(list(abscisse_sites.values()), 
#                   rotation=0, fontsize=12)
## x-axis with lat-lon values
##ax.set_xticks(data1.i_sect[::10])
##ax.set_xticklabels(abscisse_coords[::10], rotation=0, fontsize=9)
#
## set y limits (height ASL)
#if alti_type == 'asl':
#    min_ZS = section_ds['ZS'].min()
#    ax[0].set_ylim([min_ZS, max_ZS + toplevel])
#ax[0].set_ylabel(ylabel)
#
#
#### 2. Subplot of surface characteristic ---
#
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
#
##ax[1].set_xticks(ticks = data_soil.i_sect.values[::9],
##                 labels = (data_soil.i_sect.values * \
##                           line['nij_step']/1000)[::9].round(decimals=1)
##                 )
#labels_arr = np.arange(0,30,5)
#tick_pos = labels_arr/ (line['nij_step']/1000)
#ax[1].set_xticks(ticks = tick_pos,
#                 labels = labels_arr
#                 )
#ax[1].set_xlabel('distance [km]')
#
#ax[1].set_yticks([])
##ax[1].set_ylabel(surf_var)
#ax[1].set_ylabel('soil moisture')
#
### Global options
#plot_title = 'Cross section on {0}-{1}-{2}'.format(
#        wanted_date, model, wind_visu)
##plot_title = 'Cross section of ABL between irrigated and rainfed areas on July 22 at 12:00 - {0}'.format(
##        model)
##plot_title = 'Cross section on July 22 at 12:00 - {0}'.format(model)
#fig.suptitle(plot_title)
#
#if save_plot:
#    tools.save_figure(plot_title, save_folder)
