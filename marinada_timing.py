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
import metpy.calc as mcalc
from metpy.units import units
import global_variables as gv
import pandas as pd

########## Independant parameters ###############

# Simulation to show: 'irr' or 'std'
models = [
#        'irr_d1',
#        'irrswi1_d1_old',
#        'irrswi1_d1',
        'irrlagrip30_d2',
        'irrlagrip30_d1',
#        'std_d1',
        ]
#domain to consider: 1 or 2
domain_nb = 1

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
#        '20210721',
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

save_plot = True
save_folder = f'figures/marinada_front/'

###########################################
        
        
colordict = {'irr_d2': 'g', 
             'std_d2': 'r',
             'irr_d1': 'g', 
             'std_d1': 'r', 
             'irrlagrip30_d1': 'orange',
             'irrlagrip30_d2': 'yellow',
             'irrswi1_d1': 'b',
             'irr_d2_old': 'g', 
             'std_d2_old': 'r', 
             'obs': 'k'}
linestyledict = {
        '20210715': (0, (2, 8)),
        '20210716': (0, (3, 7)), 
        '20210717': (0, (4, 6)), 
        '20210718': (0, (5, 5)), 
        '20210719': (0, (1, 10)), 
        '20210720': (0, (3, 8, 1, 8)), 
        '20210721':( 0, (3, 5, 1, 5)), 
        '20210722': (0, (3, 2, 1, 2)), 
        }
        

end = (gv.whole[site_end]['lat'], gv.whole[site_end]['lon'])
start = (gv.whole[site_start]['lat'], gv.whole[site_start]['lon'])

if gv.whole[site_start]['lon'] > gv.whole[site_end]['lon']:
    raise ValueError("site_start must be west of site_end")


#%% PROCESS DATA TO FIND MAX CONVERGENCE

for model in models:
    print('-----------------------')
    print(model)
    print('-----------------------')
    
    res_filt_dict[model] = {}
    for date in date_list:
        freq='1H'  #'1H', 30T, 10T
        datetime_range = pd.date_range(pd.Timestamp(f'{date}-0700'), 
                                       pd.Timestamp(f'{date}-2300'), 
                                       freq=freq,
                                       )
        for datime in datetime_range:
            # compute date and time
    #        datime = pd.Timestamp(date) + pd.Timedelta(hour, 'h')
            print(datime)
            
            # FIND and LOAD corresponding file
            filename = tools.get_simu_filepath(model, datime, 
                                               file_suffix='dg', 
                                               out_suffix='')
            print(filename)
            ds = xr.open_dataset(filename)
            
            # pre-processing of data
            ds_sub = tools.subset_ds(ds, 
                              lat_range = [start[0], end[0]], 
                              lon_range = [start[1], end[1]],
                              nb_indices_exterior=5)    
            ds_subcen = tools.center_uvw(ds_sub)
            
            # Computation of other diagnostic variable
            ds_subcen['DIV'] = mcalc.divergence(ds_subcen['UT'], ds_subcen['VT'])
            ds_subcen['WS'], ds_subcen['WD'] = tools.calc_ws_wd(ds_subcen['UT'], ds_subcen['VT'])
            
            data_subcenred = ds_subcen[['THT', 'RVT', 'UT', 'VT', 'WT', 
    #                                   'ZS',
    #                                   'TEMP', 'PRES', 
    #                                   'DENS', 
                                       'DIV',
                                       'WS', 'WD',
    #                                   surf_var
                                       ]]
            data = data_subcenred
            
            
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
    #        max_ZS = data['ZS'].max()
            max_ZS = 2500
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
        
        # --- filter to keep a constantly progressing front ---
    #    while not df_filt.i_sect.is_monotonic_decreasing:
#        df_filt_monotonic = pd.DataFrame()
#        for i in range(len(df_filt.index)):
#            if i == 0:
#                pass
#            else:
#                i_sect_diff = df_filt.iloc[i-1].i_sect - df_filt.iloc[i].i_sect
#    #            print(i_sect_diff)
#                if i_sect_diff > 0:
#                    df_filt_monotonic = df_filt_monotonic.append(df_filt.iloc[i])
    #                df_filt_monotonic = pd.concat([df_filt_monotonic, df_filt.iloc[i]])
    #                df_filt.iloc[i]['DIV'] = 0
    #                print(df_filt_monotonic)
        # remove values
    #    df_filt = df_filt[df_filt['DIV'] == 0]
        
        # store filtered df
#        res_filt_dict[model][date] = df_filt_monotonic
        res_filt_dict[model][date] = df_filt


#%% PICKLE - LOAD or SAVE
import pickle

## SAVE
filehandler = open(f'{save_folder}/marinada_timing_res_dict_d1d2.pickle', 'wb')
pickle.dump(res_filt_dict, filehandler)
filehandler.close()

# LOAD
#filehandler = open(f'{save_folder}/marinada_timing_res_dict.pickle', 'rb')
#res_filt_dict = pickle.load(filehandler)
#filehandler.close()

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

#%% PLOT PROGRESSION of marinada
plt.figure(figsize=(7,7))

for model in res_filt_dict:

    for date in date_list:
        df_filt = res_filt_dict[model][date]
        df_filt['hour_float'] = df_filt.index.hour + df_filt.index.minute/60
        
        # --- filter to keep a constantly progressing front ---
#        marinada_started = False
#        marinada_ended = False
#        for i in range(len(df_filt.index)-1):
#            # look for start of the marinada, set to NaN values before
#            if marinada_started is False:
#                if df_filt.i_sect[i+1] >= df_filt.i_sect[i]:
#                    df_filt.hour_float[i] = np.nan
#                else:
#                    marinada_started = True
#            # look for end of the marinada, set to NaN values at the limit
#            elif marinada_ended is False:
#                if df_filt.i_sect[i+1] > df_filt.i_sect[i]:
#                    df_filt.hour_float[i+1] = np.nan
#                    marinada_ended = True
#            # marinada ended - all values after set to NaN
#            else:
#                df_filt.hour_float[i+1] = np.nan
#        # remove NaN values ()
#        df_filt = df_filt.dropna()
    
        plt.plot(df_filt.dist_coast, df_filt.hour_float,
                 label=f'{model}-{date}',
                 color=colordict[model],
                 linestyle=linestyledict[date],
                 )
    
# add line of main sites:
distances_dict = {'torredembarra': 0.5,
                  'coll_lilla': 28,
                  'serra_tallat': 43.8,
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
### Mean wind

#dict_mean_wind = {}
#
#for date in res_dict:
#    dict_mean_wind[date] = {}
#    
#    for datime in res_dict[date]['section']:
#        hour = datime.hour
#    #datime = pd.Timestamp('20210716-1800')
#        section = res_dict[date]['section'][datime]
#        mean_wind_synop = section['WPROJ'][1:10, 200:210].mean()
#    #    res_dict[date]['section'][datime]['mean_wind_synop'] = mean_wind_synop
#        dict_mean_wind[date][hour] = float(mean_wind_synop)
#    
#df_mean_wind = pd.DataFrame(dict_mean_wind)
## get wind mean synop at 12UTC
#hour=12
#df_mean_wind.loc[hour]

### Arrival at serra del tallat
#arrival_dict = {}
#
#for date in res_filt_dict:
#    arrival_dict[date] = {}
#    front_dict = res_filt_dict[date]
#    for site in distances_dict:
#        arrival_hour = np.interp(distances_dict[site], 
#                                      front_dict['dist_coast'], 
#                                      front_dict['hour'],
#                                      left=np.nan,
#                                      right=np.nan)
#        arrival_dict[date][site] = arrival_hour
#        
#df_arrival = pd.DataFrame(arrival_dict)


### Correlation analysis from dataframe 

#
##df_mean_wind.loc[hour]
#df_summary = pd.concat([
#        df_mean_wind.loc[hour],  
#        df_arrival.loc['tossal_gros']
#        ], 
#        axis=1,
#        keys=['mean_wind_synop','arrival_tossal_gros'])
#            
#df_corr = df_summary.corr()


#%% PLOT FRONT SPEED

#plt.figure(figsize=(7,7))
## plot front speed of marinada
#for model in models:
#    for date in date_list:
#        df_filt = res_filt_dict[model][date]
#        
#        # compute front speed
#        ls_dist = []
#        ls_front_speed = []
#        ls_time_mean = []
#        
#        for i in range(len(df_filt.index)-1):
#            dist_coast = (df_filt.dist_coast[i+1] + df_filt.dist_coast[i])/2
#            time_mean = (df_filt.hour_float[i+1] + df_filt.hour_float[i])/2       
#            front_speed = (df_filt.dist_coast[i+1] - df_filt.dist_coast[i]) / \
#                    (df_filt.hour_float[i+1] - df_filt.hour_float[i])   # in km/h
#            
#            ls_dist.append(dist_coast)
#            ls_front_speed.append(front_speed/3.6)  # convert to m/s
#            ls_time_mean.append(time_mean)
#    #        print('hour: ', time_mean)
#    #        print("front_speed = ", front_speed)
#        
#        df_front_speed = pd.DataFrame(list(zip(ls_dist, ls_front_speed, ls_time_mean)),
#                                      columns=['dist_coast', 'front_speed', 'hour'])
#        
#        plt.plot(df_front_speed.dist_coast, df_front_speed.front_speed,
#                 label=f'{model}-{date}')
#
#
## add line of main sites:
#distances_dict = {'torredembarra': 0.5,
#                  'coll_lilla': 28,
#                  'serra_tallat': 43.8,
#                  'els_plans': 58.75,
#                  'cendrosa': 72.81,
#                  }
#for site in distances_dict:
#    plt.vlines(distances_dict[site], 0, 20,
#               colors='k', linestyle='--')
#    plt.text(distances_dict[site]+1, 15, site, rotation=90)
#
##plt.xlim(0, df_filt.dist_coast.max())
#plt.xlim(0, 100)
#
#plt.xlabel('distance to coast [km]')
#plt.ylabel('front speed [m/s]')
#plt.legend()
#plt.grid(axis='y')

#%%
plot_title = f'marinada front progression - {date_list}'

if save_plot:
    tools.save_figure(plot_title, save_folder)
