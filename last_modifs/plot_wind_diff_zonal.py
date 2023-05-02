#!/usr/bin/env python3
"""
@author: Tanguy LUNEL

Compute wind speed difference between irrig and std run - simu only.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import tools
from windrose import WindroseAxes
import metpy.calc as mpcalc
from metpy.units import units


################%% Independant Parameters (TO FILL IN):
# center of zone
site = 'preixana'
zone_sidelength_km = 10  #km

#domain to consider for simu files: 1 or 2
domain_nb = 2

ilevel = 3   #0 is Halo, 1->2m, 2->6.12m, 3->10.49m

save_plot = False 
save_folder = './figures/winds/diff/'.format(domain_nb)
#varname_sim = 'T2M_ISBA'      # simu varname to compare obs with
#leave None for automatic attribution
########################################################

simu_folders = {
        'irr': '2.13_irr_2021_22-27/', 
        'std': '1.11_ECOII_2021_ecmwf_22-27/'
         }
father_folder = '/cnrm/surface/lunelt/NO_SAVE/nc_out/'

date = '2021-07'

colordict = {'irr': 'g', 'std': 'orange', 'obs': 'k'}


def plot_wind_speed(ws, dates=None, start_date=None, end_date=None, 
                    fig=None, label='', **kwargs):
    """
    Required input:
        ws: Wind speeds (m/s)
    Optional Input:
        wsmax: Wind gust (m/s)
        fig: Use pre-existing figure to plot on
        plot_range: Data range for making figure (list of (min,max,step))
    """
    if dates is None:
        dates = ws.time.values

    if fig is None:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
    else:
        ax1=fig
    
    if start_date is not None:
        ax1.set_xlim(start_date, end_date)
    
    ax1.plot(dates, ws, label='Wind Speed ' + label, **kwargs)
    ax1.set_ylabel('Wind Speed (m/s)', multialignment='center')
    ax1.grid(visible=True, which='major', axis='y', color='k', linestyle='--',
             linewidth=0.5)

def plot_wind_dir(wd, dates=None, start_date=None, end_date=None, 
                  fig=None, label='', **kwargs):
    """
    Required input:
        wd: Wind direction (degrees)
    Optional Input:
        wsmax: Wind gust (m/s)
        fig: Use pre-existing figure to plot on
        plot_range: Data range for making figure (list of (min,max,step))
    """
    if dates is None:
        dates = wd.time.values

    if fig is None:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
    else:
        ax1=fig
    
    if start_date is not None:
        ax1.set_xlim(start_date, end_date)
        
    ax1.plot(dates, wd, 
#             linewidth=0.5, 
             linestyle = 'dashed',
             label='Wind Direction ' + label,
             **kwargs)
    ax1.set_ylabel('Wind Direction\n(degrees)', multialignment='center')
    ax1.set_ylim(0, 360)
    ax1.set_yticks(np.arange(0, 360, 90))
    ax1.set_yticklabels(['0 (N)', '90 (E)', '180 (S)', '270 (W)'])



def plot_windrose(ws, wd, start_date=None, end_date=None, fig=None, **kwargs):
    #keep only data between  start & end:
    if start_date is not None:
        ws = ws[ws.time > start_date]
        ws = ws[ws.time < end_date]
    if end_date is not None:
        wd = wd[wd.time > start_date]
        wd = wd[wd.time < end_date]
    
    ax = WindroseAxes.from_ax()
    ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='white', **kwargs)
    ax.set_legend()



#%% Dependant Parameters

if site == 'cendrosa':
    lat = 41.6925905
    lon = 0.9285671
    varname_obs_ws = 'ws_2'
    varname_obs_wd = 'wd_2'
    datafolder = \
        '/cnrm/surface/lunelt/data_LIAISE/cendrosa/30min/'
    filename_prefix = \
        'LIAISE_LA-CENDROSA_CNRM_MTO-FLUX-30MIN_L2_'
    in_filenames_obs = filename_prefix + date
elif site == 'preixana':
    lat = 41.59373 
    lon = 1.07250
    datafolder = \
        '/cnrm/surface/lunelt/data_LIAISE/preixana/30min/'
    filename_prefix = \
        'LIAISE_PREIXANA_CNRM_MTO-FLUX-30MIN_L2_'
    in_filenames_obs = filename_prefix + date
elif site == 'elsplans':
    lat = 41.590111 
    lon = 1.029363
    datafolder = '/cnrm/surface/lunelt/data_LIAISE/elsplans/mat_50m/5min/'
    filename_prefix = 'LIAISE_'
    date = date.replace('-', '')
    in_filenames_obs = filename_prefix + date

else:
    raise ValueError('Site name not known')

#if varname_sim is None:
#    varname_sim = varname_sim_prefix + varname_sim_suffix




start_date = pd.Timestamp('20210721-0000')
end_date = pd.Timestamp('20210725-0000')


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
#ax = plt.gca()

#%% SIMU:

varname_sim = 'UT,VT'
in_filenames_sim = 'LIAIS.{0}.SEG*.001.nc'.format(domain_nb)  # use of wildcard allowed
out_filename_sim = 'LIAIS.{0}.{1}.nc'.format(domain_nb, varname_sim)

ws_mean = {}

for model in simu_folders:
#model='std'
    datafolder = father_folder + simu_folders[model]
    
    #concatenate multiple days for 1 variable
    tools.concat_simu_files_1var(datafolder, varname_sim, 
                                 in_filenames_sim, out_filename_sim)
    
    ds1 = xr.open_dataset(datafolder + out_filename_sim)
    
    # find indices from lat,lon values 
    index_lat, index_lon = tools.indices_of_lat_lon(ds1, lat, lon)

    ut_md = ds1['UT']
    vt_md = ds1['VT']
    
    # Set time abscisse axis
    start = np.datetime64('2021-07-21T01:00')
    dati_arr = np.array([start + np.timedelta64(i, 'h') for i in np.arange(0, ut_md.shape[0])])


    if domain_nb == 2:
        zone_sidelength_i = zone_sidelength_km/0.4
    elif domain_nb == 1:
        zone_sidelength_i = zone_sidelength_km/2
    
    index_lat_min = index_lat - int(zone_sidelength_i/2)
    index_lat_max = index_lat + int(zone_sidelength_i/2)
    index_lon_min = index_lon - int(zone_sidelength_i/2)
    index_lon_max = index_lon + int(zone_sidelength_i/2)
    
    ut_zone = ut_md[:, :, ilevel, 
                    index_lat_min:index_lat_max, 
                    index_lon_min:index_lon_max].squeeze()
    vt_zone = vt_md[:, :, ilevel, 
                    index_lat_min:index_lat_max, 
                    index_lon_min:index_lon_max].squeeze()
        
    
    #computation of windspeed and  winddirection following ut and vt
    ws = mpcalc.wind_speed(ut_zone * units.meter_per_second, 
                           vt_zone * units.meter_per_second)
    
    ws_mean[model] = ws.mean(dim=['nj_u', 'ni_u'])
    
#    wd = mpcalc.wind_direction(ut_1d * units.meter_per_second,
#                               vt_1d * units.meter_per_second)
    
    plot_wind_speed(ws_mean[model], dates=dati_arr, fig=ax, 
                    label=model +'_l'+str(ilevel), color=colordict[model])
#    plot_wind_dir(wd, dates=dati_arr, fig=ax[1],
#                  label=model +'_l'+str(ilevel), color=colordict[model])


    
#plot DIFF:
ws_mean_diff = ws_mean['std'] - ws_mean['irr']
plot_wind_speed(ws_mean_diff, dates=dati_arr, fig=ax, 
                label= 'diff' +'_l'+str(ilevel), color='b')


    #fig = plt.figure()
#    ax = plt.gca()
    #    ax.set_ylabel(ylabel)
    #ax.set_ylim([0, 0.4])
    
    #    ax.set_xlim([np.min(obs.time), np.max(obs.time)])
#    ax.set_xlim([np.min(dati_arr), np.max(dati_arr)])

plot_title = 'mean wind speed around {0} ({1}km x {1}km)'.format(site, 
                                     zone_sidelength_km)

fig.suptitle(plot_title)
ax.legend(loc='upper right')
ax.grid(visible=True, axis='both')


#%% Save figure

if save_plot:
    tools.save_figure(plot_title, save_folder)
