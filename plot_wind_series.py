#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021

Fonctionnement:
    Seule plusieurs sections ont besoin d'être remplies.    
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import tools
from windrose import WindroseAxes
import metpy.calc as mpcalc
from metpy.units import units
import global_variables as gv

################%% Independant Parameters (TO FILL IN):
    
site = 'cendrosa'

#domain to consider for simu files: 1 or 2
#domain_nb = 2

ilevel = 3   #0 is Halo, 1->2m, 2->6.12m, 3->10.49m

save_plot = False 
#save_folder = './figures/winds/'.format(domain_nb)
save_folder = './figures/wind_series/'
figsize = (10, 6) #small for presentation: (6,6), big: (15,9)

models = [
        'irr_d1', 
        'std_d1',
        'irrlagrip30_d1',
#        'irr_d2_old', 
#        'std_d2_old',
#        'irr_d2', 
#        'std_d2', 
         ]

errors_computation = True

########################################################


simu_folders = {key:gv.simu_folders[key] for key in models}

father_folder = '/cnrm/surface/lunelt/NO_SAVE/nc_out/'

date = '2021-07'

#colordict = {'irr_d2': 'g', 'irr_d1': 'g', 
#             'std_d2': 'r', 'std_d1': 'r', 
#             'obs': 'k'}
#styledict = {'irr_d2': '-', 'irr_d1': ':', 
#             'std_d2': '-', 'std_d1': ':', 
#             'obs': 'k'}
colordict = {'irr_d2': 'g', 
             'std_d2': 'r',
             'irr_d1': 'g', 
             'std_d1': 'r', 
             'irrlagrip30_d1': 'y',
             'irr_d2_old': 'g', 
             'std_d2_old': 'r', 
             'obs': 'k'}
styledict = {'irr_d2': '-', 
             'std_d2': '-',
             'irr_d1': '--', 
             'std_d1': '--', 
             'irrlagrip30_d1': '--',
             'irr_d2_old': ':', 
             'std_d2_old': ':', 
             'obs': '-'}

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
    
    ax1.plot(dates, ws, 
             label='Wind Speed ' + label, 
             **kwargs)
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
#             linestyle = ':',
#             marker='*', markersize=0.5,
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
    varname_obs_ws = 'ws_2'
    varname_obs_wd = 'wd_2'
    datafolder = '/cnrm/surface/lunelt/data_LIAISE/cendrosa/30min/'
    filename_prefix = 'LIAISE_LA-CENDROSA_CNRM_MTO-FLUX-30MIN_L2_'
    in_filenames_obs = filename_prefix + date
elif site == 'preixana':
    varname_obs_ws = 'ws_2'
    varname_obs_wd = 'wd_2'
    datafolder = '/cnrm/surface/lunelt/data_LIAISE/preixana/30min/'
    filename_prefix = 'LIAISE_PREIXANA_CNRM_MTO-FLUX-30MIN_L2_'
    in_filenames_obs = filename_prefix + date
elif site == 'elsplans':
    varname_obs_ws = 'UTOT_10m'
    varname_obs_wd = 'DIR_10m'
    freq = '30'  # '5' min or '30'min
    datafolder = '/cnrm/surface/lunelt/data_LIAISE/elsplans/mat_50m/{0}min/'.format(freq)
    filename_prefix = 'LIAISE_'
    date = date.replace('-', '')
    in_filenames_obs = filename_prefix + date
#    varname_sim_suffix = '_ISBA'  # or P7, but already represents 63% of _ISBA
elif site == 'irta-corn':
    varname_obs_ws = 'WS'
    varname_obs_wd = 'WD'
    datafolder = '/cnrm/surface/lunelt/data_LIAISE/irta-corn/seb/'
    in_filenames_obs = 'LIAISE_IRTA-CORN_UIB_SEB-10MIN_L2.nc'
    ilevel = 1  # because measurement were made at 3m AGL = 1m above maize
#    raise ValueError('Site name not known')
    
lat = gv.sites[site]['lat']
lon = gv.sites[site]['lon']


#%% OBS: CONCATENATE AND LOAD

if site == 'irta-corn':
    out_filename_obs = in_filenames_obs
    dat_to_nc = 'uib'
elif site == 'elsplans':
    out_filename_obs = 'CAT_' + date + filename_prefix + '.nc'
    dat_to_nc='ukmo'
else:
    out_filename_obs = 'CAT_' + date + filename_prefix + '.nc'
    dat_to_nc=None

# Concatenate multiple days
tools.concat_obs_files(datafolder, in_filenames_obs, out_filename_obs,
                       dat_to_nc=dat_to_nc)

# Load data:
obs = xr.open_dataset(datafolder + out_filename_obs)

#fig_speed=plt.figure()
#fig_dir=plt.figure()

fig, ax = plt.subplots(2, 1, figsize=figsize)

#%% PLOT OBS


ws_obs = obs[varname_obs_ws]
wd_obs = obs[varname_obs_wd]

if site == 'elsplans':
    dati_arr = pd.date_range(
            start=obs.time.min().values, 
#            start=pd.Timestamp('20210702-0000'),
            periods=len(obs[varname_obs_ws]), 
            freq='30T')
    #turn outliers into NaN
#    ws_obs_filtered = ws_obs.where(
#            (ws_obs-ws_obs.mean()) < (3*ws_obs.std()), 
#             np.nan)
    ws_obs_filtered = ws_obs.where(
            (ws_obs-ws_obs.mean()) < np.nanpercentile(ws_obs.data, 96), 
             np.nan)
else:
    ws_obs_filtered = ws_obs  # no filtering
    dati_arr = ws_obs.time
    
ax[0].plot(dati_arr, ws_obs_filtered, 
         label='obs_' + varname_obs_ws,
         color=colordict['obs'],
         linewidth=1)
ax[1].plot(dati_arr, wd_obs, 
         label='obs_' + varname_obs_wd,
         color=colordict['obs'],
         linewidth=1)

#start_date = pd.Timestamp('20210721-0000')
#end_date = pd.Timestamp('20210723-0000')
#ax[0].set_xlim(start_date, end_date)
#ax[1].set_xlim(start_date, end_date)

#plot_wind_speed(ws_obs,
#                start_date=start_date, end_date=end_date, fig=ax[0],
##                label='obs_'+ ws_obs.long_name[-6:],
#                label='obs_'+ varname_obs_ws,
#                color=colordict['obs'])
#plot_wind_dir(wd_obs,
#              start_date=start_date, end_date=end_date, fig=ax[1],
##              label='obs_'+ wd_obs.long_name[-6:],
#                label='obs_'+ varname_obs_wd,
#              color=colordict['obs'])


#plot_windrose(ws_obs, wd_obs, 
#              start_date=start_date, end_date=end_date
#              )

#%% SIMU:

diff = {}
rmse = {}
bias = {}
obs_sorted = {}
sim_sorted = {}

varname_sim = 'UT,VT'

for model in simu_folders:
    in_filenames_sim = gv.format_filename_simu[model]
    out_filename_sim = 'LIAIS.{0}.{1}.nc'.format(
            in_filenames_sim[6], varname_sim)

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
    try:
        start = ds1.time.data[0]
    except AttributeError:    
#        start = np.datetime64('2021-07-14T01:00')
        start = np.datetime64('2021-07-21T01:00')
    dati_arr = np.array([start + np.timedelta64(i, 'h') for i in np.arange(0, ut_md.shape[0])])

    # PLOT d1
    if len(ut_md.shape) == 5:
        ut_1d = ut_md[:, :, ilevel, index_lat, index_lon].data #1st index is time, 2nd is ?, 3rd is Z,..
        vt_1d = vt_md[:, :, ilevel, index_lat, index_lon].data
    elif len(ut_md.shape) == 4:
        ut_1d = ut_md[:, ilevel ,index_lat, index_lon].data #1st index is time, 2nd is ?, 3rd is Z,..
        vt_1d = vt_md[:, ilevel ,index_lat, index_lon].data
    elif len(ut_md.shape) == 3:
        var_1d = ut_md.data[:, index_lat, index_lon]
    
    #computation of windspeed and  winddirection following ut and vt
    ws = mpcalc.wind_speed(ut_1d * units.meter_per_second, 
                           vt_1d * units.meter_per_second)
    wd = mpcalc.wind_direction(ut_1d * units.meter_per_second,
                               vt_1d * units.meter_per_second)
    
    plot_wind_speed(ws, dates=dati_arr, fig=ax[0], 
                    color=colordict[model],
                    linestyle=styledict[model],
                    label=model +'_l'+str(ilevel), 
                    )
    plot_wind_dir(wd, dates=dati_arr, fig=ax[1], 
                  color=colordict[model],
                  linestyle=styledict[model],
                  label=model +'_l'+str(ilevel), 
                  )
    
    if errors_computation:
        ## Errors computation
        obs_sorted[model] = []
        sim_sorted[model] = []
        for i, date in enumerate(dati_arr):
            val = ws_obs_filtered.where(ws_obs.time == date, drop=True).data
            if len(val) != 0:
                sim_sorted[model].append(ws[i])
                obs_sorted[model].append(float(val))
        
        diff[model] = np.array(sim_sorted[model]) - np.array(obs_sorted[model])
        # compute bias and rmse, and keep values with 3 significant figures
        bias[model] = float('%.3g' % np.nanmean(diff[model]))
#        rmse[model] = np.sqrt(np.nanmean((np.array(obs_sorted[model]) - np.array(sim_sorted[model]))**2))
        rmse[model] = float('%.3g' % np.sqrt(np.nanmean(diff[model]**2)))

    #fig = plt.figure()
#    ax = plt.gca()
    #    ax.set_ylabel(ylabel)
    #ax.set_ylim([0, 0.4])
    
    #    ax.set_xlim([np.min(obs.time), np.max(obs.time)])
#    ax.set_xlim([np.min(dati_arr), np.max(dati_arr)])
    
#    plt.plot(dati_arr, var_1d, 
#             label='simu_{0}'.format(model))

plot_title = 'wind at {0}'.format(site)

fig.suptitle(plot_title)
ax[0].legend(loc='upper right')
ax[1].legend(loc='lower right')
ax[0].grid(visible=True, axis='both')
ax[1].grid(visible=True, axis='both')

# add grey zones for night
days = np.arange(1,30)
for day in days:
    # zfill(2) allows to have figures with two digits
    sunrise = pd.Timestamp('202107{0}-1930'.format(str(day).zfill(2)))
    sunset = pd.Timestamp('202107{0}-0500'.format(str(day+1).zfill(2)))
    ax[0].axvspan(sunset, sunrise, ymin=0, ymax=1, 
               color = '0.9'  #'1'=white, '0'=black, '0.8'=light gray
               )
    ax[1].axvspan(sunset, sunrise, ymin=0, ymax=1, 
               color = '0.9'  #'1'=white, '0'=black, '0.8'=light gray
               )

# add errors on graph
if errors_computation:
    ax[0].text(.01, .95, 'RMSE: {0}'.format(rmse), 
             ha='left', va='top', 
#             transform=ax.transAxes
             )
    ax[0].text(.01, .99, 'Bias: {0}'.format(bias), 
             ha='left', va='top', 
#             transform=ax.transAxes
             )
    plt.legend(loc='upper right')
else:
    plt.legend(loc='best')

#%% Save figure

if save_plot:
    tools.save_figure(plot_title, save_folder)
