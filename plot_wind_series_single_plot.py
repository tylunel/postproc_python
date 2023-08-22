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

save_plot = True 
#save_folder = './figures/winds/'.format(domain_nb)
save_folder = './figures/wind_series/'
figsize = (9, 4) #small for presentation: (6,6), big: (15,9)
plt.rcParams.update({'font.size': 11})

models = [
#        'irr_d1',
#        'irrlagrip30_d1',
#        'std_d1',
#        'irr_d2_old', 
#        'std_d2_old',
        'irr_d2', 
        'std_d2', 
         ]

varplot = 'ws'

errors_computation = False

########################################################

global_data_liaise = gv.global_data_liaise
simu_folders = {key:gv.simu_folders[key] for key in models}

#father_folder = '/cnrm/surface/lunelt/NO_SAVE/nc_out/'
global_simu_folder = gv.global_simu_folder

date = '2021-07'

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
    if ilevel == 3:
        varname_obs_ws = 'ws_2'
        varname_obs_wd = 'wd_2'
    elif ilevel == 10:
        varname_obs_ws = 'ws_4'
        varname_obs_wd = 'wd_4'
    datafolder = global_data_liaise + '/cendrosa/30min/'
    filename_prefix = 'LIAISE_LA-CENDROSA_CNRM_MTO-FLUX-30MIN_L2_'
    in_filenames_obs = filename_prefix + date
elif site == 'preixana':
    varname_obs_ws = 'ws_2'
    varname_obs_wd = 'wd_2'
    datafolder = global_data_liaise + '/preixana/30min/'
    filename_prefix = 'LIAISE_PREIXANA_CNRM_MTO-FLUX-30MIN_L2_'
    in_filenames_obs = filename_prefix + date
elif site == 'elsplans':
    if ilevel == 3:
        varname_obs_ws = 'UTOT_10m'
        varname_obs_wd = 'DIR_10m'
    elif ilevel == 10:
        varname_obs_ws = 'UTOT_50m'
        varname_obs_wd = 'DIR_50m'
    freq = '5'  # '5' min or '30'min
    datafolder = global_data_liaise + '/elsplans/mat_50m/{0}min_v4/'.format(freq)
    date = date.replace('-', '')
    in_filenames_obs = f'LIAISE_ELS-PLANS_UKMO_MTO-{str(freq).zfill(2)}MIN_L2_{date}'
elif site == 'irta-corn':
    varname_obs_ws = 'WS'
    varname_obs_wd = 'WD'
    datafolder = global_data_liaise + '/irta-corn/seb/'
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

fig, ax = plt.subplots(1, 1, figsize=figsize,
#                       gridspec_kw={'height_ratios': [4,1]}
                       )

#%% PLOT OBS

ws_obs = obs[varname_obs_ws]
wd_obs = obs[varname_obs_wd]

if site == 'elsplans':
    dati_arr_obs = pd.date_range(
            start=obs.time.min().values, 
#            start=pd.Timestamp('20210702-0000'),
            periods=len(obs[varname_obs_ws]), 
            freq=f'{freq}T')
    #turn outliers into NaN
#    ws_obs_filtered = ws_obs.where(
#            (ws_obs-ws_obs.mean()) < (3*ws_obs.std()), 
#             np.nan)
    ws_obs_filtered = ws_obs.where(
            (ws_obs-ws_obs.mean()) < np.nanpercentile(ws_obs.data, 94), 
             np.nan)
else:
    ws_obs_filtered = ws_obs  # no filtering
    dati_arr_obs = ws_obs.time

if varplot == 'ws': 
    ax.plot(dati_arr_obs, ws_obs_filtered, 
             label='obs_' + varname_obs_ws,
             color=colordict['obs'],
             linewidth=1)
elif varplot == 'wd':
    ax.plot(dati_arr_obs, wd_obs, 
             label='obs_' + varname_obs_wd,
             color=colordict['obs'],
             linewidth=1)


#%% SIMU:

diff = {}
rmse = {}
bias = {}
obs_sorted = {}
sim_sorted = {}

varname_sim_list = ['UT.OUT', 'VT.OUT']

for model in simu_folders:
    if model == 'irrlagrip30_d1':
        varname_sim_list = ['UT.OUT', 'VT.OUT']
    else:
        varname_sim_list = ['UT', 'VT']
    
    ds1 = tools.load_dataset(varname_sim_list, model, 
                             concat_if_not_existing=True)
    
    # find indices from lat,lon values 
    index_lat, index_lon = tools.indices_of_lat_lon(ds1, lat, lon)

    ut_md = ds1['UT'].squeeze()
    vt_md = ds1['VT'].squeeze()
    
    # FIX/SET time abscisse axis
    try:
        start = ds1.time.data[0]
    except IndexError:
        start = ds1.time.data
    except AttributeError:
        print('WARNING! datetime array is hard coded')
        start = np.datetime64('2021-07-21T01:00')
        
    if 'OUT' in varname_sim_list[0]:
        dati_arr_sim = np.array([start + np.timedelta64(i*30, 'm') for i in np.arange(0, ds1['record'].shape[0])])
    else:
        dati_arr_sim = np.array([start + np.timedelta64(i, 'h') for i in np.arange(0, ut_md.shape[0])])

    # PLOT d1
    if len(ut_md.shape) == 5:
        ut_1d = ut_md[:, :, ilevel, index_lat, index_lon].data #1st index is time, 2nd is ?, 3rd is Z,..
        vt_1d = vt_md[:, :, ilevel, index_lat, index_lon].data
    elif len(ut_md.shape) == 4:
        ut_1d = ut_md[:, ilevel ,index_lat, index_lon].data #1st index is time, 2nd is Z,..
        vt_1d = vt_md[:, ilevel ,index_lat, index_lon].data
    elif len(ut_md.shape) == 3:
        ut_1d = ut_md[:, index_lat, index_lon].data 
        vt_1d = vt_md[:, index_lat, index_lon].data
    
    #computation of windspeed and  winddirection following ut and vt
    ws = mpcalc.wind_speed(ut_1d * units.meter_per_second, 
                           vt_1d * units.meter_per_second)
    wd = mpcalc.wind_direction(ut_1d * units.meter_per_second,
                               vt_1d * units.meter_per_second)
    if varplot == 'ws':
        plot_wind_speed(ws, dates=dati_arr_sim, fig=ax, 
                        color=colordict[model],
                        linestyle=styledict[model],
                        label=model +'_l'+str(ilevel), 
                        )
    elif varplot == 'wd':
        plot_wind_dir(wd, dates=dati_arr_sim, fig=ax, 
                      color=colordict[model],
                      linestyle=styledict[model],
                      label=model +'_l'+str(ilevel), 
                      )
    
    if errors_computation:
        ## Errors computation
        obs_sorted[model] = {}
        sim_sorted[model] = {}
        diff[model] = {}
        bias[model] = {}
        rmse[model] = {}
        
        obs_sorted[model] = []
        sim_sorted[model] = []
        
        # interpolation
        if varplot == 'ws':
#                val = ws_obs_filtered.where(ws_obs.time == date, drop=True).data
            obs_data = ws_obs_filtered
            sim_data = np.array([elt.magnitude for elt in ws])  # remove pint.quantities units
        elif varplot == 'wd':
            obs_data = wd_obs
            sim_data = np.array([elt.magnitude for elt in wd])  # remove pint.quantities units
            
        dati_arr_sim_unix = np.float64(dati_arr_sim)/1e9
        dati_arr_obs_unix = np.float64(np.array(dati_arr_obs))/1e9
        obs_data_interp = np.interp(
                dati_arr_sim_unix, dati_arr_obs_unix, obs_data.values,
                left=np.nan, right=np.nan)
        
        diff[model] = sim_data - obs_data_interp
#            if wind_charac == 'direction':
#                diff[model][wind_charac] = np.mod(diff[model][wind_charac], 180)  # modulo funciotn
        # compute bias and rmse, and keep values with 3 significant figures
        bias[model] = float('%.3g' % np.nanmean(diff[model]))
#        rmse[model] = np.sqrt(np.nanmean((np.array(obs_sorted[model]) - np.array(sim_sorted[model]))**2))
        rmse[model] = float('%.3g' % np.sqrt(np.nanmean(diff[model]**2)))


plot_title = f'wind at {site} - level {ilevel}'

ax.set_title(plot_title)

#ax.set_xlim([np.min(dati_arr_sim), np.max(dati_arr_sim)])
ax.set_xlim([np.min(dati_arr_sim), 
             (np.max(dati_arr_sim) - pd.Timedelta(1, 'h'))])
ax.set_xlabel('time UTC')
plt.xticks(rotation=30)
ax.grid(visible=True, axis='both')

# add grey zones for night
days = np.arange(1,30)
for day in days:
    # zfill(2) allows to have figures with two digits
    sunrise = pd.Timestamp('202107{0}-1930'.format(str(day).zfill(2)))
    sunset = pd.Timestamp('202107{0}-0500'.format(str(day+1).zfill(2)))
    ax.axvspan(sunset, sunrise, ymin=0, ymax=1, 
               color = '0.9'  #'1'=white, '0'=black, '0.8'=light gray
               )

# add errors on graph
if errors_computation:
    # for direction
    ax.text(.01, .90, 'RMSE: {0}'.format({key: rmse[key] for key in rmse}), 
             ha='left', va='top', transform=ax.transAxes
             )
    ax.text(.01, .99, 'Bias: {0}'.format({key: bias[key] for key in bias}), 
             ha='left', va='top', transform=ax.transAxes
             )
    ax.legend(loc='upper right')
else:
    ax.legend(loc='best')

#plt.tight_layout()
plt.subplots_adjust(left=0.13, right=0.87, bottom=0.2) 
#%% Save figure

if save_plot:
    tools.save_figure(plot_title, save_folder)
