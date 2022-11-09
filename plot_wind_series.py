#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021

Fonctionnement:
    Seule plusieurs sections ont besoin d'être remplies, à automatiser.    
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import tools
from windrose import WindroseAxes
import metpy.calc as mpcalc
from metpy.units import units


################%% Independant Parameters (TO FILL IN):
    
site = 'cendrosa'

#domain to consider for simu files: 1 or 2
domain_nb = 2

ilevel = 3   #0 is Halo, 1->2m, 2->6.12m, 3->10.49m

save_plot = True 
save_folder = './figures/winds/domain{0}/series/'.format(domain_nb)
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
    varname_obs_ws = 'ws_2'
    varname_obs_wd = 'wd_2'
    datafolder = \
        '/cnrm/surface/lunelt/data_LIAISE/preixana/30min/'
    filename_prefix = \
        'LIAISE_PREIXANA_CNRM_MTO-FLUX-30MIN_L2_'
    in_filenames_obs = filename_prefix + date
elif site == 'elsplans':
    lat = 41.590111 
    lon = 1.029363
    varname_obs_ws = 'UTOT_10m'
    varname_obs_wd = 'DIR_10m'
    datafolder = '/cnrm/surface/lunelt/data_LIAISE/elsplans/mat_50m/5min/'
    filename_prefix = 'LIAISE_'
    date = date.replace('-', '')
    in_filenames_obs = filename_prefix + date

else:
    raise ValueError('Site name not known')

#if varname_sim is None:
#    varname_sim = varname_sim_prefix + varname_sim_suffix


#%% OBS: CONCATENATE AND LOAD

out_filename_obs = 'CAT_' + date + filename_prefix + '.nc'

# Concatenate multiple days
if not os.path.exists(datafolder + out_filename_obs):
    os.system('''
        cd {0}
        ncrcat {1}* -o {2}
        '''.format(datafolder, in_filenames_obs, out_filename_obs))

obs = xr.open_dataset(datafolder + out_filename_obs)

#fig_speed=plt.figure()
#fig_dir=plt.figure()

fig, ax = plt.subplots(2, 1, figsize=(15,9))

#%% PLOT OBS
start_date = pd.Timestamp('20210721-0000')
end_date = pd.Timestamp('20210725-0000')

ws_obs = obs[varname_obs_ws]
wd_obs = obs[varname_obs_wd]

if site == 'elsplans':
    dati_arr = pd.date_range(start=obs.time.min().values, 
                             periods=len(obs[varname_obs_ws]), 
                             freq='5T')
    #turn outliers into NaN
    ws_obs_filtered = ws_obs.where(
            (ws_obs-ws_obs.mean()) < (4*ws_obs.std()), 
             np.nan)
    ax[0].plot(dati_arr, ws_obs_filtered, 
             label='obs_'+varname_obs_ws,
             color=colordict['obs'],
             linewidth=1)
    ax[1].plot(dati_arr, wd_obs, 
             label='obs_'+varname_obs_wd,
             color=colordict['obs'],
             linewidth=1)
    ax[0].set_xlim(start_date, end_date)
    ax[1].set_xlim(start_date, end_date)
else:
    plot_wind_speed(ws_obs,
                    start_date=start_date, end_date=end_date, fig=ax[0],
    #                label='obs_'+ ws_obs.long_name[-6:],
                    label='obs_'+ varname_obs_ws,
                    color=colordict['obs'])
    plot_wind_dir(wd_obs,
                  start_date=start_date, end_date=end_date, fig=ax[1],
    #              label='obs_'+ wd_obs.long_name[-6:],
                    label='obs_'+ varname_obs_wd,
                  color=colordict['obs'])
#plot_windrose(ws_obs, wd_obs, 
#              start_date=start_date, end_date=end_date
#              )

#%% SIMU:

varname_sim = 'UT,VT'
in_filenames_sim = 'LIAIS.{0}.SEG*.001.nc'.format(domain_nb)  # use of wildcard allowed
out_filename_sim = 'LIAIS.{0}.{1}.nc'.format(domain_nb, varname_sim)

for model in simu_folders:
#model='std'

    datafolder = father_folder + simu_folders[model]
    if not os.path.exists(datafolder + out_filename_sim):
        print("creation of file: ", out_filename_sim)
        os.system('''
            cd {0}
            ncecat -v {1} {2} {3}
            '''.format(datafolder, varname_sim, 
                       in_filenames_sim, out_filename_sim))
    #command 'cdo -select,name={1} {2} {3}' may work as well, but not always...
    
    ds1 = xr.open_dataset(datafolder + out_filename_sim)
    
    # find indices from lat,lon values 
    index_lat, index_lon = tools.indices_of_lat_lon(ds1, lat, lon)

    ut_md = ds1['UT']
    vt_md = ds1['VT']
    
    # Set time abscisse axis
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
                    label=model +'_l'+str(ilevel), color=colordict[model])
    plot_wind_dir(wd, dates=dati_arr, fig=ax[1], 
                  label=model +'_l'+str(ilevel), color=colordict[model])

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



#%% Save figure

if save_plot:
    tools.save_figure(plot_title, save_folder)
