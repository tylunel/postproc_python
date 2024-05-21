#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021
    
"""

#import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import tools
import global_variables as gv

############# Independant Parameters #############
    
site = 'elsplans'

varname_obs = 'THT_10m'  # RHO_10m, THT_10m, UTOT_10m, DIR_10m
# -- For UKMO (elsplans):
# TEMP, RHO (=hus), WQ, WT, UTOT, DIR, ... followed by _2m, _10mB, _25m, _50m, _rad, _subsoil

varname_sim = 'THT'  # RVT, THT, do not work for WS, WD for now
# T2M_ISBA, LE_P4, EVAP_P9, GFLUX_P4, WG3_ISBA, WG4P9, SWI4_P9, U_STAR

# default values (can be change below)
offset_obs = 0
coeff_obs = 1  # 0.001 for RVT, 1 for THT

varname_label = 'Potential temperature [K]'  
# Specific humidity [kg kg$^{-1}$], Potential temperature [K],
# Wind Speed [m s$^{-1}$], Wind direction [°]

vmin, vmax = None, None

days_list = [16, 21]
# days_dict = {
#     16: 'irrswi1_d1_16_10min',
#     21: 'irrswi1_d1',
#     }

#If varname_sim is 3D:
ilevel =  3  #0 is Halo, 1->2m, 2->6.12m, 3->10.49m

# figsize = (7, 7) #small for presentation: (6,6), big: (15,9), paper:(7, 7)
figsize = (11,3.5)
plt.rcParams.update({'font.size': 11})

standard_time_series = False

save_plot = True
save_folder = './fig/time_series/'

models = [
#        'irr_d2_old', 
#        'std_d2_old',
#        'irr_d2', 
#        'std_d2',
#        'irr_d1',
#        'std_d1',
#        'irrlagrip30_d1',
        'irrswi1_d1',
         ]

remove_alfalfa_growth = False
errors_computation = False
compare_to_residue_corr = False

add_seb_residue = False

add_irrig_time = False

kelvin_to_celsius = False

if 'irrlagrip30_d1' in models and errors_computation:
    print("""Warning: computation of errors will be run on all of july for
          'irrlagrip30_d1' - bug to fix in code""")

######################################################

simu_folders = {key:gv.simu_folders[key] for key in models}
father_folder = gv.global_simu_folder

date = '2021-07'

colordict = gv.colordict
styledict = gv.styledict

#%% Dependant Parameters

secondary_axis = None

if site == 'cendrosa':
    datafolder = gv.global_data_liaise + '/cendrosa/30min/'
    filename_prefix = 'LIAISE_LA-CENDROSA_CNRM_MTO-FLUX-30MIN_L2_'
    in_filenames_obs = filename_prefix + date
elif site == 'preixana':
    datafolder = gv.global_data_liaise + '/preixana/30min/'
    filename_prefix = 'LIAISE_PREIXANA_CNRM_MTO-FLUX-30MIN_L2_'
    in_filenames_obs = filename_prefix + date
elif site == 'elsplans':
    freq = '5'  # '5' min or '30'min
    datafolder = gv.global_data_liaise + '/elsplans/mat_50m/{0}min_v4/'.format(freq)
    filename_prefix = 'LIAISE_'
    date = date.replace('-', '')
    in_filenames_obs = filename_prefix + date
#    varname_sim_suffix = '_ISBA'  # or P7, but already represents 63% of _ISBA
elif site in ['irta-corn', 'irta-corn-real',]:
    datafolder = gv.global_data_liaise + '/irta-corn/seb/'
    in_filenames_obs = 'LIAISE_IRTA-CORN_UIB_SEB-10MIN_L2.nc'
else:  # SMC
    freq = '30'
    datafolder = gv.global_data_liaise + '/SMC/ALL_stations_july/' 
    
lat = gv.whole[site]['lat']
lon = gv.whole[site]['lon']


#%% OBS: LOAD and COMPUTE SOME DIAGs
if varname_obs != '':
    if site in ['irta-corn', 'irta-corn-real']:
        out_filename_obs = in_filenames_obs
    #    dat_to_nc = 'uib'  #To create a new netcdf file
        dat_to_nc = None   #To keep existing netcdf file
    elif site == 'elsplans':
        out_filename_obs = 'CAT_' + date + filename_prefix + '.nc'
        dat_to_nc = 'ukmo'
    #    dat_to_nc = None   #To keep existing netcdf file
    elif site == 'cendrosa':
        out_filename_obs = 'CAT_' + date + filename_prefix + '.nc'
        dat_to_nc = None
    else:  # SMC case
        out_filename_obs = f'{site}.nc'
        dat_to_nc = None
        
    # CONCATENATE multiple days
    tools.concat_obs_files(datafolder, in_filenames_obs, out_filename_obs, 
                           dat_to_nc=dat_to_nc)
    
    obs = xr.open_dataset(datafolder + out_filename_obs)
    
    # DIAG - process other variables:
    if site in ['preixana', 'cendrosa']:
        # net radiation
        obs['rn'] = obs['swd'] + obs['lwd'] - obs['swup'] - obs['lwup']
        # bowen ratio -  diff from bowen_ratio_1
        obs['bowen'] = obs['shf_1'] / obs['lhf_1']
        obs['SEB_RESIDUE'] = obs['rn']-obs['lhf_1']-obs['shf_1']-obs['soil_heat_flux']
        obs['EVAP_FRAC'] = obs['lhf_1'] / (obs['lhf_1'] + obs['shf_1'])
        obs['EVAP_FRAC_FILTERED'] = obs['EVAP_FRAC'].clip(min=0, max=1)
        for i in [1,2,3]:
            obs['swi_{0}'.format(i)] = tools.calc_swi(
                    obs['soil_moisture_{0}'.format(i)],
                    gv.wilt_pt[site][i],
                    gv.field_capa[site][i],) 
    elif site in ['irta-corn', 'irta-corn-real']:
        for i in [1,2,3,4,5]:
            site = 'irta-corn'
            obs['swi_{0}'.format(i)] = tools.calc_swi(
                    obs['VWC_{0}0cm_Avg'.format(i)],
                    gv.wilt_pt[site][i],
                    gv.field_capa[site][i],)
        obs['Q_1_1_1'] = tools.psy_ta_rh(
            obs['TA_1_1_1'], 
            obs['RH_1_1_1'],
            obs['PA']*1000)['hr']
        obs['air_density'] = obs['PA']*1000/(287.05*(obs['TA_1_1_1']+273.15))
        obs['U_STAR'] = np.sqrt(obs['TAU']/obs['air_density'])
        obs['SEB_RESIDUE'] = obs['NETRAD']-obs['LE']-obs['H']-obs['G_plate_1_1_1']
        obs['EVAP_FRAC'] = obs['LE'] / (obs['LE'] + obs['H'])
        obs['EVAP_FRAC_FILTERED'] = obs['EVAP_FRAC'].clip(min=0, max=1)
    elif site == 'elsplans':
        ## Flux calculations
        obs['H_2m'] = obs['WT_2m']*1200  # =Cp_air * rho_air
        obs['LE_2m'] = obs['WQ_2m']*2264000  # =L_eau
        obs['NETRAD'] = obs['SWDN_rad'] + obs['LWDN_rad'] - obs['SWUP_rad'] - obs['LWUP_rad']
        obs['SEB_RESIDUE'] = obs['NETRAD']-obs['LE_2m']-obs['H_2m']-obs['SFLXA_subsoil']
        obs['SEB_RESIDUE'] = obs['SEB_RESIDUE'].where(
                    obs['SEB_RESIDUE']>-1000, 
                    np.nan)
        obs['EVAP_FRAC'] = obs['LE_2m'] / (obs['LE_2m'] + obs['H_2m'])
        obs['EVAP_FRAC_FILTERED'] = obs['EVAP_FRAC'].clip(min=0, max=1)
        ## Webb Pearman Leuning correction
        obs['BOWEN_2m'] = obs['H_2m'] / obs['LE_2m']
        #obs['WQ_2m_WPL'] = obs['WQ_2m']*(1.016)*(0+(1.2/300)*obs['WT_2m'])  #eq (25)
        obs['LE_2m_WPL'] = obs['LE_2m']*(1.010)*(1+0.051*obs['BOWEN_2m'])  #eq (47) of paper WPL
        obs['THT_2m'] = (obs['TEMP_2m']+273.15)*tools.exner_function(obs['PRES_subsoil']*100)
        obs['THT_10m'] = (obs['TEMP_10m']+273.15)*tools.exner_function(obs['PRES_subsoil']*100)
        
        for i in [10,20,30,40]:
            obs['SWI{0}_subsoil'.format(i)] = tools.calc_swi(
                    obs['PR{0}_subsoil'.format(i)]*0.01,  #conversion from % to decimal
                    gv.wilt_pt[site][i],
                    gv.field_capa[site][i],)
        
#%% OBS PLOT:

if varname_obs != '':
    if site == 'elsplans':
        ## create datetime array
    #    dati_arr = pd.date_range(start=obs.time.min().values, 
        dati_arr_obs = pd.date_range(
#                pd.Timestamp('20210701-0000'),
                pd.Timestamp(obs[varname_obs]['time'][0].values),
                periods=len(obs[varname_obs]), 
                freq='{0}T'.format(freq))
        
        obs['time']=dati_arr_obs
        
        if varname_obs == 'RHO_2m':
            obs = obs.where(obs.time>pd.Timestamp('20210715T1200'), drop=True)
        
        # filter outliers (turn into NaN)
        obs_var_filtered = obs[varname_obs].where(
                (obs[varname_obs]-obs[varname_obs].mean()) < (4*obs[varname_obs].std()),
#                (obs[varname_obs] < 400),
#                drop=True,
                np.nan
                )
        obs_var_filtered = obs_var_filtered.where(
                obs_var_filtered>200, 
                np.nan
                )
        if varname_obs == 'RAIN_subsoil':
            obs_var_filtered = obs[varname_obs]
        
        obs_var_corr = (obs_var_filtered+offset_obs)*coeff_obs
            
        if standard_time_series:
            plt.plot(obs_var_corr.time, obs_var_corr, 
                     label='obs_'+varname_obs,
                     color=colordict['obs'])
    else:
        if remove_alfalfa_growth:
            if varname_obs in ['lhf_1', 'shf_1'] and site == 'cendrosa':  # because of growth of alfalfa
                obs = obs.where(obs.time>pd.Timestamp('20210721T0100'), drop=True)
        
        if site == 'irta-corn':
            obs = obs.where(~obs.time.isnull(), drop=True)
        
        # filter outliers (turn into NaN)
        obs_var_filtered = obs[varname_obs].where(
                (obs[varname_obs]-obs[varname_obs].mean()) < (4*obs[varname_obs].std()), 
                np.nan)
        # apply correction for comparison with models
        obs_var_corr = ((obs_var_filtered+offset_obs)*coeff_obs)
    
        # plot
        if standard_time_series:
            
            fig = plt.figure(figsize=figsize)
            plt.plot(obs_var_corr.time, obs_var_corr, 
                     label='obs_'+varname_obs,
                     color=colordict['obs'])
#        obs_var_corr.plot(label='obs_'+varname_obs,
#                          color=colordict['obs'],
#                          linewidth=1)


#%% SIMU - LOAD and PLOT:

if varname_sim == 'U_STAR':
    varname_sim_preproc = ['FMU_ISBA', 'FMV_ISBA']
elif varname_sim in ['WS', 'WD']:
    varname_sim_preproc = ['UT', 'VT', 'THT']
else:
    varname_sim_preproc = [varname_sim,]

for model in simu_folders:
# model = 'irrswi1_d1'

    out_suffix = '.OUT'
    file_suffix = ''
    
    ds = tools.load_series_dataset(
            varname_sim_preproc, model, concat_if_not_existing=True,
            out_suffix=out_suffix, file_suffix=file_suffix)
    
    try:
        index_lat, index_lon = tools.indices_of_lat_lon(ds, lat, lon)
    except AttributeError:  # if the data does not have lat-lon data, merge with another that have it
        ds = tools.load_series_dataset(['H_ISBA',] + varname_sim_preproc, model)
        # and now, try again:
        index_lat, index_lon = tools.indices_of_lat_lon(ds, lat, lon)
    
    # Compute other diag variables
    if varname_sim == 'U_STAR':
        ds['U_STAR'] = tools.calc_u_star_sim(ds)
    elif varname_sim in ['WS', 'WD']:
        # ds = tools.center_uvw(ds)
        ut_1d = ds['UT'].isel(ni_u=index_lon).isel(nj_u=index_lat).isel(level=ilevel)
        vt_1d = ds['VT'].isel(ni_v=index_lon).isel(nj_v=index_lat).isel(level=ilevel)
        ws_1d, wd_1d = tools.calc_ws_wd(ut_1d, vt_1d)
    # else:

    
    # Set time abscisse axis
    try:
        start = ds.time.data[0]
    except IndexError:
        start = ds.time[0].data
    except AttributeError:
        start = np.datetime64('2021-07-21T01:00')
    
    if out_suffix == '.OUT':
        freq_simu = gv.output_freq_dict[model]
    else:
        freq_simu = 60
    
    dati_arr_simu = pd.date_range(start=pd.Timestamp(start), 
                                  periods=len(ds['record']),
                                  freq=f'{freq_simu}T')

    ds = ds.squeeze()
    ds['record'] = dati_arr_simu.to_numpy()
    ds = ds.drop_vars(['time'])
    ds = ds.rename({'record': 'time'})
    
    # keep variable of interest
    var_md = ds[varname_sim]
    
    if kelvin_to_celsius:
        var_md = var_md - 273.15
    
    var_1d = var_md.isel(ni=index_lon).isel(nj=index_lat).isel(level=ilevel).squeeze()
    
    # PLOT
    if standard_time_series:
        plt.plot(ds.time, var_1d, 
                 color=colordict[model],
#                 colordict[model],
                 label=f'simu_{model}_{varname_sim}',
#                 label=f'simu_{model}',
                 )
    

#%% PLOT MULTI DAYS
plot_multi_days = True

if plot_multi_days:
    
    linestyle_list = ['--', ':']
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    #ws_obs_filtered = ws_obs  # no filtering
    #dati_arr_obs = ws_obs.time
    
    for i, day in enumerate(days_list):
        
        # OBS
        dati_arr_obs = pd.DatetimeIndex(obs_var_corr.time)
        day_data = obs_var_corr[dati_arr_obs.day == day]
        
        minute_dati_arr_obs = dati_arr_obs[dati_arr_obs.day == day].minute / 60
        hour_dati_arr_obs = dati_arr_obs[dati_arr_obs.day == day].hour
        time_dati_arr_obs = minute_dati_arr_obs + hour_dati_arr_obs
    
        ax.plot(time_dati_arr_obs, day_data, 
             label=f'obs_{day}',
             color=colordict['obs'],
             linewidth=1.5,
             linestyle=linestyle_list[i],
             )
        
        # SIMU
        model = models[0]
        
        dati_arr_simu = pd.DatetimeIndex(ds.time)
        day_simu = var_1d[dati_arr_simu.day == day]
        
        minute_dati_arr_simu = dati_arr_simu[dati_arr_simu.day == day].minute / 60
        hour_dati_arr_simu = dati_arr_simu[dati_arr_simu.day == day].hour
        time_dati_arr_simu = minute_dati_arr_simu + hour_dati_arr_simu
    
        ax.plot(time_dati_arr_simu, day_simu, 
             label=f'simu_{day}',
             color=colordict[model],
             linewidth=1.5,
             linestyle=linestyle_list[i],
             )

#    ax.set_ylabel('potential temperature [K]', multialignment='center')
#    ax.set_ylim([293, 311])
        
    ax.set_ylabel(varname_label, multialignment='center')

    ax.grid(visible=True, which='major', axis='y', color='k', linestyle='--',
             linewidth=0.5)

    ax.set_xlabel('Hour UTC')
    ax.set_xlim([9, 22])
    
    ax.grid(visible=True, axis='both')    
    # ax.legend(loc='upper left')
    
    # add grey zones for night
    ax.axvspan(19.5, 24, ymin=0, ymax=1, 
               color = '0.9')  #'1'=white, '0'=black, '0.8'=light gray


fig.subplots_adjust(bottom=0.2)
#plt.tight_layout()  # ensure that all figure elements fit in frame
#%% Save figure
save_title = f'time_series_16_21_{varname_sim}'

if save_plot:
    tools.save_figure(save_title, save_folder)
#    tools.save_figure(plot_title, '/d0/images/lunelt/figures/')
