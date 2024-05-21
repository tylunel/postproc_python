#!/usr/bin/env python3
"""
@author: Tanguy LUNEL

Difference with the classical plot_time_series_compare_days.py:
    - only plots obs
    
"""

#import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import tools
import global_variables as gv

############# Independant Parameters (TO FILL IN):
    
site = 'irta-corn'

file_suffix = 'dg'  # '' or 'dg'

varname_obs = 'WS'
# -- For CNRM:
# ta_5, hus_5, hur_5, soil_moisture_3, soil_temp_3, u_var_3, w_var_3, swd,... 
# w_h2o_cov, h2o_flux[_1], shf_1, u_star_1
# from données lentes: 1->0.2m, 2->2m, 3->10m, 4->25m, 5->50m
# from eddy covariance measures: 1->3m, 2->25m, 3->50m
# -- For UKMO (elsplans):
# TEMP, RHO (=hus), WQ, WT, UTOT, DIR, ... followed by _2m, _10mB, _25m, _50m, _rad, _subsoil
# RAIN, PRES, ST01 (=soil_temp), SWDN ... followed by _2m, _10mB, _25m, _50m, _rad, _subsoil
# ST01, ST04, ST10, ST17, ST35_subsoil with number being depth in cm, SFLXA=soil flux
# PR10, PR20, PR40_subsoil (=vol water content), SWI10, SWI40_subsoil
# LE_2m(_WPL) and H_2m also available by calculation
# -- For IRTA-corn
#LE, H, FC_mass, WS, WD, Ux,
#VWC_40cm_Avg: Average volumetric water content at 35 cm (m3/m3) 
#T_20cm_Avg (_Std for standard deviation)
#TA_1_1_1, RH_1_1_1 Temp. and relative humidity 360cm above soil (~2m above maize)
#Q_1_1_1

vmin, vmax = None, None

figsize = (7, 7) #small for presentation: (6,6), big: (15,9), paper:(7, 7)
plt.rcParams.update({'font.size': 11})

save_plot = True
save_folder = './fig/compa_days/{0}/'.format(site)

add_seb_residue = True

standard_plot = False
remove_alfalfa_growth = False
add_irrig_time = False
kelvin_to_celsius = False

plot_multi_days = True
days_list = [
    15,
    17,
    ]

######################################################

date = '2021-07'

colordict = {'irr_d2': 'g', 
              'std_d2': 'r',
              'irr_d1': 'g', 
              'std_d1': 'r',
              'irrswi1_d1': 'b',
              'irrlagrip30_d1': 'y',
              'irr_d2_old': 'g', 
              'std_d2_old': 'r', 
              'obs': 'k'}
styledict = {'irr_d2': '-', 
              'std_d2': '-',
              'irr_d1': '--', 
              'std_d1': '--',
              'irrswi1_d1': '--',
              'irrlagrip30_d1': '--',
              'irr_d2_old': ':', 
              'std_d2_old': ':', 
              'obs': '-'}
    

#%% Dependant Parameters

# default values (can be change below)
offset_obs = 0
coeff_obs = 1
secondary_axis = None

if varname_obs in ['LE', ]:
    ymin, ymax = -50, 750
    secondary_axis = 'evap'
    figsize = (6, 6)
elif varname_obs in ['NETRAD', ]:
    ymin, ymax = -350, 750
    add_seb_residue = False
elif varname_obs in ['TA_1_1_1',]:
    offset_obs = 273.15
    ymin, ymax = 280, 315
    add_seb_residue = False
    figsize = (6, 2.5)
elif varname_obs in ['RH_1_1_1',]:
    ymin, ymax = 0, 100
    add_seb_residue = False
    figsize = (6, 2.5)
elif varname_obs in ['WS',]:
    ymin, ymax = 0, 4
    add_seb_residue = False
    figsize = (6, 2.5)


if site == 'cendrosa':
    datafolder = gv.global_data_liaise + '/cendrosa/30min/'
    filename_prefix = 'LIAISE_LA-CENDROSA_CNRM_MTO-FLUX-30MIN_L2_'
    in_filenames_obs = filename_prefix + date
elif site == 'preixana':
    datafolder = gv.global_data_liaise + '/preixana/30min/'
    filename_prefix = 'LIAISE_PREIXANA_CNRM_MTO-FLUX-30MIN_L2_'
    in_filenames_obs = filename_prefix + date
elif site == 'elsplans':
    freq = '30'  # '5' min or '30'min
    datafolder = gv.global_data_liaise + '/elsplans/mat_50m/{0}min/'.format(freq)
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
if standard_plot:
    fig = plt.figure(figsize=figsize)

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
#        obs_var_filtered = obs_var_filtered.where(
#                obs_var_filtered>200, 
#                np.nan
#                )
        if varname_obs == 'RAIN_subsoil':
            obs_var_filtered = obs[varname_obs]
        obs_var_corr = (obs_var_filtered+offset_obs)*coeff_obs
        if standard_plot:
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
        obs_var_corr = ((obs[varname_obs]+offset_obs)*coeff_obs)
    
        # plot
        if standard_plot:
            plt.plot(obs_var_corr.time, obs_var_corr, 
                     label='obs_'+varname_obs,
                     color=colordict['obs'])
#        obs_var_corr.plot(label='obs_'+varname_obs,
#                          color=colordict['obs'],
#                          linewidth=1)
        
    if add_seb_residue and standard_plot:
        
        obs_uncertainty = obs['SEB_RESIDUE'].data
        
        if varname_obs in ['LE', 'LE_2m', 'LE_2m_WPL', 'lhf_1']:
            obs_residue_corr = obs_var_corr + obs['SEB_RESIDUE']*obs['EVAP_FRAC_FILTERED'].data
        elif varname_obs in ['H', 'H_2m', 'shf_1']:
            obs_residue_corr = obs_var_corr + obs['SEB_RESIDUE']*(1-obs['EVAP_FRAC_FILTERED'].data)
        else:
            raise ValueError('add_seb_residue available only on LE and H')
            
        obs_residue_corr.plot(
            label='obs_adjust',
            color=colordict['obs'],
            linestyle=':',
            linewidth=1)
        
        plt.fill_between(obs_var_corr.time, 
                          obs_var_corr.data,
                          obs_var_corr.data + obs_uncertainty.data,
                          alpha=0.2, 
                          facecolor=colordict['obs'],
                          )
    

#%% Add irrigation datetime
if add_irrig_time and varname_obs != '' and standard_plot:
    if site == 'irta-corn':
        sm_var = obs['VWC_40cm_Avg']
    if site == 'cendrosa':
        sm_var = obs['soil_moisture_3']
    if site == 'preixana':
        sm_var = None  # not irrigated, but could represent rain
    if site == 'elsplans':
        sm_var = None  # not irrigated, but could represent rain
    dati_list = tools.get_irrig_time(sm_var)
    plt.vlines(dati_list, 
               ymin=obs_var_corr.min().data, 
               ymax=obs_var_corr.max().data, 
               label='irrigation')

#%% Plot esthetics
if standard_plot:
    try:
        ylabel = obs[varname_obs].long_name
    except AttributeError:
        try:
            ylabel = ds[varname_sim].comment
        except (AttributeError, KeyError, NameError):
            ylabel = varname_obs
    
    plot_title = '{0} at {1}'.format(ylabel, site)
    ax = plt.gca()
    ax.set_ylabel(ylabel)
    ax.set_ylim([vmin, vmax])
    
    try:
        ax.set_xlim([np.min(dati_arr_simu), 
                     (np.max(dati_arr_simu) - pd.Timedelta(1, 'h'))])
    except NameError:  #if 'dati_arr_simu' is not defined
        pass
    
    ax.set_xlabel('time UTC')
    
    # add grey zones for night
    days = np.arange(1,30)
    for day in days:
        # zfill(2) allows to have figures with two digits
        sunrise = pd.Timestamp('202107{0}-1930'.format(str(day).zfill(2)))
        sunset = pd.Timestamp('202107{0}-0430'.format(str(day+1).zfill(2)))
        ax.axvspan(sunset, sunrise, ymin=0, ymax=1, 
                   color = '0.9'  #'1'=white, '0'=black, '0.8'=light gray
                   )
    
    # add secondary axis on the right, relative to the left one - (for LE)
    if secondary_axis == 'le':
        axes = plt.gca()
        secax = axes.secondary_yaxis("right",                              
            functions=(lambda evap: evap*2264000,
                       lambda le: le/2264000))
        secax.set_ylabel('latent heat flux [W/m²]')
    if secondary_axis == 'evap':
        axes = plt.gca()
        secax = axes.secondary_yaxis("right",                              
            functions=(lambda le: (le/2264000)*3600,
                       lambda evap: (evap*2264000)/3600))
        secax.set_ylabel('evapotranspiration [mm/h]')
    
    plt.legend(loc='best')
    plt.title(plot_title)
    plt.grid()
    
    # keep only hours as X axis
    #plt.xticks(dati_arr[1:25:2], labels=np.arange(2,25,2))
    plt.xticks(rotation=30)

#%% PLOT MULTI DAYS

month = int(date[-2:])

if plot_multi_days:

    linestyle_list = ['--', ':']
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    #ws_obs_filtered = ws_obs  # no filtering
    #dati_arr_obs = ws_obs.time
    
    for i, day in enumerate(days_list):
        
        datimax = pd.Timestamp(f'{date}-{day}T2350')
        datimin = pd.Timestamp(f'{date}-{day}T0010')
        
        obs_day = obs.where(obs.time < datimax, drop=True)
        obs_day = obs_day.where(obs_day.time > datimin, drop=True)
        
        obs_var_corr = ((obs_day[varname_obs]+offset_obs)*coeff_obs)
        # OBS
        # ---- OLD ----
        # # select month
        # dati_arr_obs= pd.DatetimeIndex(obs_var_corr.time)
        # month_data = obs_var_corr[dati_arr_obs.month == month]
        # # select day in the selected month
        # dati_arr_obs_month = pd.DatetimeIndex(month_data.time)
        # day_data = month_data[dati_arr_obs_month.day == day]
        
        # minute_dati_arr_obs = dati_arr_obs_month[dati_arr_obs_month.day == day].minute / 60
        # hour_dati_arr_obs = dati_arr_obs_month[dati_arr_obs_month.day == day].hour
        # time_dati_arr_obs = minute_dati_arr_obs + hour_dati_arr_obs
    
        dati_arr_obs= pd.DatetimeIndex(obs_var_corr.time)
        minute_dati_arr_obs = dati_arr_obs[dati_arr_obs.day == day].minute/60
        hour_dati_arr_obs = dati_arr_obs[dati_arr_obs.day == day].hour
        time_dati_arr_obs = minute_dati_arr_obs + hour_dati_arr_obs
          
        if add_seb_residue:
            
            obs_uncertainty = obs_day['SEB_RESIDUE'].data
            
            if varname_obs in ['LE', 'LE_2m', 'LE_2m_WPL', 'lhf_1']:
                obs_residue_corr = obs_var_corr + obs_day['SEB_RESIDUE']*obs_day['EVAP_FRAC_FILTERED'].data
            elif varname_obs in ['H', 'H_2m', 'shf_1']:
                obs_residue_corr = obs_var_corr + obs_day['SEB_RESIDUE']*(1-obs_day['EVAP_FRAC_FILTERED'].data)
            else:
                raise ValueError('add_seb_residue available only on LE and H')
                
            # obs_residue_corr.plot(
            #     label='obs_adjust',
            #     color=colordict['obs'],
            #     linestyle=linestyle_list[i],
            #     linewidth=1)
            
            ax.plot(time_dati_arr_obs, obs_residue_corr, 
                  label=f'obs_residue_adjust_{day}',
                  color=colordict['obs'],
                  linewidth=1.5,
                  linestyle=linestyle_list[i],
                  )
            
            # plt.fill_between(time_dati_arr_obs, 
            #                   obs_var_corr.data,
            #                   obs_var_corr.data + obs_uncertainty.data,
            #                   alpha=0.2, 
            #                   facecolor=colordict['obs'],
            #                   )
        else:
            ax.plot(time_dati_arr_obs, obs_var_corr.data, 
                  label=f'obs_{day}',
                  color=colordict['obs'],
                  linewidth=1.5,
                  linestyle=linestyle_list[i],
                  )
        

#    ax.set_ylim([293, 311])
    ylabel = 'specific humidity [kg kg$^{-1}$]'
    ylabel = varname_obs
    ax.set_ylabel(ylabel, 
                  multialignment='center')

    ax.grid(visible=True, which='major', axis='y', color='k', linestyle='--',
             linewidth=0.5)

    ax.set_xlabel('Hour UTC')
    ax.set_xlim([1, 23])
    ax.set_ylim([ymin, ymax])
    
    ax.grid(visible=True, axis='both')    
    ax.legend(loc='upper left')
    
    # add grey zones for night
    ax.axvspan(0, 4.5, ymin=0, ymax=1, 
               color = '0.9')  #'1'=white, '0'=black, '0.8'=light gray
    ax.axvspan(19.5, 24, ymin=0, ymax=1, 
               color = '0.9')  #'1'=white, '0'=black, '0.8'=light gray


fig.subplots_adjust(bottom=0.2)
#plt.tight_layout()  # ensure that all figure elements fit in frame

# ---- Save figure
save_title = f'{site}_{varname_obs}_compare{days_list}'
if save_plot:
    tools.save_figure(save_title, save_folder)
#    tools.save_figure(plot_title, '/d0/images/lunelt/figures/')
