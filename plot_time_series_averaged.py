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

############# Independant Parameters (TO FILL IN):
    
site = 'XM'

file_suffix = 'dg'  # '' or 'dg'

varname_obs = 'T'
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
#TA_1_1_1, RH_1_1_1 Temperature and relative humidity 360cm above soil (~2m above maize)
#Q_1_1_1

varname_sim_list = ['T2M_ISBA']
# T2M_ISBA, LE_P4, EVAP_P9, GFLUX_P4, WG3_ISBA, WG4P9, SWI4_P9
# U_STAR, BOWEN

#If varname_sim is 3D:
ilevel = 1   #0 is Halo, 1->2m, 2->6.12m, 3->10.49m

figsize = (7, 7) #small for presentation: (6,6), big: (15,9)
save_plot = True
save_folder = './figures/time_series_averaged/{0}/domain1/'.format(site)

models = [
#        'irr_d2_old', 
#        'std_d2_old',
#        'irr_d2', 
#        'std_d2', 
        'irrswi1_d1', 
        'std_d1',
#        'irrlagrip30_d1',
#        'lagrip100_d1',
         ]

stdtype = None  # 'fillbetween' or 'errorbars' or None
hspace = 4  # error bars horizontal spacing
errors_computation = True
add_seb_residue = False

######################################################

simu_folders = {key:gv.simu_folders[key] for key in models}
father_folder = gv.global_simu_folder

date = '2021-07'

# colordict = {'irr_d2': 'g', 
#              'std_d2': 'r',
#              'irr_d1': 'g', 
#              'std_d1': 'r', 
#              'irrlagrip30_d1': 'y',
#              'irr_d2_old': 'g', 
#              'std_d2_old': 'r', 
#              'obs': 'k'}
# styledict = {'irr_d2': '-', 
#              'std_d2': '-',
#              'irr_d1': '--', 
#              'std_d1': '--', 
#              'irrlagrip30_d1': '--',
#              'irr_d2_old': ':', 
#              'std_d2_old': ':', 
#              'obs': '-'}

colordict = gv.colordict
styledict = gv.styledict
    
#%% Dependant Parameters

# default values (can be change below)
offset_obs = 0
coeff_obs = 1
secondary_axis = None
    
if varname_obs in ['soil_moisture_1', 'soil_moisture_2', 'soil_moisture_3']:
    ylabel = 'soil moisture [m3/m3]'
elif varname_obs in ['soil_temp_1', 'soil_temp_2', 'soil_temp_3',
                     'ST01_subsoil', 'ST04_subsoil', 'ST10_subsoil',
                     'ST17_subsoil', 'ST35_subsoil',
                     'T_10cm_Avg', 'T_20cm_Avg', 'T_30cm_Avg', 'T_40cm_Avg',
                     'T_50cm_Avg']:
    ylabel = 'soil temperature [K]'
    offset_obs = 273.15
elif varname_obs in ['swd']:
    ylabel = 'shortwave downward radiation [W/m2]'
elif varname_obs in ['lmon_1', 'lmon_2', 'lmon_3']:
    ylabel = 'monin-obukhov length [m]'
elif varname_obs in ['h2o_flux_1', 'h2o_flux_2', 'h2o_flux']:  #this includes Webb Pearman Leuning correction on w_h2o_cov
    ylabel = 'h2o flux [kg.m-2.s-1]'
    coeff_obs = 0.001
    secondary_axis = 'le'
elif varname_obs in ['co2_flux_1', 'co2_flux_2', 'co2_flux']:  #this includes Webb Pearman Leuning correction on w_h2o_cov
    ylabel = 'co2 flux [kg.m-2.s-1]'
    coeff_obs = 44e-9  # from umol/m2/s to kgCO2/m2/s
elif varname_obs in ['w_h2o_cov_1', 'w_h2o_cov_2', 'w_h2o_cov']:
    ylabel = 'h2o turbulent flux [kg.m-2.s-1]'
    coeff_obs = 0.001
    secondary_axis = 'le'
elif varname_obs in ['WQ_2m', 'WQ_10m']:
    ylabel = 'h2o turbulent flux [kg.m-2.s-1]'
    secondary_axis = 'le'
elif varname_obs in ['ta_1', 'ta_2', 'ta_3', 'ta_4', 'ta_5', 'TEMP_2m',
                     'TA_1_1_1', 'T']:
    ylabel = 'air temperature [K]'
    offset_obs = 273.15
#    offset_obs = 0
elif varname_obs in ['hus_1', 'hus_2', 'hus_3', 'hus_4', 'hus_5', 'RHO_2m']:
    ylabel = 'specific humidity [kg/kg]'
    coeff_obs = 0.001
elif varname_obs in ['FC_mass']:
    ylabel = 'CO2 flux in kg/m2/s'
    coeff_obs = 0.000001
elif varname_obs in ['WT_2m']:
    ylabel = 'turbulent sensible temperature flux [K.m-1.s-1]'
elif varname_obs in ['H_2m']:
    ylabel = 'turbulent sensible heat flux [W.m-2]'
elif varname_obs in ['LE_2m']:
    ylabel = 'turbulent latent heat flux [W.m-2]'
    secondary_axis = 'evap'
elif varname_obs in ['LE_2m_WPL']:
    ylabel = 'turbulent latent heat flux WPL corrected [W.m-2]'
    secondary_axis = 'evap'
if varname_obs in ['lhf_1', 'lhf']:
    secondary_axis = 'evap'
else:
    ylabel = varname_obs
    pass
#    raise ValueError("nom de variable d'observation inconnue"), 'WQ_2m', 'WQ_10m'


if site == 'cendrosa':
    datafolder = gv.global_data_liaise + '/cendrosa/30min/'
    filename_prefix = 'LIAISE_LA-CENDROSA_CNRM_MTO-FLUX-30MIN_L2_'
    in_filenames_obs = filename_prefix + date
    time_shift_correction = None

elif site == 'preixana':
    datafolder = gv.global_data_liaise + '/preixana/30min/'
    filename_prefix = 'LIAISE_PREIXANA_CNRM_MTO-FLUX-30MIN_L2_'
    in_filenames_obs = filename_prefix + date
    time_shift_correction = None

elif site == 'elsplans':
    freq = '30'  # '5' min or '30'min
    datafolder = gv.global_data_liaise + '/elsplans/mat_50m/{0}min/'.format(freq)
    filename_prefix = 'LIAISE_'
    date = date.replace('-', '')
    in_filenames_obs = filename_prefix + date
    time_shift_correction = None

elif site in ['irta-corn', 'irta-corn-real']:
    datafolder = gv.global_data_liaise + '/irta-corn/seb/'
    in_filenames_obs = 'LIAISE_IRTA-CORN_UIB_SEB-10MIN_L2.nc'
    time_shift_correction = None

else: # SMC stations
    freq = '30'
    datafolder = gv.global_data_liaise + '/SMC/ALL_stations_july/' 
    in_filenames_obs = f'{site}.nc'
    time_shift_correction = -30
    # raise ValueError('Site name not known')
    
lat = gv.whole[site]['lat']
lon = gv.whole[site]['lon']


#%% OBS: Concatenate and plot data
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
    out_filename_obs = in_filenames_obs
    dat_to_nc = None
    
# CONCATENATE multiple days
tools.concat_obs_files(datafolder, in_filenames_obs, out_filename_obs, 
                       dat_to_nc=dat_to_nc)

obs = xr.open_dataset(datafolder + out_filename_obs)

# process other variables:
if site in ['preixana', 'cendrosa']:
    # net radiation
    obs['rn'] = obs['swd'] + obs['lwd'] - obs['swup'] - obs['lwup']
    # bowen ratio -  diff from bowen_ratio_1
    obs['bowen'] = obs['shf_1'] / obs['lhf_1']
    obs['SEB_RESIDUE'] = obs['rn']-obs['lhf_1']-obs['shf_1']-obs['soil_heat_flux']
    obs['EVAP_FRAC'] = obs['lhf_1'] / (obs['lhf_1'] + obs['shf_1'])
    EF_temp_min0 = [max(0, val) for val in obs.EVAP_FRAC.data]
    EF_temp_max1 = [min(1, val) for val in EF_temp_min0]
    obs['EVAP_FRAC_FILTERED'] = EF_temp_max1
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
    EF_temp_min0 = [max(0, val) for val in obs.EVAP_FRAC.data]
    EF_temp_max1 = [min(1, val) for val in EF_temp_min0]
    obs['EVAP_FRAC_FILTERED'] = EF_temp_max1
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
    EF_temp_min0 = [max(0, val) for val in obs.EVAP_FRAC.data]
    EF_temp_max1 = [min(1, val) for val in EF_temp_min0]
    obs['EVAP_FRAC_FILTERED'] = EF_temp_max1
    ## Webb Pearman Leuning correction
    obs['BOWEN_2m'] = obs['H_2m'] / obs['LE_2m']
    #obs['WQ_2m_WPL'] = obs['WQ_2m']*(1.016)*(0+(1.2/300)*obs['WT_2m'])  #eq (25)
    obs['LE_2m_WPL'] = obs['LE_2m']*(1.010)*(1+0.051*obs['BOWEN_2m'])  #eq (47) of paper WPL
    for i in [10,20,30,40]:
        obs['SWI{0}_subsoil'.format(i)] = tools.calc_swi(
                obs['PR{0}_subsoil'.format(i)]*0.01,  #conversion from % to decimal
                gv.wilt_pt[site][i],
                gv.field_capa[site][i],)
else:  # SMC stations
    obs['datetime'] = [pd.Timestamp(str((elt.data))) for elt in obs['datetime']]
    obs = obs.rename({'datetime': 'time'})


if time_shift_correction not in [None, 0]:
    obs['time'] = obs['time'] - pd.Timedelta(time_shift_correction, 'm')        


# PLOT:
fig = plt.figure(figsize=figsize)

if varname_obs != '':
    if site == 'elsplans':
        ## create datetime array
    #    dati_arr = pd.date_range(start=obs.time.min().values, 
        dati_arr = pd.date_range(
#                pd.Timestamp('20210701-0000'),
                pd.Timestamp(obs[varname_obs]['time'][0].values),
                periods=len(obs[varname_obs]), 
                freq='{0}T'.format(freq))
        
#        dati_arr = pd.date_range(pd.Timestamp('20210701-0000'),
#                                 periods=len(obs[varname_obs]), 
#                                 freq='{0}T'.format(freq))
        obs['time']=dati_arr
        
        # filter outliers (turn into NaN)
        obs_var_filtered = obs[varname_obs].where(
                (obs[varname_obs]-obs[varname_obs].mean()) < (4*obs[varname_obs].std()), 
                np.nan)
        if varname_obs == 'RAIN_subsoil':
            obs_var_filtered = obs[varname_obs]
        obs_var_corr = (obs_var_filtered+offset_obs)*coeff_obs
#        plt.plot(dati_arr, obs_var_corr, 
#                 label='obs_'+varname_obs,
#                 color=colordict['obs'])
    else:
        # filter outliers (turn into NaN)
        obs_var_filtered = obs[varname_obs].where(
                (obs[varname_obs]-obs[varname_obs].mean()) < (4*obs[varname_obs].std()), 
                np.nan)
        # apply correction for comparison with models
        obs_var_corr = ((obs[varname_obs]+offset_obs)*coeff_obs)
        # plot
#        obs_var_corr.plot(label='obs',
##                          label='obs_'+varname_obs,
#                          color=colordict['obs'],
#                          linewidth=1)
            
    # compute mean day profile 
    df = obs_var_corr.to_dataframe()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index)
    
    df['hour_minute'] = df.index.time
    df['day'] = df.index.day
    if site =='cendrosa' and varname_obs in ['lhf_1', 'shf_1']:
        df_filt = df[df['day']>20]
    else:
        df_filt = df[df['day']>14]
    
    mean_dict = {}
    std_dict = {}
    for hm in df_filt['hour_minute']:
        mean_dict[hm] = df_filt[df_filt['hour_minute'] == hm][varname_obs].mean()
        std_dict[hm] = df_filt[df_filt['hour_minute'] == hm][varname_obs].std()
    
    pds_mean = pd.Series(mean_dict)
    pds_std = pd.Series(std_dict)
    pds_mean.name = 'mean'
    pds_std.name = 'std'
    
    df_mean_profile = pd.merge(pds_mean, pds_std, 
                               left_index=True, right_index=True)
    df_mean_profile.sort_index(inplace=True)
    df_mean_profile['minutes'] = [time.minute for time in df_mean_profile.index]
    df_mean_profile['hour'] = [time.hour for time in df_mean_profile.index] + \
        df_mean_profile['minutes']/60
    
    plt.plot(df_mean_profile['hour'], df_mean_profile['mean'],
             color=colordict['obs'],
             label='obs_'+varname_obs)
    if stdtype == 'fillbetween':
        plt.fill_between(df_mean_profile['hour'], 
                          df_mean_profile['mean']-df_mean_profile['std'],
                          df_mean_profile['mean']+df_mean_profile['std'],
                          alpha=0.2, 
                          facecolor=colordict['obs'],
                          )
    if stdtype == 'errorbar':
        plt.errorbar(df_mean_profile['hour'][::hspace],
                     df_mean_profile['mean'][::hspace],
                     yerr=df_mean_profile['std'][::hspace],
                     fmt='none',
                     elinewidth=0.5, capsize=2,
                     color=colordict['obs'])
        
    # add residue on graph
    if add_seb_residue:
        
        obs_uncertainty = obs['SEB_RESIDUE']
        
        if varname_obs in ['LE', 'LE_2m', 'LE_2m_WPL', 'lhf_1']:
            obs_residue_corr = obs_var_corr + obs['SEB_RESIDUE']*obs['EVAP_FRAC_FILTERED'].data
        elif varname_obs in ['H', 'H_2m', 'shf_1']:
            obs_residue_corr = obs_var_corr + obs['SEB_RESIDUE']*(1-obs['EVAP_FRAC_FILTERED'].data)
        else:
            raise ValueError('add_seb_residue available only on LE and H')
        
        df_res = obs_uncertainty.to_dataframe()
        df_res['hour_minute'] = df_res.index.time
        df_res['day'] = df_res.index.day
        if site =='cendrosa' and varname_obs in ['lhf_1', 'shf_1']:
            df_filt = df_res[df_res['day']>20]
        else:
            df_filt = df_res[df_res['day']>14]
        
        df_res_corr = obs_residue_corr.to_dataframe(name='residue_corr')
        df_res_corr['hour_minute'] = df_res_corr.index.time
        df_res_corr['day'] = df_res_corr.index.day
        if site =='cendrosa' and varname_obs in ['lhf_1', 'shf_1']:
            df_filt_corr = df_res_corr[df_res_corr['day']>20]
        else:        
            df_filt_corr = df_res_corr[df_res_corr['day']>14]
        
        mean_dict = {}
        std_dict = {}
        for hm in df_filt['hour_minute']:
            mean_dict[hm] = df_filt_corr[df_filt_corr['hour_minute'] == hm]['residue_corr'].mean()
            std_dict[hm] = df_filt[df_filt['hour_minute'] == hm]['SEB_RESIDUE'].mean()
        
        pds_mean = pd.Series(mean_dict)
        pds_std = pd.Series(std_dict)
        pds_mean.name = 'residue_corr'
        pds_std.name = 'residue'
        
        df_mean_res_profile = pd.merge(pds_mean, pds_std, 
                                   left_index=True, right_index=True)
        df_mean_res_profile.sort_index(inplace=True)
        
        df_mean_res_profile['minutes'] = [time.minute for time in df_mean_res_profile.index]
        df_mean_res_profile['hour'] = [time.hour for time in df_mean_res_profile.index] + \
            df_mean_res_profile['minutes']/60
        
        plt.plot(df_mean_res_profile['hour'], df_mean_res_profile['residue_corr'],
                 color=colordict['obs'],
                 label=f'obs_{varname_obs}_residue_corr',
                 linestyle=':',
                 linewidth=1)
        plt.fill_between(df_mean_res_profile['hour'], 
                          df_mean_profile['mean'],
                          df_mean_profile['mean']+df_mean_res_profile['residue'],
                          alpha=0.2, 
                          facecolor=colordict['obs'],
                          )
        
    
#%% SIMU:
diff = {}
rmse = {}
bias = {}
obs_sorted = {}
sim_sorted = {}
esthetic_shift = 0

for  varname_sim in varname_sim_list:

    if varname_sim == 'U_STAR':
        varname_sim_preproc = ['FMU_ISBA', 'FMV_ISBA']
    elif varname_sim == 'BOWEN':
        varname_sim_preproc = ['H_ISBA', 'LE_ISBA']
    else:
        varname_sim_preproc = [varname_sim,]
    
    for model in simu_folders:

        ds = tools.load_series_dataset(varname_sim_preproc, model)
        
        try:
            index_lat, index_lon = tools.indices_of_lat_lon(ds, lat, lon)
        except AttributeError:  #if the data does not have lat-lon data, merge with another that have it
            ds = tools.load_series_dataset(['H_ISBA',] + varname_sim_preproc, model)
            # and now, try again:
            index_lat, index_lon = tools.indices_of_lat_lon(ds, lat, lon)
        
        # Compute other diag variables
        if varname_sim == 'U_STAR':
            ds['U_STAR'] = tools.calc_u_star_sim(ds)
        elif varname_sim == 'BOWEN':
            ds['BOWEN'] = tools.calc_bowen_sim(ds)
        
        # keep variable of interest
        var_md = ds[varname_sim]
        
        # Set time abscisse axis
        try:
            start = ds.time.data[0]
        except AttributeError:
            start = np.datetime64('2021-07-21T01:00')
        
        dati_arr = np.array([start + np.timedelta64(i, 'h') for i in np.arange(0, var_md.shape[0])])
        
        var_md = var_md.squeeze()  # removes dimension with 1 value only
        
        # find indices from lat,lon values 
        index_lat, index_lon = tools.indices_of_lat_lon(ds, lat, lon)
        
        if len(var_md.shape) == 5:
            var_1d = var_md[:, :, ilevel, index_lat, index_lon].data #1st index is time, 2nd is ?, 3rd is Z,..
        elif len(var_md.shape) == 4:
            var_1d = var_md[:, ilevel, index_lat, index_lon].data #1st index is time, 2nd is Z,..
        elif len(var_md.shape) == 3:
            var_1d = var_md[:, index_lat, index_lon].data
        
        # PLOT
#        plt.plot(dati_arr, var_1d, 
#    #             color=colordict[model],
#                 colordict[model],
#                 label=f'simu_{model}_{varname_sim}',
##                 label=f'simu_{model}',
#                 )
        df_simu = pd.DataFrame(var_1d, index=dati_arr, columns=[varname_sim])
        df_simu['hour_minute'] = df_simu.index.time
        df_simu['day'] = df_simu.index.day
        df_filt = df_simu[df_simu['day']>14]
        
        mean_dict_simu = {}
        std_dict_simu = {}
        for hm in df_filt['hour_minute']:
            mean_dict_simu[hm] = df_filt[df_filt['hour_minute'] == hm][varname_sim].mean()
            std_dict_simu[hm] = df_filt[df_filt['hour_minute'] == hm][varname_sim].std()
            
        pds_mean = pd.Series(mean_dict_simu)
        pds_std = pd.Series(std_dict_simu)
        pds_mean.name = 'mean'
        pds_std.name = 'std'
        
        df_mean_profile = pd.merge(pds_mean, pds_std, 
                                   left_index=True, right_index=True)
        
        df_mean_profile['minutes'] = [time.minute for time in df_mean_profile.index]
        df_mean_profile['hour'] = [time.hour for time in df_mean_profile.index] + \
            df_mean_profile['minutes']/60
        
        plt.plot(df_mean_profile['hour'], df_mean_profile['mean'],
                 color=colordict[model],
                 label=f'simu_{model}')
        if stdtype == 'fillbetween':
            plt.fill_between(df_mean_profile['hour'], 
                              df_mean_profile['mean']-df_mean_profile['std'],
                              df_mean_profile['mean']+df_mean_profile['std'],
                              alpha=0.2, 
                              facecolor=colordict[model],
                              )
        if stdtype == 'errorbar':
            # plt.errorbar(df_mean_profile['hour'], df_mean_profile['mean'], 
            #              yerr=df_mean_profile['std'], 
            #              elinewidth=0.5, capsize=2)
            esthetic_shift += 0.15
            plt.errorbar(df_mean_profile['hour'][::hspace] + esthetic_shift,
                         df_mean_profile['mean'][::hspace],
                         yerr=df_mean_profile['std'][::hspace],
                         fmt='none',
                         elinewidth=0.5, capsize=2,
                         color=colordict[model])
        
        ax = plt.gca()        
    #    ax.set_xlim([np.min(obs.time), np.max(obs.time)])
#        ax.set_xlim([np.min(dati_arr), np.max(dati_arr)])
        
        if errors_computation and varname_obs != '':
            ## Errors computation
            obs_sorted[model] = []
            sim_sorted[model] = []
            for i, date in enumerate(dati_arr):
                val = obs_var_corr.where(obs.time == date, drop=True).data
                if len(val) != 0:
                    sim_sorted[model].append(var_1d[i])
                    obs_sorted[model].append(float(val))
            
            diff[model] = np.array(sim_sorted[model]) - np.array(obs_sorted[model])
            # compute bias and rmse, and keep values with 3 significant figures
            bias[model] = float('%.3g' % np.nanmean(diff[model]))
    #        rmse[model] = np.sqrt(np.nanmean((np.array(obs_sorted[model]) - np.array(sim_sorted[model]))**2))
            rmse[model] = float('%.3g' % np.sqrt(np.nanmean(diff[model]**2)))
    

#%% Plot esthetics

if varname_obs == '':
    try:
        ylabel = ds[varname_sim].comment
    except AttributeError:
        ylabel = varname_sim
else:
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
ax.set_xlabel('hour UTC')
plt.xticks([0,4, 8,12,16,20,24])

# add grey zones for night
#days = np.arange(1,30)
#for day in days:
#    # zfill(2) allows to have figures with two digits
#    sunrise = pd.Timestamp('202107{0}-1930'.format(str(day).zfill(2)))
#    sunset = pd.Timestamp('202107{0}-0500'.format(str(day+1).zfill(2)))
#    ax.axvspan(sunset, sunrise, ymin=0, ymax=1, 
#               color = '0.9'  #'1'=white, '0'=black, '0.8'=light gray
#               )

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

# add errors values on graph
if errors_computation:
    plt.text(.01, .95, 'RMSE: {0}'.format(rmse), 
             ha='left', va='top', transform=ax.transAxes)
    plt.text(.01, .99, 'Bias: {0}'.format(bias), 
             ha='left', va='top', transform=ax.transAxes)
    plt.legend(loc='upper right')
else:
    plt.legend(loc='best')

plot_title = '{0} at {1}'.format(varname_sim, site)
plt.title(plot_title)
#plt.title('test')
plt.grid()

# keep only hours as X axis
#plt.xticks(dati_arr[1:25:2], labels=np.arange(2,25,2))
#plt.tick_params(rotation=0)

#%% Save figure

if save_plot:
    tools.save_figure(plot_title, save_folder)
#    tools.save_figure(plot_title, '/d0/images/lunelt/figures/')
