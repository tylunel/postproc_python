#!/usr/bin/env python3
"""
@author: Tanguy LUNEL

Plot time series for outputs of Surfex offline runs
    
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

varname_sim = 'LE'
varname_obs = 'LE'
# -- For CNRM:
# ta_5, hus_5, hur_5, soil_moisture_3, soil_temp_3, u_var_3, w_var_3, swd,... 
# w_h2o_cov, h2o_flux[_1], shf_1, u_star_1
# from données lentes: 1->0.2m, 2->2m, 3->10m, 4->25m, 5->50m
# from eddy covariance measures: 1->3m, 2->25m, 3->50m
# -- For UKMO (elsplans):
# TEMP, RHO (=hus), WQ, WT, UTOT, DIR, ... followed by _2m, _10mB, _25m, _50m, _rad, _subsoil
# RAIN, PRES, ST01 (=soil_temp), SWDN ... followed by _2m, _10mB, _25m, _50m, _rad, _subsoil
# ST01, ST04, ST10, ST17, ST35_subsoil with number being depth in cm
# PR10, PR20, PR40_subsoil (=vol water content), SWI10, SWI40_subsoil
# LE_2m(_WPL) and H_2m also available by calculation
# -- For IRTA-corn
#LE, H, FC_mass, WS, WD, Ux,
#VWC_40cm_Avg: Average volumetric water content at 35 cm (m3/m3) 
#T_20cm_Avg (_Std for standard deviation)
#TA_1_1_1, RH_1_1_1 Temperature and relative humidity 360cm above soil (~2m above maize)
#Q_1_1_1


add_irrig_time = True
figsize = (15, 9) #small for presentation: (6,6), big: (15,9)
save_plot = True
save_folder = './figures/time_series_sfx/{0}/domain1/'.format(site)

models = [
        'irr_d1', 
        'std_d1',
         ]

models_idnumber = {'irr_d1': '2.15',
                   'std_d1': '1.15',
                   }

errors_computation = True

add_fao56_et = False
######################################################

simu_folders = {key:gv.simu_folders[key] for key in models}
father_folder = gv.global_simu_folder

date = '2021-07'

colordict = {'irr_d2': 'g', 
             'std_d2': 'r',
             'irr_d1': 'g', 
             'std_d1': 'r',
             'obs': 'k'}

if site in ['cendrosa', 'irta-corn']:
    folder = '/cnrm/surface/lunelt/NO_SAVE/sfx_out/{0}/'.format(site)
else:
    raise KeyError('No simulation data for this site')


#%% Dependant Parameters

# default values (can be change below)
offset_obs = 0
coeff_obs = 1
secondary_axis = None
    
if varname_sim in ['LE', 'LE_ISBA']:
    ylabel = 'latent heat flux [m3/m3]'
    secondary_axis = 'evap'
else:
    ylabel = varname_sim
    pass
#    raise ValueError("nom de variable d'observation inconnue"), 'WQ_2m', 'WQ_10m'

if varname_sim in ['U_STAR',]:
    varname_sim_preproc = 'FMU_ISBA,FMV_ISBA'
elif varname_sim in ['BOWEN',]:
    varname_sim_preproc = 'H_ISBA,LE_ISBA'
else:
    varname_sim_preproc = varname_sim

if site == 'cendrosa':
    datafolder = '/cnrm/surface/lunelt/data_LIAISE/cendrosa/30min/'
    filename_prefix = 'LIAISE_LA-CENDROSA_CNRM_MTO-FLUX-30MIN_L2_'
    in_filenames_obs = filename_prefix + date
elif site == 'preixana':
#    varname_sim_suffix = '_ISBA'
    datafolder = '/cnrm/surface/lunelt/data_LIAISE/preixana/30min/'
    filename_prefix = 'LIAISE_PREIXANA_CNRM_MTO-FLUX-30MIN_L2_'
    in_filenames_obs = filename_prefix + date
elif site == 'elsplans':
    freq = '30'  # '5' min or '30'min
    datafolder = '/cnrm/surface/lunelt/data_LIAISE/elsplans/mat_50m/{0}min/'.format(freq)
    filename_prefix = 'LIAISE_'
    date = date.replace('-', '')
    in_filenames_obs = filename_prefix + date
#    varname_sim_suffix = '_ISBA'  # or P7, but already represents 63% of _ISBA
elif site == 'irta-corn':
    datafolder = '/cnrm/surface/lunelt/data_LIAISE/irta-corn/seb/'
    in_filenames_obs = 'LIAISE_IRTA-CORN_UIB_SEB-10MIN.nc'
#    raise ValueError('Site name not known')
    
lat = gv.sites[site]['lat']
lon = gv.sites[site]['lon']


#%% OBS: Concatenate and plot data
if site == 'irta-corn':
    out_filename_obs = in_filenames_obs
#    dat_to_nc = 'uib'  #To create a new netcdf file
    dat_to_nc = None   #To keep existing netcdf file
elif site == 'elsplans':
    out_filename_obs = 'CAT_' + date + filename_prefix + '.nc'
#    dat_to_nc = 'ukmo'
    dat_to_nc = None   #To keep existing netcdf file
else:
    out_filename_obs = 'CAT_' + date + filename_prefix + '.nc'
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
    # potential evapotranspiration
    obs['lhf_0_fao56'] = tools.calc_fao56_et_0(
        obs['rn'], 
        obs['ta_2'], 
        obs['ws_1'], 
        obs['hur_2'], 
        obs['pa']*100,
        gnd_flx=obs['soil_heat_flux'])['LE_0']
    for i in [2,3]:
        obs['swi_{0}'.format(i)] = tools.calc_swi(
                obs['soil_moisture_{0}'.format(i)],
                gv.wilt_pt[site][i],
                gv.field_capa[site][i],) 
elif site == 'irta-corn':
    for i in [2,3,4,5]:
        obs['swi_{0}'.format(i)] = tools.calc_swi(
                obs['VWC_{0}0cm_Avg'.format(i)],
                gv.wilt_pt[site][i],
                gv.field_capa[site][i],)
    obs['Q_1_1_1'] = tools.psy_ta_rh(
        obs['TA_1_1_1'], 
        obs['RH_1_1_1'],
        obs['PA']*1000)['hr']
    obs['LE_0_FAO56'] = tools.calc_fao56_et_0(
        obs.NETRAD, 
        obs.TA_1_1_1, 
        obs.WS, 
        obs.RH_1_1_1, 
        obs.PA*1000,
        gnd_flx=obs.G_plate_1_1_1)['LE_0']
elif site == 'elsplans':
    ## Flux calculations
    obs['H_2m'] = obs['WT_2m']*1200  # =Cp_air * rho_air
    obs['LE_2m'] = obs['WQ_2m']*2264000  # =L_eau
    
    ## Webb Pearman Leuning correction
    obs['BOWEN_2m'] = obs['H_2m'] / obs['LE_2m']
    #obs['WQ_2m_WPL'] = obs['WQ_2m']*(1.016)*(0+(1.2/300)*obs['WT_2m'])  #eq (25)
    obs['LE_2m_WPL'] = obs['LE_2m']*(1.010)*(1+0.051*obs['BOWEN_2m'])  #eq (47) of paper WPL
    for i in [10,20,30,40]:
        obs['SWI{0}_subsoil'.format(i)] = tools.calc_swi(
                obs['PR{0}_subsoil'.format(i)]*0.01,  #conversion from % to decimal
                gv.wilt_pt[site][i],
                gv.field_capa[site][i],)
    
# PLOT:
fig = plt.figure(figsize=figsize)

if varname_obs != '':
    if site == 'elsplans':
        ## create datetime array
    #    dati_arr = pd.date_range(start=obs.time.min().values, 
        dati_arr = pd.date_range(pd.Timestamp('20210701-0000'),
                                 periods=len(obs[varname_obs]), 
                                 freq='{0}T'.format(freq))
        
        # filter outliers (turn into NaN)
        obs_var_filtered = obs[varname_obs].where(
                (obs[varname_obs]-obs[varname_obs].mean()) < (4*obs[varname_obs].std()), 
                np.nan)
        obs_var_corr = (obs_var_filtered+offset_obs)*coeff_obs
        plt.plot(dati_arr, obs_var_corr, 
                 label='obs_'+varname_obs,
                 color=colordict['obs'])
    else:
        # filter outliers (turn into NaN)
        obs_var_filtered = obs[varname_obs].where(
                (obs[varname_obs]-obs[varname_obs].mean()) < (4*obs[varname_obs].std()), 
                np.nan)
        # apply correction for comparison with models
        obs_var_corr = ((obs[varname_obs]+offset_obs)*coeff_obs)
        # plot
        obs_var_corr.plot(label='obs_'+varname_obs,
                          color=colordict['obs'],
                          linewidth=1)
        if add_fao56_et:
            obs['LE_0_FAO56'].plot(label='obs_LE_0',
                                   color=colordict['obs'],
                                   linestyle=':',
                                   linewidth=1)


#%% SIMU:
diff = {}
rmse = {}
bias = {}
obs_sorted = {}
sim_sorted = {}

#fig = plt.figure(figsize=figsize)


for model in simu_folders:

    datafolder = folder + 'forcing_{0}/'.format(models_idnumber[model])
    
    ds = xr.open_dataset(datafolder + 'SURF_ATM_DIAGNOSTICS.OUT.nc')
    
    # Compute other diag variables
    if varname_sim == 'U_STAR':
        ds['U_STAR'] = tools.calc_u_star_sim(ds)
    elif varname_sim == 'BOWEN':
        ds['BOWEN'] = tools.calc_bowen_sim(ds)
    elif add_fao56_et:
        ds_forcing = xr.open_dataset(datafolder + 'FORCING_{0}_{1}.nc'.format(
                site, models_idnumber[model]))
        ds_forcing['REHU'] = tools.rel_humidity(
                ds_forcing['Qair'], 
                ds_forcing['Tair']-273.15,
                ds_forcing['PSurf'],
                )
        ds['LE_0_ISBA'] = tools.calc_fao56_et_0(
            ds['RN'], 
            ds_forcing['Tair'], 
            ds_forcing['Wind'], 
            ds_forcing['REHU'], 
            ds_forcing['PSurf'],
            gnd_flx= ds['GFLUX'])['LE_0'].dropna(dim='time', how='any')
        LE_0 =  ds['LE_0_ISBA'].dropna(dim='time', how='any').squeeze()
    
    # keep variable of interested
    var_1d = ds[varname_sim].squeeze()
    
    # Set time abscisse axis
#    try:
#        start = ds.time.data[0]
#    except AttributeError:    
##        start = np.datetime64('2021-07-14T01:00')
#        start = np.datetime64('2021-07-21T01:00')
#    
#    dati_arr = np.array([start + np.timedelta64(i, 'h') for i in np.arange(0, var_md.shape[0])])
    
    # PLOT
    plt.plot(var_1d.time, var_1d, 
             color=colordict[model],
#             colordict[model],
             linestyle='--',
             label='simu_{0}_{1}'.format(model, varname_sim),
             )
    if add_fao56_et:
        plt.plot(LE_0.time, LE_0, 
             color=colordict[model],
#             colordict[model],
             linestyle=':',
             label='simu_{0}_LE_0_FAO56'.format(model),
             )

    ax = plt.gca()
    
#    ax.set_xlim([np.min(obs.time), np.max(obs.time)])
    ax.set_xlim([np.min(var_1d.time), np.max(var_1d.time)])
    
    if errors_computation:
        ## Errors computation
        obs_sorted[model] = []
        sim_sorted[model] = []
        for i, date in enumerate(ds.time):
            val = obs_var_corr.where(obs.time == date, drop=True).data
            if len(val) == 0:
                pass
            if len(val) == 1:
                sim_sorted[model].append(var_1d[i])
                obs_sorted[model].append(float(val))
            elif len(val) == 2:
                sim_sorted[model].append(var_1d[i])
                obs_sorted[model].append(float(val[0]))
                print('Weird: val = {0} at date {1}'.format(val, date))
        
        diff[model] = np.array(sim_sorted[model]) - np.array(obs_sorted[model])
        bias[model] = np.nanmean(diff[model])
        rmse[model] = np.sqrt(np.nanmean((np.array(obs_sorted[model]) - np.array(sim_sorted[model]))**2))
    

#%% Add irrigation datetime
if add_irrig_time:
    if site == 'irta-corn':
        sm_var = obs['VWC_40cm_Avg']
    if site == 'cendrosa':
        sm_var = obs['soil_moisture_3']
    if site == 'preixana':
        sm_var = None
    if site == 'elsplans':
        sm_var = None
    dati_list = tools.get_irrig_time(sm_var)
    plt.vlines(dati_list, 
               ymin=obs_var_corr.min().data, 
               ymax=obs_var_corr.max().data, 
               label='irrigation')

#%% Plot esthetics


#if varname_obs == '':
#    try:
#        ylabel = ds[varname_sim].comment
#    except AttributeError:
#        ylabel = varname_sim
#else:
#    try:
#        ylabel = obs[varname_obs].long_name
#    except AttributeError:
#        try:
#            ylabel = ds[varname_sim].comment
#        except (AttributeError, KeyError, NameError):
#            ylabel = varname_obs

plot_title = '{0} at {1}'.format(ylabel, site)
ax = plt.gca()
ax.set_ylabel(ylabel)

# add grey zones for night
days = np.arange(14,30)
for day in days:
    sunrise = pd.Timestamp('202107{0}-1930'.format(day))
    sunset = pd.Timestamp('202107{0}-0500'.format(day+1))
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
        functions=(lambda le: le/2264000,
                   lambda evap: evap*2264000))
    secax.set_ylabel('evapotranspiration [kg/m²/s]')

# add errors values on graph
if errors_computation:
    plt.text(.01, .95, 'RMSE: {0}'.format(rmse), 
             ha='left', va='top', transform=ax.transAxes)
    plt.text(.01, .99, 'Bias: {0}'.format(bias), 
             ha='left', va='top', transform=ax.transAxes)
    plt.legend(loc='upper right')
else:
    plt.legend(loc='best')

plt.title(plot_title)
plt.grid()


#%% Save figure

if save_plot:
    tools.save_figure(plot_title, save_folder)
