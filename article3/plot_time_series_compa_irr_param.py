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
import matplotlib


############# Independant Parameters (TO FILL IN):
    
site = 'irta-corn'

varname_sim = 'HU2M_ISBA'    # TG3_ISBA, LE_ISBA, LWU_ISBA, TG2_ISBA, U_STAR, SWI4_ISBA
varname_obs = 'RH_1_1_1'  # T_10cm_Avg, NETRAD, LE, LW_OUT
# -- For CNRM:
# ta_5, hus_5, hur_5, soil_moisture_3, soil_temp_3, u_var_3, w_var_3, swd,... 
# w_h2o_cov, h2o_flux[_1], shf_1, u_star_1
# from données lentes: 1->0.2m, 2->2m, 3->10m, 4->25m, 5->50m
# from eddy covariance measures: 1->3m, 2->25m, 3->50m
# -- For IRTA-corn
#LE, H, FC_mass, WS, WD, U_STAR, SW_OUT, NETRAD, Bowen_ratio
#VWC_40cm_Avg: Average volumetric water content at 35 cm (m3/m3), 'swi_4'
#T_20cm_Avg (_Std for standard deviation)
#TA_1_1_1, RH_1_1_1 Temperature and relative humidity 360cm above soil (~2m above maize)
#Q_1_1_1

figsize = (7, 6) #small for presentation: (6,6), big: (15,9)
save_plot = False
save_folder = './fig/'

forcing_file = 'obs'
options = 'AST'

models = [
        # 'IRRLAGRIP10_THLD05',
        # 'IRRLAGRIP30_THLD05',
        # 'IRRLAGRIP50_THLD05',
        # 'IRRLAGRIP70_THLD05',
        # 'IRRLAGRIP100_THLD05',        
        # 'IRRLAGRIP50_THLD07',
        # 'IRRLAGRIP30_THLD07',
        # 'IRRLAGRIP100_THLD07',
        # 'IRRSWI12',
        # 'IRRSWI12TEST',
        'IRRSWI10',
        # 'IRRSWI08',
        # 'IRRSWI07',
        # 'IRRSWI06', 
        # 'IRRSWI05',
        # 'IRRSWI04',
        # 'IRRSWI03',
        # 'IRRSWI02',
        # 'IRRSWI00',
        # 'MASTER',
         ]

errors_computation = False
remove_alfalfa_growth = False
add_fao56_et = False
add_seb_residue = True
error_on_residue = True
add_irrig_time = False
time_shift_correction = 30  # in minutes
one_day_focus = None    # remove days different from this day
one_day_zoom_plot = None  # just zoom on one day (do not necessarily remove the rest)
kelvin_to_celsius_simu_data = False  # for TG2_ISBA, TG3_SIBA
time_window_limits = None  #[14, 31], None  - time window for error computation

if varname_obs == '':
    errors_computation = False
if varname_obs not in ['LE', 'lhf_1', 'H', 'shf_1']:
    add_seb_residue = False
    error_on_residue = False
if varname_sim[0] != 'T' and kelvin_to_celsius_simu_data:
    print('Warning: -273.15 applied to simu data!')
if one_day_focus not in [0, None]:
    one_day_zoom_plot = one_day_focus
if site == 'irta-corn' and varname_sim in ['TG2_ISBA', 'TG3_ISBA']:
    kelvin_to_celsius_simu_data = True

######################################################

#simu_folders = {key:gv.simu_folders[key] for key in models}
#father_folder = gv.global_simu_folder

date = '2021-07'

cmap = matplotlib.cm.get_cmap('Blues')
rgba = cmap(0)

colordict = {
        'IRRLAGRIP30_THLD05': '#008000ff',  #std green
        'IRRLAGRIP100': '#73d216ff',
        'IRRSWI12': cmap(0.999),  # std blue
        'IRRSWI12TEST': cmap(0.999),  # std blue
        'IRRSWI12PL98': cmap(0.999),  # std blue
        'IRRSWI10': cmap(0.999),
        'IRRSWI08': cmap(0.9),
        'IRRSWI07': cmap(0.85),
        'IRRSWI06': cmap(0.8),
        'IRRSWI05': cmap(0.75),
        'IRRSWI04': cmap(0.7),
        'IRRSWI03': cmap(0.65),
        'IRRSWI02': cmap(0.6),
        'IRRSWI00': cmap(0.5),
        'MASTER': 'r',
        'obs': 'k',
        'IRRSWI10_LAI30_Z001': 'b',
        'IRRSWI10_LAI30_Z003': 'b',
        }
linedict = {
        'IRRLAGRIP30_THLD05': (0, (1, 10)),
        # 'IRRLAGRIP30_THLD05': 'None',
        'IRRLAGRIP100': ':',
        'IRRSWI12': (0, (3, 2, 1, 2)),
        'IRRSWI12PL98': (0, (2.5, 7.5)),
        'IRRSWI12TEST': (0, (2.5, 7.5)),
        'IRRSWI10': (0, (3, 3, 1, 3)),
        # 'IRRSWI10': '--',
        'IRRSWI08': (0, (3, 7)),
        'IRRSWI07': (0, (3.5, 6.5)),
        'IRRSWI06': (0, (4, 6)),
        'IRRSWI05': (0, (3, 2)),
        'IRRSWI04': (0, (4, 1)),
        'IRRSWI03': (0, (3, 2)),
        'IRRSWI02': (0, (7, 3)),
        'IRRSWI00': (0, (7.5, 2.5)),
        # 'IRRSWI00': '--',
        'MASTER': (0, (3, 3, 1, 1, 1, 1)),
        # 'MASTER': '-.',
        'obs': '-',
        'IRRSWI10_LAI30_Z001': '--',
        'IRRSWI10_LAI30_Z003': '--',
        }

markerdict = {
        'IRRLAGRIP30_THLD05': '.',
        }

#if site in ['cendrosa', 'irta-corn', 'elsplans']:
#    folder = '/cnrm/surface/lunelt/NO_SAVE/sfx_out/{0}/comp_irrig_param_forc_obs/{1}/'.format(site, model)
#else:
#    raise KeyError('No simulation data for this site')


#%% Dependant Parameters

# default values (can be change below)
offset_obs = 0
coeff_obs = 1
secondary_axis = None
    
if varname_sim in ['LE', 'LE_ISBA']:
    ylabel = 'Latent heat flux [W m$^{-2}$]'
    secondary_axis = 'evap'
    ymin, ymax = -100, 800
elif varname_sim in ['H', 'H_ISBA']:
    ylabel = 'Sensible heat flux [W m$^{-2}$]'
    ymin, ymax = -300, 600
elif varname_sim in ['TG2_ISBA', 'TG3_ISBA',]:
    ylabel = 'Soil temperature [°C]'
    ymin, ymax = 10, 50
elif varname_sim in ['RN_ISBA',]:
    ylabel = 'Net radiation [W m$^{-2}$]'
    ymin, ymax = -200, 850
elif varname_sim in ['U_STAR',]:
    ylabel = 'Friction velocity [m s$^{-1}$]'
    ymin, ymax = 0, 0.6
elif varname_sim in ['HU2M_ISBA',]:
    ylabel = 'Relative humidity [%]'
    coeff_obs = 0.01
    ymin, ymax = 0, 1
else:
    ylabel = varname_sim
    ymin, ymax = None, None
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
    in_filenames_obs = 'LIAISE_IRTA-CORN_UIB_SEB-10MIN_L2.nc'
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
    obs['SEB_RESIDUE'] = obs['rn']-obs['lhf_1']-obs['shf_1']-obs['soil_heat_flux']
    # bowen ratio -  diff from bowen_ratio_1
    obs['bowen'] = obs['shf_1'] / obs['lhf_1']
    obs['EVAP_FRAC'] = obs['lhf_1'] / (obs['lhf_1'] + obs['shf_1'])
    obs['EVAP_FRAC_FILTERED'] = np.clip(obs[varname_obs], 0, 1)
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
    obs['air_density'] = obs['PA']*1000/(287.05*(obs['TA_1_1_1']+273.15))
    obs['U_STAR'] = np.sqrt(obs['TAU']/obs['air_density'])
    obs['SEB_RESIDUE'] = obs['NETRAD']-obs['LE']-obs['H']-obs['G_plate_1_1_1']
    obs['EVAP_FRAC'] = obs['LE'] / (obs['LE'] + obs['H'])
    obs['EVAP_FRAC_FILTERED'] = np.clip(obs['EVAP_FRAC'], 0, 1)
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

obs_complete = obs


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
        if remove_alfalfa_growth and site == 'cendrosa':  # because of growth of alfalfa
            obs = obs.where(obs.time>pd.Timestamp('20210721T0100'), drop=True)
        if one_day_focus not in [0, None]:
            # obs = obs.where(obs.time>pd.Timestamp(f'202107{str(one_day_focus).zfill(2)}T0000'), drop=True)
            # obs = obs.where(obs.time<pd.Timestamp(f'202107{str(one_day_focus).zfill(2)}T2330'), drop=True)            
            obs = obs.where(obs.time>pd.Timestamp(f'202107{str(one_day_focus).zfill(2)}T0530'), drop=True)
            obs = obs.where(obs.time<pd.Timestamp(f'202107{str(one_day_focus).zfill(2)}T1930'), drop=True)
            
        # filter outliers (turn into NaN)
        obs_var_filtered = obs[varname_obs].where(
                (obs[varname_obs]-obs[varname_obs].mean()) < (4*obs[varname_obs].std()), 
                np.nan)
        # apply correction for comparison with models
        obs_var_corr = ((obs[varname_obs]+offset_obs)*coeff_obs)
        # plot
        obs_var_corr.plot(
#                label='obs_'+varname_obs,
                label='obs',
                color=colordict['obs'],
                linewidth=1)
            
        if add_seb_residue:
            # get EVAP_FARC_between 0 and 1
            # EF_temp_min0 = [max(0, val) for val in obs.EVAP_FRAC.data]
            # EF_temp_max1 = [min(1, val) for val in EF_temp_min0]
            # obs['EVAP_FRAC_FILTERED'] = EF_temp_max1
            obs['EVAP_FRAC_FILTERED'] = np.clip(obs[varname_obs], 0, 1)
            
            obs_uncertainty = obs['SEB_RESIDUE'].data
            
            if varname_obs in ['LE', 'lhf_1']:
                obs_residue_corr = obs_var_corr + obs['SEB_RESIDUE']*obs['EVAP_FRAC_FILTERED'].data
            elif varname_obs in ['H', 'shf_1']:
                obs_residue_corr = obs_var_corr + obs['SEB_RESIDUE']*(1-obs['EVAP_FRAC_FILTERED'].data)
            else:
                raise ValueError('add_seb_residue available only on LE, lhf_1, H, shf_1')
                
            obs_residue_corr.plot(
                label='obs_residue_adj',
                color=colordict['obs'],
                linestyle=':',
                linewidth=1)
            
            plt.fill_between(obs_var_corr.time, 
                              obs_var_corr.data,
                              obs_var_corr.data + obs_uncertainty.data,
                              alpha=0.2, 
                              facecolor=colordict['obs'],
                              )
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


for model in models:
    datafolder = (
        '/cnrm/surface/lunelt/NO_SAVE/sfx_out/' + \
        'et_overestimation/{0}/forcing_{1}_{2}/{3}/'.format(
            site, forcing_file, options, model))
#    datafolder = folder + 'forcing_{0}/'.format(models_idnumber[model])
    
    # ds = xr.open_dataset(datafolder + 'SURF_ATM_DIAGNOSTICS.OUT.nc')
    ds = xr.open_dataset(datafolder + 'ISBA_DIAGNOSTICS.OUT.nc',
                          decode_times=False)
    ds['time'] = xr.open_dataset(datafolder + 'SURF_ATM_DIAGNOSTICS.OUT.nc').time
    if time_shift_correction not in [None, 0]:
        ds['time'] = ds['time'] - pd.Timedelta(time_shift_correction, 'm')
    
    # Compute other diag variables
    if varname_sim == 'U_STAR':
        ds['U_STAR'] = tools.calc_u_star_sim(ds['FMU_ISBA'], ds['FMV_ISBA'])
    elif varname_sim == 'BOWEN':
        ds['BOWEN'] = tools.calc_bowen_sim(ds)
    elif add_fao56_et:
        ds_forcing = xr.open_dataset(datafolder + 'FORCING_{0}_{1}.nc'.format(
                site, forcing_file))
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
    
    if kelvin_to_celsius_simu_data:
        var_1d = var_1d - 273.15
    
    # PLOT
    if 'SWI' in model:
        legend_label = f'simu_{model[3:]}'
    elif 'LAGRIP' in model:
        # new_name = model.replace('LAGRIP','_WAT')
        new_name = model.replace('LAGRIP','_THLD_')[:-7]
        legend_label = f'simu_{new_name}'
    elif model == 'MASTER':
        legend_label = 'simu_NOIRR'
    else:
        legend_label = f'simu_{model}'
    
    plt.plot(var_1d.time, var_1d,
             color=colordict.get(model, 'purple'),
             linestyle=linedict.get(model, '--'),
             marker=markerdict.get(model, ','),
             markersize=8,
             linewidth=2,
#             label='simu_{0}_{1}'.format(model, varname_sim),
             label=legend_label,
             )
    if add_fao56_et:
        plt.plot(LE_0.time, LE_0, 
             color=colordict.get(model, 'purple'),
             linestyle=linedict[model],
             label='simu_{0}_LE_0_FAO56'.format(model),
             )

    ax = plt.gca()
    
#    ax.set_xlim([np.min(obs.time), np.max(obs.time)])
    ax.set_xlim([np.min(var_1d.time), np.max(var_1d.time)])
    
    if errors_computation:
        ## Errors computation
        obs_sorted[model] = []
        sim_sorted[model] = []
        # compute on restricted time window or full time window
        # - 1 day window:
#        day = 10
#        timestep = 0.5  # in hour
#        time_window = ds.time[:][(day-1)*int(24/timestep) : (day)*int(24/timestep)]
        # - multiple days window:
        if time_window_limits != None:
            day_start = time_window_limits[0]
            day_end = time_window_limits[1]
            timestep = 0.5  # in hour
            time_window = ds.time[:][(day_start-1)*int(24/timestep) : (day_end)*int(24/timestep)]
        else:
            time_window = ds.time
    
        for i, date in enumerate(time_window):
            if error_on_residue:
                val = obs_residue_corr.where(obs.time == date, drop=True).data
            else:
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
        # compute bias and rmse, and keep values with 3 significant figures
        bias[model] = float('%.3g' % np.nanmean(diff[model]))
#        rmse[model] = np.sqrt(np.nanmean((np.array(obs_sorted[model]) - np.array(sim_sorted[model]))**2))
        rmse[model] = float('%.3g' % np.sqrt(np.nanmean(diff[model]**2)))
    

#%% Add irrigation datetime
if add_irrig_time:
    if site == 'irta-corn':
        sm_var = obs_complete['VWC_40cm_Avg']
    if site == 'cendrosa':
        sm_var = obs_complete['soil_moisture_3']
    if site == 'preixana':
        sm_var = None
    if site == 'elsplans':
        sm_var = None
    dati_list = tools.get_irrig_time(sm_var)
    plt.vlines(dati_list, 
               ymin=obs_var_corr.min().data, 
               ymax=obs_var_corr.max().data+1, 
               label='irrigation',
               color='b')

#%% Plot esthetics

if one_day_zoom_plot not in [0, None]:
    xmin = pd.Timestamp(f'202107{str(one_day_zoom_plot).zfill(2)}T0000')
    xmax = pd.Timestamp(f'202107{str(one_day_zoom_plot).zfill(2)}T2330')
elif remove_alfalfa_growth: 
    xmin = pd.Timestamp('20210721T0100')
    xmax = pd.Timestamp('20210730T2300')
else:
    xmin = obs.time.min(skipna=True)
    xmax = obs.time.max(skipna=True)

ax = plt.gca()
ax.set_ylabel(ylabel)
ax.set_ylim(ymin, ymax)
ax.set_xlabel('time UTC')
ax.set_xlim(xmin, xmax)
# ax.set_xbound(xmin, xmax)
# ax.set_xlim([datetime.date(2021, 7, 15), datetime.date(2021, 7, 17)])

# add grey zones for night
days = np.arange(1,31)
for day in days:
    # zfill(2) allows to have figures with two digits
    sunrise = pd.Timestamp('202107{0}-1930'.format(str(day).zfill(2)))
    sunset = pd.Timestamp('202107{0}-0500'.format(str(day+1).zfill(2)))
    ax.axvspan(sunset, sunrise, ymin=0, ymax=1, 
               color = '0.9'  #'1'=white, '0'=black, '0.8'=light gray
               )

# add secondary axis on the right, relative to the left one - (for LE)
if secondary_axis == 'le':
    axes = plt.gca()
    secax = axes.secondary_yaxis("right",                              
        functions=(lambda evap: evap*2264000,
                   lambda le: le/2264000))
    secax.set_ylabel('Latent heat flux [W m$^{-2}$]')
if secondary_axis == 'evap':
    axes = plt.gca()
    # secax = axes.secondary_yaxis("right",                              
    #     functions=(lambda le: le/2264000,
    #                lambda evap: evap*2264000))
    # secax.set_ylabel('Evapotranspiration [kg/m²/s]')
    secax = axes.secondary_yaxis("right",                              
        functions=(lambda le: (le*3600)/2264000,
                   lambda evap: (evap*2264000)/3600))
    secax.set_ylabel('Evapotranspiration [mm h$^{-1}$]')

# add errors values on graph
if errors_computation:
    plt.text(.01, .95, 'RMSE: {0}'.format(rmse), 
             ha='left', va='top', transform=ax.transAxes)
    plt.text(.01, .99, 'Bias: {0}'.format(bias), 
             ha='left', va='top', transform=ax.transAxes)
    plt.legend(loc='upper right')
    print('RMSE:') 
    print(rmse)
    print('Bias:') 
    print(bias)
else:
    plt.legend(loc='best')

# plt.title(plot_title)
plt.grid()
plt.subplots_adjust(bottom=0.15)


#%% other computation
# obs_residue_corr
# LE_energy = obs_residue_corr.sum() * 10*60  # in J/m2
# LE_mm = LE_energy/2264000
# LE_mm_tot = LE_mm.sum()
# print(LE_mm_tot)


#%% Save figure
save_title = '{0}_{1}_{2}'.format(varname_sim, site, one_day_zoom_plot)
if save_plot:
    tools.save_figure(save_title, save_folder)
