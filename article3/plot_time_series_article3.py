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
    
site = 'irta-corn'

file_suffix = 'dg'  # '' or 'dg'

varname_obs = ''
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
#LE, H, FC_mass, WS, WD, Ux, NETRAD
#VWC_40cm_Avg: Average volumetric water content at 35 cm (m3/m3) 
#T_20cm_Avg (_Std for standard deviation)
#TA_1_1_1, RH_1_1_1 Temperature and relative humidity 360cm above soil (~2m above maize)
#Q_1_1_1

varname_sim = 'WS'
# T2M_ISBA, LE_P4, EVAP_P9, GFLUX_P4, WG3_ISBA, WG4P9, SWI4_P9
# U_STAR, BOWEN

vmin, vmax = None, None

#If varname_sim is 3D:
ilevel =  10  #0 is Halo, 1->2m, 2->6.12m, 3->10.49m, 10 -> 50m

figsize = (6, 6) #small for presentation: (6,6), big: (15,9), paper:(7, 7)
plt.rcParams.update({'font.size': 11})

save_plot = True
save_folder = './fig/time_series/{0}/'.format(site)

models = [
        # 'std_d1',
        'noirr_lai_d1',
        # 'irrlagrip30_d1',
        # 'irrlagrip30thld07_d1',
        'irrswi1_d1',
        # 'irr_d1',
         ]

remove_alfalfa_growth = False
errors_computation = False
compare_to_residue_corr = False

add_seb_residue = True

add_irrig_time = False

kelvin_to_celsius = False

if 'irrlagrip30_d1' in models and errors_computation:
    print("""Warning: computation of errors will be run on all of july for
          'irrlagrip30_d1' - bug to fix in code""")

xmin = pd.Timestamp('20210715T00')
xmax = pd.Timestamp('20210716T00')

######################################################

simu_folders = {key:gv.simu_folders[key] for key in models}
father_folder = gv.global_simu_folder

date = '2021-07'

colordict = {
    'irr_d2': 'g', 
    'std_d2': 'r',
    'irr_d1': 'g',
    'irrswi1_d1': 'b', 
    'noirr_lai_d1': 'r', 
    'std_d1': 'r',
    'irrlagrip30_d1': 'y',
    'obs': 'k'}

models_name = {
    'irrswi1_d1': 'IRR_FC',
    # 'std_d1',
    'noirr_lai_d1': 'NOIRR',
#        'irrlagrip30_d1',
    }

#%% Dependant Parameters

# default values (can be change below)
offset_obs = 0
coeff_obs = 1
secondary_axis = None
ylabel = varname_sim

if varname_sim in ['LE', 'LE_ISBA']:
    vmin, vmax = -50, 750
    secondary_axis = 'evap'
    figsize = (6, 6)
    ylabel = 'Latent heat flux [W m$^{-2}$]'
elif varname_sim in ['RN', 'RN_ISBA']:
    vmin, vmax = -350, 750
    add_seb_residue = False
    ylabel = 'Net radiation [W m$^{-2}$]'
elif varname_sim in ['T2M_ISBA',]:
    vmin, vmax = 285, 310
    add_seb_residue = False
    ylabel = 'Air temperature at 2 m [K]'
    figsize = (6, 2.5)
elif varname_sim in ['HU2M_ISBA',]:
    vmin, vmax = 0, 1
    add_seb_residue = False
    ylabel = 'Relative humidity at 2 m []'
    figsize = (6, 2.5)
elif varname_sim in ['WS',]:
    vmin, vmax = 0, 7
    add_seb_residue = False
    ylabel = 'Wind speed [m s$^{-1}$]'
    figsize = (6, 3)


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
    if site in ['preixana']:
        # net radiation
        obs['rn'] = obs['swd'] + obs['lwd'] - obs['swup'] - obs['lwup']
        # bowen ratio -  diff from bowen_ratio_1
        obs['bowen'] = (obs['shf'] / obs['lhf']).clip(min=-0.5, max=10)
        # obs['bowen'] = np.clip(obs['bowen'], -0.5, 10)
        obs['SEB_RESIDUE'] = obs['rn']-obs['lhf']-obs['shf']-obs['soil_heat_flux']
        obs['EVAP_FRAC'] = obs['lhf'] / (obs['lhf'] + obs['shf'])
        obs['EVAP_FRAC_FILTERED'] = obs['EVAP_FRAC'].clip(min=0, max=1)
        for i in [1,2,3]:
            obs['swi_{0}'.format(i)] = tools.calc_swi(
                    obs['soil_moisture_{0}'.format(i)],
                    gv.wilt_pt[site][i],
                    gv.field_capa[site][i],) 
    elif site in ['cendrosa']:
        # net radiation
        obs['rn'] = obs['swd'] + obs['lwd'] - obs['swup'] - obs['lwup']
        obs['albedo'] = obs['swup']/obs['swd']
        # bowen ratio -  diff from bowen_ratio_1
        obs['bowen'] = (obs['shf_1'] / obs['lhf_1']).clip(min=-0.5, max=10)
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
        
        for i in [10,20,30,40]:
            obs['SWI{0}_subsoil'.format(i)] = tools.calc_swi(
                    obs['PR{0}_subsoil'.format(i)]*0.01,  #conversion from % to decimal
                    gv.wilt_pt[site][i],
                    gv.field_capa[site][i],)
        
#%% OBS PLOT:
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
        if varname_obs == 'RAIN_subsoil':
            obs_var_filtered = obs[varname_obs].where(
                obs[varname_obs] < 10, 
                np.nan)
        else:
            obs_var_filtered = obs[varname_obs].where(
                (obs[varname_obs]-obs[varname_obs].mean()) < (4*obs[varname_obs].std()), 
                np.nan)
        # if varname_obs == 'RAIN_subsoil':
        #     obs_var_filtered = obs[varname_obs]
        obs_var_corr = (obs_var_filtered+offset_obs)*coeff_obs
        plt.plot(obs_var_corr.time, obs_var_corr, 
                 label='obs_'+varname_obs,
                 color=colordict['obs'])
    else:
        if remove_alfalfa_growth and site == 'cendrosa':  # because of growth of alfalfa
            obs = obs.where(obs.time>pd.Timestamp('20210721T0100'), drop=True)
        
        # to remove intense rainfall
        if varname_obs == 'rain_cumul':
            obs = obs.where(obs.time<pd.Timestamp('20210725T0100'), drop=True)
        
        if site == 'irta-corn':
            obs = obs.where(~obs.time.isnull(), drop=True)
        
        # filter outliers (turn into NaN)
        obs_var_filtered = obs[varname_obs].where(
                (obs[varname_obs]-obs[varname_obs].mean()) < (4*obs[varname_obs].std()), 
                np.nan)
        # apply correction for comparison with models
        obs_var_corr = ((obs[varname_obs]+offset_obs)*coeff_obs)
    
        # plot
        plt.plot(obs_var_corr.time, obs_var_corr, 
                 label='obs_'+varname_obs,
                 color=colordict['obs'])
#        obs_var_corr.plot(label='obs_'+varname_obs,
#                          color=colordict['obs'],
#                          linewidth=1)
        
    if add_seb_residue:
        
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


#%% SIMU - LOAD and PLOT:
diff = {}
rmse = {}
bias = {}
obs_sorted = {}
sim_sorted = {}


out_suffix = ''
file_suffix = 'dg'

if varname_sim == 'U_STAR':
    varname_sim_preproc = ['FMU_ISBA', 'FMV_ISBA']
elif varname_sim == 'BOWEN':
    varname_sim_preproc = ['H_ISBA', 'LE_ISBA']
elif varname_sim in ['WS', 'WD']:
    varname_sim_preproc = ['UT', 'VT']
    # out_suffix = '.OUT'
    # file_suffix = ''
else:
    varname_sim_preproc = [varname_sim,]

for model in simu_folders:

    ds = tools.load_series_dataset(varname_sim_preproc, model,
                                   out_suffix=out_suffix,
                                   file_suffix=file_suffix)
    
    try:
        index_lat, index_lon = tools.indices_of_lat_lon(ds, lat, lon)
    except AttributeError:  #if the data does not have lat-lon data, merge with another that have it
        ds = tools.load_series_dataset(['H_ISBA',] + varname_sim_preproc, model)
        # and now, try again:
        index_lat, index_lon = tools.indices_of_lat_lon(ds, lat, lon)
    
    # Compute other diag variables
    if varname_sim == 'U_STAR':
        ds['U_STAR'] = tools.calc_u_star_sim(ds['FMU_ISBA'], ds['FMV_ISBA'])
    elif varname_sim == 'BOWEN':
        ds['BOWEN'] = tools.calc_bowen_sim(ds)
    elif varname_sim in ['WS', 'WD']:
        ds = ds.isel(level=ilevel)
        print(ds)
        print('centering UT and VT...')
        ds = tools.center_uvw(ds)
        print('computing WS and WD...')
        ds['WS'], ds['WD'] = tools.calc_ws_wd(ds['UT'], ds['VT'])
    
    # Set time abscisse axis
    try:
        start = ds.time.data[0]
    except IndexError:
        start = ds.time.data
    except AttributeError:
        print('WARNING: time array is hardcoded')
        start = np.datetime64('2021-07-21T01:00')
    
    dati_arr_sim = np.array([start + np.timedelta64(i, 'h') for i in np.arange(0, ds[varname_sim].shape[0])])

    ds = ds.squeeze()
    
    ds['record'] = dati_arr_sim
    ds = ds.drop_vars(['time'])
    ds = ds.rename({'record': 'time'})
        
    # find indices from lat,lon values 
    index_lat, index_lon = tools.indices_of_lat_lon(ds, lat, lon)
    
    # keep variable of interest
    var_md = ds[varname_sim]
    
    # to compare performance score on only 2 last weeks of july - BUG
#        if model == 'irrlagrip30_d1':
#            var_md = var_md.where(var_md.time > pd.Timestamp('20210714T0100'), drop=True)
    
    if kelvin_to_celsius:
        var_md = var_md - 273.15
    
    # if len(var_md.shape) == 5:
    #     var_1d = var_md[:, :, ilevel, index_lat, index_lon].data  #1st index is time, 2nd is ?, 3rd is Z,..
    # elif len(var_md.shape) == 4:
    #     var_1d = var_md[:, ilevel, index_lat, index_lon].data  #1st index is time, 2nd is Z,..
    # elif len(var_md.shape) == 3:
    #     var_1d = var_md[:, index_lat, index_lon].data
    var_1d = var_md.isel(nj=index_lat, ni=index_lon)
    
    # PLOT
    plt.plot(ds.time, var_1d, 
             color=colordict[model],
             linestyle='--',
             label=f'simu_{models_name[model]}',
#                 label=f'simu_{model}',
             )
   
    if errors_computation and varname_obs != '':
        ## Errors computation
        obs_sorted[model] = []
        sim_sorted[model] = []
        
        if compare_to_residue_corr:
            obs_var_corr = obs_residue_corr
        
        # interp obs on datetime array of simu
        dati_arr_sim_unix = np.float64(ds.time)/1e9
        dati_arr_obs_unix = np.float64(np.array(obs.time))/1e9
        obs_data_interp = np.interp(
                dati_arr_sim_unix, dati_arr_obs_unix, obs_var_corr.values,
                left=np.nan, right=np.nan)
        
        diff[model] = var_1d - obs_data_interp

        # compute bias and rmse, and keep values with 3 significant figures
        bias[model] = float('%.3g' % np.nanmean(diff[model]))
#        rmse[model] = np.sqrt(np.nanmean((np.array(obs_sorted[model]) - np.array(sim_sorted[model]))**2))
        rmse[model] = float('%.3g' % np.sqrt(np.nanmean(diff[model]**2)))
    

#%% Add irrigation datetime
if add_irrig_time and varname_obs != '':
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


plot_title = '{0} at {1}'.format(ylabel, site)
ax = plt.gca()
ax.set_ylabel(ylabel)
ax.set_ylim([vmin, vmax])


if xmin is None:
    try:
        xmin = np.min(dati_arr_sim)
        xmax = np.max(dati_arr_sim)  - pd.Timedelta(1, 'h')
    except:
        xmin = None
        xmax = None

ax.set_xlim([xmin, xmax])
ax.set_xlabel('time UTC')

# add grey zones for night
days = np.arange(1,30)
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


# plt.title(plot_title)
plt.grid()

# keep only hours as X axis
#plt.xticks(dati_arr[1:25:2], labels=np.arange(2,25,2))
plt.xticks(rotation=30)


#plt.tight_layout()  # ensure that all figure elements fit in frame
#%% Save figure
save_title = f'{varname_sim}_{site}'
if save_plot:
    tools.save_figure(save_title, save_folder)
#    tools.save_figure(plot_title, '/d0/images/lunelt/figures/')
