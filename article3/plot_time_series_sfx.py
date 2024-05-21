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
import pickle


############# Independant Parameters (TO FILL IN):
    
site = 'irta-corn'
vegtype = 'C3'

option = ''  # fracveg070

varname_sim = 'LETR_ISBA'  #

veruser_list = [
    # 'MASTER',
    # 'IRRSWI00',
    # 'IRRSWI01',
    # 'IRRSWI02',
    # 'IRRSWI03',
    # 'IRRSWI04',
    # 'IRRSWI05',
    # 'IRRSWI06',
    # 'IRRSWI07',
    # 'IRRSWI08',
    # 'IRRSWI09',
    'IRRSWI10',
    # 'IRRSWI12',
    # 'IRRLAGRIP30',
    ]
cphoto_list = [
    'AST',
    'NON',
    ]

forc_models = [
        'irrswi1_d1', 
        # 'std_d1',
        'noirr_lai_d1',
#        'irrlagrip30_d1',
         ]

varname_obs = ''
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

plot_figure = False
add_irrig_time = False
figsize = (6, 6) #small for presentation: (6,6), big: (15,9)
save_plot = False
save_folder = f'./fig/compa_forcings/{site}/'
to_pickle = False

errors_computation = False
add_seb_residue = False
add_fao56_et = True
######################################################

#simu_folders = {key:gv.simu_folders[key] for key in forc_models}
father_folder = gv.global_simu_folder

date = '2021-07'

forc_models_idnumber = {
    'irr_d1': '2.15',
    'std_d1': '1.15',
    'noirr_lai_d1': '3.01',
    'irrlagrip30_d1': '7.15',
    'irrswi1_d1': '8.16',
    }

colordict = {
    'irr_d2': 'g', 
    'std_d2': 'r',
    'irr_d1': 'g',
    'irrswi1_d1': 'g', 
    'noirr_lai_d1': 'r', 
    'std_d1': 'r',
    'irrlagrip30_d1': 'y',
    'obs': 'k'}

if site == 'irta-corn-real':
    site = 'irta-corn'

if site in ['cendrosa', 'irta-corn', 'elsplans']:
    folder = f'/cnrm/surface/lunelt/NO_SAVE/sfx_out/et_overestimation/{option}/{site}/'
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

# if site == 'cendrosa':
#     datafolder = '/cnrm/surface/lunelt/data_LIAISE/cendrosa/30min/'
#     filename_prefix = 'LIAISE_LA-CENDROSA_CNRM_MTO-FLUX-30MIN_L2_'
#     in_filenames_obs = filename_prefix + date
# elif site == 'preixana':
# #    varname_sim_suffix = '_ISBA'
#     datafolder = '/cnrm/surface/lunelt/data_LIAISE/preixana/30min/'
#     filename_prefix = 'LIAISE_PREIXANA_CNRM_MTO-FLUX-30MIN_L2_'
#     in_filenames_obs = filename_prefix + date
# elif site == 'elsplans':
#     freq = '30'  # '5' min or '30'min
#     datafolder = '/cnrm/surface/lunelt/data_LIAISE/elsplans/mat_50m/{0}min/'.format(freq)
#     filename_prefix = 'LIAISE_'
#     date = date.replace('-', '')
#     in_filenames_obs = filename_prefix + date
# #    varname_sim_suffix = '_ISBA'  # or P7, but already represents 63% of _ISBA
# elif site == 'irta-corn':
#     datafolder = '/cnrm/surface/lunelt/data_LIAISE/irta-corn/seb/'
#     in_filenames_obs = 'LIAISE_IRTA-CORN_UIB_SEB-10MIN_L2.nc'
# #    raise ValueError('Site name not known')
    
lat = gv.sites[site]['lat']
lon = gv.sites[site]['lon']


#%% OBS: Concatenate and plot data
# if site == 'irta-corn':
#     out_filename_obs = in_filenames_obs
# #    dat_to_nc = 'uib'  #To create a new netcdf file
#     dat_to_nc = None   #To keep existing netcdf file
# elif site == 'elsplans':
#     out_filename_obs = 'CAT_' + date + filename_prefix + '.nc'
# #    dat_to_nc = 'ukmo'
#     dat_to_nc = None   #To keep existing netcdf file
# else:
#     out_filename_obs = 'CAT_' + date + filename_prefix + '.nc'
#     dat_to_nc = None
    
# # CONCATENATE multiple days
# tools.concat_obs_files(datafolder, in_filenames_obs, out_filename_obs, 
#                        dat_to_nc=dat_to_nc)

# obs = xr.open_dataset(datafolder + out_filename_obs)

# # process other variables:
# if site in ['preixana', 'cendrosa']:
#     # net radiation
#     obs['rn'] = obs['swd'] + obs['lwd'] - obs['swup'] - obs['lwup']
#     # bowen ratio -  diff from bowen_ratio_1
#     obs['bowen'] = obs['shf_1'] / obs['lhf_1']
#     # potential evapotranspiration
#     obs['lhf_0_fao56'] = tools.calc_fao56_et_0(
#         obs['rn'], 
#         obs['ta_2'], 
#         obs['ws_1'], 
#         obs['hur_2'], 
#         obs['pa']*100,
#         gnd_flx=obs['soil_heat_flux'])['LE_0']
#     for i in [1,2,3]:
#         obs['swi_{0}'.format(i)] = tools.calc_swi(
#                 obs['soil_moisture_{0}'.format(i)],
#                 gv.wilt_pt[site][i],
#                 gv.field_capa[site][i],) 
#     obs['SEB_RESIDUE'] = obs['rn']-obs['lhf_1']-obs['shf_1']-obs['soil_heat_flux']
#     obs['EVAP_FRAC'] = obs['lhf_1'] / (obs['lhf_1'] + obs['shf_1'])
#     EF_temp_min0 = [max(0, val) for val in obs.EVAP_FRAC.data]
#     EF_temp_max1 = [min(1, val) for val in EF_temp_min0]
#     obs['EVAP_FRAC_FILTERED'] = EF_temp_max1
# elif site == 'irta-corn':
#     for i in [2,3,4,5]:
#         obs['swi_{0}'.format(i)] = tools.calc_swi(
#                 obs['VWC_{0}0cm_Avg'.format(i)],
#                 gv.wilt_pt[site][i],
#                 gv.field_capa[site][i],)
#     obs['Q_1_1_1'] = tools.psy_ta_rh(
#         obs['TA_1_1_1'], 
#         obs['RH_1_1_1'],
#         obs['PA']*1000)['hr']
#     obs['LE_0_FAO56'] = tools.calc_fao56_et_0(
#         obs.NETRAD, 
#         obs.TA_1_1_1, 
#         obs.WS, 
#         obs.RH_1_1_1, 
#         obs.PA*1000,
#         gnd_flx=obs.G_plate_1_1_1)['LE_0']
#     obs['air_density'] = obs['PA']*1000/(287.05*(obs['TA_1_1_1']+273.15))
#     obs['U_STAR'] = np.sqrt(obs['TAU']/obs['air_density'])
#     obs['SEB_RESIDUE'] = obs['NETRAD']-obs['LE']-obs['H']-obs['G_plate_1_1_1']
#     obs['EVAP_FRAC'] = obs['LE'] / (obs['LE'] + obs['H'])
#     EF_temp_min0 = [max(0, val) for val in obs.EVAP_FRAC.data]
#     EF_temp_max1 = [min(1, val) for val in EF_temp_min0]
#     obs['EVAP_FRAC_FILTERED'] = EF_temp_max1
# elif site == 'elsplans':
#     ## Flux calculations
#     obs['H_2m'] = obs['WT_2m']*1200  # =Cp_air * rho_air
#     obs['LE_2m'] = obs['WQ_2m']*2264000  # =L_eau
    
#     ## Webb Pearman Leuning correction
#     obs['BOWEN_2m'] = obs['H_2m'] / obs['LE_2m']
#     #obs['WQ_2m_WPL'] = obs['WQ_2m']*(1.016)*(0+(1.2/300)*obs['WT_2m'])  #eq (25)
#     obs['LE_2m_WPL'] = obs['LE_2m']*(1.010)*(1+0.051*obs['BOWEN_2m'])  #eq (47) of paper WPL
#     for i in [10,20,30,40]:
#         obs['SWI{0}_subsoil'.format(i)] = tools.calc_swi(
#                 obs['PR{0}_subsoil'.format(i)]*0.01,  #conversion from % to decimal
#                 gv.wilt_pt[site][i],
#                 gv.field_capa[site][i],)
    
# # PLOT:
# if plot_figure:
    
#     if varname_obs != '':
        
#         fig = plt.figure(figsize=figsize)
        
#         if site == 'elsplans':
#             ## create datetime array
#         #    dati_arr = pd.date_range(start=obs.time.min().values, 
#             dati_arr = pd.date_range(pd.Timestamp('20210701-0000'),
#                                      periods=len(obs[varname_obs]), 
#                                      freq='{0}T'.format(freq))
#             obs['time']=dati_arr
#     #        # filter outliers (turn into NaN)
#     #        obs_var_filtered = obs[varname_obs].where(
#     #                (obs[varname_obs]-obs[varname_obs].mean()) < (4*obs[varname_obs].std()), 
#     #                np.nan)
#     #        obs_var_corr = (obs_var_filtered+offset_obs)*coeff_obs
#     ##        plt.plot(dati_arr, obs_var_corr, 
#     ##                 label='obs_'+varname_obs,
#     ##                 color=colordict['obs'])
#     #        obs_var_corr.plot(
#     ##            label='obs_'+varname_obs,
#     #            label='obs',
#     #            color=colordict['obs'],
#     #            linewidth=1)
#     #    else:
#         # filter outliers (turn into NaN)
#         obs_var_filtered = obs[varname_obs].where(
#                 (obs[varname_obs]-obs[varname_obs].mean()) < (4*obs[varname_obs].std()), 
#                 np.nan)
#         # apply correction for comparison with models
#     #    obs_var_corr = ((obs[varname_obs] + offset_obs)*coeff_obs)
#         obs_var_corr = ((obs_var_filtered + offset_obs)*coeff_obs)
        
#         # plot
#         obs_var_corr.plot(
#     #            label='obs_'+varname_obs,
#             label='obs',
#             color=colordict['obs'],
#             linewidth=1)
            
#         if site != 'elsplans':
#             if add_fao56_et:
#                 pass  #LE_0 from FAO has no sense hourly, plot horizontal bar at best
#     #            obs['LE_0_FAO56'].plot(label='obs_LE_0',
#     #                                   color=colordict['obs'],
#     #                                   linestyle=':',
#     #                                   linewidth=1)
#             if add_seb_residue:
                
#                 obs_uncertainty = obs['SEB_RESIDUE'].data
                
#                 if varname_obs in ['LE', 'lhf_1']:
#                     obs_residue_corr = obs_var_corr + obs['SEB_RESIDUE']*obs['EVAP_FRAC_FILTERED'].data
#                 elif varname_obs in ['H', 'shf_1']:
#                     obs_residue_corr = obs_var_corr + obs['SEB_RESIDUE']*(1-obs['EVAP_FRAC_FILTERED'].data)
#                 else:
#                     raise ValueError('add_seb_residue available only on LE and H')
                    
#                 obs_residue_corr.plot(
#                     label='obs_residue_corr',
#                     color=colordict['obs'],
#                     linestyle=':',
#                     linewidth=1)
                
#                 plt.fill_between(obs_var_corr.time, 
#                                   obs_var_corr.data,
#                                   obs_var_corr.data + obs_uncertainty.data,
#                                   alpha=0.2, 
#                                   facecolor=colordict['obs'],
#                                   )


#%% SIMU:

fao_mean_LE_0 = {}

diff_val = {}
diff_percent_dict = {}
res_dict = {}

for veruser in veruser_list:
    res_dict[veruser] = {}
    diff_val[veruser] = {}
    diff_percent_dict[veruser] = {}
    
    for cphoto in cphoto_list:
        res_dict[veruser][cphoto] = {}
        diff_val[veruser][cphoto] = {}
        diff_percent_dict[veruser][cphoto] = {}

        for forc_model in forc_models:
            # datafolder = folder + 'forcing_{0}_{1}/{2}/'.format(
            #                                forc_models_idnumber[forc_model],
            #                                cphoto,
            #                                veruser)
            datafolder = f'{folder}/{vegtype}_{cphoto}/' + \
                f'forcing_{forc_models_idnumber[forc_model]}/{veruser}/'
            
            # ds = xr.open_dataset(datafolder + 'SURF_ATM_DIAGNOSTICS.OUT.nc')
            ds = xr.open_dataset(datafolder + 'ISBA_DIAGNOSTICS.OUT.nc',
                                 decode_times=False)
            
            # Compute other diag variables
            if varname_sim == 'U_STAR':
                ds['U_STAR'] = tools.calc_u_star_sim(ds)
            elif varname_sim == 'BOWEN':
                ds['BOWEN'] = tools.calc_bowen_sim(ds)
            elif add_fao56_et:
                runfolder = '/home/lunelt/Logiciels/SFX/VERSION_V81/MY_RUN/et_overestimation/'
                ds_forcing = xr.open_dataset(
                    runfolder + 'FORCING_{0}_{1}.nc'.format(
                    site, forc_models_idnumber[forc_model]))
                ds_forcing['REHU'] = tools.rel_humidity(
                        ds_forcing['Qair'], 
                        ds_forcing['Tair']-273.15,
                        ds_forcing['PSurf'],
                        )
                
                daily_LE = []
                for i in range(int(len(ds_forcing['Tair'])/24)): 
                    daily_LE.append(tools.calc_fao56_et_0(
                        ds['RN_ISBA'][i*24:(i+1)*24].mean()*3600*24/1e6, 
                        (ds_forcing['Tair'][i*24:(i+1)*24].max() + \
                             ds_forcing['Tair'][i*24:(i+1)*24].min())/2 - 273.15, 
                        ds_forcing['Wind'][i*24:(i+1)*24].mean(), 
                        ds_forcing['REHU'][i*24:(i+1)*24].mean()*100, 
                        ds_forcing['PSurf'][i*24:(i+1)*24].mean(),
                        gnd_flx= ds['GFLUX_ISBA'][i*24:(i+1)*24].mean())['LE_0'])
                
                fao_mean_LE_0[forc_model] = np.mean(daily_LE)
            
            # test
            # cut = 24*2
            # val1 = tools.calc_fao56_et_0(
            #     ds['RN_ISBA'][:cut].mean()*3600*24/1e6, 
            #     (ds_forcing['Tair'][:cut].max()+ds_forcing['Tair'][:cut].min())/2 - 273.15, 
            #     ds_forcing['Wind'][:cut].mean(), 
            #     ds_forcing['REHU'][:cut].mean()*100, 
            #     ds_forcing['PSurf'][:cut].mean(),
            #     gnd_flx= ds['GFLUX_ISBA'][:cut].mean())['LE_0']
            # val2 = tools.calc_fao56_et_0(
            #     ds['RN_ISBA'][cut:].mean()*3600*24/1e6, 
            #     (ds_forcing['Tair'][cut:].max()+ds_forcing['Tair'][cut:].min())/2 - 273.15, 
            #     ds_forcing['Wind'][cut:].mean(), 
            #     ds_forcing['REHU'][cut:].mean()*100, 
            #     ds_forcing['PSurf'][cut:].mean(),
            #     gnd_flx= ds['GFLUX_ISBA'][cut:].mean())['LE_0']
            # print((val1*(cut/814) + val2*((814-cut)/814)))
            
    #            .dropna(dim='time', how='any')
    #         LE_0 =  ds['LE_0_ISBA']
    # #            .dropna(dim='time', how='any').squeeze()
    #         fao_mean_LE_0[forc_model] = float(LE_0)
        
        # Set time abscisse axis
    #    try:
    #        start = ds.time.data[0]
    #    except AttributeError:    
    ##        start = np.datetime64('2021-07-14T01:00')
    #        start = np.datetime64('2021-07-21T01:00')
    #    
    #    dati_arr = np.array([start + np.timedelta64(i, 'h') for i in np.arange(0, var_md.shape[0])])
    
            res_dict[veruser][cphoto][forc_model] = ds.squeeze()


#%% PROCESS DIFF between forcing files
            
        var_irrad = 'SWD_ISBA'  # SWD, c
        diff_swd = res_dict[veruser][cphoto][forc_models[0]][var_irrad] - \
            res_dict[veruser][cphoto][forc_models[0]][var_irrad]
        mask_swd = np.abs(diff_swd) < 10
        
        mean = {}
        
        for forc_model in forc_models:
            # keep variable of interested
            var_1d_raw = res_dict[veruser][cphoto][forc_model][varname_sim]
            # remove data where SWD is too different
            var_1d = var_1d_raw.where(mask_swd, drop=False)
            # compute value of varname
            mean[forc_model] = var_1d.mean()

#%% FORCING FILE COMPARISON
    
        diff = float(mean[forc_models[0]] - mean[forc_models[1]])
        diff_val[veruser][cphoto] = diff
        
        if float(mean[forc_models[0]]) == 0:
            diff_percent = np.nan
        else:
            diff_percent = 100*diff / float(mean[forc_models[0]])
            
        diff_percent_dict[veruser][cphoto] = diff_percent
        
        verbose = True
        if verbose:
            print('--------')
            print('VER_USER: ', veruser)
            print('CPHOTO: ', cphoto)
            # print('Mean LE: ')
            # print(mean)
            print(f'Diff in percent: {forc_models[0]} - {forc_models[1]}')
            print(diff_percent)
        
if add_fao56_et:
    print('FAO mean LE_0: ', fao_mean_LE_0)
    diff_percent_dict['FAO56'] = (fao_mean_LE_0['noirr_lai_d1']-fao_mean_LE_0['irrswi1_d1']) / fao_mean_LE_0['irrswi1_d1'] *100
    print(diff_percent_dict['FAO56'])
    

# Create pandas dataframe
diff_percent_df = pd.DataFrame(diff_percent_dict)
diff_df = pd.DataFrame(diff_val)


#%% PLOT

varname_plot = 'LETR_ISBA'
ylabel = varname_plot

veruser_to_plot = [
    # 'MASTER',
    # 'IRRSWI00',
    # 'IRRSWI02',
    # 'IRRSWI03',
    # 'IRRSWI04',
    # 'IRRSWI05',
    # 'IRRSWI06',
    # 'IRRSWI07',
    # 'IRRSWI08',
    'IRRSWI10',
    # 'IRRSWI12',
    'IRRLAGRIP30',
    ]
cphoto_to_plot = [
    'AST',
    # 'NON',
    ]


if plot_figure:
    for veruser in veruser_to_plot:
        
        for cphoto in cphoto_to_plot:
            
            fig = plt.figure(figsize=figsize)
            for forc_model in forc_models:
                # keep variable of interested
                var_1d_raw = res_dict[veruser][cphoto][forc_model][varname_plot]
                # remove data where SWD is too different
                var_1d = var_1d_raw.where(mask_swd, drop=False)
                
                # set the time scale
                dati_arr = pd.date_range(
                        pd.Timestamp('20210714-0100'),
                        periods=len(var_1d), 
                        freq='30T')
                var_1d['time'] = dati_arr

                plt.plot(var_1d.time, var_1d, 
                          color=colordict[forc_model],
                          linestyle='--',
            #             label='simu_{0}_{1}'.format(forc_model, varname_sim),
                          label='simu_{0}'.format(forc_model),
                          )
                
            ax = plt.gca()
        #   ax.set_xlim([np.min(obs.time), np.max(obs.time)])
            # ax.set_xlim([np.min(var_1d.time), np.max(var_1d.time)])
            ax.set_xlim([pd.Timestamp('2021-07-17T00:30'),
                         pd.Timestamp('2021-07-19T23:30')])
            
            ax.set_ylabel(ylabel)    
            
            plt.legend(loc='best')
            plot_title = f'{veruser} - {cphoto}'
            plt.title(plot_title)
            ax.grid()
            plt.tick_params(rotation=30)
            
            # add grey zones for night
            days = np.arange(14,31)
            for day in days:
                sunrise = pd.Timestamp('202107{0}-1930'.format(day))
                sunset = pd.Timestamp('202107{0}-0500'.format(day+1))
                ax.axvspan(sunset, sunrise, ymin=0, ymax=1, 
                           color = '0.9'  #'1'=white, '0'=black, '0.8'=light gray
                           )
    
    if add_fao56_et:
        pass
    #            plt.plot(LE_0.time, LE_0, 
    #                 color=colordict[forc_model],
    #    #             colordict[forc_model],
    #                 linestyle=':',
    #                 label='simu_{0}_LE_0_FAO56'.format(model),
    #                 )
    
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
                       lambda evap: evap*2264000/3600))
        secax.set_ylabel('evapotranspiration [mm/h]')
    
    # # add errors values on graph
    # if errors_computation:
    #     plt.text(.01, .95, 'RMSE: {0}'.format(rmse), 
    #              ha='left', va='top', transform=ax.transAxes)
    #     plt.text(.01, .99, 'Bias: {0}'.format(bias), 
    #              ha='left', va='top', transform=ax.transAxes)
    #     plt.legend(loc='upper right')
    # else:
    plt.legend(loc='best')
    
    plt.title(plot_title)
    plt.grid()
    plt.tick_params(rotation=30)

#%% Save figure
if to_pickle:
    pickle_file_name = f'diff_percent_df_{site}_{vegtype}.pkl'
    # Open a file and use dump() 
    with open(pickle_file_name, 'wb') as file: 
        # A new file will be created 
        pickle.dump(diff_percent_df, file)
    print('File created: ', pickle_file_name)

if save_plot:
    tools.save_figure(plot_title, save_folder)
