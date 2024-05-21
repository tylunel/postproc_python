#!/usr/bin/env python3
"""
@author: Tanguy LUNEL

Plot time series for outputs of Surfex offline runs
This V2 do not include the possibility of plotting observation, 
-> cf plot_time_series_sfx.py for that
    
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
    # 'IRRLAGRIP30_THLD05',
    # 'IRRLAGRIP100_THLD05',
    ]
cphoto_list = [
    'AST',
    'NON',
    ]
vegtype_list = [
    'C4',
    'C3',
    ]

forc_models = [
        'irrswi1_d1', 
        # 'std_d1',
        'noirr_lai_d1',
#        'irrlagrip30_d1',
         ]

to_pickle = True

errors_computation = False
add_seb_residue = False
add_fao56_et = True

# --/!\-- All setup variables for plots are available in the corresponding section
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
    
#    raise ValueError("nom de variable d'observation inconnue"), 'WQ_2m', 'WQ_10m'

if varname_sim in ['U_STAR',]:
    varname_sim_preproc = 'FMU_ISBA,FMV_ISBA'
elif varname_sim in ['BOWEN',]:
    varname_sim_preproc = 'H_ISBA,LE_ISBA'
else:
    varname_sim_preproc = varname_sim
    
lat = gv.sites[site]['lat']
lon = gv.sites[site]['lon']

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

        for vegtype in vegtype_list:
            res_dict[veruser][cphoto][vegtype] = {}
            diff_val[veruser][cphoto][vegtype] = {}
            diff_percent_dict[veruser][cphoto][vegtype] = {}

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
        
                res_dict[veruser][cphoto][vegtype][forc_model] = ds.squeeze()


#%% PROCESS DIFF between forcing files
                
            var_irrad = 'SWD_ISBA'  # SWD, c
            diff_swd = res_dict[veruser][cphoto][vegtype][forc_models[0]][var_irrad] - \
                res_dict[veruser][cphoto][vegtype][forc_models[0]][var_irrad]
            mask_swd = np.abs(diff_swd) < 10
            
            mean = {}
            
            for forc_model in forc_models:
                # keep variable of interested
                var_1d_raw = res_dict[veruser][cphoto][vegtype][forc_model][varname_sim]
                # remove data where SWD is too different
                var_1d = var_1d_raw.where(mask_swd, drop=False)
                # compute value of varname
                mean[forc_model] = var_1d.mean()
    
#%% FORCING FILE COMPARISON
        
            diff = float(mean[forc_models[0]] - mean[forc_models[1]])
            diff_val[veruser][cphoto][vegtype] = diff
            
            if float(mean[forc_models[0]]) == 0:
                diff_percent = np.nan
            else:
                diff_percent = 100*diff / float(mean[forc_models[0]])
                
            diff_percent_dict[veruser][cphoto][vegtype] = diff_percent
            
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
newdict = {}
for key in diff_percent_dict:
    if key == 'FAO56':
        newdict[key] = diff_percent_dict[key]
    else:
        newdict[key] = pd.DataFrame(diff_percent_dict[key])
    
diff_percent_df = pd.DataFrame(diff_percent_dict)
diff_df = pd.DataFrame(diff_val)


#%% Save dict to pickle
if to_pickle:
    # Open a file and use dump() 
    fn = f'diff_percent_df_{site}_{varname_sim}.pkl'
    with open(fn, 'wb') as file: 
        # A new file will be created 
        pickle.dump(diff_percent_df, file)
        print(f'Saved: {fn}')


#%% PLOT

plot_figure = True
save_plot = True
save_folder = f'./fig/compa_forcings/{site}/'

figsize = (6, 6) #small for presentation: (6,6), big: (15,9)
one_day_zoom_plot = 15  # just zoom on one day  # 18 and 22 are interesting
add_irrig_time = False

varname_plot = 'LE_ISBA'

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
    # 'IRRLAGRIP30_THLD05',
    # 'IRRLAGRIP100_THLD05',
    ]
cphoto_to_plot = [
    'AST',
    # 'NON',
    ]
vegtype_to_plot = [
    # 'C3',
    'C4',
    ]

colordict = {
    'irr_d2': 'g', 
    'std_d2': 'r',
    'irr_d1': 'g',
    'irrswi1_d1': 'b', 
    'noirr_lai_d1': 'r', 
    'std_d1': 'r',
    'irrlagrip30_d1': 'y',
    'obs': 'k'}

forc_model_name = {
    'irrswi1_d1': 'IRR_FC',
    # 'std_d1',
    'noirr_lai_d1': 'NOIRR',
#        'irrlagrip30_d1',
    }

if varname_plot in ['LE', 'LE_ISBA']:
    ylabel = 'Latent heat flux [W m$^{-2}$]'
    secondary_axis = 'evap'
    ymin, ymax = -50, 750
elif varname_plot in ['LETR_ISBA',]:
    ylabel = 'Latent heat flux from transpiration [W m$^{-2}$]'
    secondary_axis = 'evap'
    ymin, ymax = -50, 450
else:
    ylabel = varname_plot
    pass


if plot_figure:
    for veruser in veruser_to_plot:
        for cphoto in cphoto_to_plot:
            for vegtype in vegtype_to_plot:
                fig = plt.figure(figsize=figsize)
                for forc_model in forc_models:
                    # keep variable of interested
                    var_1d_raw = res_dict[veruser][cphoto][vegtype][forc_model][varname_plot]
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
                #             label='simu_{0}_{1}'.format(forc_model, varname_plot),
                              label='atmo_file_{0}'.format(forc_model_name[forc_model]),
                              )
                
                ax = plt.gca()
            #   ax.set_xlim([np.min(obs.time), np.max(obs.time)])
                # ax.set_xlim([np.min(var_1d.time), np.max(var_1d.time)])
                
                if one_day_zoom_plot not in [0, None]:
                    xmin = pd.Timestamp(f'202107{str(one_day_zoom_plot).zfill(2)}T0000')
                    xmax = pd.Timestamp(f'202107{str(one_day_zoom_plot).zfill(2)}T2330')
                else:
                    xmin = None
                    xmax = None
                
                ax.set_xlim([xmin, xmax])
                
                ax.set_ylabel(ylabel)
                ax.set_ylim([ymin, ymax])
                
                ax.legend(loc='best')
                # plot_title = f'{veruser} - {cphoto}'
                # plt.title(plot_title)
                ax.grid()
                ax.tick_params(rotation=30)
                
                # add grey zones for night
                days = np.arange(14,31)
                for day in days:
                    sunrise = pd.Timestamp('202107{0}-1930'.format(day))
                    sunset = pd.Timestamp('202107{0}-0430'.format(day+1))
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

    
    # add secondary axis on the right, relative to the left one - (for LE)
    if secondary_axis == 'le':
        axes = plt.gca()
        secax = axes.secondary_yaxis("right",                              
            functions=(lambda evap: evap*2264000,
                       lambda le: le/2264000))
        secax.set_ylabel('Latent heat flux [W m$^{-2}$]')
    if secondary_axis == 'evap':
        axes = plt.gca()
        secax = axes.secondary_yaxis("right",                              
            functions=(lambda le: (le/2264000)*3600,
                       lambda evap: evap*2264000/3600))
        secax.set_ylabel('Evapotranspiration [mm h$^{-1}$]')
    
    # # add errors values on graph
    # if errors_computation:
    #     plt.text(.01, .95, 'RMSE: {0}'.format(rmse), 
    #              ha='left', va='top', transform=ax.transAxes)
    #     plt.text(.01, .99, 'Bias: {0}'.format(bias), 
    #              ha='left', va='top', transform=ax.transAxes)
    #     plt.legend(loc='upper right')
    # else:
        
    # plt.legend(loc='best')
    # plt.title(plot_title)
    # plt.grid()
    # plt.tick_params(rotation=30)

# -- Save figure

save_title = f'{veruser_to_plot}-{cphoto_to_plot}-{vegtype_to_plot}-{site}-{varname_plot}_{one_day_zoom_plot}July'
if save_plot:
    tools.save_figure(save_title, save_folder)
