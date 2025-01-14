#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021

Module gathering various functions used in different part of the codes

"""
import os
#from scipy.stats import circmean
import copy
import time
import numpy as np
from scipy import interpolate, integrate, stats
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
# from metpy.units import units
#import metpy.calc as mcalc
import re
import global_variables as gv
#from datetime import datetime as dt
from shapely.geometry import Point
#from difflib import SequenceMatcher
import warnings
import epygram

# from dask import compute
# from dask.delayed import delayed
# import dask


#%% Load MNH/SFX/AROME files

def load_series_dataset(varname_sim_list, model, concat_if_not_existing=True,
                        out_suffix='', file_suffix='dg'):
    """
    Load a simulation file in netcdf format. If not already existing,
    tries to concatenate multiple instantaneous output files into a series.
    
    Parameters:
    varname_sim_list: ex:['T2M_ISBA', 'H_P9']
    model: 'std_d1', 'irr_d1', 'irrlagrip30_d1'
    concat_if_not_existing: Try to concatenate or not if the concatenated file
        does not exist yet
    """
    global_simu_folder = gv.global_simu_folder
    
    for i, varname_item in enumerate(varname_sim_list):
        # get format of file to concatenate
#        in_filenames_sim = gv.format_filename_simu_wildcards[model]
        in_filenames_sim = gv.format_filename_simu_new[model].format(
            seg_nb='??', output_nb_3f='???', output_nb_2f='??',
            out_suffix=out_suffix, file_suffix=file_suffix)
        gridnest_nb = in_filenames_sim[6]  # integer: 1 or 2 in my case
        # set name of concatenated output file
        out_filename_sim = f'LIAIS.{gridnest_nb}.{varname_item}{out_suffix}.nc'
        # path of file
        datafolder = global_simu_folder + gv.simu_folders[model]
        # concatenate multiple days if not already existing
        if concat_if_not_existing:
            concat_simu_files_1var(datafolder, varname_item, 
                                   in_filenames_sim, out_filename_sim)
        
        if i == 0:      # if first data, create new dataset
            ds = xr.open_dataset(datafolder + out_filename_sim)
        else:           # append data to existing dataset
            ds_temp = xr.open_dataset(datafolder + out_filename_sim)
            ds = ds.merge(ds_temp)
    
    return ds


def load_dataset(varname_sim_list, model, concat_if_not_existing=True,
                 out_suffix='', file_suffix='dg'):
    warnings.warn(
        """Deprecation warning:
        'load_dataset' should be replaced by 'load_series_dataset'.
        """)
    return load_series_dataset(varname_sim_list, 
                               model, concat_if_not_existing=True,
                               out_suffix='', file_suffix='dg')


def open_budget_file(filename, budget_type):
    """
    Add longitude, latitude, ni, nj coordinates to a dataset which does not
    have it.
    
    /!\ the longitude, latitude, ni, nj coordinates may not be the same than
    those of outer domain (may correspond to latitude_u, ni_u, 
    of outer domain etc for ex.)
    """
    
    meta = xr.open_dataset(filename)  # metadata of the entire simulation domain, not only the budget one
    ds = xr.open_dataset(filename, group=f"Budgets/{budget_type}")
    
    if budget_type == 'UU':
        cart_nj_X = 'cart_nj_u'
        cart_ni_X = 'cart_ni_u'
        cart_level_X = 'cart_level'
        nj_X = 'nj_u'
        ni_X = 'ni_u'
        latitude_X = 'latitude_u'
        longitude_X = 'longitude_u'
    elif budget_type == 'VV':
        cart_nj_X = 'cart_nj_v'
        cart_ni_X = 'cart_ni_v'
        cart_level_X = 'cart_level'
        nj_X = 'nj_v'
        ni_X = 'ni_v'
        latitude_X = 'latitude_v'
        longitude_X = 'longitude_v'
    elif budget_type == 'WW':
        cart_nj_X = 'cart_nj'
        cart_ni_X = 'cart_ni'
        cart_level_X = 'cart_level_w'
        nj_X = 'nj'
        ni_X = 'ni'
        latitude_X = 'latitude'
        longitude_X = 'longitude'
    elif budget_type in ['TH', 'RV', 'TK']:
        cart_nj_X = 'cart_nj'
        cart_ni_X = 'cart_ni'
        cart_level_X = 'cart_level'
        nj_X = 'nj'
        ni_X = 'ni'
        latitude_X = 'latitude'
        longitude_X = 'longitude'
    
    lon_arr = np.zeros([len(meta[cart_nj_X]), len(meta[cart_ni_X])])
    lat_arr = np.zeros([len(meta[cart_nj_X]), len(meta[cart_ni_X])])
    nj_arr = np.zeros([len(meta[cart_nj_X]),])
    ni_arr = np.zeros([len(meta[cart_ni_X]),])
    
    for i, ni_val in enumerate(meta[cart_ni_X].values):
        for j, nj_val in enumerate(meta[cart_nj_X].values):
            if budget_type == 'UU':
                pt = meta.sel(ni_u=ni_val, nj_u=nj_val)
            elif budget_type == 'VV':
                pt = meta.sel(ni_v=ni_val, nj_v=nj_val)
            elif budget_type in ['TH', 'RV', 'TK', 'WW']:
                pt = meta.sel(ni=ni_val, nj=nj_val)
            lat_arr[j, i] = float(pt[latitude_X])
            lon_arr[j, i] = float(pt[longitude_X])
            nj_arr[j] = nj_val
            ni_arr[i] = ni_val
    
    ds[longitude_X] = xr.DataArray(lon_arr, 
        coords={cart_nj_X: ds[cart_nj_X].values,
                cart_ni_X: ds[cart_ni_X].values,})
    ds[latitude_X] = xr.DataArray(lat_arr, 
        coords={cart_nj_X: ds[cart_nj_X].values, 
                cart_ni_X: ds[cart_ni_X].values})
    ds[ni_X] = xr.DataArray(ni_arr, 
        coords={cart_ni_X: ds[cart_ni_X].values})
    ds[nj_X] = xr.DataArray(nj_arr, 
        coords={cart_nj_X: ds[cart_nj_X].values})
    
    ds[cart_level_X] = xr.DataArray(meta[cart_level_X], 
        coords={cart_level_X: meta[cart_level_X]})
    
    ds = ds.set_coords([latitude_X, longitude_X, nj_X, ni_X])
    ds = ds.swap_dims({cart_nj_X: nj_X, cart_ni_X: ni_X,})
    ds = ds.drop([cart_nj_X, cart_ni_X])
    
    # rename for simplicity in processing in other functions, but may be risky...
    ds = ds.rename({cart_level_X: 'level'})
    
    if budget_type in ['VV', 'UU']:
        
        meta_subset = subset_ds(meta,
                  lat_range=[ds[latitude_X].min(), ds[latitude_X].max()],
                  lon_range=[ds[longitude_X].min(), ds[longitude_X].max()],
                  )
        
        # interpolate on mass points (center of grid)
        if budget_type == 'UU':
            ds = ds.interp(ni_u=meta_subset.ni.values, nj_u=meta_subset.nj.values)
        elif budget_type == 'VV':
            ds = ds.interp(ni_v=meta_subset.ni.values, nj_v=meta_subset.nj.values)
        
        ds = ds.rename({nj_X: 'nj', ni_X: 'ni',
                        longitude_X: 'longitude', latitude_X: 'latitude'})
    
    return ds


def open_multiple_budget_file(filename_bu, budget_type_list=['UU', 'VV'],
                              wanted_datetime=None):
    """
    Create xarray.Dataset with compound budget, typically UU and VV data
    (code need adaptation in order to work with var different from UU and VV).
    
    The variables of the budget datasets merged are renamed:
        [varname] -> [varname]_[budget_type]
        ex: 'PRES' becomes 'PRES_UU'
    
    return:
        xarray.Dataset
    
    """
    budget_file_dict = {}
    
    # Load separately the different budget files
    for bu in budget_type_list:
        budget_file_dict[bu] = open_budget_file(filename_bu, bu)
    
    # Ensure that dataset are of the same size: -> NOT NEEDED I THINK: 'join' = outer anyway for the merge step
#    ds_bu_UU == ds_bu_UU.where(ds_bu_VV.nj==ds_bu_UU.nj).where(ds_bu_VV.ni==ds_bu_UU.ni)
#    ds_bu_VV = ds_bu_VV.where(ds_bu_VV.nj==ds_bu_UU.nj).where(ds_bu_VV.ni==ds_bu_UU.ni)

    # Change names of variable by adding the budget type
    for bu in budget_type_list:
        for var_name in list(budget_file_dict[bu].keys()):
            budget_file_dict[bu] = budget_file_dict[bu].rename({var_name: f'{var_name}_{bu}'})
    
    # Merge the datasets in 1
    # fix unsignificant differences in lat/lon coords between datasets
    # note that 1e-5=0.00001° in latitude or longitude ~= 1m
    for i, bu in enumerate(budget_type_list):
        if i==0:
            ds_out = budget_file_dict[bu]
        else:
            if np.abs(budget_file_dict[bu].latitude - ds_out.latitude).max() < 1e-5:
                budget_file_dict[bu]['latitude'] = ds_out.latitude
            if np.abs(budget_file_dict[bu].longitude - ds_out.longitude).max() < 1e-5:
                budget_file_dict[bu]['longitude'] = ds_out.longitude
            # concatenate the datasets together
            ds_out = ds_out.merge(budget_file_dict[bu])
    
    # if a specific datetime is requested:
    if wanted_datetime is not None:
        hour = pd.Timestamp(wanted_datetime).hour
        ds_out = ds_out.isel(time_budget=hour)
        ds_out = ds_out.assign_coords({'time': pd.Timestamp(wanted_datetime)})

    return ds_out


def compound_budget_file(filename_bu, budget_type_list=['UU', 'VV']):
    warnings.warn(
        """Deprecation warning:
        'compound_budget_file' should be replaced by 'open_multiple_budget_file'.
        """)
    return open_multiple_budget_file(filename_bu, 
                                     budget_type_list=['UU', 'VV'])


def open_relevant_file(model, wanted_date, var_simu):
    """
    Choose the file to open depending on the variable wanted.
    """
    # --- Retrieve and open file
    # Budget variables
    if var_simu[-3:] in ['_TH',  '_RV', '_TK', '_UU', '_VV', '_WW',]:
        filename_simu = get_simu_filepath(model, wanted_date, 
                                          filetype='000',)
        ds = open_multiple_budget_file(filename_simu, [var_simu[-2:],],
                                       wanted_datetime=wanted_date)
    elif var_simu[-3:] in ['_UV',]:
        raise NameError('to be coded')
        # code below could do?
#        filename_simu = tools.get_simu_filepath(model, wanted_date, 
#                                                filetype='000',)
#        ds = tools.open_multiple_budget_file(filename_simu, ['UU', 'VV'])
#        ds[f'{var_simu[:-3]}}_VAL'], ds_bu[f'{var_simu[:-3]}}_DIR'] = tools.calc_ws_wd(
#            ds_bu[f'{var_simu[:-3]}_UU'], ds_bu[f'{var_simu[:-3]}_VV'])
        
    # Variable available in OUTPUT files:
    elif var_simu in ['RVT', 'THT', 'THTV', 'WT', 'UT', 'VT', 'WS', 'TKET']:
        output_freq = gv.output_freq_dict[model]
        try:
            filename_simu = get_simu_filepath(model, wanted_date, 
                                              file_suffix='',  #'dg' or ''
                                              out_suffix='.OUT',
                                              output_freq=output_freq)
        except FileNotFoundError:
            filename_simu = get_simu_filepath(model, wanted_date, 
                                              file_suffix='dg',  #'dg' or ''
                                              out_suffix='',)
            
        ds = xr.open_dataset(filename_simu)
        ds['THTV'] = ds['THT']*(1 + 0.61*ds['RVT'])
        if var_simu == 'WS':
            ds = center_uvw(ds)
            ds['WS'], _ = calc_ws_wd(ds['UT'], ds['VT'])
    # Other variables:
    else:
        filename_simu = get_simu_filepath(model, wanted_date, 
                                          file_suffix='dg',  #'dg' or ''
                                          out_suffix='',)
        ds = xr.open_dataset(filename_simu)
    
    return ds


def get_simu_filepath_old(model, date='20210722-1200', 
                      filetype='synchronous',  # synchronous or diachronic (=000)
                      file_suffix='dg',  #'dg' or ''
                      out_suffix='',  #'.OUT' or ''
                      output_freq=None,  # [min]
                      global_simu_folder=gv.global_simu_folder,
                      verbose=False,
                      ):
    """
    Returns the whole string for filename and path. 
    Chooses the right SEG (segment) number and suffix corresponding to date
    
    Parameters:
        model: str, 
            'irr_d2' or 'std_d1'
        wanted_date: str with format accepted by pd.Timestamp ('20210722-1200')
            or pd.Timestamp
    
    Returns:
        filename: str, full path and filename
    """
    pd_date = pd.Timestamp(date)
    seg_nb = str(pd_date.day)
    
    if filetype in ['synchronous', '']:
        if output_freq is None:  # set default according to file types
            if out_suffix == '' and file_suffix == 'dg':
                output_freq = 60
            elif out_suffix == '.OUT' and file_suffix == '':
                output_freq = 30
            else:
                print('output_freq not defined')
        
        if out_suffix == '' and file_suffix == 'dg' and output_freq==60:
            output_nb = str(pd_date.hour)
            
        elif out_suffix == '.OUT' and file_suffix == '':
            minute_total_day = pd_date.minute + pd_date.hour*60
            output_nb = str(int(minute_total_day/output_freq))
            print('output_nb = {0}'.format(output_nb))
            
        else:
            raise ValueError('out_suffix and file_suffix values not consistent')
        
        # Reformat the output_nb with 2 and 3 digits:
        output_nb_2f = output_nb.zfill(2)
        output_nb_3f = output_nb.zfill(3)
        
    elif filetype in ['diachronic', '000']:
        output_nb_2f = str(pd_date.hour).zfill(2)
        output_nb_3f = '000'
        file_suffix = ''
        out_suffix = ''
        
    # Midnight case
    if output_nb_2f == "00":
        output_nb_2f = "24"
        output_nb_3f = "024"
        seg_nb = str(pd_date.day - 1)
    
    # retrieve filename
    filename = gv.format_filename_simu[model].format(
            seg_nb=seg_nb, 
            output_nb_2f=output_nb_2f, 
            output_nb_3f=output_nb_3f, 
            file_suffix=file_suffix,
            out_suffix=out_suffix)
    
    filepath = global_simu_folder + gv.simu_folders[model] + filename
    
    #check if nomenclature of filename is ok
    if filetype in ['synchronous', '']:
        check_filename_datetime(filepath)
    
    if verbose:
        print(filepath)
    
    return filepath


def get_simu_filepath(
        model, wanted_date='20210722-1200', 
        output_type='backup',  # backup, output or diachronic (=000)
        global_simu_folder=gv.global_simu_folder,
        verbose=False,
        ):
    """
    Returns the whole string for filename and path. 
    Chooses the right SEG (segment) number and suffix corresponding to date
    
    Parameters:
        model: str, 
            'irr_d2' or 'std_d1'
        wanted_date: str with format accepted by pd.Timestamp ('20210722-1200')
            or pd.Timestamp
    
    Returns:
        filename: str, full path and filename
    """
    pd_date = pd.Timestamp(wanted_date)
    seg_nb = str(pd_date.day)
    
    name_rules = gv.filename_simu_name_rules[model][output_type]
    
    # Suffixes
    out_suffix = name_rules['out_suffix']
    file_suffix = name_rules['file_suffix']
    # SEG number
    if name_rules['seg_nb'] == '01':
        seg_nb = name_rules['seg_nb']
    # Output number
    if name_rules['output_nb'] == 'HH':
        output_nb = pd_date.strftime('%H')
    elif name_rules['output_nb'] == 'Hx6':
        output_freq = 10
        minute_total_day = pd_date.minute + pd_date.hour*60
        output_nb = str(int(minute_total_day/output_freq))
    
    # Midnight case
    if output_nb == "00":
        output_nb = "24"
    
    # Reformat the output_nb with 2 and 3 digits:
    output_nb_2f = output_nb.zfill(2)
    output_nb_3f = output_nb.zfill(3)
    
    # retrieve filename
    filename = gv.format_filename_simu[model].format(
            seg_nb=seg_nb, 
            output_nb_2f=output_nb_2f, 
            output_nb_3f=output_nb_3f, 
            file_suffix=file_suffix,
            out_suffix=out_suffix)
    
    filepath = global_simu_folder + gv.simu_folders[model] + filename
    
    #check if nomenclature of filename is ok
    if output_type in ['backup', 'output']:
        check_filename_datetime(filepath, wanted_date)
    
    if verbose:
        print(filepath)
    
    return filepath


def extract_profile_arome(site, wanted_date, varname,  
                          max_height=3000, 
                          folder_path='/home/lunelt/Data/analysis/',
                          file_prefix='arome.AN.',
                          # save_to_pickle=True,
                          # save_folder_pickle='/home/lunelt/Data/data_NEMO/arome_profiles/',
                          verbose=True):
    """
    Parameters
    ----------
    site : string
        name of site above to perform the extraction. 
        Must be in global_variables
    wanted_date : string with format YYYYMMDD.HH
        ex: 20230725.00
    varname : string
        must be in [tke, t, u, v, q, ws, wd]
    max_height : TYPE, optional
        DESCRIPTION. The default is 3000.
    folder_path : TYPE, optional
        DESCRIPTION. The default is '/home/lunelt/Data/analysis/'.
    file_prefix : TYPE, optional
        DESCRIPTION. The default is 'arome.AN.'.

    Returns
    -------
    verti_profiles: dict of dict, first key is the varname(s),
        second are the heights as keys

    """
    
    try:
        # Convert the input string to a pandas timestamp
        timestamp = pd.to_datetime(wanted_date)
        # Format the timestamp to the desired format
        formatted_date = timestamp.strftime('%Y%m%d.%H')
    except Exception as e:
        raise ValueError(f"Invalid input wanted_date: {wanted_date}. Error: {e}")
    
    epygram.init_env()

    lon = gv.whole[site]['lon']
    lat = gv.whole[site]['lat']
    
    filename = f'{file_prefix}{formatted_date}'

    file = epygram.formats.resource(filename=f'{folder_path}/{filename}',
                                    openmode='r')
    
    temp_res = {}
    verti_profile = {}
    verti_profiles = {}

    if verbose:
        print('processing of level nb:')
        
    # if varname in ['ws', 'wd', 'wind']:
    #     for level, height in gv.level_height_AROME_mean.items():
    #         if height > max_height:
    #             pass
    #         else:
    #             if verbose:
    #                 print(level)
    #             # u component
    #             u = file.find_fields_in_resource(f'shortName:u, level:{level}')
    #             field_u = file.readfields(u[0])[0]
    #             pt_u = float(field_u.extract_point(lat=lat, lon=lon).data)
    #             # v component
    #             v = file.find_fields_in_resource(f'shortName:v, level:{level}')
    #             field_v = file.readfields(v[0])[0]
    #             pt_v = float(field_v.extract_point(lat=lat, lon=lon).data)
    #             # ws,wd computation
    #             temp_res['ws'], temp_res['wd'] = calc_ws_wd(pt_u, pt_v)
    #             if varname == 'wind':
    #                 verti_profile[height] = temp_res['ws'], temp_res['wd']
    #             else:
    #                 verti_profile[height] = temp_res[varname]
    #     if varname == 'wind':
    #         # Initialize two new dictionaries
    #         dict_ws = {}
    #         dict_wd = {}
    #         # Loop through the original dictionary
    #         for altitude, values in verti_profile.items():
    #             dict_ws[altitude] = values[0]  # First value of the tuple
    #             dict_wd[altitude] = values[1]
    #         verti_profiles['ws'] = dict_ws
    #         verti_profiles['wd'] = dict_wd
    #     else:
    #         verti_profiles[varname] = verti_profile
    # else:     
    for level, height in gv.level_height_AROME_mean.items():
        if height > max_height:
            pass
        else:
            if verbose:
                print(level)
            fid = file.find_fields_in_resource(f'shortName:{varname}, level:{level}')
            field = file.readfields(fid[0])[0]
            verti_profile[height] = float(field.extract_point(lat=lat, lon=lon).data)
    
    verti_profiles[varname] = verti_profile
    
    # if save_to_pickle:
    #     # -- Save profile to pickle
    #     formatted_date = wanted_date.strftime('%Y%m%d.%H')
    #     filename_pickle = f"verti_profile_arome_{site}_{formatted_date}"
        
    #     # Format pd.DataFrame:
    #     df = pd.DataFrame(verti_profile)
    #     df.to_pickle(f'{save_folder_pickle}/{filename_pickle}.pickle')
        
    return verti_profiles


def extract_profile_arome_parallel(site, wanted_date, varname,  
    max_height=3000, folder_path='/home/lunelt/Data/analysis/',
    file_prefix='arome.AN.'):
    """
    Not working yet, need to fix issue with Dask
    
    Parameters
    ----------
    site : string
        name of site above to perform the extraction. 
        Must be in global_variables
    wanted_date : string with format YYYYMMDD.HH
        ex: 20230725.00
    varname : string
        must be in [tke, t, u, v, q, ws, wd]
    max_height : TYPE, optional
        DESCRIPTION. The default is 3000.
    folder_path : TYPE, optional
        DESCRIPTION. The default is '/home/lunelt/Data/analysis/'.
    file_prefix : TYPE, optional
        DESCRIPTION. The default is 'arome.AN.'.

    Returns
    -------
    dict verti_profile with height as keys

    """
    try:
        # Convert the input string to a pandas timestamp
        timestamp = pd.to_datetime(wanted_date)
        # Format the timestamp to the desired format
        formatted_date = timestamp.strftime('%Y%m%d.%H')
    except Exception as e:
        raise ValueError(f"Invalid input wanted_date: {wanted_date}. Error: {e}")
    
    epygram.init_env()

    lon = gv.sites[site]['lon']
    lat = gv.sites[site]['lat']
    
    filename = f'{file_prefix}{wanted_date}'

    file = epygram.formats.resource(filename=f'{folder_path}/{filename}',
                                    openmode='r')
    
    # Create a dictionary to hold the delayed tasks
    verti_profile_tasks = {}
    
    # Iterate through the levels and heights in parallel using Dask's delayed
    for level, height in gv.level_height_AROME_mean.items():
        if height > max_height:
            continue
    
        @delayed
        def process_level(level, height):
            print(level)
            fid = file.find_fields_in_resource(f'shortName:{varname}, level:{level}')
            field = file.readfields(fid[0])[0]
            return height, float(field.extract_point(lat=lat, lon=lon).data)
    
        verti_profile_tasks[level] = process_level(level, height)
    
    # Compute all delayed tasks in parallel
    results = compute(*verti_profile_tasks.values(), scheduler='threads')
    
    # Assemble the results into the final dictionary
    verti_profile = {height: value for height, value in results}
    
    return verti_profile




#%% Load OBS data
def open_ukmo_mast(datafolder, filename, create_netcdf=True, remove_modi=True,
                   remove_outliers=False):
    """
    Open the 50m mast from UK MetOffice formatted under .txt,
    create netcdf file, and returns a xarray dataset.
    """
    filename_modi = filename + '_modi'
    
    #replace multi space by tab
    os.system('''
        cd {0}
        cp {1} {2}
        sed -i 's/ \+ /\t/g' {2}
        '''.format(datafolder, filename, filename_modi))
    
    #Version de Guylaine :
    #sed -i -e 's/ |     /;/g' LIAISE_20210614_30.dat 
    #sed -i -e 's/ |    /;/g' LIAISE_20210614_30.dat 
    #sed -i -e 's/ m     /;/g' LIAISE_20210614_30.dat
    #sed -i -e 's/ D     /;/g' LIAISE_20210614_30.dat 
    #sed -i -e 's/ D    /;/g' LIAISE_20210614_30.dat 
    #sed -i -e 's/ X     /;/g' LIAISE_20210614_30.dat 
    
    #read header
    with open(datafolder + filename_modi) as f:
        fulltext = f.readlines()    # read all lines
        
        #Get date of file
        line = fulltext[1]
        strings = line.split(sep=' ')
        day = strings[2][:2]
        month = strings[2][3:5]
        year = strings[2][6:10]
        datetime = pd.Timestamp('{0}-{1}-{2}'.format(year, month, day))
        
        #Get length of header (part starting by '!')
        head_len = np.array([line.startswith('!') for line in fulltext]).sum()
        
        #Get variables
        var = fulltext[head_len-1].split(sep='\t')
        
        #gGt height of measures
        meas_height = fulltext[head_len-2].split(sep='\t')
        
        columns=[]
        for i in range(len(var)):
            columns.append(var[i] + '_' + meas_height[i])

    #removes flags corresponding to each measure
    os.system('''
    cd {0}
    sed -i -e 's/|//g' -e 's/D//g' -e 's/X//g' -e 's/m//g' {2}
    sed -i 's/ \+ /\t/g' {2}
    '''.format(datafolder, filename, filename_modi))
    
    #read data
    obs_ukmo = pd.read_table(datafolder + filename_modi,
                        sep="\t",
                        header=head_len-1,
                        names=columns)
    
    obs_ukmo.replace(['nan ', 'nan', ' '], np.nan, inplace=True)
    # set data to numeric type
    obs_ukmo = obs_ukmo.astype(float)
    
    # add time column with datetime
#    time_list = [(pd.Timedelta(val, unit='s') + datetime).round(freq='T') for val in obs_ukmo['HOUR_time']]
    time_list = [(pd.Timedelta(val, unit='h') + datetime) for val in obs_ukmo['HOUR_time']]
    obs_ukmo['time'] = pd.to_datetime(time_list)
#    obs_ukmo.set_index('time', inplace=True)
    
    #drop NaN column
    obs_ukmo.dropna(axis=1, how='all', inplace=True)
#    obs_ukmo.drop('!_!', axis=1, inplace=True)
    obs_ukmo.drop('VIS\n_vis\n', axis=1, inplace=True)
    #set time as index
    obs_ukmo.set_index('time', inplace =True)

#    obs_ukmo.rename(columns = {
#            'TEMP_2m':'ta_2', 'TEMP_10m':'ta_3', 'TEMP_25m':'ta_4',
#            'UTOT_2m':'ws_2', 'UTOT_10m':'ws_3', 'UTOT_25m':'ws_4',
#            'DIR_2m':'wd_2', 'DIR_10m':'wd_3', 'DIR_25m':'wd_4',
#            'RHUM_2m':'hur_2', 'RHUM_10m':'hur_3', 'RHUM_25m':'hur_4',
##            'WQ_2m' #
#            }, inplace=True)
    
    #remove outliers lying outside of 4 standard deviation
    if remove_outliers:
        obs_ukmo = obs_ukmo[(obs_ukmo-obs_ukmo.mean()) <= (4*obs_ukmo.std())]
    
    obs_ukmo_xr = xr.Dataset.from_dataframe(obs_ukmo
#        [['time', 'TEMP_2m','RHUM_10mB']]
        )
    
    if create_netcdf:
        obs_ukmo_xr.to_netcdf(datafolder+filename.replace('.dat', '.temp'), 
                              engine='netcdf4')
        os.system('''
              cd {0}
              ncks -O --mk_rec_dmn time {1} {2}
              '''.format(datafolder, filename.replace('.dat', '.temp'),
              filename.replace('.dat', '.nc')))
              #we can add 'rm -f {1}' after to remove temp file
    
    if remove_modi:
        os.system('''
        cd {0}
        rm -f {1}'''.format(datafolder, filename_modi))

    return obs_ukmo_xr
    

def open_ukmo_rs(datafolder, filename, add_metpy_units=False):
    """
    Open the radiosounding from UK MetOffice formatted under .txt,
    and return it into xarray dataset with same variable names 
    than .nc file from cnrm.
    """
    
    filename_modi = filename + '_modi'
    
    os.system('''
        cd {0}
        sed -e 's/Latitude/Lat  /g' -e 's/Longitude/Lon  /g' {1} > {2}
        sed -i 's/ \+ /\t/g' {2}
        '''.format(datafolder, filename, filename_modi))
    
    #read header
    with open(datafolder + filename_modi) as f:
        fulltext = f.readlines()
        for i in [0, 4]:
            line = fulltext[i]
            key, value = line.split(sep='\t')
            if i == 0:
                datetime = pd.Timestamp(value)
            if i == 4:
                start_alti = float(value.replace(' m', ''))
    
    #reard data
    obs_ukmo = pd.read_table(datafolder + filename_modi,
                        sep="\t",
                        header=7)
    
    obs_ukmo.dropna(axis=1, how='all', inplace=True)
    obs_ukmo.rename(columns = {'Time':'seconds', 
                               'Height':'height', 
                               'Pres':'pressure', 
                               'Temp':'temperature', 
                               'RH':'humidity',
                               'Dewp':'dewPoint', 
                               'MixR':'mixingRatio',
                               'Dir':'windDirection', 
                               'Speed':'windSpeed',
                               'Lat':'latitude', 
                               'Lon':'longitude', 
    #                           'Range', 
    #                           'Azimuth'
                               }, inplace=True)
    
    units_row = obs_ukmo.iloc[0]    #retrieve first row (of units)
    units_row.replace('deg C', 'degC', inplace=True)
    obs_ukmo.drop(index=0, inplace=True)     #drop first row (units)
    
    #convert all string number to float
    obs_ukmo = obs_ukmo.astype(float)
    
    #compute altitude ASL from height AGL
    obs_ukmo['altitude'] = obs_ukmo['height'] + start_alti
    units_row['altitude'] = units_row['height']
    
    #compute datetimes from time
    obs_ukmo['time'] = pd.to_timedelta(obs_ukmo['seconds'], 'S') + datetime
    
    obs_ukmo_xr = xr.Dataset.from_dataframe(obs_ukmo)
    # add units
    if add_metpy_units:
        raise DeprecationWarning("""metpy not used anymore in here. 
                                 No units added.""")
        # for key in units_row.index:
        #     obs_ukmo_xr[key] = obs_ukmo_xr[key]*units(units_row[key])

    return obs_ukmo_xr


def open_ukmo_lidar(datafolder,
                    filter_low_data=True, level_low_filter=100,
                    create_netcdf=True,):
    """
    Open the lidar doppler data from UK MetOffice formatted under .txt,
    and return it into xarray dataset with same variable names 
    than .nc file from cnrm.
    
    Parameters:
        datafolder: str, contains multiple files that will be concatenated
            together
        filter_low_data: bool, to remove or not the lowest data 
            that may be inacurrate
        level_low_filter: float, UKMO mentions that below 100m, data may be
            inacurrate.
        create_netcdf: bool
    """
    
    fnames = os.listdir(datafolder)
    fnames = [f for f in fnames if '.hpl' in f]  # keep only .hpl files
    fnames.sort()
    
    dict_ws = {}
    dict_wd = {}
    
    for filename in fnames:
        datetime = pd.Timestamp(filename[58:73])
#        datetime = filename[58:73]  # string format
        obs_ukmo = pd.read_table(datafolder + filename,
                                 skiprows=1,
                                 delim_whitespace=True,  # one or more space as delimiter
                                 names=['level_agl', 'WD', 'WS'])
        #drop rows that are null
        obs_ukmo = obs_ukmo[obs_ukmo['WS'] != 0]
        if filter_low_data:
            obs_ukmo = obs_ukmo[obs_ukmo['level_agl'] > level_low_filter]
        obs_ukmo.set_index(['level_agl'], inplace=True)
        
        dict_ws[datetime] = obs_ukmo['WS']
        dict_wd[datetime] = obs_ukmo['WD']
        
    df_ws = pd.DataFrame(dict_ws)
    df_wd = pd.DataFrame(dict_wd)
    
    obs_ukmo_xr = xr.merge([xr.DataArray(df_ws, name='WS'), 
                            xr.DataArray(df_wd, name='WD')],)
    obs_ukmo_xr = obs_ukmo_xr.rename({'dim_1': 'time'})
        
    if create_netcdf:
        out_filename = filename[:64] + '.nc'  # out_filename= 'LIAISE_[...]-30MIN_L1_202107.nc'
        obs_ukmo_xr.to_netcdf(datafolder + out_filename, 
                              engine='netcdf4')

    return obs_ukmo_xr


def open_uib_seb(QC_threshold = 9, create_netcdf=True):
    """
    Convert the csv files coming from UIB (universidad de los baleares)
    into netCDF files to be used in other functions
    
    QC_threshold: int,
        Threshold below which function does not keep the data.
        1 is the highest quality data, 9 the worst
    """
    filename1_orig = 'LIAISE_IRTA-CORN_UIB_SEB1-10MIN_L2.dat_orig'
    filename2_orig = 'LIAISE_IRTA-CORN_UIB_SEB2-10MIN_L2.dat_orig'
    datafolder = '/cnrm/surface/lunelt/data_LIAISE/irta-corn/seb/'
    
    filename1 = filename1_orig.replace('_orig', '')
    filename2 = filename2_orig.replace('_orig', '')
    
    #replace some characters for file 1
    os.system('''
        cd {0}
        cp {1} {2}
        sed -i 's/\"//g' {2}
        sed -i 's/\//_per_/g' {2}
        sed -i 's/%/percent/g' {2}
        '''.format(datafolder, filename1_orig, filename1)
        ) #         sed -i 's/\s/_/g' {2}
    #replace character for file 2
    os.system('''
        cd {0}
        cp {1} {2}
        sed -i 's/\"//g' {2}
        sed -i 's/\//_per_/g' {2}
        sed -i 's/%/percent/g' {2}
        '''.format(datafolder, filename2_orig, filename2)
        ) #        sed -i 's/\s/_/g' {2}
    
#    open_uib_seb('/cnrm/surface/lunelt/data_LIAISE/irta-corn/seb/', filename)
    units_list = []
    
    with open(datafolder + filename1) as f:
        fulltext = f.readlines()   # read all lines
        
        headers_line = fulltext[1]
        headers_list = headers_line.replace('\"', '').split(sep=',')
        units_line = fulltext[2]
        units_list.extend(units_line.replace('\"', '').split(sep=','))
        
        # gather data in lists inside another list
        data_list = []
        for i in range(4, len(fulltext)):
            data_list.append(fulltext[i].split(sep=','))
        
        df1 = pd.DataFrame(data_list, columns=headers_list)
        
    with open(datafolder + filename2) as f:
        fulltext = f.readlines()    # read all lines
        
        headers_line = fulltext[1]
        headers_list = headers_line.replace('\"', '').split(sep=',')
        units_line = fulltext[2]
        units_list.extend(units_line.replace('\"', '').split(sep=','))
        
        # gather data in lists inside another list
        data_list = []
        for i in range(4, len(fulltext)):
            data_list.append(fulltext[i].split(sep=','))
        
        df2 = pd.DataFrame(data_list, columns=headers_list)
    
    # merge the 2 dataframe into one
#    df3 = pd.merge(df1, df2, on='TIMESTAMP')
    df3 = pd.merge(df1, df2, how='outer', on='TIMESTAMP')
    df3 = df3.sort_values(by='TIMESTAMP')
    
    # convert TIMESTAMP strings into pd.Timestamp
    res = []
    for dati in df3['TIMESTAMP']:
        year = int(dati[0:4])
        month = int(dati[5:7])
        day = int(dati[8:10])
        hour = int(dati[11:13])
        minu = int(dati[14:15])
        if hour == 24:
            # convert T24:00 to d+1T00:00 / BUT PB if join 'outer' (repeat twice the midnight timestamp)
#            date = pd.Timestamp(year, month, day)
#            unixdate = (date - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
#            unixtime = unixdate + hour*3600 + minu*60
#            res.append(pd.to_datetime(unixtime, unit='s'))
            pass
        else:
            res.append(pd.Timestamp(dati))
#        except:
#            print(dati)
#            res.append(res[-1])
    
#    df3['time'] = [pd.Timestamp(dati) for dati in df3['TIMESTAMP']]
#    df3['time'] = [dt.strptime(dati, '%Y-%m-%d %H:%M:%S') for dati in df3['TIMESTAMP']]
    df3['time'] = pd.Series(res)
    
    
    # set time as index
    df4 = df3.set_index('time')
    df4 = df4.sort_values(by='time')
    
    df5_list = [pd.to_numeric(df4[key], errors='coerce') for key in df4.columns]
    df5 = pd.concat(df5_list, axis=1)
    
    # convert to xarray dataset and sets units
    ds = df5.to_xarray()
    for i, key in enumerate(list(ds.keys())):
        ds[key].attrs['unit'] = units_list[i]
    
    # remove last columns that contains \n everywhere and is problematic
    # for creating netcdf
    ds = ds.drop_vars(['LWmV_Avg\n', 'batt_volt_Min\n'])
    
    # Remove bad quality data - filter with Quality Check:
#    if QC_threshold < 9:
    ds['LE'] = ds['LE'].where(ds['LE_QC'] <= QC_threshold)
    ds['H'] = ds['H'].where(ds['H_QC'] <= QC_threshold)
    ds['FC_mass'] = ds['FC_mass'].where(ds['FC_QC'] <= QC_threshold)
    
    if create_netcdf:
        # BUG: if file already existing, it raises 'ascii' codec rerror
        ds.to_netcdf(datafolder + 'LIAISE_IRTA-CORN_UIB_SEB-10MIN_L2.nc', 
                     engine='netcdf4')
    
    return ds

def extract_profile_lidar(wanted_date, site='planier',
                                folder_path=gv.global_data_nemo):
    """
    Functions that extract a vertical profile from lidar data under csv.

    Parameters
    ----------
    wanted_date : string that can be interpreted by pd.Timestamp
        DESCRIPTION.
    site: string
        DESCRIPTION
    folder_path: string
        DESCRIPTION

    Returns
    -------
    lidar_data_dict : dict
        DESCRIPTION.

    """
    
    if site not in ['planier']:
        print('No lidar data available for this site')
        return None
    
    lidar_filepath_dict = {
        'ws': f'{folder_path}/{site}/30-min/wind_speed.csv',
        'wd': f'{folder_path}/{site}/30-min/wind_direction.csv',
        'tke_along': f'{folder_path}/{site}/30-min/TKE_along_wind_standard_method.csv',
        'tke_cross': f'{folder_path}/{site}/30-min/TKE_cross_wind_standard_method.csv',
        'tke_verti': f'{folder_path}/{site}/30-min/TKE_vertical_wind_standard_method.csv',
        }

    lidar_data_list = ['ws', 'wd', 'tke_along', 'tke_cross', 'tke_verti']
    lidar_data_dict = {}
    
    # find nearest date:
    data_temp = pd.read_csv(lidar_filepath_dict[lidar_data_list[0]], 
                            sep='\t')
    datetime_list = data_temp.time
    nearest_date, i = nearest(pd.DatetimeIndex(datetime_list), pd.Timestamp(wanted_date))
    print('nearest date found is ', nearest_date)
    
    for var in lidar_data_list:
        data_temp = pd.read_csv(lidar_filepath_dict[var], 
                                sep='\t')
        data_temp.columns = ['time', 60, 80, 100, 120, 140, 
                      160, 180, 200, 220, 240]
        data_temp.index = pd.DatetimeIndex(data_temp.time)
        data_temp = data_temp.drop(columns=['time'])
        data_temp_wanted_date = data_temp.loc[pd.Timestamp(nearest_date)]
        lidar_data_dict[var] = data_temp_wanted_date

    # add 3d TKE:
    lidar_data_dict['tke'] = lidar_data_dict['tke_along'] + \
        lidar_data_dict['tke_cross'] + lidar_data_dict['tke_verti']
    
    return lidar_data_dict


def extract_profile_rs_mf(site, wanted_date,
                          max_height=3000, 
                          folder_path='/home/lunelt/Data/data_NEMO/rs/',
                          filename='DALTI2024000515169.csv'):
    """
    Function that return the radiosounding for the given site
    
    Parameters
    ----------
    site : string
        DESCRIPTION.
    wanted_date : string that can be interpreted by pd.Timestamp
        DESCRIPTION.
    max_height : float, optional
        DESCRIPTION. The default is 3000.
    folder_path : string, optional
        DESCRIPTION. The default is global_data_nemo.
    filename : TYPE, optional
        DESCRIPTION. The default is 'DALTI2024000515169.csv'.

    Returns
    -------
    pd.Dataframe with available vars

    """
    try:
        # Convert the input string to a pandas timestamp
        timestamp = pd.to_datetime(wanted_date)
        # Format the timestamp to the desired format
        formatted_date = timestamp.strftime('%Y%m%d%H%M')
    except Exception as e:
        raise ValueError(f"Invalid input wanted_date: {wanted_date}. Error: {e}")
    
    ## check site is available or go fetch the closest rs
    if site not in ['nimes']:
        print('No radiosounding data available for this site')
        return None
    
    ## load radiosounding file
    
    datelist = []
    rs_dict = {}
    
    with open(f'{folder_path}/{filename}') as file:
        counterline = 0
        newdate = True  # first line contain a date
        
        for line in file:
            # print(line)
            counterline += 1
            if newdate:
                newdate = False
                currentdate = line.split(sep=';')[1]
                datelist.append(currentdate)
                firstline_of_rs = counterline
            if line in ['\n', '\r\n']:
                newdate = True
                if currentdate[-4::] in ['0600', '1800']:
                    pass
                else:
                    try:
                        rs_dict[currentdate] = pd.read_csv(
                            f'{folder_path}/{filename}',
                            sep=';', 
                            names=gv.dict_rs_mf_metadata.keys(),
                            skiprows = firstline_of_rs, 
                            nrows = counterline-1-firstline_of_rs)
                    except pd.errors.EmptyDataError:
                        pass
    
    ## subset data (date and lowest part of rs)
    rs = rs_dict[formatted_date]
    rs_low = rs.where(rs['ALTI']<max_height).dropna(how='all')
    
    ## diagnostic data 
    rs_low['theta'] = potential_temperature_from_temperature(
        rs['P']*100, rs['T']+273.15)
    rs_low['theta_v'] = rs_low['theta']*(1 + 0.61*rs_low['RV'])
    
    return rs_low





def get_obs_filename_from_date(datafolder, wanted_date='20210722-1200', 
                       dt_threshold=pd.Timedelta('0 days 01:30:00'),
                       regex_date='202107\d\d.\d\d\d\d'):
    """
    Function returning obs filename corresponding to closest datetime. 
    Obs filename have varying filename depending on launch time
    for radiosoundings.
    
    Parameters:
        datafolder: str
            folder containing the obs
        wanted_datetime: str, that can be convert to pd.Timestamp
            ex: '20210722-1200', '20210722T1200'
    
    Returns:
        filename: str
        distance: pd.Timedelta, time diff between wanted and found file.
    """
    fnames = os.listdir(datafolder)
    fnames.sort()
    dates = []
    
    for f in fnames:
        match = re.search(regex_date, f)
        stringdate = f[match.start():match.end()]
        try:
            datetime = pd.Timestamp(stringdate)
        except ValueError as e:
            if 'could not convert string to Timestamp' not in e.args:
                raise e
            stringdate = stringdate.replace('_', '-')
            datetime = pd.Timestamp(stringdate)
        dates.append(datetime)
    
    #convert str to timestamp
    wanted_datetime = pd.Timestamp(wanted_date)
    #find nearest datetime
    nearest_date, i_near = nearest(dates, wanted_datetime)
    print("The nearest date found in files is: " + str(nearest_date))
    
    #compute time distance between wanted datetime and available obs
    distance = abs(nearest_date - wanted_datetime)
    # check it is below threshold
    if distance > dt_threshold:
        raise FileNotFoundError("No obs at wanted datetime (or not close enough)")
    
    filename = fnames[i_near]
    
    return filename



#%% Misc

def distance_from_lat_lon(lat_lon_1, lat_lon_2):
    """
    compute distance in km between 2 pts given their coordinates in degrees
    """
    lat1_rad = np.radians(lat_lon_1[0])
    lon1_rad = np.radians(lat_lon_1[1])
    lat2_rad = np.radians(lat_lon_2[0])
    lon2_rad = np.radians(lat_lon_2[1])
    radius_earth = 6371.01  #km
    
    dist = radius_earth * np.arccos(
            np.sin(lat1_rad)*np.sin(lat2_rad) + np.cos(lat1_rad)*np.cos(lat2_rad)*np.cos(lon1_rad - lon2_rad))
    
    return dist


def compute_errors(y_ref, y_to_eval, x_ref=None, x_to_eval=None,
                   method='direct'):
    """
    Computes the errors of an output compared to a reference.

    Parameters
    ----------
    y_ref : TYPE
        DESCRIPTION.
    y_to_eval : TYPE
        DESCRIPTION.
    x_ref : TYPE, optional
        DESCRIPTION. The default is None.
    x_to_eval : TYPE, optional
        DESCRIPTION. The default is None.
    method : TYPE, optional
        DESCRIPTION. The default is 'direct'.

    Returns
    -------
    errors: dict

    """ 
    
    errors = {}
    
    if method in ['interp', 'direct']:
        if method == 'interp':
            if (x_ref is None) or (x_to_eval is None):
                raise NameError('x-ref and x_to_eval cannot be None')
            else:
                y_to_eval = np.interp(x_ref, x_to_eval, y_to_eval)
        
        # compute bias and rmse, and keep values with 3 significant figures
        diff = np.array(y_to_eval) - np.array(y_ref)
        errors['bias'] = float('%.3g' % np.nanmean(diff))
        errors['rmse'] = float('%.3g' % np.sqrt(np.nanmean(diff**2)))
    
    elif method == 'integral':
        print('Not ready yet')

        xrange_intersec = intersec_of_intervals(x_ref, x_to_eval)
        
        x_ref_new = np.array(x_ref).clip(xrange_intersec[0], xrange_intersec[1])
        x_ref_new = np.append(xrange_intersec[0], x_ref_new, xrange_intersec[1])
        
        y_to_eval = np.interp(x_ref_new, x_to_eval, y_to_eval)
        y_ref = np.interp(x_ref_new, x_to_eval, y_to_eval)
                
        integral_ref = integrate.cumtrapz(y_ref, x=x_ref)[-1]
        integral_to_eval = integrate.cumtrapz(y_to_eval, x=x_to_eval)[-1]
        
        errors['bias'] = (integral_ref - integral_to_eval) / (xrange_intersec[1] - xrange_intersec[0])

    return errors


def moving_average(arr, window_size = 3):
    """computes a moving average on the array given"""
    i = 0
    moving_averages = []
    # if windows_size in odd
    window_side = int((window_size - 1)/2)
    for i in range(len(arr)):
        if i < window_side or i >= (len(arr) - window_side):
            moving_averages.append(arr[i])
#        elif i >= len(arr) - window_side:
        else:
            # Calculate the average of current window
            window_average = round(np.sum(arr[i-window_side:i+1+window_side]) / window_size, 2)
            # append
            moving_averages.append(window_average)
        i += 1
      
    return np.array(moving_averages)

def nearest(items, pivot):
    """ 
    Returns nearest element and corresponding index 
    
    Parameters:
        items: iterable
        pivot: the target value
    
    Returns:
        tuple containing:
            1. The nearest value in items
            2. The index for this value in items
            
    """
    # compute a list of distance between each item and the pivot
    dist = [abs(elt - pivot) for elt in items]
    
    return items[np.argmin(dist)], np.argmin(dist)


def intersec_of_intervals(interval1, interval2):
    """
    Find the intersection between two intervals
    
    Parameters:
        interval1: list of float,
            if len(interval1)>2, only the first and last values are considered
        interval2: list of float,
            if len(interval2)>2, only the first and last values are considered
    
    Returns:
        intersec: list of two values (min and max)
        
    """
    
    intersec = []
    if (interval1[0] > interval2[-1] or interval1[-1] < interval2[0]):
        return None    
    else:
        intersec.append(max(interval1[0], interval2[0]))
        intersec.append(min(interval1[-1], interval2[-1]))
 
    return intersec

#%% Manipulation of NetCDF files

def concat_obs_files(datafolder, in_filenames, out_filename, 
                     dat_to_nc=None, overwrite=False):
    """
    Check if file already exists. If not create it by concatenating files
    that correspond to "in_filenames*". It uses shell script, and the command
    'ncrcat' for it.
    
    Parameters:
        datafolder: 
            str, absolute path
        in_filenames: 
            str, common partial name of files to be concatenated.
            Used with wildcard "*" in the shell script run.
        out_filename:
            str, name of output file
        dat_to_nc: 
            str 'uib' or 'ukmo' or None, 
            convert .dat files into .nc files prior to concatenation
            
    Returns: Nothing in python, but it should have created the output file
    in the datafolder if not existing before.        
    """
    
    if not os.path.exists(datafolder + out_filename) or overwrite:
        # CREATE netCDF file from .dat
        if dat_to_nc == 'ukmo':
            fnames = os.listdir(datafolder)
            for f in fnames:
                if '.dat' not in f:
                    pass
                else:
                    open_ukmo_mast(datafolder, f, 
                                   create_netcdf=True, remove_modi=True)
        elif dat_to_nc == 'uib':
            open_uib_seb(create_netcdf=True)
    
    # CONCATENATE
        print("creation of file: ", out_filename)
        out = os.system('''
            cd {0}
            ncrcat {1}*.nc -o {2}
            '''.format(datafolder, in_filenames, out_filename))
        if out == 0:
            print('Creation of file {0} successful !'.format(out_filename))
        else:
            print('Creation of file {0} failed !'.format(out_filename))


def concat_simu_files_1var(datafolder, varname_sim, in_filenames, out_filename):
    """
    Check if file already exists. If not create it by concatenating files
    that correspond to "in_filenames". "in_filenames" must contain wildcards
    like * in order to gather more than one file. It concatenates the values
    for only one or few variables in varname_sim in order to get a reasonable
    siez for the output file.
    It then uses shell script, and the command 'ncecat' to concatenate.
    
    Parameters:
        datafolder: 
            str, absolute path
        varname_sim:
            str, variable name that will be concatenated over the files
            ex: 'T2M', 'UT,VT'
        in_filenames: 
            str, common partial name of files to be concatenated 
            with wildcard (?, *, ...)
        out_filename:
            str, name of output file
            
    Returns: Nothing in python, but it should have created the output file
    in the datafolder if not existing before.
    
    # ncecat -v {1},time,latitude,longitude {2} {3}
    ncrcat cannot be used in place of ncecat because it does not attach 
    the values 'time' dimension to the variable of interest.
    
    """
    
    if not os.path.exists(datafolder + out_filename):
        print('in_filenames = ', in_filenames)
        print("creation of file: ", out_filename)
        out = os.system('''
            cd {0}
            ncecat -v {1},time,latitude,longitude {2} {3}
            '''.format(datafolder, varname_sim, 
                       in_filenames, out_filename))
        if out == 0:
            print('Creation of file {0} successful!'.format(out_filename))
        else:
            print('Creation of file {0} failed!'.format(out_filename))
    else:
        print('file {0} already exists.'.format(out_filename))
    #command 'cdo -select,name={1} {2} {3}' may work as well, but not always...
    
    
#%% Manipulation of MNH/SFX files

def check_filename_datetime(filepath, wanted_date, error_processing='warning',
                            verbose=False):
    """
    Check that filename segment number is day and suffix number is hour
    """
    ds = xr.open_dataset(filepath)
    datetime_file = pd.Timestamp(ds.time.data[0])
    
    datetime_correspondance = (datetime_file == pd.Timestamp(wanted_date))
    
    if not datetime_correspondance:
        if verbose:
            print(f'Datetime of file is: {datetime_file}')
            print(f'Wanted datetime is: {pd.Timestamp(wanted_date)}')
    
        if error_processing == 'warning':
            print('Wanted datetime and effective datetime do not match')
        else:
            raise ValueError(
                'Wanted datetime and effective datetime do not match')
    
    return datetime_correspondance
        

def check_filename_datetime_old(filepath, fix=False):
    """
    Check that filename segment number is day and suffix number is hour
    """
    ds = xr.open_dataset(filepath)
    datetime = pd.Timestamp(ds.time.data[0])
    #get filename SEG nummber
    filepath_nosuf = filepath.replace('dg.nc','').replace('.nc','')  #remove filetype
    seg_nb = filepath_nosuf[-6:-4]  #get SEG number
    suffix_nb = filepath_nosuf[-3::]  #get suffix nb
    
    #correct values based on datetime variable
    correct_seg_nb = str(datetime.day)
    correct_suffix_nb = str(datetime.hour)
    
    # special case for midnight
    if correct_suffix_nb == '0':
        correct_seg_nb = str(int(correct_seg_nb) - 1)
        correct_suffix_nb = '24'
    
    # format suffix with 3 digits:
    if len(correct_suffix_nb) == 1:
        correct_suffix_nb = '00'+ correct_suffix_nb
    elif len(correct_suffix_nb) == 2:
        correct_suffix_nb = '0'+ correct_suffix_nb
    
    if seg_nb != correct_seg_nb:
        if fix:
            new_fpath = re.sub('SEG..\.0..', 'SEG{0}.{1}'.format(
                    correct_seg_nb, correct_suffix_nb), filepath)
            out = os.system('''
                mv --backup=numbered {0} {1}
                '''.format(filepath, new_fpath))
            print('OUT = {0}'.format(out))
            print("* File '{0}' RENAMED IN: '{1}'".format(filepath, new_fpath))
            return False
        else:
            return False
#            raise NameError('the date in simu file do not match filename SEG')
    
    if suffix_nb != correct_suffix_nb:
        if fix:
            new_fpath = re.sub('SEG..\.0..', 'SEG{0}.{1}'.format(
                    correct_seg_nb, correct_suffix_nb), filepath)
            out = os.system('''
                mv {0} {1}
                '''.format(filepath, new_fpath))
            print('OUT = {0}'.format(out))
            print("* File '{0}' RENAMED IN: '{1}'".format(filepath, new_fpath))
            return False
        else:
            return False
#            raise NameError('the time in simu file do not match filename suffix')
    
    return True


def check_folder_filenames(folder='/cnrm/surface/lunelt/NO_SAVE/nc_out/1.11_std_2021_21-24/'):
    """
    Check that the naming of the netcdf output is consistent 
    with its internal datetime.
    
    """
    
    ls = os.listdir(folder)
    
    regex = re.compile(r'LIAIS.2.SEG.*[^0]dg.nc$')
#    numbered_re = re.compile(r'~1~')
#    numbered = [fn for fn in ls if numbered_re.match(fn)]
    
    filtered = [fn for fn in ls if regex.match(fn)]
    filtered.sort()
    print(filtered)
    
    filtered = ['LIAIS.2.SEG21.017dg.nc', 'LIAIS.2.SEG21.018dg.nc']
    
    pb = []
    for fn in filtered:      
        filepath = folder + fn
        if not check_filename_datetime(filepath, fix=False):
            pb.append(fn)
            print('issue with ', fn)
        else:
            print(fn + ' is OK')
    


def get_indices_of_lat_lon(ds, lat, lon, verbose=True):
    """ 
    Find indices corresponding to latitude and longitude values
    for a given file.
    
    ds: xarray.DataSet
        Dataset containing fields of netcdf file
    lat: float, 
        Latitude
    lon: float,
        Longitude
    
    Return:
        tuple [index_lat, index_lon]
    
    """
    # Get latitude and longitude data
    try:
        lat_dat = ds['latitude'].data
        lon_dat = ds['longitude'].data
    except KeyError:
        try:
            lat_dat = ds['latitude_u'].data
            lon_dat = ds['longitude_u'].data
        except KeyError:
            try:
                lat_dat = ds['latitude_v'].data
                lon_dat = ds['longitude_v'].data
            except KeyError:
                raise AttributeError("""this dataset does not have 
                                     latitude-longitude coordinates""")
    
    # Gross evaluation of lat, lon (because latitude lines are curved)
    distance2lat = np.abs(lat_dat - lat)
    index_lat = np.argwhere(distance2lat <= np.nanmin(distance2lat))[0,0]
    
    distance2lon = np.abs(lon_dat - lon)
    index_lon = np.argwhere(distance2lon <= np.nanmin(distance2lon))[0,1]
    
#    print("Before refinement : index_lat={0}, index_lon={1}".format(
#            index_lat, index_lon))
    
    # refine as long as optimum not reached
    opti_reached = False
    n = 0  #count of iteration
    while opti_reached is False:
        if (np.abs(lon_dat[index_lat, index_lon] - lon) > \
            np.abs(lon_dat[index_lat, index_lon+1] - lon) ):
            index_lon = index_lon+1
        elif (np.abs(lon_dat[index_lat, index_lon] - lon) > \
            np.abs(lon_dat[index_lat, index_lon-1] - lon) ):
            index_lon = index_lon-1
        elif (np.abs(lat_dat[index_lat, index_lon] - lat) > \
            np.abs(lat_dat[index_lat+1, index_lon] - lat) ):
            index_lat = index_lat+1
        elif (np.abs(lat_dat[index_lat, index_lon] - lat) > \
            np.abs(lat_dat[index_lat-1, index_lon] - lat) ):
            index_lat = index_lat-1
        elif n > 20:
            raise ValueError("""loop does not converge, 
                             check manually for indices.""")
        else:
            opti_reached = True
    if verbose:
        print("For lat={0}, lon={1} : index_lat={2}, index_lon={3}".format(
            lat, lon, index_lat, index_lon))
    
    return index_lat, index_lon


def get_points_in_polygon(data_in, polygon):
    """
    Return a list of all the points data points that are in a given polygon
    
    parameters:
        data_in: array-like, xarray.dataset or xarray.dataarray
        
        polygon: 2D Polygon shape from shapely.geometry
        
    returns:
        list of points contained in the polygon
    
    """
    try:  #or: if 'ni_u' inlist(ds1.coords)
        ds = center_uvw(data_in)
    except (ValueError, KeyError):
        ds = data_in
        
    inside_points = []
    
    # get one of the dataarray in the dataset, and retrieve its shape
    if type(ds) == xr.core.dataset.Dataset:
#        random_dataarray = ds[list(ds.keys())[0]]
        random_dataarray = ds['latitude']  # latitude expected to be present in all (could be latitude_u also)
    elif type(ds) == xr.core.dataarray.DataArray:
        random_dataarray = ds
    else:
        raise TypeError('ds is not xr.dataset or xr.dataarray')
    
    # get shape of data
    try:
        shape_ds = random_dataarray[0, :, :].shape
    except IndexError:
        shape_ds = random_dataarray[:, :].shape
    
    # filter points that are inside the polygon
    for coords in np.ndindex(shape_ds):
        # get elements one by one
        elt = ds.isel(nj=coords[0], ni=coords[1])
#        elt = ds.isel(level=ilevel, nj=coords[0], ni=coords[1])

        # create location point
        location_elt = Point(elt.longitude, elt.latitude)
        if polygon.contains(location_elt):
            inside_points.append(elt)
            
    return inside_points


def get_surface_type(site, domain_nb, model='irr',
                     ds=None, translate_patch = True, 
                     nb_patch=12):
    """
    Returns the type and characteristics of land surface in SURFEX 
    for a given site
    
    """

    if ds is None:
        if model == 'irr':
            if domain_nb==2:
                ds = xr.open_dataset('/cnrm/surface/lunelt/NO_SAVE/nc_out/' + \
                                 '2.13_irr_d1d2_21-24/LIAIS.2.SEG22.012dg.nc')
            elif domain_nb==1:
                ds = xr.open_dataset('/cnrm/surface/lunelt/NO_SAVE/nc_out/' + \
                                 '2.13_irr_d1d2_21-24/LIAIS.1.SEG22.012dg.nc')
        if model == 'std':
            if domain_nb==2:
                ds = xr.open_dataset('/cnrm/surface/lunelt/NO_SAVE/nc_out/' + \
                                 '1.11_std_d1d2_21-24/LIAIS.2.SEG22.012dg.nc')
            elif domain_nb==1:
                ds = xr.open_dataset('/cnrm/surface/lunelt/NO_SAVE/nc_out/' + \
                                 '1.11_std_d1d2_21-24/LIAIS.1.SEG22.012.nc')
    
    # get indices of sites in ds
    index_lat, index_lon = indices_of_lat_lon(ds, 
                                              gv.sites[site]['lat'], 
                                              gv.sites[site]['lon'])
    
    res = {}
    weight = []
    if nb_patch == 12:
        patch_dict = {'PATCHP1': 'P1_NO',
                     'PATCHP2': 'P2_ROCK',
                     'PATCHP3': 'P3_SNOW',
                     'PATCHP4': 'P4_BROADLEAF_TREES',
                     'PATCHP5': 'P5_NEEDLELEAF_TREES', 
                     'PATCHP6': 'P6_TRBE_tropi_broad_evergreen', 
                     'PATCHP7': 'P7_C3',
                     'PATCHP8': 'P8_C4',
                     'PATCHP9': 'P9_IRRC4',
                     'PATCHP10': 'P10_GRASSC3',
                     'PATCHP11': 'P11_GRASSC4',
                     'PATCHP12': 'P12_BOG_PARK',
                     }
    else:
        raise ValueError('Not adapted yet to this patch number - TO DO')
        
    for key in patch_dict:
        val = float(ds[key][index_lat, index_lon].data)
        if val == 999:
            val=0
        if translate_patch:
            res[patch_dict[key]] = val
        else:
            res[key] = val
        weight.append(val)
        
    # keys already aggregated
    for key in ['LAI_ISBA', 'Z0_ISBA', 'Z0H_ISBA', 
#                'EMIS', 
                'CD_ISBA',  # coeff trainee
                'CH_ISBA',  # coeff echange turbulent pour sensible heat
                'CE_ISBA',  # coeff echange turbulent pour latent heat
                ]:
        sq_ds = ds[key] #.squeeze()
        res[key] = float(sq_ds[index_lat, index_lon].data)
    
    
    # keys given by patch, aggregated hereafter
    for key in ['CV',  # vegetation thermal inertia
                'ALBNIR', 'ALBUV', 'ALBVIS',  # albedo
                'DROOT_DIF',  # root (fraction?)
                ]:
        temp = []
        for p_nb in np.arange(1, 13):  # patch number
            keyP = '{0}P{1}'.format(key, p_nb)
            temp.append(float(ds[keyP][index_lat, index_lon].data))
        # remove 999 from list (replaced by 0)
        temp = [0 if i == 999 else i for i in temp]
        aggr_key = '{0}_ISBA'.format(key)
        
        res[aggr_key] = np.sum(np.array(weight) * np.array(temp))
        
    # keys by patch
    for key in ['TALB_P', 'LAIP', 'WWILT', 'WFC',
                ]:
        for p_nb in np.arange(1, 13):  # patch number
            keyP = f'{key}{p_nb}'
            val = float(ds[keyP][index_lat, index_lon].data) #.squeeze()
            if val == 999:
                res[keyP] = 0
            else:
                res[keyP] = float(val)
    
    # Get soil type from pgd
    if domain_nb == 2:
        pgd = xr.open_dataset(
            gv.global_simu_folder + \
            '2.01_pgds_irr/PGD_400M_CovCor_v26_ivars.nc')
    elif domain_nb == 1:
        pgd = xr.open_dataset(
            gv.global_simu_folder + \
            '2.01_pgds_irr/PGD_2KM_CovCor_v26_ivars.nc')
    
    for key in ['CLAY', 'SAND', ]:
        val = float(pgd[key][index_lat, index_lon].data)
        res[key] = val
        
    return res


def get_irrig_time(soil_moist, stdev_threshold=5):
    """
    soil_moist: xarray Dataarray,
        Variable representing soil moisture of other variable 
        strongly dependant on irrigation. Must have 'time' as coordinate.
    """
    # if soil_moist is None (not irrigated site for ex.)
    if soil_moist is None:
        return []
    # derivate of variable with time
    soil_moist_dtime = soil_moist.differentiate(coord='time')
    # keep absolute values
    soil_moist_dtime_abs = np.abs(soil_moist_dtime)
    
    # compute threshold to determine the outliers
    avrg = soil_moist_dtime_abs.mean()
    stdev = soil_moist_dtime_abs.std()
    thres = avrg + stdev_threshold*stdev
    
    # only keep outliers
    irr_dati = soil_moist_dtime_abs.where(soil_moist_dtime_abs > thres, drop=True).time
    
    # remove close datetimes
    tdelta_min = pd.Timedelta(6, unit='h')  #minimum time delta between 2 irrig
    irr_dati_filtered = []
    
    for i, dati in enumerate(irr_dati):
        if i == 0:
            continue
        else:
            if (irr_dati[i].data - irr_dati[i-1].data) > tdelta_min:
                irr_dati_filtered.append(dati.data)
    
    return irr_dati_filtered

   
def get_line_coords(data, 
                start_pt = (41.6925905, 0.9285671), # cendrosa
                end_pt = (41.590111, 1.029363), #els plans
                nb_indices_exterior=10,
                verbose=True):     
    """
    data: xarray dataset
    start_pt: tuple with (lat, lon) coordinates of start point
    start_pt: tuple with (lat, lon) coordinates of end point
    nb_indices_exterior: int, number of indices to take resp. before and after
        the start and end point.
        
    Returns:
        Dict with following keys:
            'ni_range':
            'nj_range':
            'slope':
            'index_distance': numbers of indices between the start and end pt
            'ni_step':
            'nj_step':
            'nij_step': distance in [m] between 2 pts of the line
            'ni_start':
            'ni_end':
            'nj_start':
            'nj_end':
    """
    # get ni, nj values (distance to borders of first domain in m)
    # for Start site
    index_lat_start, index_lon_start = indices_of_lat_lon(data, *start_pt, 
                                                          verbose=verbose)
    ni_start = data.ni[index_lon_start].values     # ni corresponds to longitude
    nj_start = data.nj[index_lat_start].values     # nj corresponds to latitude
    # for End site
    index_lat_end, index_lon_end = indices_of_lat_lon(data, *end_pt,
                                                      verbose=verbose)
    ni_end = data.ni[index_lon_end].values     # ni corresponds to longitude
    nj_end = data.nj[index_lat_end].values     # nj corresponds to latitude
    
    if verbose:
        print('ni, nj start:', ni_start, nj_start)
        print('ni, nj end:', ni_end, nj_end)
    
    #get line formula:
    if ni_end != ni_start:
        slope = (nj_end - nj_start)/(ni_end - ni_start)
        y_intercept = nj_start - slope * ni_start
        
        if verbose:
            print('slope and y-intercept :', slope, y_intercept)

        # approximate distance between start and end in term of indices
        index_distance = np.ceil(np.sqrt((index_lon_end - index_lon_start)**2 + \
                                         (index_lat_end - index_lat_start)**2))
        
        # distance between start and end in term of meters ...
        # .. on i (longitude)
        ni_step = (ni_end - ni_start)/(index_distance-1)
        # .. on j (latitude)
        nj_step = (nj_end - nj_start)/(index_distance-1)
        
        ni_range = np.linspace(ni_start - ni_step*nb_indices_exterior, 
                               ni_end + ni_step*nb_indices_exterior,
                               num=int(index_distance)+2*nb_indices_exterior
                               )
        
        nj_range = y_intercept + slope * ni_range
        
    else:  # ni_end == ni_start   -   vertical line - following meridian
        index_distance = np.abs(index_lat_end - index_lat_start) + 1
        nj_step = (nj_end - nj_start)/(index_distance - 1)
        ni_step = 0
        nj_range = np.linspace(nj_start - nj_step*nb_indices_exterior, 
                               nj_end + nj_step*nb_indices_exterior,
                               num=int(index_distance)+2*nb_indices_exterior)
        ni_range = np.array([ni_end]*len(nj_range))
        slope='vertical'
    
    # distance in meters between two points on section line
    nij_step = np.sqrt(ni_step**2 + nj_step**2)
    
    return {'ni_range': ni_range, 'nj_range': nj_range, 
            'slope': slope, 'index_distance': index_distance, 
            'ni_step': ni_step, 'nj_step': nj_step, 'nij_step': nij_step,
            'ni_start': ni_start, 'ni_end': ni_end,
            'nj_start': nj_start, 'nj_end': nj_end}

def subset_ds(ds, zoom_on=None, lat_range=[], lon_range=[],
              nb_indices_exterior=0):
    """
    Extract a subset of a domain.
    
    """
    
    if zoom_on != None:
        prop = gv.zoom_domain_prop[zoom_on]
        lat_range = prop['lat_range']
        lon_range = prop['lon_range']
    elif zoom_on == None:
        if lat_range == []:
            # raise ValueError("Argument 'zoom_on' or lat_range & lon_range must be given")
            print('No subsetting done on the dataset.')
            return ds
        # else:
        #     lat_range = lat_range
        #     lon_range = lon_range
        
    nj_min, ni_min = indices_of_lat_lon(ds, 
                                        np.min(lat_range), np.min(lon_range),
                                        verbose=False)
    nj_max, ni_max = indices_of_lat_lon(ds, 
                                        np.max(lat_range), np.max(lon_range),
                                        verbose=False)
    
    nj_min -= nb_indices_exterior
    ni_min -= nb_indices_exterior
    nj_max += nb_indices_exterior
    ni_max += nb_indices_exterior
    
#    # if need to be more flexible to coordinates: TODO
#    latitude_X = [key for key in list(ds.coords) if 'ni' in key][0]
#    lastletters = latitude_X[-2::]
#    if lastletters == '_u':
#            ds_subset = ds.isel(ni_u=np.arange(ni_min, ni_max), 
#                                nj_u=np.arange(nj_min, nj_max))
#    elif lastletters == 'de':
        
    ds_subset = ds.isel(ni=np.arange(ni_min, ni_max), 
                        nj=np.arange(nj_min, nj_max))
    
    return ds_subset.squeeze()

def center_uvw(data, data_type='wind', budget_type='UU', varname_bu='PRES'):
    """
    Interpolate in middle of grid for variable UT, VT and WT,
    rename the associated coordinates, and squeezes the result.
    Useful for operation on winds in post processing.
    
    Inputs:
        data: xarray.Dataset, containing variables 'UT', 'VT', 'WT'
    
    Returns:
        data: xarray.Dataset,
            same dataset but with winds positionned in the center of grid
    """
    if data_type == 'wind':
        data['UT'] = data['UT'].interp(ni_u=data.ni.values, nj_u=data.nj.values).rename(
                {'ni_u': 'ni', 'nj_u': 'nj'})
        data['VT'] = data['VT'].interp(ni_v=data.ni.values, nj_v=data.nj.values).rename(
                {'ni_v': 'ni', 'nj_v': 'nj'})
        # remove useless coordinates
        data_new = data.drop(['ni_u', 'nj_u', 'ni_v', 'nj_v'])
        try:
            data_new = data.drop(['latitude_u', 'longitude_u', 
                                  'latitude_v', 'longitude_v',])
        except ValueError:
            pass
    # for components UU and VV of MNH budgets ()
    elif data_type == 'budget':
        if budget_type == 'UU':
            data[varname_bu] = data[varname_bu].interp(ni_u=data.ni.values, nj_u=data.nj.values).rename(
                {'ni_u': 'ni', 'nj_u': 'nj'})
            # remove useless coordinates
            data_new = data.drop(['ni_u', 'nj_u', 
                                  'latitude_u', 'longitude_u', ])
                
        if budget_type == 'VV':
            data[varname_bu] = data[varname_bu].interp(ni_v=data.ni.values, nj_v=data.nj.values).rename(
                {'ni_v': 'ni', 'nj_v': 'nj'})
            # remove useless coordinates
            data_new = data.drop(['ni_v', 'nj_v',
                                  'latitude_v', 'longitude_v', ])
    
    # TRY loop in case no WT found in file
    try:
        data['WT'] = data['WT'].interp(level_w=data.level.values).rename(
                {'level_w': 'level'})
        data['ALT'] = data['ALT'].interp(level_w=data.level.values).rename(
                {'level_w': 'level'})
        # remove useless coordinates
        data_new = data.drop(['level_w'])
    except KeyError:
        pass
    
    # consider time no longer as a dimension but just as a single coordinate
    data_new = data_new.squeeze()
    
    return data_new


def interp_iso_asl(alti_asl, ds, varname, verbose=True):
    """
    Interpolate values at same altitude ASL
    
    Returns:
        np.array
    
    TODO: try fastening the function with scipy.interpn, or map()
        -> It did not work with map(),
        and I did not manage to use interpn (not adapted?)
    
    """
    
    ds_sub = ds[[varname, 'ZS', 'ALT']].squeeze()
    # ds_sub = ds_sub.where(ds_sub.ni<40000, drop=True)
    # change coordinates of ALT: level_w -> level
    ds_sub['ALT'] = ds_sub['ALT'].interp(level_w=ds_sub[varname].level.values).rename(
            {'level_w': 'level'})
    
    # make 2D array with same asl value everywhere
    alti_grid = ds_sub['ZS']*0 + alti_asl
    
    # make empty 2D array for the result
    res_layer = alti_grid*0
    
    ds_alt = ds_sub['ALT']
    ds_var = ds_sub[varname]
    
    # compute len() before loop allow to gain computation time
    nj_len = len(ds_sub.nj)
    ni_len = len(ds_sub.ni)
    
    for j in range(nj_len):
        if verbose:
            print('{0}/{1}'.format(j, nj_len))
            
        for i in range(ni_len):
            
            res_layer[j,i] = np.interp(
                # alti_grid.isel(ni=i, nj=j),  # 0d value of alti ASL wanted
                alti_asl,
                ds_alt.isel(ni=i, nj=j),  # 1d column of alti ASL
                ds_var.isel(ni=i, nj=j),  # 1d column of varname
                left = np.nan,
                right = None)
    
    
# OLD version: ---------
    
#     ds = ds[[varname, 'ZS']].squeeze()
    
#     # make with same asl value everywhere
#     alti_grid = np.array([[alti_asl]*len(ds.ni)]*len(ds.nj))
#     # compute the corresponding height AGL to keep iso alti ASL on each pt
#     level_grid = alti_grid - ds['ZS'].data
#     # initialize the result layer with same shape same than alti_grid
#     res_layer = level_grid*0
    
#     # get column of correspondance between level and height AGL
# #    level = ds[varname][:, :, :].level.data 
#     level = ds.level.data
    
#     # interpolation
#     for j in range(len(ds.nj)):
#         if verbose:
#             print('{0}/{1}'.format(j, len(ds.nj)))
#         for i in range(len(ds.ni)):
#             if level_grid[j,i] < 0:
#                 res_layer[j,i] = np.nan
#             else:
#     #            res_layer[i,j] = ds['PRES'].interp(
#     #                ni=ds.ni[i], nj=ds.nj[j], level=level_grid[i,j])
# #                res_layer[j,i] = np.interp(level_grid[j,i], 
# #                                           level, 
# #                                           ds[varname][:, j, i])
#                 f = interpolate.interp1d(level, 
#                                          ds[varname][:, j, i],
#                                          fill_value='extrapolate')
#                 res_layer[j,i] = f(level_grid[j,i])

    # new shorter loop: TODO
#    for tuple_coord in np.ndindex(ds[varname].isel(level=0).shape):
#        res_layer[tuple_coord] = np.interp(level_grid[tuple_coord], 
#                                           level, 
#                                           ds[varname][:, j, i])

    return res_layer


def interp_iso_agl(alti_agl, ds_asl, varname, verbose=True):
    """
    Interpolate values at same altitude ASL
    
    Returns:
        np.array
    
    """
    
    ds_asl = ds_asl[[varname, 'ZS']].squeeze()
    
    # make with same asl value everywhere
    agl_grid = np.array([[alti_agl]*len(ds_asl.ni)]*len(ds_asl.nj))
    # compute the corresponding height AGL to keep iso alti ASL on each pt
    asl_grid = agl_grid + ds_asl['ZS'].data
    # initialize the result layer with same shape same than asl_grid
    res_layer = (agl_grid*0).astype('float')
    
    # get column of correspondance between level and height AGL
#    level_asl = ds_asl.level.data
    
    # interpolation
    for j in range(len(ds_asl.nj)):
        if verbose:
            print('{0}/{1}'.format(j, len(ds_asl.nj)))
        for i in range(len(ds_asl.ni)):
            var_column = ds_asl[varname][:, j, i].dropna(dim='level')
            level_column = var_column.level
#            if np.isnan(ds_asl[varname][:, j, i]).all():
            if len(var_column) < 3:
                res_layer[j,i] = np.nan
            else:
                #TODO turn into np.interp
                f = interpolate.interp1d(level_column, 
                                         var_column,
                                         fill_value='extrapolate')
                res_layer[j,i] = f(asl_grid[j,i])
    
    return res_layer

def flux_pt_to_mass_pt(ds, verbose=True):
    """
    Change flux point coordinates of dataset into mass point coordinates.
    Done through interpolation between flux points.
    Generalization of center_uvw() function.
    
    parameter:
        ds: xarray.dataset
    """
    for var in list(ds.keys()):
        
        # NOT WORKING
#        for coord in ['nj_v', 'nj_u', 'ni_v', 'ni_u', 'level_w']:
#            new_coord = coord[:-2]
#            ds[var] = ds[var].interp(coords={coord:ds[new_coord].values}).rename(
#                {coord: new_coord})
#            if verbose:
#                print(f"{var}: coordinates '{coord}' changed to '{new_coord}'")
            
        # change coordinates for 'nj_v' flux points
        if 'nj_v' in list(ds[var].coords):
            ds[var] = ds[var].interp(nj_v=ds.nj.values).rename(
                {'nj_v': 'nj'})
            if verbose:
                print(f"{var}: coordinates 'nj_v' changed to 'nj'")
        
        # change coordinates for 'ni_v' flux points
        if 'ni_v' in list(ds[var].coords):
            ds[var] = ds[var].interp(ni_v=ds.ni.values).rename(
                {'ni_v': 'ni'})
            if verbose:
                print(f"{var}: coordinates 'ni_v' changed to 'ni'")
        
        # change coordinates for 'ni_u' flux points
        if 'ni_u' in list(ds[var].coords):
            ds[var] = ds[var].interp(ni_u=ds.ni.values).rename(
                {'ni_u': 'ni'})
            if verbose:
                print(f"{var}: coordinates 'ni_u' changed to 'ni'")
        
        # change coordinates for 'nj_u' flux points
        if 'nj_u' in list(ds[var].coords):
            ds[var] = ds[var].interp(nj_u=ds.nj.values).rename(
                {'nj_u': 'nj'})
            if verbose:
                print(f"{var}: coordinates 'nj_u' changed to 'nj'")
        
        # change coordinates for '_w' flux points
        if 'level_w' in list(ds[var].coords):
            ds[var] = ds[var].interp(level_w=ds.level.values).rename(
                {'level_w': 'level'})
            if verbose:
                print(f"{var}: coordinate 'level_w' changed to 'level'")
    
        # ALT coordinate badly encoded ('level_w not defnied as coords)
        if var == 'ALT':
            ds[var] = ds[var].interp(level_w=ds.level.values).rename(
                {'level_w': 'level'})
            if verbose:
                print(f"{var}: coordinate 'level_w' changed to 'level'")
    
    # remove flux coordinates
    try:
        ds_new = ds.drop(['latitude_u', 'longitude_u', 'ni_u', 'nj_u',
                      'latitude_v', 'longitude_v', 'ni_v', 'nj_v',
                      'level_w'
                      ])
    except:
        if verbose:
            print("no old coordinate dropped - error during ds.drop()")
        ds_new = ds

    return ds_new


def agl_to_asl_coords(ds_agl, alti_asl_arr=np.arange(25,1000,25)):
    """
    Change coordinates of dataset from terrain-following (i.e. levels 
    are above ground level - AGL), to flat (i.e. levels are above sea 
    level - ASL).
    
    ds: xarray.Dataset containing 3D variables, ZS and coordinate level.
    
    N.B.:
        Computation time: 
            3s per alti if domain is 20x20 (zoom_on 'liaise')
            ...s per alti if domain is 47x52 (zoom_on 'urgell')
            ...s per alti if domain is 90x108 (zoom_on 'd2')
            ...s per alti if domain is 164x290 (zoom_on 'd1' or None)
        
    """
    var_list = list(ds_agl.keys())
    var3d_list = var_list
    var3d_list.remove('ZS')
    
    # SOME PRELIMINARY CHECK
    if len(alti_asl_arr) < 2:
        raise ValueError('alti_asl_arr must be greater than 1, use inter_iso_asl for 1 layer only')
    
    var_temp_construc = var_list[0]
    if 'pint.quantity' in str(type(ds_agl[var_temp_construc].data)):  # issue if type pint.Quantity
        var_temp_construc = var_list[1]
    
    # keep only layers of interest: (to gain computation time)
    #TODO
#    alti_agl_max = float(alti_asl_arr.max() - ds_agl_in.ZS.min())
    #ds_agl_in.where(ds_agl_in.level < alti_agl_max, drop=True)
    
    # CREATE NEW DATASET based on AGL dataset
    ds_agl = ds_agl.squeeze()
    ds_asl = ds_agl[[var_temp_construc, 'ZS']].isel(level=np.arange(len(alti_asl_arr)))
    
    for var in var3d_list:
        ds_asl[f'{var}_ASL'] = ds_asl[var_temp_construc]*0
    
    # change levels from AGL to ASL
    ds_asl['level'] = alti_asl_arr
    # remove vars that are not ASL
    ds_asl = ds_asl.drop_vars([var_temp_construc])  # Now added in var_list
    
    
    # INTERPOLATION
    t0 = time.time()

    for ilevel, alti in enumerate(alti_asl_arr):
        print(f'Computation for {alti}m')
        for var in var3d_list:
            layer_interp = interp_iso_asl(alti, ds_agl, var, verbose=False)
            ds_asl[f'{var}_ASL'][ilevel,:,:] = layer_interp
            
    print('interpolation time:', time.time()-t0)
    
    return ds_asl


def asl_to_agl_coords(ds_asl, alti_agl_arr=np.arange(0,200,25)):
    """
    Change coordinates of dataset from flat (i.e. levels are above sea 
    level - ASL) to terrain-following (i.e. levels 
    are above ground level - AGL).
    
    ds: xarray.Dataset containing 3D variables, ZS and coordinate level.
    
    N.B.:
        Computation time: 
            0.003s per alti if domain is 20x20 (zoom_on 'liaise')
            1.5s per alti if domain is 47x52 (zoom_on 'urgell')
            10s per alti if domain is 90x108 (zoom_on 'd2')
            100s per alti if domain is 164x290 (zoom_on 'd1' or None)
        
    """
    var_list_ASL = list(ds_asl.keys())
    var3d_list_ASL = var_list_ASL
    var3d_list_ASL.remove('ZS')
    
    # SOME PRELIMINARY CHECK
    if len(alti_agl_arr) < 2:
        raise ValueError('alti_agl_arr must be greater than 1, use inter_iso_agl for 1 layer only')

    
    # keep only layers of interest: (to gain computation time)
    #TODO

    ds_asl = ds_asl.squeeze()

#    # OLD WAY
#    # CREATE NEW DATASET based on AGL dataset
#    var_temp_construc = var_list_ASL[0]
#    if var_temp_construc == 'ZS':  # issue if type pint.Quantity
#        var_temp_construc = var_list_ASL[1]
#    ds_agl = ds_asl[[var_temp_construc, 'ZS']]
#
#    for var_ASL in var_list_ASL:
#        var_agl = var_ASL[:-4]  # remove '_asL' from varnames 'THT_ASL' -> 'THT'
#        ds_agl[var_agl] = ds_agl[var_temp_construc]*0
#    
#    # change levels from AGL to ASL
#    ds_agl['level'] = alti_agl_arr
#    # remove vars that are not ASL
#    ds_agl = ds_agl.drop_vars([var_temp_construc])
    
    # NEW WAY
    # CREATE NEW DATASET based on AGL dataset
    var3d_list_AGL = [key.replace('_ASL', '') for key in var3d_list_ASL]  # remove '_ASL' from varnames 'THT_ASL' -> 'THT'
    dataarray3d_prop = (('level', 'nj', 'ni'), np.zeros((len(alti_agl_arr), 
                                                      len(ds_asl.nj), 
                                                      len(ds_asl.ni))
                                                        ))
    ds_agl = xr.Dataset()
    for var3d in var3d_list_AGL:
        var_dict = {var3d: dataarray3d_prop}
        ds_agl = ds_agl.merge(xr.Dataset(data_vars=var_dict).copy(deep=True))
        
    ds_agl = ds_agl.merge(ds_asl['ZS'])
    ds_agl['level'] = alti_agl_arr
    
#    if ds_agl['THT'].data is ds_agl['PABST'].data:
#        raise ValueError('same ID')
    
    # INTERPOLATION
    t0 = time.time()
    
#    var3d_list = var_list_AGL
#    var3d_list.remove('ZS')
    
    for ilevel, alti in enumerate(alti_agl_arr):
        print(f'Computation for {alti}m AGL')
        for var_agl in var3d_list_AGL:
            layer_interp = interp_iso_agl(alti, ds_asl, f'{var_agl}_ASL', 
                                          verbose=False)
            ds_agl[var_agl][ilevel,:,:] = layer_interp

    print('interpolation time:', time.time()-t0)
    
    return ds_agl


def merge_standard_and_budget_files(ds, ds_bu, join='outer'):
    """
    Merges standard and budget datasets from a same simulation.
    """
    # Put all data in mass point coordinates
    print("'flux_pt_to_mass_pt()' for ds")
    ds = flux_pt_to_mass_pt(ds, verbose=False)
    print("'flux_pt_to_mass_pt()' for ds_bu")
    ds_bu = flux_pt_to_mass_pt(ds_bu, verbose=False)
    
    # Merge
    print("Merging datasets")
    try:
        ds_out = ds.merge(ds_bu)
    except xr.MergeError:
        if np.abs(ds_bu.latitude - ds.latitude).max() < 1e-5:
                ds_bu['latitude'] = ds.latitude
        if np.abs(ds_bu.longitude - ds.longitude).max() < 1e-5:
                ds_bu['longitude'] = ds.longitude
        # Retry
        ds_out = ds.merge(ds_bu, join=join)
    
    return ds_out
    

#%% Matplotlib tools
def save_figure(plot_title, save_folder='./figures/', verbose=True):
    """
    Save the plot
    """
    #split the save_folder name at each /
    for i in range(2, len(save_folder.split('/'))):
        # recreate the path step by step and check its existence at each step.
        # If not existing, make a new folder
        path = '/'.join(save_folder.split('/')[0:i])
        if not os.path.isdir(path):
            print('Make directory: {0}'.format(path))
            os.mkdir(path)
    filename = (plot_title)
    filename = filename.replace('=', '').replace('(', '').replace(')', '')
    filename = filename.replace(' ', '_').replace(',', '').replace('.', '_')
    filename = filename.replace('/', 'over')
    plt.savefig(save_folder + '/' + str(filename), dpi=300)
    if verbose:
        print('figure {0} saved in {1}'.format(plot_title, save_folder))
        

#%% Diagnostics

def calc_apparent_temperature(tdb, rh, v=0, q=None, **kwargs):
    """Calculates the Apparent Temperature (AT). The AT is defined as the
    temperature at the reference humidity level producing the same amount of
    discomfort as that experienced under the current ambient temperature,
    humidity, and solar radiation [17]_. In other words, the AT is an
    adjustment to the dry bulb temperature based on the relative humidity
    value. Absolute humidity with a dew point of 14°C is chosen as a reference.

    [16]_. It includes the chilling effect of the wind at lower temperatures.

    Two formulas for AT are in use by the Australian Bureau of Meteorology: one includes
    solar radiation and the other one does not (http://www.bom.gov.au/info/thermal_stress/
    , 29 Sep 2021). Please specify q if you want to estimate AT with solar load.

    taken from pythermalcomfort, few modifs.
    
    Parameters
    ----------
    tdb : float
        dry bulb air temperature,[°C]
    rh : float
        relative humidity, [%]
    v : float
        wind speed 10m above ground level, [m/s]
        
    ERASED:
    q : float
        Net radiation absorbed per unit area of body surface [W/m2]


    Returns
    -------
    float
        apparent temperature, [°C]
        
    References
    ----------
    [16]	Blazejczyk, K., Epstein, Y., Jendritzky, G., Staiger, H., Tinz, B., 2012. Comparison of UTCI to selected thermal indices. Int. J. Biometeorol. 56, 515–535. https://doi.org/10.1007/s00484-011-0453-2
    [17]	Steadman RG, 1984, A universal scale of apparent temperature. J Appl Meteorol Climatol 23:1674–1687
    """
#    default_kwargs = {
#        "round": True,
#    }
#    kwargs = {**default_kwargs, **kwargs}

    # dividing it by 100 since the at eq. requires p_vap to be in hPa
    p_vap = psy_ta_rh(tdb, rh)["p_vap"] / 100

    # equation sources [16] and http://www.bom.gov.au/info/thermal_stress/#apparent
#    if q:
#        t_at = tdb + 0.348 * p_vap - 0.7 * v + 0.7 * q / (v + 10) - 4.25
#    else:
    t_at = tdb + 0.33 * p_vap - 0.7 * v - 4.00

    return t_at


def calc_partial_vap_pres(tdb, rh):
    """Calculates partial vapour pressure values of air 
    based on dry bulb air temperature and relative humidity.
    
    Comes from package 'pythermalcomfort' that returns more values
    like wet bulb temperature [°C], dew point temperature [°C],
    enthalpy [J/kg dry air]
    
    For more accurate results we recommend the use of the the Python package
    `psychrolib`_.
    .. _psychrolib: https://pypi.org/project/PsychroLib/

    Parameters
    ----------
    tdb: float
        air temperature, [°C]
    rh: float
        relative humidity, [%]

    Returns
    -------
    p_vap: float
        partial pressure of water vapor in moist air, [Pa]
    """
    p_saturation = calc_sat_vap_pres(tdb)
    p_vap = rh / 100 * p_saturation

    return p_vap
    
    
def calc_mixing_ratio(tdb, rh, p_atm=101325):
    """Calculates mixing ratio of air based on dry bulb air temperature and
    relative humidity.
    
    Comes from package 'pythermalcomfort' that returns more values
    like wet bulb temperature [°C], dew point temperature [°C],
    enthalpy [J/kg dry air]
    
    For more accurate results we recommend the use of the Python package
    `psychrolib`_.
    .. _psychrolib: https://pypi.org/project/PsychroLib/

    Parameters
    ----------
    tdb: float
        air temperature, [°C]
    rh: float
        relative humidity, [%]
    p_atm: float
        atmospheric pressure, [Pa]

    Returns
    -------
    mixing_ratio: float
        humidity ratio, [kg water/kg dry air]
    """

    p_vap = calc_partial_vap_pres(tdb, rh)
    mixing_ratio = 0.62198 * p_vap / (p_atm - p_vap)

    return mixing_ratio

    
def calc_sat_vap_pres(tdb):
    """Calculates saturated vapour pressure of water at different temperatures

    Parameters
    ----------
    tdb: float
        air temperature, [°C]

    Returns
    -------
    p_sat: float
        saturated vapour pressure, [Pa]
    """

    ta_k = tdb + 273.15

    # Low temperature case commented to allow for calculation on array of ta_k
    # (if not "np.array < 273.15" is ambiguous)
#    if ta_k < 273.15:
#        c1 = -5674.5359
#        c2 = 6.3925247
#        c3 = -0.9677843e-2
#        c4 = 0.62215701 * np.pow(10, -6)
#        c5 = 0.20747825 * np.pow(10, -8)
#        c6 = -0.9484024 * np.pow(10, -12)
#        c7 = 4.1635019
#        pascals = np.exp(
#            c1 / ta_k
#            + c2
#            + ta_k * (c3 + ta_k * (c4 + ta_k * (c5 + c6 * ta_k)))
#            + c7 * np.log(ta_k)
#        )
#    else:
    c8 = -5800.2206
    c9 = 1.3914993
    c10 = -0.048640239
    c11 = 0.41764768e-4
    c12 = -0.14452093e-7
    c13 = 6.5459673
    
    pascals = np.exp(
        c8 / ta_k
        + c9
        + ta_k * (c10 + ta_k * (c11 + ta_k * c12))
        + c13 * np.log(ta_k)
    )

    return np.round(pascals, 1)

    
def calc_rel_humidity(spec_humidity, t_dry_bulb, pressure):
    """Calculates the relative humidity.

    Parameters
    ----------
    t_dry_bulb: float
        dry bulb air temperature, [°C]
    spec_hum: float
        specific humidity, (kg/kg)

    Returns
    -------
    rel_humidity: float
        relative humidity
    """
    sat_mix_ratio = sat_mixing_ratio(pressure, t_dry_bulb)
    rel_humidity = spec_humidity / ((1-spec_humidity)*sat_mix_ratio)
    return rel_humidity
    
  
def calc_sat_mixing_ratio(pressure, t_dry_bulb):
    """
    Calculates the saturation_mixing_ratio
        Parameters
    ----------
    t_dry_bulb: float
        dry bulb air temperature, [°C]
    pressure: float
        atmospheric pressure, (Pa)

    Returns
    -------
    saturation mixing ratio: float
        (kg/kg)
    """
    sat_mixing_ratio = 0.622*p_sat(t_dry_bulb) / (pressure-p_sat(t_dry_bulb))
    return sat_mixing_ratio


def calc_u_star_sim(fmu_md, fmv_md):
    """
    Compute friction velocity u* from zonal and meridian wind stress FMU_ISBA
    and FMV_ISBA.
    
    Parameters:
        ds: xarray.Dataset containing FMU_ISBA and FMV_ISBA
    
    Returns:
        xarray.DataArray
    
    """

    tau = np.sqrt(fmu_md**2 + fmv_md**2)
    u_star = np.sqrt(tau)
    
    return u_star

def calc_bowen_sim(ds):
    """
    Compute Bowen ratio from H and LE.
    
    Parameters:
        ds: xarray.Dataset containing H_ISBA and LE_ISBA
    
    Returns:
        xarray.DataArray
    
    """
    bowen = ds['H_ISBA']/ds['LE_ISBA']
    
    return bowen

def calc_longwave_up_sim(tsrad, emis):
    """
    Compute far infrared long wave radiation up from 
    radiative surface temperature and emissivity.
    
    Parameters:
        tsrad: xarray.Dataset for radiative surface temp in [K]
        emis: xarray.Dataset for emissivity in [.] (no unit)
    
    Returns:
        xarray.DataArray
    
    """
    const = 5.670374e-8  # stephan boltzamann constante
    lwup = const * emis * tsrad**4
    
    return lwup
    

def calc_swi(sm, wilt_pt=None, field_capa=None):
    """
    inputs:
        sm: soil_moisture
        wilt_pt: wilting_point
        field_capa: field capacity
    
    returns:
        soil water index (swi)
    
    Proposed values: see global_variables.py

    
    """
    
    if None in [wilt_pt, field_capa]:
        plt.plot(sm)
        raise ValueError('Fill in wilting_point and field_capa values.')
    return (sm - wilt_pt)/(field_capa - wilt_pt)

def calc_ws_wd(ut, vt):
    """
    Compute wind speed and wind direction from ut and vt
    If ut and vt are xarray.dataarray, be careful with the coordinates,
    it may be needed to center it with center_uvw().

    """
    ws = np.sqrt(ut**2 + vt**2)
    wd_temp = np.arctan2(-ut, -vt)*180/np.pi
    
    try:
        # works with xarray datatypes and conserve the format
        wd = xr.where(wd_temp<0, wd_temp+360, wd_temp)
    except:
        # work with more formats
        wd = np.where(wd_temp<0, wd_temp+360, wd_temp)
    
    return ws, wd

def calc_u_v(ws, wd):
    """
    Compute wind components ut and vt from wind speed and direction.
    
    wd must be in degrees (0°=N, 90°=E, 180°=S, 270°=W).

    """
    vt = -np.cos((wd/360)*2*np.pi) * ws
    ut = -np.sin((wd/360)*2*np.pi) * ws

    return ut, vt


def calc_thetav(theta, spec_humidity):
    return theta + 0.61*spec_humidity*theta


def calc_mslp(ds, ilevel=1, z0=0):
    """
    Extrapolate the a level pressure for given level of model 
    and at wanted height. With ilevel=1 and z0=0, it corresponds to the 
    mean sea level pressure.
    N.B.: Comes from MNH routine 'write_lfifm1_for_diag.f90' in which 
    the MSLP diagnostic is done.
    
    parameters:
        ds with variables 'THT', 'RVT', 'PABST', 'ZS'
        Units: PABST [Pa], THT [K], RVT [kg/kg] and ZS [m].
    
    ilevel: int, screen level, the level at which we want to extrapolate the
        data to the sea level. If None, computation is done for all levels
        
    z0: float, altitude aslin m at which we want to extrapolate. Default is 0m
        for sea level.
        
    return:
        xarray.DataArray with values in Pa
    """
    lapse_rate = -6.5e-3    # standard atmo lapse rate in K/m
    R0 = 8.314462           # gas constant for dry air J⋅K−1⋅mol−1
    M = 0.028969            # Molar mass of dry air kg.mol-1
    Rd = R0/M       # gas constant for dry air J⋅K−1⋅kg-1
    Cpd = 1005              # heat capacity of dry air at constant pressure J⋅K−1⋅kg-1
    P0 = 100000     # standard reference surface pressure, usually taken as 1000 hPa
#    g = 9.80665     # m.s-2
    g = 9.80665 * (6371000**2)/((6371000+ds.level+ds['ZS'])**2)
    
    # from MNH routine "write_lfifm1_for_diag.f90"
#    # Exner function at the first mass point
#    exner(:,:) = (PABST(:,:,ilevel) / P0)**(Rd/Cpd)
#    #!  virtual temperature at the first mass point
#    #!  virtual temperature at P0 for the first mass point
#    thetav_p0(:,:) = exner(:,:) * thetav(:,:,ilevel)
#    #!  virtual temperature at ground level (height = 0m agl, not 1st level)
#    alti_agl = XZZ(:,:,ilevel) - ZS(:,:)     # XZZ is the altitude asl of each mass point
#    thetav_gl(:,:) = thetav_p0(:,:) - lapse_rate*alti_agl
#    #!  virtual temperature at sea level
#    thetav_sl(:,:) = thetav_gl(:,:) - lapse_rate*ZS(:,:)
#    #!  average underground virtual temperature
#    thetav_underground(:,:) = 0.5*(thetav_gl(:,:) + thetav_sl(:,:))
#    #!  surface pressure
#    pres_screenlevel(:,:) = PABST(:,:,ilevel)
#    #!  sea level pressure (hPa)
##    mslp(:,:) = 1e-2*pres_screenlevel(:,:)*np.exp(g*ZS(:,:)/(Rd*thetav_underground(:,:)))  # in hPa
    
    #perso version:
    # selection of level if ilevel different from None
    ds = ds.squeeze()
    if ilevel != None:
        ds = ds.isel(level=ilevel)
        
    thetav = calc_thetav(ds['THT'], ds['RVT'])
#    thetav = ds['THT']
    
    pres_screenlevel = ds['PABST']
    exner = (pres_screenlevel / P0)**(Rd/Cpd)
    thetav_p0 = exner * thetav
    
    # version in MNH:
    # virtual temperature at ground level (height = 0m agl, not 1st level)
    tempv_gl = thetav_p0
    # virtual temperature at sea level
    tempv_sl = tempv_gl - lapse_rate * ds['ZS']
    # average underground virtual temperature
    tempv_underground = 0.5*(tempv_gl + tempv_sl)
    T0 = tempv_underground
    
    # version perso (from my understanding, but do not seems to be reliable in mountains...)
#    T0 = thetav_p0
    
    height_asl_screenlevel = ds.level + ds['ZS'] - z0
    
    mslp = pres_screenlevel*np.exp(g*height_asl_screenlevel/(Rd*T0))
    
    return mslp


def height_to_pressure_std(height, p0=101325):
    r"""Convert height data to pressures using the U.S. standard atmosphere [NOAA1976]_.

    The implementation inverts the formula outlined in [Hobbs1977]_ pg.60-61.

    Parameters
    ----------
    height : Atmospheric height [m]
    p0 : reference pressure at sea level [Pa]

    Returns
    -------
    Corresponding pressure value(s) [Pa]

    Notes
    -----
    from metpy, but units check removed
    .. math:: p = p_0 e^{\frac{g}{R \Gamma} \text{ln}(1-\frac{Z \Gamma}{T_0})}

    """
    t0 = 288  #kelvin
    gamma = 0.0065  #'K/m'
    g = 9.81  #m/s-2
    R = 8.314462618  #'J / mol / K' - molar gas constant
    Md = 28.96546e-3  # 'kg / mol' - dry_air_molecular_weight
    Rd = R / Md  # dry_air_gas_constant
    
    return p0 * (1 - (gamma / t0) * height) ** (g / (Rd * gamma))


def exner_function(pressure, reference_pressure = 100000):
    r"""Calculate the Exner function.

    .. math:: \Pi = \left( \frac{p}{p_0} \right)^\kappa

    This can be used to calculate potential temperature from temperature (and visa-versa),
    since:

    .. math:: \Pi = \frac{T}{\theta}

    Parameters
    ----------
    pressure:
        Total atmospheric pressure [Pa]

    reference_pressure:
        The reference pressure against which to calculate the Exner function, 
        defaults to 100,000 Pa

    Returns
    -------
        Value of the Exner function at the given pressure

    See Also
    --------
    potential_temperature
    temperature_from_potential_temperature

    """
#    reference_pressure = 100000  # Pa
    R = 8.314462618  #'J / mol / K' - molar gas constant
    Md = 28.96546e-3  # 'kg / mol' - dry_air_molecular_weight
    Rd = R / Md  # dry_air_gas_constant
    dry_air_spec_heat_ratio = 1.4  #'dimensionless'
    Cp_d = dry_air_spec_heat_ratio * Rd / (dry_air_spec_heat_ratio - 1)
    kappa = (Rd / Cp_d)  #'dimensionless'
    
    return (reference_pressure / pressure)**kappa


def temperature_from_potential_temperature(pressure, potential_temperature,
                                           reference_pressure=100000):
    r"""Calculate the temperature from a given potential temperature.

    Uses the inverse of the Poisson equation to calculate the temperature from a
    given potential temperature at a specific pressure level.

    Parameters
    ----------
    pressure:
        Total atmospheric pressure [Pa]

    potential_temperature:
        Potential temperature [K]

    Returns
    -------
        Temperature corresponding to the potential temperature and pressure

    """
    return potential_temperature / exner_function(pressure, reference_pressure)


def potential_temperature_from_temperature(pressure, temperature,
                                           reference_pressure=100000):
    r"""Calculate the temperature from a given potential temperature.

    Uses the inverse of the Poisson equation to calculate the temperature from a
    given potential temperature at a specific pressure level.

    Parameters
    ----------
    pressure:
        Total atmospheric pressure [Pa]

    temperature:
        Potential temperature [K]

    Returns
    -------
        Temperature corresponding to the potential temperature and pressure
    """
    return temperature * exner_function(pressure, reference_pressure)

def calc_fao56_et_0(rn_MJ, t_2m, ws_2m, rh_2m, p_atm, gnd_flx=0, gamma=66):
    """
    Compute wind speed and wind direction from ut and vt.
    
    rn = Net irradiance (MJ.m-2.day-1). /!\ cannot be used on 1 hour, equation is not linear
    t_2m = mean daily air temperature at 2m (°C), defined as (Tmax+Tmin)/2
    ws_2m = Wind speed at 2m height (m−1)
    rh_2m = relative humidity (%)
    p_atm = atmospheric pressure (Pa)
    
    gnd_flx = Ground heat flux (MJ.m-2.day-1), usually taken as zero on a day
    gamma = Psychrometric constant (γ ≈ 66 Pa K−1)
    
    Returns:
        ET_0: float
            Potential evapotranspiration in mm.day-1
    """
    psychro = psy_ta_rh(t_2m, rh_2m, p_atm)
    
    vap_pres_deficit = (psychro['p_vap_sat'] - psychro['p_vap'])/1000  #in kPa
    delta = p_sat(t_2m+0.5) - p_sat(t_2m-0.5)
    
    # Convert W.m-2 to MJ.m-2.day-1
#    rn_MJ = rn * 3600*24 / 1e6
    gnd_flx_MJ = gnd_flx * 3600*24 / 1e6
    
    # potential ET [mm.day-1] (exactly as in FAO 56)
    ET_0 = (0.408*delta*(rn_MJ - gnd_flx_MJ) + \
            (900/(t_2m + 273.15))*gamma*ws_2m*vap_pres_deficit) / \
           (delta + gamma*(1 + 0.34*ws_2m))
    
    # potential ET [mm.s-1 = kg/m2/s]
#    ET_0 = ET_0 / (24*3600)
    
    # equivalent mean Latent Heat Flux [W/m2] (2450 J/g is latent heat of vaporization at 20°C according to FAO guide)
    LE_0 = 1000 * 2450 * ET_0 /(3600*24)
    
    return {'ET_0': ET_0, 'LE_0': LE_0}


def calc_soil_prop(PCLAY=0.35, PSAND=0.15, 
                   CPEDO_FUNCTION='CO84', CSCOND='PL98', CISBA='DIF',
                   vol_water_content=np.nan):
    """
    Function from SURFEX routine, giving soil properties according to
    fraction of sand and clay, and depending on the pedotransfer function.
    in SFX: see soil.F90
    author: Marti Belen, Lunel Tanguy
    
    parameters:
        PCLAY: float between 0 and 1, clay fraction
        PSAND: float between 0 and 1, sand fraction
        HPEDOTF: pedotransfer function name
        PCLAY: fraction of clay
        PSAND: fraction of sand
    
    returns: dict with following keys:
        - general property:
        W_sat: water at saturation = porosity [m3/m3]
        - hydraulic properties:
        W_wilt: Volumetric water content of Wilting point [m3/m3]
        W_fc: Volumetric water content of Field Capacity [m3/m3]
        b: soil water b-parameter (from CH78 and CO84 papers)
        psi_sat: Matric potential at saturation
        K_sat: hydraulic conductivity at saturation
        - thermal properties:
        CG_wilt: volumetric soil heat capacity at wilting point [J/m3/K]
        CG_fc: volumetric soil heat capacity [J/m3/K]
        thermal_cond_sat: thermal conductivity at saturation [W/m/K]
        thermal_cond_fc: thermal conductivity at field capacity [W/m/K]
        thermal_cond_wilt: thermal conductivity at wilting point [W/m/K]
            if vol_water_content != NaN:
        CG: volumetric soil heat capacity [J/m3/K]
        thermal_cond: thermal conductivity [W/m/K]
        
    """

    res_dict = {}
    
    if CPEDO_FUNCTION == 'CH78':
        # W_sat:
        res_dict['W_sat'] = 0.001 * (-108.*PSAND+494.305)
        # W_wilt: Volumetric water content of Wilting point
        res_dict['W_wilt'] = 37.1342E-3*(PCLAY*100.)**0.5
        # W_fc: Volumetric water content of Field Capacity
        if CISBA == 'DIF':
            res_dict['W_fc'] = 0.2298915119 - 0.4062575773*PSAND + 0.0874218705*PCLAY \
                     + 0.2942558675*PSAND**(1./3.)+0.0413771051*PCLAY**(1./3.)
        else:
            res_dict['W_fc'] = 89.0467E-3*(PCLAY*100.)**0.3496
        # b:
        res_dict['b'] = 13.7*PCLAY + 3.501       
        # psisat, or matpotsat = water matrix potential at saturation
        res_dict['psi_sat'] = -0.01*(10.**(1.85 - 0.88*PSAND))
        # ksat, hydcondsat = hydraulic conductivity at saturation ?
        res_dict['K_sat'] = 1.0e-6*(10.0**(0.161874E+01                   \
                            - 0.581989E+01*PCLAY    - 0.907123E-01*PSAND    \
                            + 0.529268E+01*PCLAY**2 + 0.120332E+01*PSAND**2))
    elif CPEDO_FUNCTION == 'CO84':
        # W_sat
        res_dict['W_sat'] = 0.505-0.142*PSAND-0.037*PCLAY 
        # W_wilt: Volumetric water content of Wilting point
        res_dict['W_wilt'] = 0.15333-0.147*PSAND+0.33*PCLAY-0.102*(PCLAY**2)
        # W_fc: Volumetric water content of Field Capacity
        if CISBA == 'DIF':
            res_dict['W_fc'] = 0.2016592588 - 0.5785747196*PSAND + 0.1113006987*PCLAY    \
                         + 0.4305771483*PSAND**(1./3.)-0.0080618093*PCLAY**(1./3.)
        else:
            res_dict['W_fc'] = 0.1537-0.1233*PSAND+0.2685*PCLAY**(1./3.)
        # b:
        res_dict['b'] = 3.10+15.7*PCLAY-0.3*PSAND
        # psisat, or matpotsat = matrix potential at saturation
        res_dict['psi_sat'] = -0.01*(10.0**(1.54-0.95*PSAND+0.63*(1.-PSAND-PCLAY)))
        # ksat, hydcondsat = hydraulic conductivity at saturation ?
        res_dict['K_sat'] = 0.0254*(10.0**(-0.6+1.26*PSAND-0.64*PCLAY))/3600
    
    #
    if CSCOND == 'NP89':
        raise ValueError("""CSCOND='NP89' not avalaible yet, 
                         check soil.F90 surfex routine""")
        # Ground heat capacity at saturation
        #res_dict['Cg_sat'] = (-1.5571*PSAND - 1.441*PCLAY + 4.70217 )*1e-6
        # ...
    elif CSCOND  == 'PL98':
        mass_soil_heat_capa = 733  #[J/kg/K] soil specific heat, SPHSOIL in SFX
        soil_compact_weight = 2700  #[kg/m3]  DRYWGHT in SFX
        vol_soil_heat_capa_compact = mass_soil_heat_capa * soil_compact_weight  #=1979100 J/m3/K
        vol_water_heat_capa = 4180000  # [J/m3/K]
        # calculate Volumetric soil heat capacity
        res_dict['CG_wilt'] = (
                (1 - res_dict['W_sat'])*vol_soil_heat_capa_compact \
                + res_dict['W_wilt'] * vol_water_heat_capa)
        res_dict['CG_fc'] = (
                (1 - res_dict['W_sat'])*vol_soil_heat_capa_compact \
                + res_dict['W_fc'] * vol_water_heat_capa)
        res_dict['CG'] = (
                (1 - res_dict['W_sat'])*vol_soil_heat_capa_compact \
                + vol_water_content * vol_water_heat_capa)

        # calculate soil thermal conductivity
        PQUARTZ = 0.038 + 0.95*PSAND
        CONDQRTZ = 7.7    # W/(m K) Quartz thermal conductivity
        CONDWTR = 0.57   # W/(m K)  Water thermal conductivity
        if PQUARTZ > 0.2:
            CONDOTH = 2.0    # W/(m K)  Other thermal conductivity
        else:
            CONDOTH = 3.0
        # soil solids conductivity
        CONDSLD = (CONDQRTZ**PQUARTZ) * (CONDOTH**(1.0-PQUARTZ))
        # soil dry conductivity
        GAMMAD = (1.0-res_dict['W_sat'])*soil_compact_weight  # soil dry density
        CONDDRY = (0.135*GAMMAD + 64.7)/(soil_compact_weight - 0.947*GAMMAD) 
        # Saturated thermal conductivity:
        res_dict['thermal_cond_sat'] = (CONDSLD**(1.0-res_dict['W_sat']))* (CONDWTR) 
        # effective thermal conductivity
        SATDEG = np.max([0.1, vol_water_content/res_dict['W_sat']])  # saturation degree
        KERSTEN  = np.log10(SATDEG) + 1.0
        res_dict['thermal_cond'] = KERSTEN*(res_dict['thermal_cond_sat']-CONDDRY) + CONDDRY
        # thermal conductivity at Field capacity
        SATDEG = np.max([0.1, res_dict['W_wilt']/res_dict['W_sat']])
        KERSTEN  = np.log10(SATDEG) + 1.0
        res_dict['thermal_cond_wilt'] = KERSTEN*(res_dict['thermal_cond_sat']-CONDDRY) + CONDDRY
        # thermal conductivity at Field capacity
        SATDEG = np.max([0.1, res_dict['W_fc']/res_dict['W_sat']])
        KERSTEN  = np.log10(SATDEG) + 1.0
        res_dict['thermal_cond_fc'] = KERSTEN*(res_dict['thermal_cond_sat']-CONDDRY) + CONDDRY
        
    return res_dict
    

def calc_mean_wd(data):
    """
    Computes the circular mean of wind direction data found between 0 and 360
    
    N.B.: Computing the mean of circular data is not trivial 
    (cf https://en.wikipedia.org/wiki/Directional_statistics)
    
    cf function 'circmean', 'circstd' from scipy.stats    
    """
    return stats.circmean(data, high=360, low=0)


def log_wind_profile(ws, ws_height, new_height, z_0, d=0):
    """
    Calculates the wind speed at a new height using a logarithmic wind profile.

    The logarithmic height equation is used. There is the possibility of
    including the height of the surrounding obstacles in the calculation.

    Parameters
    ----------
    ws : pandas.Series or array
        Wind speed time series.
    ws_height : float
        Height for which the parameter `ws` applies.
    new_height : float
        Hub height of wind turbine.
    z_0 : pandas.Series or array or float
        Roughness length in m
    d: float
        Displacement height. Default: 0.

    Returns
    -------
    pandas.Series or array
        Wind speed at new height.

    Notes
    -----
    `ws_height`, `z_0`, `new_height` and `d` must be of the same unit.
    
    see: https://en.wikipedia.org/wiki/Log_wind_profile
    
    This function comes from windpowerlib library.

    """
    if d > ws_height:
        raise ValueError(
            'Displacement height of {0}m is superior to ws_height.'.format(d))
        
    ws_new = ws * np.log((new_height - d)/z_0) / np.log((ws_height - d)/z_0)
    
    return ws_new


def cls_wind(ws, z_uv, Cd, CdN, Ri, new_height):
    """
    /!\To revise, not working
    
    ChatGPT traduction of cls_wind.f90
    
    ws:     wind speed
    z_uv:   atmospheric level height (wind)
    Cd:     drag coefficient for momentum
    CdN:    neutral drag coefficient
    Ri:     Richardson number
    new_height: height of diagnostic (m)
    !
    !*      0.2    declarations of local variables
    !
    ZBN,ZBD,ZRU
    REAL, DIMENSION(SIZE(z_uv)) :: ZLOGU,ZCORU,ZIV
    REAL(KIND=JPRB) :: ZHOOK_HANDLE

    """
    XKARMAN = 0.4 # constant value
    
    # preparatory calculations
    ZBN = XKARMAN / np.sqrt(CdN)
    ZBD = XKARMAN / np.sqrt(Cd)
    
    if new_height <= z_uv:
        ZRU = np.min(new_height / z_uv, 1.0)
    else:
        ZRU = np.min(z_uv / new_height, 1.0)
    
    ZLOGU = np.log(1.0 + ZRU * (np.exp(ZBN) - 1.0))
    
    # stability effect
    if (Ri >= 0):
        ZCORU = ZRU * (ZBN - ZBD)
    else:
        ZCORU = np.log(1.0 + ZRU * (np.exp(np.maximum(0.0, ZBN - ZBD)) - 1.0))
    
    # interpolation of dynamical variables
    IV = np.max(0.0, np.min(1.0, (ZLOGU - ZCORU) / ZBD))
    
    if new_height <= z_uv:
        ws10M = ws * IV
    else:
        ws10M = ws / np.maximum(1.0, IV)
    
    return ws10M


def cls_t(TA, HT, Z0, CD, CH, RI, TS, H):
    """
    conversion of cls_t.f90 of Surfex into python
    
    Interpolation of T value at height N following laws of the CLS 
    (couche limite de surface).
    
    Parameters:
        TA    ! atmospheric temperature
        HT    ! atmospheric level height (temp)
        CD    ! drag coefficient for momentum
        CH    ! drag coefficient for heat
        RI    ! Richardson number
        TS    ! surface temperature
        Z0    ! roughness length
        H     ! height of diagnostic
    returns:
        TNM   ! temperature at n meters
        
    intermediate variables:
        Z0H   ! roughness length for heat (for now Z0H = Z0/10)
    """
    
    Z0H = Z0/10
    
    # 1. Preparatory calculations
    BNH = np.log(HT/Z0H)
    BH = 0.41 * np.sqrt(CD) / CH
    RS = min(H / HT, 1.0)
    LOGS = np.log(1.0 + RS * (np.exp(BNH) - 1.0))
    print('LOGS =', LOGS)
        
    # 2. Stability effects
    Ri_pos = RI[:] >= 0  # in MNH, Ri = - thermal_prod/dynamic_prod
    Ri_neg = ~Ri_pos
    CORS = RI*0  # init of CORS dataset
    # when consumption of TKE by temperature
    CORS.values[Ri_pos] = \
        RS * (BNH.values[Ri_pos] - BH.values[Ri_pos])
    # when production of TKE (temp or dynamic)
    CORS.values[Ri_neg] = np.log(
        1.0 \
        + RS * (np.exp(np.maximum(0.0, BNH.values[Ri_neg]- BH.values[Ri_neg]))\
                - 1.0))
    print('CORS =', CORS)
    
    # 3. Interpolation of thermodynamical variables
    IV = np.maximum(0.0, np.minimum(1.0, (LOGS - CORS) / BH))
    
    TNM = TS + IV * (TA - TS)
    
    return TNM


def cls_tq(TA, QA, HT, Z0, CD, CH, RI, TS, QS, H, PNM):
    """
    conversion of cls_tq.f90 of Surfex into python, 
    for the part concerning Q.
    
    Interpolation of Q value at height N following laws of the CLS 
    (couche limite de surface).
    
    Parameters:
        TA    ! atmospheric temperature
        QA    ! atmospheric humidity (kg/kg)
        HT    ! atmospheric level height (temp)
        CD    ! drag coefficient for momentum
        CH    ! drag coefficient for heat
        RI    ! Richardson number
        Z0    ! roughness length
        H     ! height of diagnostic
        PNM   ! pressure at N meters AGL
        QS    ! surface specific humidity
        TNM   ! temperature at n meters
    
    returns:
        QNM   ! specific humidity at n meters
        
    intermediate variables:
        Z0H   ! roughness length for heat (for now Z0H = Z0/10)
    """
    
    
    
    # 1. Preparatory calculations
    print('1st step: preparatory calculations')
    Z0H = Z0/10
    BNH = np.log(HT/Z0H)
    BH = 0.41 * np.sqrt(CD) / CH
    RS = min(H / HT, 1.0)
    LOGS = np.log(1.0 + RS * (np.exp(BNH) - 1.0))
    
    
    # 2. Stability effects
    print('2nd step: Stability effects calculations')
    Ri_pos = RI[:] >= 0  # in MNH, Ri = - thermal_prod/dynamic_prod
    Ri_neg = ~Ri_pos
    CORS = RI*0  # init of CORS dataset
    # when consumption of TKE by temperature
    CORS.values[Ri_pos] = \
        RS * (BNH.values[Ri_pos] - BH.values[Ri_pos])
    # when production of TKE (temp or dynamic)
    CORS.values[Ri_neg] = np.log(
        1.0 \
        + RS * (np.exp(np.maximum(0.0, BNH.values[Ri_neg]- BH.values[Ri_neg]))\
                - 1.0))
    
    # 3. Interpolation of thermodynamical variables
    print('3rd step: interpolation of thermodynamical vars')
    IV = np.maximum(0.0, np.minimum(1.0, (LOGS - CORS) / BH))
    
    # 4. Interpolation of temperature
    TNM = TS + IV*(TA-TS)
    
    # 5. Interpolation of relative humidity
    QNM = QS + IV*(QA-QS)
    
    # correction for saturation
    print('4rd step: correction for humidity saturation')
    QsatNM = sat_mixing_ratio(PNM, (TNM - 273.15))
    QNM   = np.minimum(QsatNM, QNM) #must be below saturation
    
    return TNM, QNM


def windvec_verti_proj(u, v, level, angle):
    """Compute the projected horizontal wind vector on an axis 
    with a given angle w.r.t. the x/ni axes (West-East)
    
    Copied from MNHPy.
    
    Parameters
    ----------
    u : array 3D
        U-wind component

    v : array 3D
        V-wind component

    level : array 1D
        level dimension array

    angle : float
        angle (radian) of the new axe w.r.t the x/ni axes (West-East). angle = 0 for (z,x) sections, angle=pi/2 for (z,y) sections

    Returns
    -------

    projected_wind : array 3D
        a 3D wind component projected on the axe
    """
    projected_wind = copy.deepcopy(u)
    for k in range(len(level)):
#        projected_wind[k, :, :] = u[k, :, :] * math.cos(angle) + v[k, :, :] * math.sin(angle)
        projected_wind[k, :, :] = u[k, :, :] * np.cos(angle) + v[k, :, :] * np.sin(angle)

    return projected_wind


def diag_lowleveljet_height(ds, top_layer_agl=1000, wind_var='WS',
                             new_height_var='H_LOWJET', 
                             upper_bound=0.95):
    """
    Evaluate the upper height of the low level jet in the low layer of ds. 
    It is considered to be the height above at which the maximum speed of the jet
    is reduced by 5%.
    
    ds must contain ZS variable (ds['ZS'])
    
    Parameters:
        - new_height_var: nariable name that will be added in the output data
    

    """
    ds[new_height_var] = ds['ZS']*0
    
#    length = len(ds.ni)
    
#    dict_coords = {}
#    for i, co in enumerate(coords):
#        dict_coords[co] = data['ZS'].shape[i]
    coords = ds['ZS'].dims
    coords_value = {}
    for index, x in np.ndenumerate(ds['ZS']):
        print(index[0], '/', ds['ZS'].shape[0])
        
        for i, coord in enumerate(coords):
            coords_value[coord] = index[i]
        
        column = ds.isel(coords_value)
        
    # OLD CODE
#    for i, ni in enumerate(ds.ni):
#        print(f'i = {i}/{length}')
#        for j, nj in enumerate(ds.nj):
#        column = ds.isel(nj=j, ni=i)
    
        column['dWSdz'] = xr.DataArray(coords={'level':ds.level}, data=np.gradient(column[wind_var]))
        column['d2WSd2z'] = xr.DataArray(coords={'level':ds.level}, data=np.gradient(column['dWSdz']))
        
        column = column.where(column.level<top_layer_agl, drop=True)
        
        jet_level_indices = []  # indices of jet speed max
        jet_top_indices = []    # indices of jet speed max - 5%
        research_jet_top = False
        for ilevel, level_agl in enumerate(column.level):
            # first look at the jet height
            if not research_jet_top:
                if ilevel<3:
                    pass
                else:
                    sign_temp = column['dWSdz'][ilevel] * column['dWSdz'][ilevel-1]
                    if sign_temp < 0:  # sign change
                        if column['d2WSd2z'][ilevel] < 0:  # is a maximum
                            jet_level_indices.append(ilevel)
                            jet_speed = column.isel(level=ilevel)[wind_var]
                            jet_speed_upperlim = jet_speed*upper_bound
                            research_jet_top = True
            # second step where we look at the height at which 5% threshold is found
            elif research_jet_top:
                if column.isel(level=ilevel)[wind_var] < jet_speed_upperlim:
                    jet_top_indices.append(ilevel)
                    research_jet_top = False
        
        try:
            H_LOWJET = float(column.isel(level=jet_top_indices[0]).level)
        except IndexError:
            H_LOWJET = np.nan
            
        ds[new_height_var][coords_value] = H_LOWJET
        
    return ds

    
#%% Deprecated functions
def indices_of_lat_lon(ds, lat, lon, verbose=True):
    raise NameError("Deprecated function, use get_indices_of_lat_lon() instead")

def line_coords(data, 
                 start_pt = (41.6925905, 0.9285671), # cendrosa
                 end_pt = (41.590111, 1.029363), #els plans
                 nb_indices_exterior=10,
                 verbose=True):
    raise NameError("Deprecated function, use get_line_coords() instead")
    
def psy_ta_rh(tdb, rh, p_atm=101325):
    raise NameError(
        """ Deprecated function, use calc_partial_vap_pres() or 
            calc_mixing_ratio() instead""")

def apparent_temperature(tdb, rh, v=0, q=None, **kwargs):
    raise NameError("Deprecated function, use calc_apparent_temperature() instead")

def sat_mixing_ratio(pressure, t_dry_bulb):
    raise NameError("Deprecated function, use calc_sat_mixing_ratio() instead")
    
def p_sat(tdb):
    raise NameError("Deprecated function, use calc_sat_vap_pres() instead")
    
def rel_humidity(spec_humidity, t_dry_bulb, pressure):
    raise NameError("Deprecated function, use calc_rel_humidity() instead")
    
def get_simu_filename(*args, **kwargs):
    raise NameError("""
                     Deprecated function get_simu_filename().
                     Use get_simu_filepath() instead.
                     """)

def get_simu_filename_000(model, date='20210722-1200'):
    raise NameError(
            """Function 'get_simu_filename_000()' is deprecated 
            and should be replaced by get_simu_filepath.
            """)

#%% Main
if __name__ == '__main__':
    pass
    
    


