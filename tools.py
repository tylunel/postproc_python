#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021

Code to get x and y coordinate in a netcdf file according to lat and lon.
Useful for example for XY station coordinate to put in MNH - &NAM_STATIONn

"""
import os
from scipy.stats import circmean
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from metpy.units import units
import re
import global_variables as gv
#from difflib import SequenceMatcher


def indices_of_lat_lon(ds, lat, lon, verbose=True):
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
    # Gross evaluation of lat, lon (because latitude lines are curved)
    lat_dat = ds.latitude.data
    lon_dat = ds.longitude.data
    
    distance2lat = np.abs(lat_dat - lat)
    index_lat = np.argwhere(distance2lat <= distance2lat.min())[0,0]
    
    distance2lon = np.abs(lon_dat - lon)
    index_lon = np.argwhere(distance2lon <= distance2lon.min())[0,1]
    
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


def open_ukmo_mast(datafolder, filename, create_netcdf=True, remove_modi=True):
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
    

def open_ukmo_rs(datafolder, filename):
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
    for key in units_row.index:
        obs_ukmo_xr[key] = obs_ukmo_xr[key]*units(units_row[key])
#        obs_ukmo_xr[key]['units'] = units_row[key]

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
    df3 = pd.merge(df1, df2, on='TIMESTAMP')
    # convert TIMESTAMP strings into pd.Timestamp
    df3['time'] = [pd.Timestamp(dati) for dati in df3['TIMESTAMP']]
    # set time as index
    df4 = df3.set_index('time')
    
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
    ds['LE'] = ds['LE'].where(ds['LE_QC'] <= QC_threshold)
    ds['H'] = ds['H'].where(ds['H_QC'] <= QC_threshold)
    ds['FC_mass'] = ds['FC_mass'].where(ds['FC_QC'] <= QC_threshold)
    
    if create_netcdf:
        # BUG: if file already existing, it raises 'ascii' codec rerror
        ds.to_netcdf(datafolder + 'LIAISE_IRTA-CORN_UIB_SEB-10MIN_L2.nc', 
                     engine='netcdf4')
    
    return ds
    

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
    """
    dist = [abs(elt - pivot) for elt in items]
    return items[np.argmin(dist)], np.argmin(dist)

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
        raise ValueError("No obs at wanted datetime (or not close enough)")
    
    filename = fnames[i_near]
    
    return filename

def get_simu_filename_old(model, date='20210722-1200', domain=2,
                      file_suffix='001dg'):
    """
    Returns the whole string for filename and path. 
    Chooses the right SEG (segment) number corresponding to date
    
    Parameters:
        model: str, 
            'irr' or 'std'
        wanted_date: str, with format accepted by pd.Timestamp,
            ex: '20210722-1200'
        domain: int,
            domain number in case of grid nesting.
            ex: domain 1 is the biggest, domain 2 smaller, etc
    
    Returns:
        filename: str, full path and filename
    """
    pd_date = pd.Timestamp(date)
    seg_nb = (pd_date - pd.Timestamp('20210721-0000')).components.hours +\
        (pd_date - pd.Timestamp('20210721-0000')).components.days*24        
    seg_nb_str = str(seg_nb)
    # add 0 before if SEG01 - SEG09
    if len(seg_nb_str) == 1:
        seg_nb_str = '0'+ seg_nb_str
        
    print("SEG number must be SEG{0}".format(seg_nb_str))

    father_folder = '/cnrm/surface/lunelt/NO_SAVE/nc_out/'
    simu_filelist = {
        'std': '1.11_ECOII_2021_ecmwf_22-27/LIAIS.{0}.SEG{1}.{2}.nc'.format(
                domain, seg_nb_str, file_suffix),
        'irr': '2.13_irr_2021_22-27/LIAIS.{0}.SEG{1}.{2}.nc'.format(
                domain, seg_nb_str, file_suffix),
#        'irr_d2': '2.13_irr_2021_22-27/LIAIS.{0}.SEG{1}.{2}.nc'.format(
#                domain, seg_nb_str, file_suffix),
#        'irr_d1': '2.14_irr_15-30/LIAIS.{0}.SEG{1}.{2}.nc'.format(
#                domain, seg_nb_str, file_suffix)
        }
    
    filename = father_folder + simu_filelist[model]
    return filename

def get_simu_filename(model, date='20210722-1200', file_suffix='dg'):
    """
    
    Returns the whole string for filename and path. 
    Chooses the right SEG (segment) number and suffix corresponding to date
    
    Parameters:
        model: str, 
            'irr_d2' or 'std_d1'
        wanted_date: str, with format accepted by pd.Timestamp,
            ex: '20210722-1200'
    
    Returns:
        filename: str, full path and filename
    """
    pd_date = pd.Timestamp(date)
    day_nb = str(pd_date.day)        
    hour_nb = str(pd_date.hour)
    # format suffix with 2 digits:
    if len(hour_nb) == 1:
        hour_nb_2f = '0'+ hour_nb
    elif len(hour_nb) == 2:
        hour_nb_2f = hour_nb
    # format suffix with 3 digits:
    hour_nb_3f = '0'+ hour_nb_2f


    father_folder = '/cnrm/surface/lunelt/NO_SAVE/nc_out/'
    simu_filelist = {
        'std_d2': 'LIAIS.1.S{0}{1}.001{2}.nc'.format(
                day_nb, hour_nb_2f, file_suffix),
        'irr_d2': 'LIAIS.1.S{0}{1}.001{2}.nc'.format(
                day_nb, hour_nb_2f, file_suffix),
        'std_d2_old': 'LIAIS.2.SEG{0}.{1}{2}.nc'.format(
                day_nb, hour_nb_3f, file_suffix),
        'irr_d2_old': 'LIAIS.2.SEG{0}.{1}{2}.nc'.format(
                day_nb, hour_nb_3f, file_suffix),
        'irr_d1': 'LIAIS.1.SEG{0}.{1}{2}.nc'.format(
                day_nb, hour_nb_3f, file_suffix),
        'std_d1': 'LIAIS.1.SEG{0}.{1}{2}.nc'.format(
                day_nb, hour_nb_3f, file_suffix)
        }
    
    filename = father_folder + gv.simu_folders[model] + simu_filelist[model]
    
    #check if nomenclature of filename is ok
    check_filename_datetime(filename)
    
    return filename

def get_simu_filename_000(model, date='20210722-1200'):
    """
    Returns the whole string for filename and path. 
    Chooses the right SEG (segment) number and suffix corresponding to date
    
    Parameters:
        model: str, 
            'irr_d2' or 'std_d1'
        wanted_date: str, with format accepted by pd.Timestamp,
            ex: '20210722-1200'
    
    Returns:
        filename: str, full path and filename
    """
    pd_date = pd.Timestamp(date)
    day_nb = str(pd_date.day)        
    hour_nb = str(pd_date.hour)
    # format suffix with 2 digits:
    if len(hour_nb) == 1:
        hour_nb_2f = '0'+ hour_nb
    elif len(hour_nb) == 2:
        hour_nb_2f = hour_nb

    simu_filelist = {
        'std_d2': 'LIAIS.1.S{0}{1}.000.nc'.format(
                day_nb, hour_nb_2f),
        'irr_d2': 'LIAIS.1.S{0}{1}.000.nc'.format(
                day_nb, hour_nb_2f),
        'std_d2_old': 'LIAIS.2.SEG{0}.000.nc'.format(
                day_nb),
        'irr_d2_old': 'LIAIS.2.SEG{0}.000.nc'.format(
                day_nb),
        'irr_d1': 'LIAIS.1.SEG{0}.000.nc'.format(
                day_nb),
        'std_d1': 'LIAIS.1.SEG{0}.000.nc'.format(
                day_nb)
        }
    
    filename = gv.global_simu_folder + gv.simu_folders[model] + simu_filelist[model]
    
    return filename

def check_filename_datetime(filepath, fix=False):
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
    
    

def concat_obs_files(datafolder, in_filenames, out_filename, 
                     dat_to_nc=None):
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
    
    # CREATE netCDF file from .dat if not existing yet
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
    if not os.path.exists(datafolder + out_filename):
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
    """
    
    if not os.path.exists(datafolder + out_filename):
        print("creation of file: ", out_filename)
        out = os.system('''
            cd {0}
            ncecat -v {1},time {2} {3}
            '''.format(datafolder, varname_sim, 
                       in_filenames, out_filename))
        if out == 0:
            print('Creation of file {0} successful !'.format(out_filename))
        else:
            print('Creation of file {0} failed !'.format(out_filename))
    else:
        print('file {0} already exists.'.format(out_filename))
    #command 'cdo -select,name={1} {2} {3}' may work as well, but not always...





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

#def interpolation_level(data, height=[2,10]):
    #interp à 2m et 10m par défaut
#    data.interp(level=height)



    
    
 
def line_coords(data, 
                start_pt = (41.6925905, 0.9285671), # cendrosa
                end_pt = (41.590111, 1.029363), #els plans
                nb_indices_exterior=10):     
    """
    data: xarray dataset
    start_pt: tuple with (lat, lon) coordinates of start point
    start_pt: tuple with (lat, lon) coordinates of end point
    nb_indices_exterior: int, number of indices to take resp. before and after
        the start and end point.
    """
    # get ni, nj values (distance to borders of first domain in m)
    index_lat_start, index_lon_start = indices_of_lat_lon(data, *start_pt)
    ni_start = data.ni[index_lon_start].values     # ni corresponds to longitude
    nj_start = data.nj[index_lat_start].values     # nj corresponds to latitude
    print('ni, nj start:', ni_start, nj_start)
    index_lat_end, index_lon_end = indices_of_lat_lon(data, *end_pt)
    ni_end = data.ni[index_lon_end].values     # ni corresponds to longitude
    nj_end = data.nj[index_lat_end].values     # nj corresponds to latitude
    print('ni, nj end:', ni_end, nj_end)
    
    #get line formula:
    if ni_end != ni_start:
        slope = (nj_end - nj_start)/(ni_end - ni_start)
        y_intercept = nj_start - slope * ni_start
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
    
    return {'ni_range': ni_range, 'nj_range': nj_range, 'slope': slope,
            'index_distance': index_distance, 'ni_step': ni_step,
            'nj_step': nj_step, 'nij_step': nij_step,
            'ni_start': ni_start, 'ni_end': ni_end,
            'nj_start': nj_start, 'nj_end': nj_end}


def center_uvw(data):
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
    
    data['UT'] = data['UT'].interp(ni_u=data.ni.values, nj_u=data.nj.values).rename(
            {'ni_u': 'ni', 'nj_u': 'nj'})
    data['VT'] = data['VT'].interp(ni_v=data.ni.values, nj_v=data.nj.values).rename(
            {'ni_v': 'ni', 'nj_v': 'nj'})
    # remove useless coordinates
    data_new = data.drop(['latitude_u', 'longitude_u', 
                          'latitude_v', 'longitude_v',
                          'ni_u', 'nj_u', 'ni_v', 'nj_v'])
    
    # TRY loop in case no WT found in file
    try:
        data['WT'] = data['WT'].interp(level_w=data.level.values).rename(
                {'level_w': 'level'})
        # remove useless coordinates
        data_new = data.drop(['level_w'])
    except KeyError:
        pass
    
    # consider time no longer as a dimension but just as a single coordinate
    data_new = data_new.squeeze()
    
    return data_new

def interp_iso_asl(alti_asl, ds, varname):
    """
    Interpolate values at same altitude ASL
    
    Returns:
        np.array
    
    """
    
#    fn = get_simu_filename_d1('irr_d1', date='20210722-1200')
#    ds = xr.open_dataset(fn)
    ds = ds[[varname, 'ZS']].squeeze()
    
    # make with same asl value everywhere
    alti_grid = np.array([[alti_asl]*len(ds.ni)]*len(ds.nj))
    # compute the corresponding height AGL to keep iso alti ASL on each pt
    level_grid = alti_grid - ds['ZS'].data
    # initialize the result layer with same shape same than alti_grid
    res_layer = level_grid*0
    
    # get column of correspondance between level and height AGL
    level = ds[varname][:, :, :].level.data 
    
    # interpolation
    for j in range(len(ds.nj)):
        print('{0}/{1}'.format(j, len(ds.nj)))
        for i in range(len(ds.ni)):
            if level_grid[j,i] < 0:
                res_layer[j,i] = np.nan
            else:
    #            res_layer[i,j] = ds['PRES'].interp(
    #                ni=ds.ni[i], nj=ds.nj[j], level=level_grid[i,j])
                res_layer[j,i] = np.interp(level_grid[j,i], 
                                           level, ds['PRES'][:, j, i])
    
    return res_layer


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
    plt.savefig(save_folder + '/' + str(filename))
    if verbose:
        print('figure {0} saved in {1}'.format(plot_title, save_folder))


def subset(latmin=41.35, lonmin=0.40, latmax=41.9, lonmax=1.4):
    """
    SUBSETING DOMAIN - Work in progress
    
    (to zoom just use ylim or set_ylim)
    """
    #ind_lat, ind_lon = indices_of_lat_lon(var2d, 
    #                                      latmin,
    #                                      lonmin)
    #ni_min = var2d.ni[ind_lon]
    #nj_min = var2d.nj[ind_lat]
    #var2d = var2d.where(var2d.ni > ni_min, drop=True)
    #var2d = var2d.where(var2d.nj > nj_min, drop=True)
    #
    #ind_lat, ind_lon = indices_of_lat_lon(var2d, 
    #                                      latmax,
    #                                      lonmax)
    #ni_max = var2d.ni[ind_lon]
    #nj_max = var2d.nj[ind_lat]
    #var2d = var2d.where(var2d.ni < ni_max, drop=True)
    #var2d = var2d.where(var2d.nj < nj_max, drop=True)
    return None

## FROM pythermalcomfort, few modifs
    
def apparent_temperature(tdb, rh, v=0, q=None, **kwargs):
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


def psy_ta_rh(tdb, rh, p_atm=101325):
    """Calculates psychrometric values of air based on dry bulb air temperature and
    relative humidity.
    For more accurate results we recommend the use of the the Python package
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
    p_vap: float
        partial pressure of water vapor in moist air, [Pa]
    hr: float
        humidity ratio, [kg water/kg dry air]
    t_wb: float
        wet bulb temperature, [°C]
    t_dp: float
        dew point temperature, [°C]
    h: float
        enthalpy [J/kg dry air]
    """
    p_saturation = p_sat(tdb)
    p_vap = rh / 100 * p_saturation
    hr = 0.62198 * p_vap / (p_atm - p_vap)
    tdp = t_dp(tdb, rh)
    twb = t_wb(tdb, rh)
#    h = enthalpy(tdb, hr)

    return {
        "p_sat": p_saturation,
        "p_vap": p_vap,
        "hr": hr,
        "t_wb": twb,
        "t_dp": tdp,
#        "h": h,
    }
    
def p_sat(tdb):
    """Calculates vapour pressure of water at different temperatures

    Parameters
    ----------
    tdb: float
        air temperature, [°C]

    Returns
    -------
    p_sat: float
        operative temperature, [Pa]
    """

    ta_k = tdb + 273.15

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

def t_dp(tdb, rh):
    """Calculates the dew point temperature.

    Parameters
    ----------
    tdb: float
        dry bulb air temperature, [°C]
    rh: float
        relative humidity, [%]

    Returns
    -------
    t_dp: float
        dew point temperature, [°C]
    """

    c = 257.14
    b = 18.678
    d = 234.5

    gamma_m = np.log(rh / 100 * np.exp((b - tdb / d) * (tdb / (c + tdb))))

    return np.round(c * gamma_m / (b - gamma_m), 1)

def t_wb(tdb, rh):
    """Calculates the wet-bulb temperature using the Stull equation [6]_

    Parameters
    ----------
    tdb: float
        air temperature, [°C]
    rh: float
        relative humidity, [%]

    Returns
    -------
    tdb: float
        wet-bulb temperature, [°C]
    """
    twb = np.round(
        tdb * np.arctan(0.151977 * (rh + 8.313659) ** (1 / 2))
        + np.arctan(tdb + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh ** (3 / 2) * np.arctan(0.023101 * rh)
        - 4.686035,
        1,
    )
    return twb

def check_folder_filenames():
    folder = '/cnrm/surface/lunelt/NO_SAVE/nc_out/1.11_std_2021_21-24/'
#    folder = '/cnrm/surface/lunelt/NO_SAVE/nc_out/2.13_irr_2021_21-24/'
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


def get_surface_type(site, domain_nb, ds=None, translate_patch = True, 
                     nb_patch=12):
    """
    Returns the type and characteristics of surface for a given site
    
    """
    
    if ds is None:
        if domain_nb==2:
            ds = xr.open_dataset('/cnrm/surface/lunelt/NO_SAVE/nc_out/' + \
                             '2.13_irr_d1d2_21-24/LIAIS.2.SEG22.012dg.nc')
        elif domain_nb==1:
            ds = xr.open_dataset('/cnrm/surface/lunelt/NO_SAVE/nc_out/' + \
                             '2.13_irr_d1d2_21-24/LIAIS.1.SEG22.012dg.nc')
    
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
        
    # keys aggregated
    for key in ['LAI_ISBA', 'Z0_ISBA', 'Z0H_ISBA', 
#                'EMIS', 
                'CD_ISBA',  # coeff trainee
                'CH_ISBA',  # coeff echange turbulent pour sensible heat
                'CE_ISBA',  # coeff echange turbulent pour latent heat
                ]:
        sq_ds = ds[key] #.squeeze()
        res[key] = float(sq_ds[index_lat, index_lon].data)
    # keys by patch, to aggregate
    
    for key in ['CV',
                'ALBNIR',
                'ALBUV',
                'ALBVIS',
                'DROOT_DIF',]:
        temp = []
        for p_nb in np.arange(1, 13):  # patch number
            keyP = '{0}P{1}'.format(key, p_nb)
            temp.append(float(ds[keyP][index_lat, index_lon].data))
        # remove 999 from list (replaced by 0)
        temp = [0 if i == 999 else i for i in temp]
        aggr_key = '{0}_ISBA'.format(key)
        
        res[aggr_key] = np.sum(np.array(weight) * np.array(temp))
    
    # Get soil type from pgd
    if domain_nb == 2:
        pgd = xr.open_dataset(
            '/cnrm/surface/lunelt/NO_SAVE/nc_out/2.01_pgds_irr/' + \
            'PGD_400M_CovCor_v26_ivars.nc')
    elif domain_nb == 1:
        pgd = xr.open_dataset(
            '/cnrm/surface/lunelt/NO_SAVE/nc_out/2.01_pgds_irr/' + \
            'PGD_2KM_CovCor_v26_ivars.nc')
    
    for key in ['CLAY', 'SAND', ]:
        val = float(pgd[key][index_lat, index_lon].data)
        res[key] = val
        
    return res

def calc_u_star_sim(ds):
    """
    Compute friction velocity u* from zonal and meridian wind stress FMU_ISBA
    and FMV_ISBA.
    
    Parameters:
        ds: xarray.Dataset containing FMU_ISBA and FMV_ISBA
    
    Returns:
        xarray.DataArray
    
    """
    fmu_md = ds['FMU_ISBA']
    fmv_md = ds['FMV_ISBA']

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
    """
    
    ws = np.sqrt(ut**2 + vt**2)
    wd_temp = np.arctan2(-ut, -vt)*180/np.pi
    wd = xr.where(wd_temp<0, wd_temp+360, wd_temp)
    return ws, wd
    
def wind_direction_mean(data):
    """
    Computing the mean of circular data is not trivial (cf https://en.wikipedia.org/wiki/Directional_statistics)
    
    use function 'circmean', 'circstd' from scipy.stats    
    """
    
    
if __name__ == '__main__':
    print('nothing to run')
