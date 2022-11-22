#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021

Code to get x and y coordinate in a netcdf file according to lat and lon.
Useful for example for XY station coordinate to put in MNH - &NAM_STATIONn

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from metpy.units import units
import re
import math   # mais on pourrait s'en passer et rendre le tout plus lisible
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


def open_ukmo_mast(datafolder, filename, create_netcdf=True, remove_modi=False):
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
        datetime = pd.Timestamp(strings[2])
        
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
    time_list = [(pd.Timedelta(val, unit='s') + datetime).round(freq='T') for val in obs_ukmo['HOUR_time']]
#    time_list = [(pd.Timedelta(val, unit='h') + datetime) for val in obs_ukmo['HOUR_time']]
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

def get_simu_filename(model, date='20210722-1200', domain=2,
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
        'irr_d2': '2.13_irr_2021_22-27/LIAIS.{0}.SEG{1}.{2}.nc'.format(
                domain, seg_nb_str, file_suffix),
#        'irr_d1': '2.14_irr_15-30/LIAIS.{0}.SEG{1}.{2}.nc'.format(
#                domain, seg_nb_str, file_suffix)
        }
    
    filename = father_folder + simu_filelist[model]
    return filename

def get_simu_filename_d1(model, date='20210722-1200', domain=1,
                         file_suffix='dg'):
    """
    --------------
    To merge with get simu filename
    --------------
    
    Returns the whole string for filename and path. 
    Chooses the right SEG (segment) number and suffix corresponding to date
    
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
    seg_nb = str(pd_date.day)        
    suffix_nb = str(pd_date.hour)
    # format suffix with 3 digits:
    if len(suffix_nb) == 1:
        suffix_nb = '00'+ suffix_nb
    elif len(suffix_nb) == 2:
        suffix_nb = '0'+ suffix_nb

    father_folder = '/cnrm/surface/lunelt/NO_SAVE/nc_out/'
    simu_filelist = {
#        'std': '1.11_ECOII_2021_ecmwf_22-27/LIAIS.{0}.SEG{1}.{2}.nc'.format(
#                domain, seg_nb_str, file_suffix),
#        'irr': '2.13_irr_2021_22-27/LIAIS.{0}.SEG{1}.{2}.nc'.format(
#                domain, seg_nb_str, file_suffix),
#        'irr_d2': '2.13_irr_2021_22-27/LIAIS.{0}.SEG{1}.{2}.nc'.format(
#                domain, seg_nb_str, file_suffix),
        'irr_d1': '2.14_irr_15-30/LIAIS.{0}.SEG{1}.{2}{3}.nc'.format(
                domain, seg_nb, suffix_nb, file_suffix)
        }
    
    filename = father_folder + simu_filelist[model]
    
    #check if nomenclature of filename is ok
    check_filename_datetime(filename)
    
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
                mv {0} {1}
                '''.format(filepath, new_fpath))
            print('OUT = {0}'.format(out))
            print("* File '{0}' RENAMED IN: '{1}'".format(filepath, new_fpath))
            return False
        else:
            raise NameError('the date in simu file do not match filename SEG')
    
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
            raise NameError('the time in simu file do not match filename suffix')
    
    return True
    
    

def concat_obs_files(datafolder, in_filenames, out_filename, 
                     create_ukmo_nc=False):
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
            
    Returns: Nothing in python, but it should have created the output file
    in the datafolder if not existing before.        
    """
    
    # CREATE netCDF file from .dat if not existing yet
    if create_ukmo_nc:
        fnames = os.listdir(datafolder)
        for f in fnames:
            if '.dat' not in f:
                pass
            else:
                open_ukmo_mast(datafolder, f, 
                               create_netcdf=True, remove_modi=True)
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


def sm2swi(sm, wilt_pt=None, field_capa=None):
    """
    inputs:
        sm: soil_moisture
        wilt_pt: wilting_point
        field_capa: field capacity
    
    returns:
        soil water index (swi)
    """
    
    if None in [wilt_pt, field_capa]:
        plt.plot(sm)
        raise ValueError('Fill in wilting_point and field_capa values.')
    return (sm - wilt_pt)/(field_capa - wilt_pt)


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
    Interpolate in middle of grid for variable UT, VT and WT
    and rename the associated coordinates.
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
    data['WT'] = data['WT'].interp(level_w=data.level.values).rename(
            {'level_w': 'level'})
    
    # Create new datarray with everything centered in the middle
    #data_new = xr.merge([data_UT, data_VT, data_WT, 
    #                 data['THT'], data['RVT'], data['ZS'],
    ##                 data[surf_var],
    #                 ])
    
    # remove useless coordinates
    data_new = data.drop(['latitude_u', 'longitude_u', 
                          'latitude_v', 'longitude_v',
                          'ni_u', 'nj_u', 'ni_v', 'nj_v', 'level_w'])
    
    # consider time no longer as a dimension but just as a single coordinate
    data_new = data_new.squeeze()
    
    return data_new


def save_figure(plot_title, save_folder='./figures/', verbose=True):
    """
    Save the plot
    """
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    filename = (plot_title)
    filename = filename.replace('=', '').replace('(', '').replace(')', '')
    filename = filename.replace(' ', '_').replace(',', '').replace('.', '_')
    filename = filename.replace('/', 'over')
    plt.savefig(save_folder +str(filename))
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
    c1 = -5674.5359
    c2 = 6.3925247
    c3 = -0.9677843 * math.pow(10, -2)
    c4 = 0.62215701 * math.pow(10, -6)
    c5 = 0.20747825 * math.pow(10, -8)
    c6 = -0.9484024 * math.pow(10, -12)
    c7 = 4.1635019
    c8 = -5800.2206
    c9 = 1.3914993
    c10 = -0.048640239
    c11 = 0.41764768 * math.pow(10, -4)
    c12 = -0.14452093 * math.pow(10, -7)
    c13 = 6.5459673

    if ta_k < 273.15:
        pascals = math.exp(
            c1 / ta_k
            + c2
            + ta_k * (c3 + ta_k * (c4 + ta_k * (c5 + c6 * ta_k)))
            + c7 * math.log(ta_k)
        )
    else:
        pascals = math.exp(
            c8 / ta_k
            + c9
            + ta_k * (c10 + ta_k * (c11 + ta_k * c12))
            + c13 * math.log(ta_k)
        )

    return round(pascals, 1)

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

    gamma_m = math.log(rh / 100 * math.exp((b - tdb / d) * (tdb / (c + tdb))))

    return round(c * gamma_m / (b - gamma_m), 1)

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
    twb = round(
        tdb * math.atan(0.151977 * (rh + 8.313659) ** (1 / 2))
        + math.atan(tdb + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * rh ** (3 / 2) * math.atan(0.023101 * rh)
        - 4.686035,
        1,
    )
    return twb


if __name__ == '__main__':
    nb_points_beyond = 10
    site_end = 'cendrosa'
    end = (41.0, 0.90)
    site_start = 'arbeca'
    start = (41.04, 0.9)
    line = line_coords(data, start, end, 
                       nb_indices_exterior=0)
    
#    open_ukmo_rs('/cnrm/surface/lunelt/data_LIAISE/elsplans/mat_50m/30min/', filename)
#    
#    ds = xr.open_dataset('/cnrm/surface/lunelt/NO_SAVE/nc_out/1.11_ECOII_2021_ecmwf_22-27/LIAIS.1.SEG21.021.nc')
    
    
#%% FOR COMPA WITH NC FILES
    
    #%% TEST for line_coords
#    ds = xr.open_dataset(
#    '/cnrm/surface/lunelt/NO_SAVE/nc_out/1.11_ECOII_2021_ecmwf_22-27/' + \
#    'LIAIS.2.SEG14.001dg.nc')
#    
#    ni_line, nj_line = line_coords(ds, 
#                       start_pt = (41.41, 0.84),
#                       end_pt = (41.705, 1.26))
    
    #%%TEST for open_ukmo_mast
#    obs = xr.open_dataset(
#            '/cnrm/surface/lunelt/data_LIAISE/cendrosa/30min/' + \
#            'LIAISE_LA-CENDROSA_CNRM_MTO-FLUX-30MIN_L2_2021-07-22_V2.nc')
#
#    datafolder = '/cnrm/surface/lunelt/data_LIAISE/elsplans/mat_50m/5min/'
#    for filename in os.listdir('/cnrm/surface/lunelt/data_LIAISE/elsplans/mat_50m/5min/'):
#        print(filename)
#        if '.dat' in filename:        
#            obs_ukmo_out = open_ukmo_mast(
#            '/cnrm/surface/lunelt/data_LIAISE/elsplans/mat_50m/5min/',
#            filename, create_netcdf=True, remove_modi=True )
    
#    obs_ukmo_out = open_ukmo_mast(
#            '/cnrm/surface/lunelt/data_LIAISE/elsplans/mat_50m/5min/',
#            'LIAISE_20210723_05.dat', create_netcdf=True, remove_modi=True )
    
    #%% TEST for open_ukmo_rs
#    datafolder = \
#    '/cnrm/surface/lunelt/data_LIAISE/elsplans/radiosoundings/'
#    filename = 'LIAISE_ELS_PLANS_20210715_050013.txt'
#    obs_ukmo = open_ukmo_rs(datafolder, filename)
    
    
    #%% TEST for filename_from_date
#    for site in ['cendrosa', 'elsplans']:
#        datafolder = \
#            '/cnrm/surface/lunelt/data_LIAISE/'+ site +'/radiosoundings/'
#        wanted_datetime = pd.Timestamp('20210721-1200')
#            
#        fname = filename_from_date(datafolder, wanted_datetime)
#        print('filename is: ' + fname)
    
    
    #%% TEST for indices_of_lat_lon 
#    ds = xr.open_dataset(
#        '/cnrm/surface/lunelt/NO_SAVE/nc_out/1.11_ECOII_2021_ecmwf_22-27/' + \
#        'LIAIS.2.SEG14.001.nc')
#    datetime = ds.time.values[0]   
#    #la cendrosa:
#    lat=41.6925905
#    lon=0.9285671  
#    index_lat, index_lon = indices_of_lat_lon(ds, lat, lon)    
#    # latitude and longitude
#    lat_dat = ds.latitude.data
#    lon_dat = ds.longitude.data
#    print('lat & lon available in file are:')
#    lat_grid = lat_dat[index_lat, index_lon]
#    lon_grid = lon_dat[index_lat, index_lon]
#    print(lat_grid, lon_grid)   
#    # find X and Y corresponding to lat and lon of grid point
#    X_grid = ds.XHAT.data[index_lon]
#    Y_grid = ds.YHAT.data[index_lat]
#    print("X_grid={0}, Y_grid={1}".format(X_grid, Y_grid))    
#    # find X and Y corresponding to lat and lon with interpolation
#    X = X_grid + (ds.XHAT.data[index_lon] - ds.XHAT.data[index_lon-1])*(
#            (lon_dat[index_lat, index_lon] - lon)/ \
#            (lon_dat[index_lat, index_lon] - lon_dat[index_lat, index_lon-1]))   
#    Y = Y_grid + (ds.YHAT.data[index_lat] - ds.YHAT.data[index_lat-1])*(
#            (lat_dat[index_lat, index_lon] - lat)/ \
#            (lat_dat[index_lat, index_lon] - lat_dat[index_lat-1, index_lon]))   
#    print("X={0}, Y={1}".format(X, Y))
