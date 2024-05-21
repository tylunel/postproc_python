#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 09:05:40 2022

@author: martib, lunelt

N.B.:
RUN "extract_concat_vars.sh" before
"""

import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
import xarray as xr
import global_variables as gv
import tools
import os

############################

site = 'irta-corn'

model = 'noirr_lai_d1'
        # 'std_d1',
        # 'noirr_lai_d1',
        # 'irrlagrip30_d1',
        # 'irrlagrip30thld07_d1',
        # 'irrswi1_d1',
        # 'irr_d1',
         
simu_folder = gv.simu_folders[model]

## Data path
data_path = '/cnrm/surface/lunelt/NO_SAVE/nc_out/'

mnh_fn = '{0}/EXTRACTED_FILES/LIAIS.1.CAT_{0}'.format(simu_folder[:-1])

## Check if CAT file exist and create it if not existing
if os.path.isfile(data_path + mnh_fn):
    print(f"Concatenated file {mnh_fn} already exists")
    ds_mnh_xr = xr.open_dataset(data_path + mnh_fn)
    ds_mnh = nc.Dataset(data_path + mnh_fn, 
                        mode='r')
else:
    print("""
        Extraction and concatenation of model outputs.
        May take one hour or more.
        """)
    out = os.system(f"""
        cd /home/lunelt/postproc_python/generate_forcing/
        bash extract_concat_vars.sh {simu_folder[:-1]}
        """)
    if out == 0:
        print('Extraction and concatenation successful!')
    else:
        print('Extraction and concatenation failed!')
    # open the new file
    ds_mnh_xr = xr.open_dataset(data_path + mnh_fn)
    ds_mnh = nc.Dataset(data_path + mnh_fn, 
                        mode='r')

## Create NetCDF File
ncdf_new = nc.Dataset(data_path + 'generate_forcing/FORCING_{0}_{1}.nc'.format(
    site, simu_folder[:4]), mode='w')

plot_data = False

################################

index_lat, index_lon = tools.indices_of_lat_lon(ds_mnh_xr, 
                                                gv.sites[site]['lat'],
                                                gv.sites[site]['lon'])
filelength = len(ds_mnh_xr.record)
#filelength = len(ds_mnh_xr.time)

# get forcing from cendrosa for getting CO2 concentration
fn = 'generate_forcing/FORCING_cendrosa_obs.nc'
ds = nc.Dataset(data_path + fn, 
                mode='r')

# TIME
fn = '/cnrm/surface/lunelt/for_tanguy/in_out_generate_nc/CAT_2021-07LIAISE_LA-CENDROSA_CNRM_MTO-FLUX-30MIN_L2_.nc'
cendrosa = nc.Dataset(fn)
init_time = dt.strptime('2021-07-01-10','%Y-%m-%d-%H')  # corrected to 07-14 later
timestamp = init_time + cendrosa['time'][1::2][:filelength]*timedelta(seconds=1)

# Create dimensions: Number of points
points_dim = ncdf_new.createDimension('Number_of_points', 1 )
#tim_dim = ncdf_new.createDimension('time', len(cendrosa["time"]))
tim_dim = ncdf_new.createDimension('time', filelength)

# Create variables
time_var = ncdf_new.createVariable('time', np.float64, ('time'))
#time_var.units = 'seconds'  
time_var.units = 'seconds since 1970-01-01 00:00:00'  # to do
time_var.standard_name = 'time'
time_var.long_name = 'seconds since 1970-01-01 00:00:00 unixtime UTC'

#ncdf_new["time"][:] = cendrosa["time"][:] + 1625097600. # add time from beggining cendrosa to unixtime
#ncdf_new["time"][:] = cendrosa['time'][1::2][:filelength] + 1626256800 # unixtime for 2021-07-14T10
ncdf_new["time"][:] = cendrosa['time'][1::2][:filelength] - 3600 + 1626224400 # unixtime for 2021-07-14T01
init_time_final= dt.strptime('1970-01-01','%Y-%m-%d')
timestamp_final= init_time_final+ ncdf_new['time'][:]*timedelta(seconds=1)


#%% 0D Variables
forc_var = ncdf_new.createVariable('FRC_TIME_STP', np.float32, ())
forc_var.long_name = 'Forcing_Time_Step'
ncdf_new["FRC_TIME_STP"][0]=3600.

lon_var = ncdf_new.createVariable('LON', np.float32, ('Number_of_points'))
lon_var.long_name = 'Longitude'
ncdf_new["LON"][0]=0.92841

lat_var = ncdf_new.createVariable('LAT', np.float32, ('Number_of_points'))
lat_var.long_name = 'Latitude'
ncdf_new["LAT"][0]=41.69336

ZS_var = ncdf_new.createVariable('ZS', np.float32, ('Number_of_points'))
ZS_var.long_name = 'Surface_Orography'
ncdf_new["ZS"][0]=240.

ZREF_var = ncdf_new.createVariable('ZREF', np.float32, ('Number_of_points'))
ZREF_var.long_name = 'Reference_Height'
ZREF_var.units = 'm'  
ncdf_new["ZREF"][0]=2. # height of T and HUM

UREF_var = ncdf_new.createVariable('UREF', np.float32, ('Number_of_points'))
UREF_var.long_name = 'Reference_Height_for_Wind' # Height of wind 
UREF_var.units = 'm'  
ncdf_new["UREF"][0]=2. # we also have one at 10m


#%% 1D variables
Tair_var = ncdf_new.createVariable('Tair', np.float32, ('time','Number_of_points'))
Tair_var.long_name = 'Near_Surface_Air_Temperature'
Tair_var.measurement_heigh = '2m'
Tair_var.units = 'K'
ncdf_new["Tair"][:,0] = ds_mnh['T2M_ISBA'][:,index_lat,index_lon]

Qair_var = ncdf_new.createVariable('Qair', np.float32, ('time','Number_of_points'))
Qair_var.long_name = 'Near_Surface_Specific_Humidity'
Qair_var.measurement_heigh = '2m'
Qair_var.units = 'Kg/Kg'
ncdf_new["Qair"][:,0] = ds_mnh['Q2M_ISBA'][:,index_lat,index_lon]

PSurf_var = ncdf_new.createVariable('PSurf', np.float32, ('time','Number_of_points'))
PSurf_var.long_name = 'Surface_Pressure'
PSurf_var.units = 'Pa'
ncdf_new["PSurf"][:,0] = ds_mnh['PRES'][:,0,0,index_lat,index_lon]*100

dirswdown_var = ncdf_new.createVariable('DIR_SWdown', np.float32, ('time','Number_of_points'))
dirswdown_var.long_name = 'Surface_Incicent_Direct_Shortwave_Radiation'
dirswdown_var.units = 'W/m2' 
ncdf_new["DIR_SWdown"][:,0] = np.sum(ds_mnh['DIRFLASWD'], axis=2)[:,0,index_lat,index_lon]

sca_swdown_var = ncdf_new.createVariable('SCA_SWdown', np.float32, ('time','Number_of_points'))
sca_swdown_var.long_name = 'Surface_Incident_Diffuse_Shortwave_Radiation'
sca_swdown_var.units = 'W/m2'
ncdf_new["SCA_SWdown"][:,0] = np.sum(ds_mnh['SCAFLASWD'], axis=2)[:,0,index_lat,index_lon]

lwdown_var = ncdf_new.createVariable('LWdown', np.float32, ('time','Number_of_points'))
lwdown_var.long_name = 'Surface_Incident_Diffuse_Shortwave_Radiation'
lwdown_var.units = 'W/m2'
ncdf_new["LWdown"][:,0] = ds_mnh['LWD'][:,0,index_lat,index_lon]

rainf_var = ncdf_new.createVariable('Rainf', np.float32, ('time','Number_of_points'))
rainf_var.long_name = 'Rainfall_Rate'
rainf_var.units = 'Kg/m2/s'
ncdf_new["Rainf"][:,0] = ds_mnh['RAINF_ISBA'][:,index_lat,index_lon]

snowf_var = ncdf_new.createVariable('Snowf', np.float32, ('time','Number_of_points'))
snowf_var.long_name = 'Snowfall_Rate'
snowf_var.units = 'Kg/m2/s'
#ncdf_new["Snowf"][:,0] = np.zeros(filelength)
ncdf_new["Snowf"][:,0] = ds_mnh['SNOWF_ISBA'][:,index_lat,index_lon]


co2air_var = ncdf_new.createVariable('CO2air', np.float32, ('time','Number_of_points'))
co2air_var.long_name = 'Near_Surface_CO2_Concentration'
co2air_var.units = 'Kg/m3'
ncdf_new["CO2air"][:,0] = np.zeros(filelength) + ds['CO2air'][:].mean()

#%% Wind
# compute ws and wd from ut and vt
ws, wd = tools.calc_ws_wd(ds_mnh['UT'][:,0,0,index_lat,index_lon],
                          ds_mnh['VT'][:,0,0,index_lat,index_lon])

wind_var = ncdf_new.createVariable('Wind', np.float32, ('time','Number_of_points'))
wind_var.long_name = 'Wind_Speed'
wind_var.units = 'm/s'
ncdf_new["Wind"][:,0] = ws

winddir_var = ncdf_new.createVariable('Wind_DIR', np.float32, ('time','Number_of_points'))
winddir_var.long_name = 'Wind_Direction'
winddir_var.units = 'deg'
ncdf_new["Wind_DIR"][:,0] = wd

ncdf_new.close()
print(f"""Forcing file created:
    {data_path}/generate_forcing/FORCING_{site}_{simu_folder[:4]}.nc""")

#%% Plot

if plot_data:
    ncdf_new = nc.Dataset(data_path + 'FORCING_out.nc', 'r') #uncomment to consult 
    plt.figure()
    
    for var in ['Tair', 'Qair', 'PSurf', 'DIR_SWdown', 'SCA_SWdown', 'LWdown',
                'Rainf', 'Snowf', 'Wind', 'Wind_DIR', 'CO2air']:

        plt.plot(ncdf_new["time"][:], ncdf_new[var][:],
                 label=var)

