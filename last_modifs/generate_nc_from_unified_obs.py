#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 09:05:40 2022

!!!!!!!!!!!!!!!!!!!
NOT WORKING
UNIFIED DATA LACKS IRRADIANCE DATA !!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!

@author: martib, lunelt
"""

import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
import xarray as xr
import global_variables as gv
import tools
import pandas as pd
import numpy.ma as ma

############################

#site = 'elsplans'

## Data path
data_path = '/cnrm/surface/lunelt/data_LIAISE/unified_stations/30Min/EddyPro_Output/'

fn = 'eddypro_IRTA_Corn_30min_full_output.csv'
#fn = 'eddypro_LACENDROSA_Alfalfa_CNRM_30min_full_output.csv'
#fn = 'new_modif.csv'
#df = pd.read_csv(data_path + fn, encoding='utf-8')

with open(data_path + fn, 'rb') as fichier:
    linelist = []
    linenb = 0
    for line in fichier:
        linenb = linenb + 1
        try:
            linelist.append(line.decode('utf-8').replace('\r\n', '').split(','))
        except UnicodeDecodeError:
            pass

df = pd.DataFrame(linelist[3:], columns=linelist[1])
df_units = pd.DataFrame(linelist[2],linelist[1])


df['datetime'] = df['date'][:]+'T'+df['time'][:]
df['datetime'] = df['datetime'].apply(pd.Timestamp)
df['unixtime'] = (df['datetime'][:] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')



#mnh_fn = '{0}_d1_15-30/EXTRACTED_FILES/LIAIS.1.CAT_{0}.nc'.format(model)
#ds_mnh = nc.Dataset(data_path + mnh_fn, 
#                    mode='r')
#ds_mnh_xr = xr.open_dataset(data_path + mnh_fn)

## Create NetCDF File
ncdf_new = nc.Dataset(data_path + 'FORCING_{0}.nc'.format('test'), 
                      mode='w')


#plot_data = False

################################

#index_lat, index_lon = tools.indices_of_lat_lon(ds_mnh_xr, 
#                                      gv.sites[site]['lat'],
#                                      gv.sites[site]['lon'])
filelength = df.shape[0]

#%%
# TIME
#fn = '/cnrm/surface/lunelt/for_tanguy/in_out_generate_nc/CAT_2021-07LIAISE_LA-CENDROSA_CNRM_MTO-FLUX-30MIN_L2_.nc'
#cendrosa = nc.Dataset(fn)
#init_time = dt.strptime('2021-07-01-10','%Y-%m-%d-%H')  # corrected to 07-14 later
#timestamp = init_time + cendrosa['time'][1::2][:filelength]*timedelta(seconds=1)

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
#ncdf_new["time"][:] = cendrosa['time'][1::2][:filelength] - 3600 + 1626224400 # unixtime for 2021-07-14T01
marray = ma.masked_values(np.array(df['unixtime']), -9999)
ncdf_new["time"][:] = marray
#init_time_final= dt.strptime('1970-01-01','%Y-%m-%d')
#timestamp_final= init_time_final+ ncdf_new['time'][:]*timedelta(seconds=1)

#%%
# 0D Variables
forc_var = ncdf_new.createVariable('FRC_TIME_STP', np.float32, ())
forc_var.long_name = 'Forcing_Time_Step'
ncdf_new["FRC_TIME_STP"][0]=1800.

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


# 1D variables
Tair_var = ncdf_new.createVariable('Tair', np.float32, ('time','Number_of_points'))
Tair_var.long_name = 'Near_Surface_Air_Temperature'
Tair_var.measurement_heigh = '2m'
Tair_var.units = 'K'  
#Tair_mid=cendrosa["ta_2"][:]
#ncdf_new["Tair"][:,0]=Tair_mid+273.15
#ncdf_new["Tair"][:,0] = ds_mnh['T2M_ISBA'][:,index_lat,index_lon]
ncdf_new["Tair"][:,0] = df['air_temperature']

#%%
Qair_var = ncdf_new.createVariable('Qair', np.float32, ('time','Number_of_points'))
Qair_var.long_name = 'Near_Surface_Specific_Humidity'
Qair_var.measurement_heigh = '2m'
Qair_var.units = 'Kg/Kg'
#Qair_mid=cendrosa["hus_2"][:]
#ncdf_new["Qair"][:,0] = Qair_mid/1000
#ncdf_new["Qair"][:,0] = ds_mnh['Q2M_ISBA'][:,index_lat,index_lon]
ncdf_new["Qair"][:,0] = df['specific_humidity']
#
PSurf_var = ncdf_new.createVariable('PSurf', np.float32, ('time','Number_of_points'))
PSurf_var.long_name = 'Surface_Pressure'
#PSurf_var.measurement_heigh = 'Pa' # pareix una errada de l'original
PSurf_var.units = 'Pa'
#PSurf_mid=cendrosa["pa"][:]
#ncdf_new["PSurf"][:,0]=PSurf_mid*100
#ncdf_new["PSurf"][:,0] = ds_mnh['PRES'][:,0,0,index_lat,index_lon]*100
ncdf_new["PSurf"][:,0] = df['air_pressure']

dirswdown_var = ncdf_new.createVariable('DIR_SWdown', np.float32, ('time','Number_of_points'))
dirswdown_var.long_name = 'Surface_Incicent_Direct_Shortwave_Radiation'
dirswdown_var.units = 'W/m2' 
#dirswdown_mid=cendrosa["swd"][:]
#ncdf_new["DIR_SWdown"][:,0]=dirswdown_mid
#ncdf_new["DIR_SWdown"][:,0] = np.sum(ds_mnh['DIRFLASWD'], axis=2)[:,0,index_lat,index_lon]
ncdf_new["DIR_SWdown"][:,0] = df['air_pressure']

sca_swdown_var = ncdf_new.createVariable('SCA_SWdown', np.float32, ('time','Number_of_points'))
sca_swdown_var.long_name = 'Surface_Incident_Diffuse_Shortwave_Radiation'
sca_swdown_var.units = 'W/m2'
#ncdf_new["SCA_SWdown"][:,0]=np.zeros(len(cendrosa["swd"]))
#ncdf_new["SCA_SWdown"][:,0] = np.sum(ds_mnh['SCAFLASWD'], axis=2)[:,0,index_lat,index_lon]

&### Unified data ne contient pas les irradiances !!!!! ########

lwdown_var = ncdf_new.createVariable('LWdown', np.float32, ('time','Number_of_points'))
lwdown_var.long_name = 'Surface_Incident_Diffuse_Shortwave_Radiation'
lwdown_var.units = 'W/m2'
#dirlwdown_mid=cendrosa["lwd"][:]
#ncdf_new["LWdown"][:,0]=dirlwdown_mid
#ncdf_new["LWdown"][:,0] = ds_mnh['LWD'][:,0,index_lat,index_lon]


rainf_var = ncdf_new.createVariable('Rainf', np.float32, ('time','Number_of_points'))
rainf_var.long_name = 'Rainfall_Rate'
rainf_var.units = 'Kg/m2/s'
#rainf_mid=cendrosa["rain_cumul"][:]
#ncdf_new["Rainf"][:,0]=rainf_mid/1800 # preguntar aaron
#ncdf_new["Rainf"][1320:1321,0]=0
#ncdf_new["Rainf"][481:490,0]=[30/1800,30/1800,30/1800,30/1800,0,0,0,0,0]  # 8th sim, more water same time as 7th, more similar
#ncdf_new["Rainf"][1102:1111,0]=[30/1800,30/1800,30/1800,30/1800,0,0,0,0,0] 
#ncdf_new["Rainf"][:,0] = ds_mnh['RAINF_ISBA'][:,index_lat,index_lon]

snowf_var = ncdf_new.createVariable('Snowf', np.float32, ('time','Number_of_points'))
snowf_var.long_name = 'Snowfall_Rate'
snowf_var.units = 'Kg/m2/s'
#ncdf_new["Snowf"][:,0] = np.zeros(filelength)
#ncdf_new["Snowf"][:,0] = ds_mnh['SNOWF_ISBA'][:,index_lat,index_lon]

#ncdf_new.close()

#%%
# compute ws and wd from ut and vt
ws, wd = tools.calc_ws_wd(ds_mnh['UT'][:,0,0,index_lat,index_lon],
                          ds_mnh['VT'][:,0,0,index_lat,index_lon])

wind_var = ncdf_new.createVariable('Wind', np.float32, ('time','Number_of_points'))
wind_var.long_name = 'Wind_Speed'
wind_var.units = 'm/s'
#wind_mid=cendrosa["ws_2"][:] # change to ws_2 for the measurement at 10m
#ncdf_new["Wind"][:,0]=wind_mid # preguntar aaron
ncdf_new["Wind"][:,0] = ws

winddir_var = ncdf_new.createVariable('Wind_DIR', np.float32, ('time','Number_of_points'))
winddir_var.long_name = 'Wind_Direction'
winddir_var.units = 'deg'
#winddir_mid=cendrosa["wd_2"][:]  # change to wd_2 for the measurement at 10m
#ncdf_new["Wind_DIR"][:,0]=winddir_mid # preguntar aaron
ncdf_new["Wind_DIR"][:,0] = wd

co2air_var = ncdf_new.createVariable('CO2air', np.float32, ('time','Number_of_points'))
co2air_var.long_name = 'Near_Surface_CO2_Concentration'
co2air_var.units = 'Kg/m3'
#Co2_mid=cendrosa["co2_density_1"][:]
#ncdf_new["CO2air"][:,0]=Co2_mid/1000
ncdf_new["CO2air"][:,0] = np.zeros(filelength) + ds['CO2air'][:].mean()

ncdf_new.close()

#%%

if plot_data:
    ncdf_new = nc.Dataset(data_path + 'FORCING_out.nc', 'r') #uncomment to consult 
    
    plt.figure()
    #plt.plot((cendrosa["time"][:]+1625097600.)[0:1])
    #ho pintam
    plt.figure()
    plt.plot(ncdf_new["time"][:],ncdf_new["Tair"][:],'black')
    #plt.figure()
    plt.ylabel("Tair")
    #plt.gcf().autofmt_xdate()
    #plt.plot(timestamp_final,ncdf_new["Tair"][:],'black')
    
    plt.figure()
    plt.plot(ncdf_new["time"][:],ncdf_new["Qair"][:],'black')
    #plt.figure()
    plt.ylabel("Qair")
    #plt.gcf().autofmt_xdate()
    #plt.plot(timestamp_final,ncdf_new["Qair"][:],'black')
    
    plt.figure()
    plt.plot(ncdf_new["time"][:],ncdf_new["PSurf"][:],'black')
    #plt.figure()
    plt.ylabel("PSurf")
    #plt.gcf().autofmt_xdate()
    #plt.plot(timestamp_final,ncdf_new["PSurf"][:],'black')
    
    plt.figure()
    plt.plot(ncdf_new["time"][:],ncdf_new["DIR_SWdown"][:],'black')
    #plt.figure()
    plt.ylabel("DIR_SWdown")
    #plt.gcf().autofmt_xdate()
    #plt.plot(timestamp_final,ncdf_new["DIR_SWdown"][:],'black')
    
    plt.figure()
    plt.plot(ncdf_new["time"][:],ncdf_new["SCA_SWdown"][:],'black')
    #plt.figure()
    plt.ylabel("SCA_SWdown")
    #plt.gcf().autofmt_xdate()
    #plt.plot(timestamp_final,ncdf_new["SCA_SWdown"][:],'black')
    
    
    plt.figure()
    plt.plot(ncdf_new["time"][:],ncdf_new["LWdown"][:],'black')
    #plt.figure()
    plt.ylabel("LWdown")
    #plt.gcf().autofmt_xdate()
    #plt.plot(timestamp_final,ncdf_new["LWdown"][:],'black')
    
    plt.figure()
    plt.plot(ncdf_new["time"][:],ncdf_new["Rainf"][:],'black')
    #plt.figure()
    plt.ylabel("Rainf")
    #plt.gcf().autofmt_xdate()
    #plt.plot(timestamp_final,ncdf_new["Rainf"][:],'black')
    
    plt.figure()
    plt.plot(ncdf_new["time"][:],ncdf_new["Snowf"][:],'black')
    #plt.figure()
    plt.ylabel("Snowf")
    #plt.gcf().autofmt_xdate()
    #plt.plot(timestamp_final,ncdf_new["Snowf"][:],'black')
    
    plt.figure()
    plt.plot(ncdf_new["time"][:],ncdf_new["Wind"][:],'black')
    #plt.figure()
    plt.ylabel("Wind")
    #plt.gcf().autofmt_xdate()
    #plt.plot(timestamp_final,ncdf_new["Wind"][:],'black')
    
    plt.figure()
    plt.plot(ncdf_new["time"][:],ncdf_new["Wind_DIR"][:],'black')
    #plt.figure()
    plt.ylabel("Wind_DIR")
    #plt.gcf().autofmt_xdate()
    #plt.plot(timestamp_final,ncdf_new["Wind_DIR"][:],'black')
    
    plt.figure()
    plt.plot(ncdf_new["time"][:],ncdf_new["CO2air"][:],'black')
    #plt.figure()
    plt.ylabel("CO2air")
    #plt.gcf().autofmt_xdate()
    #plt.plot(timestamp_final,ncdf_new["CO2air"][:],'black')
