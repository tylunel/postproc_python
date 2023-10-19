#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 09:05:40 2022

@author: martib, lunelt
"""

import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import tools
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
import numpy.ma as ma

############################

## Data path
data_path = '/cnrm/surface/lunelt/data_LIAISE/irta-corn/seb/'

fn = 'LIAISE_IRTA-CORN_UIB_SEB-10MIN_L2.nc'
#fn = 'eddypro_LACENDROSA_Alfalfa_CNRM_30min_full_output.csv'

ds = nc.Dataset(data_path + fn, 
                    mode='r')

ncdf_new = nc.Dataset(data_path + 'FORCING_irta-corn.nc', 
                      mode='w')

plot_data = False

################################
frc_time_step = 1800

#cendrosa data - used here because trouble at sfx run with other homemade time series...
fn = '/cnrm/surface/lunelt/for_tanguy/in_out_generate_nc/CAT_2021-07LIAISE_LA-CENDROSA_CNRM_MTO-FLUX-30MIN_L2_.nc'
cendrosa = nc.Dataset(fn)
#cendrosa['time'][1::2][:filelength]
array_time = cendrosa['time'][:]

# index for removing beginning of data
i_start = 243  #242 to be on 20221-07-01 T 01:00
#i_end = 4449
i_end = len(array_time)*3 + i_start
i_step = 3

filelength = len(ds['time'][i_start:i_end:i_step])

#%% TIME dimension

# Create dimensions: Number of points
points_dim = ncdf_new.createDimension('Number_of_points', 1 )
tim_dim = ncdf_new.createDimension('time', filelength)

# Create time variable
time_var = ncdf_new.createVariable('time', np.float64, ('time')) 
time_var.units = 'seconds since 1970-01-01 00:00:00'  # to do
time_var.standard_name = 'time'
time_var.long_name = 'seconds since 1970-01-01 00:00:00 unixtime UTC'

#init_time = '2021-06-29 08:10:00.0000000'  #unixtime=1624954200
#init_time = '2021-06-29 07:40:00.0000000'  #unixtime=1624952400
#init_time = '2021-07-01 00:00:00.00000'    #unixtime=1625097600
#pd_init_time = pd.Timestamp(init_time)
#init_unixtime = (pd_init_time - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

#init_time = dt.strptime('2021-07-01-10','%Y-%m-%d-%H')

#ncdf_new["time"][:] = (ds['time'][i_start:i_end]*60).astype(float) + 1624954200

ncdf_new["time"][:] = array_time + 1625097600.  # July 1


#%% 0D Variables
forc_var = ncdf_new.createVariable('FRC_TIME_STP', np.float32, ())
forc_var.long_name = 'Forcing_Time_Step'
ncdf_new["FRC_TIME_STP"][0] = frc_time_step

lon_var = ncdf_new.createVariable('LON', np.float32, ('Number_of_points'))
lon_var.long_name = 'Longitude'
ncdf_new["LON"][0] = 0.875333

lat_var = ncdf_new.createVariable('LAT', np.float32, ('Number_of_points'))
lat_var.long_name = 'Latitude'
ncdf_new["LAT"][0] = 41.619079

ZS_var = ncdf_new.createVariable('ZS', np.float32, ('Number_of_points'))
ZS_var.long_name = 'Surface_Orography'
ncdf_new["ZS"][0] = 246.

ZREF_var = ncdf_new.createVariable('ZREF', np.float32, ('Number_of_points'))
ZREF_var.long_name = 'Reference_Height'
ZREF_var.units = 'm'  
ncdf_new["ZREF"][0] = 2. # height of T and HUM

UREF_var = ncdf_new.createVariable('UREF', np.float32, ('Number_of_points'))
UREF_var.long_name = 'Reference_Height_for_Wind' # Height of wind 
UREF_var.units = 'm'  
ncdf_new["UREF"][0] = 2. # we also have one at 10m

#%% 1D variables
Tair_var = ncdf_new.createVariable('Tair', np.float32, ('time','Number_of_points'))
Tair_var.long_name = 'Near_Surface_Air_Temperature'
Tair_var.measurement_heigh = '2m'
Tair_var.units = 'C'
ncdf_new["Tair"][:,0] = ds['TA_1_1_1'][i_start:i_end:i_step] + 273.15

Qair_var = ncdf_new.createVariable('Qair', np.float32, ('time','Number_of_points'))
Qair_var.long_name = 'Near_Surface_Specific_Humidity'
Qair_var.measurement_heigh = '2m'
Qair_var.units = 'Kg/Kg'
psy = tools.psy_ta_rh(np.array(ds['TA_1_1_1'][i_start:i_end:i_step]),
                      np.array(ds['RH_1_1_1'][i_start:i_end:i_step]), 
                      np.array(ds['PA'][i_start:i_end:i_step]*1000))
# Pb with nan values that are masked in the masked array psy, so all ds[key]
# converted to ndarray prior to computation of psy. And then remove outliers:
rvt = []
for val in psy['hr']:
    if np.isnan(val):
        rvt.append(np.nanmean(psy['hr']))
        print(val)
    else:
        rvt.append(val)

ncdf_new["Qair"][:,0] = np.array(rvt)

PSurf_var = ncdf_new.createVariable('PSurf', np.float32, ('time','Number_of_points'))
PSurf_var.long_name = 'Surface_Pressure'
PSurf_var.units = 'Pa'
ncdf_new["PSurf"][:,0] = ds['PA'][i_start:i_end:i_step]*1000

dirswdown_var = ncdf_new.createVariable('DIR_SWdown', np.float32, ('time','Number_of_points'))
dirswdown_var.long_name = 'Surface_Incident_Direct_Shortwave_Radiation'
dirswdown_var.units = 'W/m2'
ncdf_new["DIR_SWdown"][:,0] = ds['SW_IN'][i_start:i_end:i_step]

sca_swdown_var = ncdf_new.createVariable('SCA_SWdown', np.float32, ('time','Number_of_points'))
sca_swdown_var.long_name = 'Surface_Incident_Diffuse_Shortwave_Radiation'
sca_swdown_var.units = 'W/m2'
ncdf_new["SCA_SWdown"][:,0] = ds['SW_IN'][i_start:i_end:i_step]*0

lwdown_var = ncdf_new.createVariable('LWdown', np.float32, ('time','Number_of_points'))
lwdown_var.long_name = 'Surface_Incident_Diffuse_Shortwave_Radiation'
lwdown_var.units = 'W/m2'
ncdf_new["LWdown"][:,0] = ds['LW_IN'][i_start:i_end:i_step]

rainf_var = ncdf_new.createVariable('Rainf', np.float32, ('time','Number_of_points'))
rainf_var.long_name = 'Rainfall_Rate'
rainf_var.units = 'Kg/m2/s'
# No rain here. No data on SMC_ET0 site either. 
# Can be replaced by data of cendrosa, or can be considered as not significant 
# compared to the irrigation water amount
# Not significant option:
#ncdf_new["Rainf"][:,0] = ds['LW_IN'][i_start:i_end:i_step]*0
# Cendrosa data:
rain = []
for val in np.array(cendrosa['rain_cumul'])/frc_time_step:
    if np.isnan(val):
        rain.append(0)
    else:
        rain.append(val)
ncdf_new["Rainf"][:,0] = np.array(rain)

snowf_var = ncdf_new.createVariable('Snowf', np.float32, ('time','Number_of_points'))
snowf_var.long_name = 'Snowfall_Rate'
snowf_var.units = 'Kg/m2/s'
ncdf_new["Snowf"][:,0] = ds['LW_IN'][i_start:i_end:i_step]*0

wind_var = ncdf_new.createVariable('Wind', np.float32, ('time','Number_of_points'))
wind_var.long_name = 'Wind_Speed'
wind_var.units = 'm/s'
ncdf_new["Wind"][:,0] = ds['WS'][i_start:i_end:i_step]

winddir_var = ncdf_new.createVariable('Wind_DIR', np.float32, ('time','Number_of_points'))
winddir_var.long_name = 'Wind_Direction'
winddir_var.units = 'deg'
ncdf_new["Wind_DIR"][:,0] = ds['WD'][i_start:i_end:i_step]

co2air_var = ncdf_new.createVariable('CO2air', np.float32, ('time','Number_of_points'))
co2air_var.long_name = 'Near_Surface_CO2_Concentration'
co2air_var.units = 'Kg/m3'
# mean C02 choosen here to better understand the effect of the other prognostic
# variables in the simulations
CO2_density = np.zeros(filelength) + ds['CO2_density'][i_start:i_end:i_step].mean()/1e6
#CO2_density = ds['CO2_density'][i_start:i_end:i_step]/1e6
#CO2_density.mask = False
#CO2_density.fill_value = 1e+20
ncdf_new["CO2air"][:,0] = CO2_density

ncdf_new.close()

#%% Plot Data to check consistency

if plot_data:
    ncdf_new = nc.Dataset(data_path + 'FORCING_irta-corn.nc', 'r') #uncomment to consult 
    
    for key in ['Qair',
#                'PSurf',"DIR_SWdown", "SCA_SWdown", 'Tair', 
#                "LWdown", "Rainf", "Snowf", "Wind", "Wind_DIR", "CO2air"
                ]:
        plt.figure()
        plt.plot(ncdf_new["time"][:],ncdf_new[key][:],'black')
        plt.ylabel(key)

    ncdf_new.close()
