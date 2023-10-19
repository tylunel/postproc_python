#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 13:29:37 2023

@author: martib
"""


import netCDF4 as nc
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime as dt
from datetime import timedelta
import pandas as pd
#import arcpy
import csv
import xarray as xr
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='svg'


str2date = lambda x: dt.strptime(x,'%y-%m-%d %H:%M:%S')
flag = lambda y: dt.strptime(y.decode("utf-8"),'%d/%m/%y %H:%M:%S')
#vconverters = {0: str2date},
# , converters = {'hour_time': str2date}
elsplans= np.genfromtxt('data_lop_1.txt', delimiter=",", names = True, encoding='ascii', dtype=None)
elsplans['hour_time'].astype('datetime64')

elsplans= pd.DataFrame(elsplans)
#print(elsplans['hour_time'].dtype)
#(elsplans["hour_time"]).astype['datetime']
#print(elsplans.dtype.names)

#, converters = {0: str2date}
elsplans_2= np.genfromtxt('data_lop_2.txt', delimiter=",", names = True, encoding='ascii', dtype=None)
elsplans_2= pd.DataFrame(elsplans_2)
#elsplans_2['temp_10m_flag'][:]='X'

a=pd.concat([elsplans, elsplans_2],axis=0)
elsplans_3= np.genfromtxt('data_lop_3.txt', delimiter=",", names = True, encoding='ascii', dtype=None)
elsplans_3= pd.DataFrame(elsplans_3)
elsplans_4= np.genfromtxt('data_lop_4.txt', delimiter=",", names = True, encoding='ascii', dtype=None)
elsplans_4= pd.DataFrame(elsplans_4)
elsplans_5= np.genfromtxt('data_lop_5.txt', delimiter=",", names = True, encoding='ascii', dtype=None)
elsplans_5= pd.DataFrame(elsplans_5)
elsplans_6= np.genfromtxt('data_lop_6.txt', delimiter=",", names = True, encoding='ascii', dtype=None)
elsplans_6= pd.DataFrame(elsplans_6)
elsplans_7= np.genfromtxt('data_lop_7.txt', delimiter=",", names = True, encoding='ascii', dtype=None)
elsplans_7= pd.DataFrame(elsplans_7)
elsplans_9= np.genfromtxt('data_lop_9.txt', delimiter=",", names = True, encoding='ascii', dtype=None)
elsplans_9= pd.DataFrame(elsplans_9)
elsplans_10= np.genfromtxt('data_lop_10.txt', delimiter=",", names = True, encoding='ascii', dtype=None)
elsplans_10= pd.DataFrame(elsplans_10)
elsplans_11= np.genfromtxt('data_lop_11.txt', delimiter=",", names = True, encoding='ascii', dtype=None)
elsplans_11= pd.DataFrame(elsplans_11)
elsplans_12= np.genfromtxt('data_lop_12.txt', delimiter=",", names = True, encoding='ascii', dtype=None)
elsplans_12= pd.DataFrame(elsplans_12)
elsplans_13= np.genfromtxt('data_lop_13.txt', delimiter=",", names = True, encoding='ascii', dtype=None)
elsplans_13= pd.DataFrame(elsplans_13)
#df3 = pd.merge(elsplans,elsplans_2,  on = 'hour_time')
#elsplans_series=elsplans
elsplans_series=pd.concat([elsplans, elsplans_2,elsplans_3, elsplans_4,elsplans_5,elsplans_6,elsplans_7,elsplans_9,elsplans_10,elsplans_11,elsplans_12,elsplans_13],axis=0)
text=str(list(elsplans_series.columns.values))
np.savetxt("data_together.txt",elsplans_series, fmt="%s",delimiter=",",header=text)   
elsplans_series["hour_time"]= pd.to_datetime(elsplans_series['hour_time'],format='%Y-%m-%d %H:%M:%S')+1*timedelta(minutes=15)


 # Similar data files to corn TANGUY
#uib data
data_path = r'/home/martib/Documents/LIAISE_comparison/data_elsplans/UIB/'
file1= np.genfromtxt('/home/martib/Documents/LIAISE_comparison/data_elsplans/UIB/TOA5_els_plans_gradient.minutal10_2021_06_17_0930.dat', delimiter=",",skip_header=1, dtype=None,encoding='ascii',names = True)
file1=file1[2:len(file1)]
#temp_1=file1['TIMESTAMP'].astype(str)
temp_1=np.char.replace(file1['TIMESTAMP'].astype(str), ' 24', ' 00')
cond=(file1['TIMESTAMP']!=temp_1)
c=pd.to_datetime(temp_1, format='"%Y-%m-%d %H:%M:%S"')
temp_1[cond]=(c[cond]+1*timedelta(days=1)).strftime('"%Y-%m-%d %H:%M:%S"').astype(str)
#file1['TIMESTAMP']=pd.to_datetime(temp_1, format='"%Y-%m-%d %H:%M:%S"')
file1['TIMESTAMP'][cond]=(c[cond]+1*timedelta(days=1)).strftime('"%Y-%m-%d %H:%M:%S"').astype(str)
file1= pd.DataFrame(file1)
file_uib=file1
for files in os.walk(data_path):
    print()
    
files_10_min=files[2][1:17]
for filename in files_10_min:
             file_path = os.path.join(data_path, filename)         
             file_iter= np.genfromtxt(file_path, delimiter=",",skip_header=1, dtype=None,encoding='ascii',names = True)
             file_iter=file_iter[2:len(file_iter)]
             temp_2=np.char.replace(file_iter['TIMESTAMP'].astype(str), ' 24', ' 00')
             cond=(file_iter['TIMESTAMP']!=temp_2)
             c=pd.to_datetime(temp_2, format='"%Y-%m-%d %H:%M:%S"')
             temp_2[cond]=(c[cond]+1*timedelta(days=1)).strftime('"%Y-%m-%d %H:%M:%S"').astype(str)
             file_iter['TIMESTAMP'][cond]=(c[cond]+1*timedelta(days=1)).strftime('"%Y-%m-%d %H:%M:%S"').astype(str)
             file_iter= pd.DataFrame(file_iter)
             file_uib=pd.concat([file_uib, file_iter],axis=0)

file_uib['TIMESTAMP']=pd.to_datetime(file_uib['TIMESTAMP'],format='"%Y-%m-%d %H:%M:%S"')

file_uib=file_uib.replace('"NAN"',np.NaN)
cols=[i for i in file_uib.columns if i not in ["TIMESTAMP"]]
for col in cols:
    file_uib[col]=pd.to_numeric(file_uib[col], errors='coerce').fillna(100000000000.0)

file_uib_30min = file_uib.resample('30min',on="TIMESTAMP").mean().reset_index() # the time is averaged at the end of period
file_uib_30min["TIMESTAMP"]= file_uib_30min["TIMESTAMP"]+1*timedelta(minutes=30)
# fi de data uib
 # END of similar data files to corn TANGUY

#series=pd.concat([elsplans_series,file_uib_30min])
#series=elsplans_series.append(file_uib_30min, ignore_index=True)
#np.savetxt("data_groups.txt",series, fmt="%s",delimiter=",")  
series=pd.merge(
    elsplans_series,
    file_uib_30min,
    left_on="hour_time",
    right_on="TIMESTAMP")
np.savetxt("data_groups.txt",series, fmt="%s",delimiter=",")  

#'TIMESTAMP', 'RECORD', 'BattV_Min', 'PTemp_C_Avg', 'PTemp_C_Std',
#       'airT1_Avg', 'airT1_Std', 'airRH1_Avg', 'airRH1_Std', 'air_e1_Avg',
#       'air_e1_Std', 'airT2_Avg', 'airT2_Std', 'airRH2_Avg', 'airRH2_Std',
#       'air_e2_Avg', 'air_e2_Std', 
#       'WS_ms_S_WVT', 'WindDir_D1_WVT', 'WS_ms_Avg', 'WS_ms_Std'

# Data path
data_path = r''
# Create NetCDF File
output_nc = os.path.join(data_path, 'FORCING_ELSPLANS.nc')
ncdf_new = nc.Dataset(output_nc, 'w')

# Create dimensions
points_dim = ncdf_new.createDimension('Number_of_points', 1 )
tim_dim = ncdf_new.createDimension('time', len(series["hour_time"]))
#tim_dim = ncdf_new.createDimension('time', size=None)

# Create variables
time_var = ncdf_new.createVariable('time', np.float64, ('time'))
time_var.units = 'seconds since 1970-01-01 00:00:00'  # to do
time_var.standard_name = 'time'
time_var.long_name = 'seconds since 1970-01-01 00:00:00 unixtime UTC'
series["hour_time"]=pd.to_datetime(series["hour_time"][:])
timestamp_pd=pd.to_datetime(series["hour_time"][:])
df_unix_sec = (timestamp_pd - pd.Timestamp('1970-01-01')).astype('timedelta64[s]').astype('int64')
#timestamp= init_time+cendrosa['time'][:]*timedelta(seconds=1)
ncdf_new["time"][:]=df_unix_sec # add time from beggining cendrosa to unixtime
#init_time_final= dt.strptime('1970-01-01','%Y-%m-%d')
#timestamp_final= init_time_final+ ncdf_new['time'][:]*timedelta(seconds=1)
##time_var.bounds = 'time_bnds'
##time_var.cell_methods = 'time : mean'

##time_var.calendar = 'standard'
#print(ds['time'])
#print(ncdf_new['time'])
##print(ncdf_new["time"][:])

forc_var = ncdf_new.createVariable('FRC_TIME_STP', np.float32, ())
forc_var.long_name = 'Forcing_Time_Step'
ncdf_new["FRC_TIME_STP"][0]=1800.

lon_var = ncdf_new.createVariable('LON', np.float32, ('Number_of_points'))
lon_var.long_name = 'Longitude'
ncdf_new["LON"][0]=1.029363 

lat_var = ncdf_new.createVariable('LAT', np.float32, ('Number_of_points'))
lat_var.long_name = 'Latitude'
ncdf_new["LAT"][0]=41.590111 

ZS_var = ncdf_new.createVariable('ZS', np.float32, ('Number_of_points'))
ZS_var.long_name = 'Surface_Orography'
ncdf_new["ZS"][0]=334 

ZREF_var = ncdf_new.createVariable('ZREF', np.float32, ('Number_of_points'))
ZREF_var.long_name = 'Reference_Height'
ZREF_var.units = 'm'  
ncdf_new["ZREF"][0]=2. # height of T and HUM

UREF_var = ncdf_new.createVariable('UREF', np.float32, ('Number_of_points'))
UREF_var.long_name = 'Reference_Height_for_Wind' # Height of wind 
UREF_var.units = 'm'  
ncdf_new["UREF"][0]=10. # we also have one at 10m

Tair_var = ncdf_new.createVariable('Tair', np.float32, ('time','Number_of_points'))
Tair_var.long_name = 'Near_Surface_Air_Temperature'
Tair_var.measurement_heigh = '2m'
Tair_var.units = 'K'  
##ncdf_new["Tair"][0:len(cendrosa_1to30["ta_2"])-1,0]=cendrosa_1to30["ta_2"][0:len(cendrosa_1to30["ta_2"])-1].real
##Tair_mid=Tair_mid.drop('time')
##ncdf_new["Tair"][:,0]=np.arange(Tair_mid,0)

##Tair_mid=cendrosa_1to30["ta_2"][0:len(cendrosa_1to30["ta_2"])-1].real.values
times=series["hour_time"][series["rtemp_2mb"]>10000]
np.savetxt("times_temp_missing.txt",times, fmt="%s",delimiter=",",header=text)   
times=series["hour_time"][series["rho_2m"]>10000]
np.savetxt("times_hum_missing.txt",times, fmt="%s",delimiter=",",header=text)   

#plt.plot(series["rtemp_2mb"])
#plt.plot(series["airT1_Avg"])
series["rtemp_2mb"][series["rtemp_2mb"]>10000]=series["airT1_Avg"][series["rtemp_2mb"]>10000]
series["rtemp_2mb"][series["rtemp_2mb"]>10000]=25.

Tair_mid=series["rtemp_2mb"][:]
ncdf_new["Tair"][:,0]=Tair_mid+273.15

Qair_var = ncdf_new.createVariable('Qair', np.float32, ('time','Number_of_points'))
Qair_var.long_name = 'Near_Surface_Specific_Humidity'
Qair_var.measurement_heigh = '2m'
Qair_var.units = 'Kg/Kg'
##Qair_mid=cendrosa_1to30["hus_2"][0:len(cendrosa_1to30["hus_2"])-1].real.values


series["pres_subsoil"][series["pres_subsoil"]<-1]=975
series["pres_subsoil"][series["pres_subsoil"]>10000]=975

series["air_e1_Avg"][series["air_e1_Avg"]>10000]=2.0
# kpa to hpa
q=0.622*series["air_e1_Avg"]*10/(series["pres_subsoil"]-0.378*series["air_e1_Avg"]*10)
#series["rho_2m"][series["rho_2m"]>10000]=12.
#Qair_mid=series["rho_2m"][:]
Qair_mid=q
ncdf_new["Qair"][:,0]=Qair_mid/1000


PSurf_var = ncdf_new.createVariable('PSurf', np.float32, ('time','Number_of_points'))
PSurf_var.long_name = 'Surface_Pressure'
##PSurf_var.measurement_heigh = 'Pa' # pareix una errada de l'original
PSurf_var.units = 'Pa'
##PSurf_mid=cendrosa_1to30["pa"][0:len(cendrosa_1to30["pa"])-1].real.values

PSurf_mid=series["pres_subsoil"][:]
ncdf_new["PSurf"][:,0]=PSurf_mid*100

dirswdown_var = ncdf_new.createVariable('DIR_SWdown', np.float32, ('time','Number_of_points'))
dirswdown_var.long_name = 'Surface_Incicent_Direct_Shortwave_Radiation'
dirswdown_var.units = 'W/m2' 
series["swdn_rad"][series["swdn_rad"]>10000]=0
##dirswdown_mid=cendrosa_1to30["swd"][0:len(cendrosa_1to30["swd"])-1].real.values
dirswdown_mid=series["swdn_rad"][:]
ncdf_new["DIR_SWdown"][:,0]=dirswdown_mid

sca_swdown_var = ncdf_new.createVariable('SCA_SWdown', np.float32, ('time','Number_of_points'))
sca_swdown_var.long_name = 'Surface_Incident_Diffuse_Shortwave_Radiation'
sca_swdown_var.units = 'W/m2' 
##ncdf_new["SCA_SWdown"][:,0]=np.zeros(len(cendrosa_1to30["swd"])-1)
ncdf_new["SCA_SWdown"][:,0]=np.zeros(len(series["swdn_rad"]))


lwdown_var = ncdf_new.createVariable('LWdown', np.float32, ('time','Number_of_points'))
lwdown_var.long_name = 'Surface_Incident_Diffuse_Shortwave_Radiation'
lwdown_var.units = 'W/m2' 

series["lwdn_rad"][series["lwdn_rad"]>10000]=350
##dirlwdown_mid=cendrosa_1to30["lwd"][0:len(cendrosa_1to30["lwd"])-1].real.values
dirlwdown_mid=series["lwdn_rad"][:]
ncdf_new["LWdown"][:,0]=dirlwdown_mid

rainf_var = ncdf_new.createVariable('Rainf', np.float32, ('time','Number_of_points'))
rainf_var.long_name = 'Rainfall_Rate'
rainf_var.units = 'Kg/m2/s'
##rainf_mid=cendrosa_1to30["rain_cumul"][0:len(cendrosa_1to30["rain_cumul"])-1].real.values

series["rain_subsoil"][series["rain_subsoil"]>10000]=0
rainf_mid=series["rain_subsoil"][:]
ncdf_new["Rainf"][:,0]=rainf_mid/1800 # preguntar aaron
#ncdf_new["Rainf"][1320:1321,0]=0 
#ncdf_new["Rainf"][481:490,0]=[30/1800,30/1800,30/1800,30/1800,0,0,0,0,0]  # 8th sim, more water same time as 7th, more similar
#ncdf_new["Rainf"][1102:1111,0]=[30/1800,30/1800,30/1800,30/1800,0,0,0,0,0] 

#rain_check=ncdf_new["Rainf"][:]

snowf_var = ncdf_new.createVariable('Snowf', np.float32, ('time','Number_of_points'))
snowf_var.long_name = 'Snowfall_Rate'
snowf_var.units = 'Kg/m2/s'
ncdf_new["Snowf"][:,0]=np.zeros(len(series["rain_subsoil"]))

wind_var = ncdf_new.createVariable('Wind', np.float32, ('time','Number_of_points'))
wind_var.long_name = 'Wind_Speed'
wind_var.units = 'm/s'

series["utot_10mb"][series["utot_10mb"]>10000]=1
##wind_mid=cendrosa_1to30["ws_1"][0:len(cendrosa_1to30["ws_1"])-1].real.values # change to ws_2 for the measurement at 10m
wind_mid=series["utot_10mb"][:] # change to ws_2 for the measurement at 10m
ncdf_new["Wind"][:,0]=wind_mid # preguntar aaron

winddir_var = ncdf_new.createVariable('Wind_DIR', np.float32, ('time','Number_of_points'))
winddir_var.long_name = 'Wind_Direction'
winddir_var.units = 'deg'

series["dir_10mb"][series["dir_10mb"]>10000]=0
##winddir_mid=cendrosa_1to30["wd_1"][0:len(cendrosa_1to30["wd_1"])-1].real.values # change to wd_2 for the measurement at 10m
winddir_mid=series["dir_10mb"][:]  # change to wd_2 for the measurement at 10m
ncdf_new["Wind_DIR"][:,0]=winddir_mid # preguntar aaron

co2air_var = ncdf_new.createVariable('CO2air', np.float32, ('time','Number_of_points'))
co2air_var.long_name = 'Near_Surface_CO2_Concentration'
co2air_var.units = 'Kg/m3'

Co2_mid=np.ones(len(series["dir_10mb"]))*7 # does not exist for els plans of ukmo so w generate flat series
#Co2_mid=elsplans_series["co2_density_1"][:] # does not exist for els plans
ncdf_new["CO2air"][:,0]=Co2_mid/1000

ncdf_new.close()
ncdf_new = nc.Dataset(output_nc, 'r') #uncomment to consult 

#plt.figure()
#plt.plot((cendrosa["time"][:]+1625097600.)[0:1])
#ho pintam
plt.figure()
#plt.plot(ncdf_new["time"][:],ncdf_new["Tair"][:],'black')
plt.plot(timestamp_pd,series["temp_2m"][:],'black')
#plt.figure()
#plt.ylabel("Tair")
#plt.gcf().autofmt_xdate()
#plt.plot(timestamp_final,ncdf_new["Tair"][:],'black')


pio.renderers.default='browser'
fig32 = go.Figure()
fig32.add_trace(go.Scatter(  x=timestamp_pd,    y=series["temp_2m"][:],name="temp_2"))
fig32.add_trace(go.Scatter(  x=timestamp_pd,    y=series["rtemp_2mb"][:],name="rtemp_2mb"))
#fig32.add_trace(go.Scatter(  x=timestamp_pd,    y=elsplans_series["rtemp_2mb"][:],name="temp_2"))
fig32.update_layout(
    yaxis_title="T",
    legend_title="",
    font=dict(
        size=18
    ))
fig32.show()
fig32.write_html("temp.html")



pio.renderers.default='browser'
fig30 = go.Figure()
#fig30.add_trace(go.Scatter(  x=timestamp_pd,    y=elsplans_series["temp_2m"][:],name="temp_2"))
fig30.add_trace(go.Scatter(  x=timestamp_pd,    y=series["rhum_2mb"][:],name="hum_2mb"))
fig30.add_trace(go.Scatter(  x=timestamp_pd,    y=series["rhum_10mb"][:],name="hum_10mb"))
#fig32.add_trace(go.Scatter(  x=timestamp_pd,    y=elsplans_series["rtemp_2mb"][:],name="temp_2"))
fig30.update_layout(
    yaxis_title="T",
    legend_title="",
    font=dict(
        size=18
    ))
fig30.show()
fig30.write_html("hum.html")


#plt.figure()
#plt.plot(ncdf_new["time"][:],ncdf_new["Qair"][:],'black')
#plt.figure()
#plt.ylabel("Qair")
#plt.gcf().autofmt_xdate()
#plt.plot(timestamp_final,ncdf_new["Qair"][:],'black')

#plt.figure()
#plt.plot(ncdf_new["time"][:],ncdf_new["PSurf"][:],'black')
#plt.figure()
#plt.ylabel("PSurf")
#plt.gcf().autofmt_xdate()
#plt.plot(timestamp_final,ncdf_new["PSurf"][:],'black')

#plt.figure()
#plt.plot(ncdf_new["time"][:],ncdf_new["DIR_SWdown"][:],'black')
#plt.figure()
#plt.ylabel("DIR_SWdown")
#plt.gcf().autofmt_xdate()
#plt.plot(timestamp_final,ncdf_new["DIR_SWdown"][:],'black')

#plt.figure()
#plt.plot(ncdf_new["time"][:],ncdf_new["SCA_SWdown"][:],'black')
#plt.figure()
#plt.ylabel("SCA_SWdown")
#plt.gcf().autofmt_xdate()
#plt.plot(timestamp_final,ncdf_new["SCA_SWdown"][:],'black')


#plt.figure()
#plt.plot(ncdf_new["time"][:],ncdf_new["LWdown"][:],'black')
#plt.figure()
#plt.ylabel("LWdown")
#plt.gcf().autofmt_xdate()
#plt.plot(timestamp_final,ncdf_new["LWdown"][:],'black')

#plt.figure()
#plt.plot(ncdf_new["time"][:],ncdf_new["Rainf"][:],'black')
#plt.figure()
#plt.ylabel("Rainf")
#plt.gcf().autofmt_xdate()
#plt.plot(timestamp_final,ncdf_new["Rainf"][:],'black')

#plt.figure()
#plt.plot(ncdf_new["time"][:],ncdf_new["Snowf"][:],'black')
#plt.figure()
#plt.ylabel("Snowf")
#plt.gcf().autofmt_xdate()
#plt.plot(timestamp_final,ncdf_new["Snowf"][:],'black')

#plt.figure()
#plt.plot(ncdf_new["time"][:],ncdf_new["Wind"][:],'black')
#plt.figure()
#plt.ylabel("Wind")
#plt.gcf().autofmt_xdate()
#plt.plot(timestamp_final,ncdf_new["Wind"][:],'black')

#plt.figure()
#plt.plot(ncdf_new["time"][:],ncdf_new["Wind_DIR"][:],'black')
#plt.figure()
#plt.ylabel("Wind_DIR")
#plt.gcf().autofmt_xdate()
#plt.plot(timestamp_final,ncdf_new["Wind_DIR"][:],'black')

#plt.figure()
#plt.plot(ncdf_new["time"][:],ncdf_new["CO2air"][:],'black')
#plt.figure()
#plt.ylabel("CO2air")
#plt.gcf().autofmt_xdate()
#plt.plot(timestamp_final,ncdf_new["CO2air"][:],'black')

