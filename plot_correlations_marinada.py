#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: toni, adapted by tanguy
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.style
mpl.style.use('default')
import numpy as np
import pandas as pd

import warnings
import pickle
warnings.filterwarnings("ignore")

import tools
import global_variables as gv
import datetime
from scipy.stats import linregress


stations_list = ['C7','VQ']  # options are C6, C7, VQ
# C6 = castelnou de seana / C7 = tarrega
# all are west of serra del tallat, except VQ which is in Tarragones
stations_coords = pd.read_csv('./stations_SMC_coords.csv', 
                              sep='\t', decimal=',', 
                              names=['name', 'lat', 'lon', 'altitude'],)

save_plot = True
save_folder = f'/home/lunelt/postproc_python/figures/misc/'

res_dict={}
df_dict = {}
for station in stations_list:
    
    # LOAD FILE of a station
    file = open(gv.global_data_liaise + f'SMC/SMC_6stations_2003-2021/{station}_Merge.csv', 'rb')
    df = pickle.load(file)       
    df['data']=pd.to_datetime(df['data'])
    df = df.set_index('data')
    df['hour']=df.index.hour
    df = df[df.index.month==7]
    
    df = df.assign(minute=df.index.minute).assign(month=df.index.month).assign(year=df.index.year).query('month==7')
    
    # assign marinada wind depending only on wind direction
    wd_range_marinada=[(70,180)]      
    df['sector']=np.where((df.DV10 >= wd_range_marinada[0][0]) & (df.DV10 <= wd_range_marinada[0][1]),'SeaBreeze','noSeaBreeze')
    
    # other DIAG added by Tanguy:
    altitude_station = float(stations_coords[stations_coords['name'] == station]['altitude'])
    df['P_std'] = tools.height_to_pressure_std(altitude_station)
    df['theta'] = tools.potential_temperature_from_temperature(
            df['P_std'], df['T'], reference_pressure=100000)
    df['Q'] = tools.psy_ta_rh(df['T'], df['HR'], df['P_std'])['mixing_ratio']
    df['thetav'] = tools.calc_thetav(df['theta'], df['Q'])
    
    # set datetime
    dt_list = []
    for i, date in enumerate(df.index):
        dt = pd.Timestamp(date)
        dt = dt.replace(hour=df.hour[i], minute=df.minute[i])
        dt_list.append(dt)
    
    # set datetime as index and remove useless columns
    df['datetime'] = dt_list
    df.index = df.datetime
    df = df.drop(columns=['month', 'year', 'minute'])
    
    df_dict[station] = df
    
df_dict[stations_list[0]]['grad_thetav'] = df_dict[stations_list[1]]['thetav'] - df_dict[stations_list[0]]['thetav']


#%%
for station in [stations_list[0],]:
    
    df = df_dict[station]
    
    # Find indices corresponding to a wind veering
    df_reversal = df[df['sector'].eq('SeaBreeze') & df['sector'].shift(1).eq('noSeaBreeze')] 
    
    df_reversal['datime'] = df_reversal.index
    
#    first_reversal_df_1 = first_reversal_df.assign(hour=df_reversal.index.hour).query('hour>=12 and hour<=20')
    df_reversal_day = df_reversal[df_reversal.hour>=12][df_reversal.hour<=20]
    # dfmatches.groupby(dfmatches.index.date).last()
    
#    first_reversal_df = first_reversal_df.groupby(first_reversal_df.index.date).last()
    df_reversal_day_first = df_reversal_day.groupby(df_reversal_day.index.date).first()
    df_reversal_day_first.index = df_reversal_day_first.datime
    
    # Add minutes to hour column by adding 0.5 for 30minutes
    reversal_hour = []
    for i in range(df_reversal_day_first.shape[0]):
        if df_reversal_day_first.index[i].minute == 0:
            reversal_hour.append(str(df_reversal_day_first.index[i].hour)+'.0')
        else:
            reversal_hour.append(str(df_reversal_day_first.index[i].hour)+'.5')
    
    # list of wind veering time
    reversal_hour = [float(x) for x in reversal_hour]
    reversal_time = df_reversal_day_first.index.time

    # create result matrix
    res_df = pd.DataFrame()
    matriu=np.zeros([len(reversal_hour), 7])
        
    for i in range(len(reversal_hour)):
        # keep only one day
        df_filt = df[df.index.date == df_reversal_day_first.index[i].date()]
        
        # look at values at a given hour in the day
        time_morning = datetime.time(11, 0)
        
        hour_mature = reversal_time[i].hour + 1
        time_mature = datetime.time(hour_mature, reversal_time[i].minute)
        
        df_morning_stage = df_filt[df_filt.index.time == time_morning]
        df_mature_stage = df_filt[df_filt.index.time == time_mature]  # TO FIX: hour rounded to hour for now
        
        matriu[i,0] = reversal_hour[i]  # add reversal time as 2nd column
        matriu[i,1] = df_morning_stage['DV10'].values[0]  # add wind direction as 1st column
        matriu[i,2] = df_morning_stage['VV10'].values[0]  # add wind speed as 3rd column
        matriu[i,3] = df_morning_stage['thetav'].values[0]  # add theta_v as 4th column
        matriu[i,4] = df_morning_stage['grad_thetav'].values[0]
        matriu[i,6] = df_morning_stage['P'].values[0]
#        matriu[i,4] = df_mature_stage['grad_thetav'].values[0]
        matriu[i,5] = df_mature_stage['VV10'].values[0]
     
    res_df = pd.DataFrame(matriu, 
                          columns=['reversal_hour', 'DV10', 'VV10',
                                   'thetav','grad_thetav','VV10_mature', 'P'])
    df_dict[station]
    res_dict[station] = res_df



#%% ORIGINAL PLOT  -  wind veering vs wind speed at 10:00
# cf script FiguraJoanVeering in folder other_scripts


#%% NEW PLOTS
# by Tanguy

print('Correlations:')
print(res_df.corr())
corr_table = res_df.corr()


#TODO: discriminate per station

Yvar='P'
Xvar='reversal_hour'

Ylabel = f'grad thetav between {stations_list[0]} and {stations_list[1]} [°C]'
Xlabel = 'wind veering time'
#Xlabel = Xvar
#Ylabel = Yvar

X = res_df[Xvar]
Y = res_df[Yvar]

# wind veering vs theta_v
plt.figure()
plt.scatter(X, Y)
plt.xlabel(Xlabel)
plt.ylabel(Ylabel)

diag_df = pd.concat([X, Y], axis=1)
diag_df = diag_df.dropna()

regres_res = linregress(diag_df[Xvar], diag_df[Yvar], )
print(regres_res)
xval = [diag_df[Xvar].min(), diag_df[Xvar].max()]
yval = [regres_res.slope * i + regres_res.intercept for i in xval]
plt.plot(xval, yval)
R = float('%.3g' % regres_res.rvalue)
plt.text(np.min(xval), np.max(yval), f'R = {R}')

dt1, dt2 = df.index.min().date(), df.index.max().date()
plot_title = f'Relation between {Yvar} and {Xvar} \n between {dt1} and {dt2}'
plt.title(plot_title)
if save_plot:
    tools.save_figure(plot_title, save_folder)


#X1 = np.array(diag_df[Xvar]).reshape((-1,1))
#Y1 = np.array(diag_df[Yvar]).reshape(-1)
#
#from sklearn.linear_model import LinearRegression
#model = LinearRegression
#model.fit(X1, Y1)
#print(f"intercept: {model.intercept_}")
#print(f"slope: {model.coef_}")



## wind speed of marinada vs theta_v
#plt.figure()
#plt.scatter(res_df['thetav'], res_df['VV10_mature'])
#plt.xlabel('thetav [°C]')
#plt.ylabel('wind speed at mature stage')
#
## wind speed of marinada vs theta_v gradient between 
#plt.figure()
#plt.scatter(res_df['grad_thetav'], res_df['VV10_mature'])
#plt.xlabel('grad thetav between C7 and VQ [°C]')
#plt.ylabel('wind speed at mature stage')
    
    
    