#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021

Last modifications
"""
#import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import global_variables as gv
import pandas as pd
import tools
import metpy.calc as mcalc


##############################

site = 'elsplans'

wanted_datetime = '2021-07-16T00-00-00'
#wanted_month = '202107'

varname = 'UWE'
# CN2: is refractive_index_structure_coefficient
# UWE: Zonal component of flow, VSN Meridional component of flow
# WVT: Vertical component of flow
# WS, WD: Wind Speed and Wind Direction

color_map = 'coolwarm'

if varname == 'WVT':
    vmin = -2
    vmax = 2
elif varname in ['UWE', 'VSN']:
    vmin = -10
    vmax = 10
elif varname == 'WS':
    vmin = 0
    vmax = 10
else:
    vmin = None
    vmax = None

figsize = (11, 6) #small for presentation: (6,6), big: (15,9)

##############################

wanted_month = str(pd.Timestamp(wanted_datetime).month).zfill(2)  # format with 2 figures

begin_datetime = pd.Timestamp(wanted_datetime)

end_datetime = begin_datetime + pd.Timedelta(1, 'd')
#begin_datetime = np.datetime64('2021-07-15T00:00:00')
#end_datetime = np.datetime64('2021-07-23T00:00:00')

# load dataset and set parameters
if site == 'elsplans':
    ds = xr.open_dataset(
        gv.global_data_liaise + '/elsplans/UHF_low/' + \
        f'LIAISE_ELS-PLANS_LAERO_UHFWindProfiler-LowMode-2MIN_L2_2021{wanted_month}_V1.nc')
elif site == 'cendrosa':

    list_ds_temp_uhf = []
    list_ds_temp_wcube = []
    
    for day in np.arange(1, 30):
        day_frmt = str(day).zfill(2)
        # dataorigin == 'uhf':
        ds_day_uhf = xr.open_dataset(
            gv.global_data_liaise + '/cendrosa/UHF_high/' + \
            f'MF-CNRM-Toulouse_UHF-RADAR_L2B-LM-Hourly-Mean_2021-{wanted_month}-{day_frmt}T00-00-00_1D_V2-10.nc'
            )
        list_ds_temp_uhf.append(ds_day_uhf)
            
        # dataorigin == 'windcube':
        ds_day_wcube = xr.open_dataset(
            gv.global_data_liaise + '/cendrosa/lidar_windcube/' + \
            f'LIAISE_LA-CENDROSA_CNRM_LIDARwindcube-WIND_L2_2021{wanted_month}{day_frmt}_V1.nc')
        list_ds_temp_wcube.append(ds_day_wcube)
    
    # merge the temporary dataset
    ds_uhf = xr.merge(list_ds_temp_uhf)
    ds_wcube = xr.merge(list_ds_temp_wcube)
    
    # Merge the Windcube and UHF data
    # dataorigin == 'windcube':
    ds_wcube = ds_wcube.drop_dims(['level'])
    ds_wcube['level'] = xr.DataArray(ds_wcube.ff_class.data, 
        coords={'level': ds_wcube.ff_class.data,})
    # unify dimension coordinate of all variable
    for var in ds_wcube:
        ds_wcube[var] = ds_wcube[var].swap_dims({f'{var}_class': 'level'})
    # drop old level coordinates
    ds_wcube = ds_wcube.drop_dims(['ff_class', 'dd_class', 'ffmin_class', 'ffmax_class', 
                       'ffstd_class', 'data_availabily_class', 
                       'CNR_class', 'CNRmin_class'])
    # rename variables
    ds_wcube = ds_wcube.rename({'ff': 'WS', 'dd': 'WD'})
    wind_components = mcalc.wind_components(ds_wcube['WS'], ds_wcube['WD'])
    # remove pint.quantities
    ds_wcube['UWE'] = xr.DataArray(wind_components[0].values,
             coords={'time': ds_wcube.time, 'level': ds_wcube.level, })
    ds_wcube['VSN'] = xr.DataArray(wind_components[1].values,
             coords={'time': ds_wcube.time, 'level': ds_wcube.level, })
    
    # dataorigin == 'uhf':
    ds_uhf['WS'], ds_uhf['WD'] = tools.calc_ws_wd(ds_uhf['UWE'], ds_uhf['VSN'])
    
    ds = xr.merge([ds_uhf, ds_wcube])
else:
    raise KeyError("No radar data for this site")





# convert level asl to agl
alti_site = gv.sites[site]['alt']

if site == 'elsplans':
    ds = ds.rename({'level': 'level_asl'})
    ds['level_agl'] = ds.level_asl-alti_site
    ds = ds.set_coords(['level_agl'])
if site == 'cendrosa':
    ds = ds.rename({'level': 'level_agl'})

    

# select data
datarr = ds[varname]

# select time of interest
#datarr = datarr.where(datarr.time > begin_datetime)
#datarr = datarr.where(datarr.time < end_datetime)

#%% PLOT


plot_title = f'{varname} from UHF radar at {site}'

plt.figure(figsize=figsize)
plt.title(plot_title)

plt.pcolormesh(
        datarr.time,
        datarr.level_agl,
        datarr.data.transpose(),
        cmap=color_map, #default is viridis
        vmin=vmin, vmax=vmax,
        )

#plt.xlim([begin_datetime, end_datetime])
plt.ylim([0, 1500])
plt.colorbar()



