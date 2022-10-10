#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021

Last modifications
"""
#import os
import numpy as np
#import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr

#%% load dataset and set parameters
ds = xr.open_dataset(
    '/cnrm/surface/lunelt/data_LIAISE/elsplans/UHF_low/' + \
    'LIAISE_ELS-PLANS_LAERO_UHFWindProfiler-LowMode-2MIN_L2_202107_V1.nc')
datetime = ds.time.values

#get data
vari = 'CN2'
datarr = ds[vari]
print('Variable considered: ' + datarr.long_name)

#date selection
#with index for now...
#index_begin = 0
#index_end = 400

#find the indices corresponding to wanted dates
begin_datetime = np.datetime64('2021-07-22T00:00:00')
end_datetime = np.datetime64('2021-07-23T00:00:00')

dist2beg_dat = np.abs(datarr.time.data - begin_datetime)  #distance to begin datetime
index_begin = np.argwhere(dist2beg_dat <= dist2beg_dat.min())[0][0]

dist2end_dat = np.abs(datarr.time.data - end_datetime)
index_end = np.argwhere(dist2end_dat <= dist2end_dat.min())[0][0]

#transpose and reverse array for plt.imshow
#datarr_select = datarr.data[index_begin:index_end]
#datarr_reshap = np.flip(datarr_select.transpose(), axis=0)
#plt.imshow(datarr_reshap)

plt.pcolormesh(
        datarr.time.data[index_begin:index_end],
        datarr.level.data,
        datarr.data[index_begin:index_end].transpose(),
        cmap='viridis', #default is viridis
#        vmin=0.2e-14, 
        vmax=1.5e-14,
        )
plt.colorbar()



