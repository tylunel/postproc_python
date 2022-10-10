#!/usr/bin/env python3
"""
@author: lunelt

Representation des différences de répartition du COVER369 (cover correspondant
à 100% de VEGTYPE9 - IRR) dans la version avec irrig dans ECOII.

"""
#import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import tools
#import geopandas as gpd
#import cartopy.crs as ccrs

ds_covcor = xr.open_dataset(
#    '/cnrm/surface/lunelt/NO_SAVE/nc_out/2.01_pgds_irr/PGD_2KM_CovCor_v26_ivars.nc')
    '/cnrm/surface/lunelt/NO_SAVE/nc_out/2.01_pgds_irr/PGD_400M_CovCor_v26_ivars.nc')

ds_sdplm = xr.open_dataset(
#    '/cnrm/surface/lunelt/NO_SAVE/nc_out/2.01_pgds_irr/PGD_2KM.nc')
    '/cnrm/surface/lunelt/NO_SAVE/nc_out/2.01_pgds_irr/PGD_400M.nc')



#%%
#------- A CHANGER --------
titre = 'COVER369_ECOII_pgd2KM'

save_plot = False
#--------------------------

#%% single figure
#
#fig = plt.figure(figsize=(28, 16))
#
#plt.pcolormesh(DS.longitude.data, 
#               DS.latitude.data, 
#               DS.ZS.data, 
#               cmap='seismic')
#
#plt.title(titre)
#plt.savefig(titre + '.png')


#%% with subplots
#fig, axs = plt.subplots(2, figsize=(8,15))
#fig.suptitle(titre)
#
#DS = ds_sdplm
#
#axs[0].pcolormesh(DS.longitude.data, 
#               DS.latitude.data, 
#               DS.ZS.data, 
#               cmap='seismic')
#axs[0].pcolormesh(DS.longitude.data, 
#               DS.latitude.data, 
#               DS.COVER369.data, 
#               cmap='YlGn', alpha=0.5)
#
#DS = ds_covcor
#
#axs[1].pcolormesh(DS.longitude.data, 
#               DS.latitude.data, 
#               DS.ZS.data, 
#               cmap='seismic')
#axs[1].pcolormesh(DS.longitude.data, 
#               DS.latitude.data, 
#               DS.COVER369.data, 
#               cmap='YlGn', alpha=0.5)

#if save_plot:
#    plt.savefig('./figures/' + titre + '.png')

#%%

DS = ds_covcor
df = DS['COVER509'].to_dataframe()

for dtarr in DS:
    print(dtarr)
    if 'COVER4' in str(dtarr):
        print('yes')
        df = pd.concat(df, DS[dtarr].to_dataframe())
  
#%%
#preixana      
lat = 41.59373 
lon = 1.07250

indlat, indlon = tools.indices_of_lat_lon(DS.COVER509, lat, lon)
newlat = DS.COVER509[indlat, indlon].latitude
newlon = DS.COVER509[indlat, indlon].longitude

df_1d = df[(df.latitude == newlat) & (df.longitude == newlon)]
df_1d.replace(0, np.nan, inplace=True)
df_1d.dropna(axis=1, inplace=True)
print(df_1d)






