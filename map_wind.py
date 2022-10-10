#!/usr/bin/env python3
"""
@author: tylunel
Creation : 07/01/2021

Multiple attempts to use MetPy BarbPlot(), but multiple issues.
Best option seems to be the simplest as follows:
"""

#import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
import pandas as pd 


#########################################"""
model = 'irr'
ilevel = 10  #0 is Halo, 1->2m, 2->6.12m, 3->10.49m, 10->49.3m
skip_barbs = 3 # 1/skip_barbs will be printed
save_plot = False
###########################################

simu_folders = {
        'irr': '2.13_irr_2021_22-27/', 
        'std': '1.11_ECOII_2021_ecmwf_22-27/'
         }
father_folder = '/cnrm/surface/lunelt/NO_SAVE/nc_out/'

datafolder = father_folder + simu_folders[model]

# load file, dataset and set parameters
ds1 = xr.open_dataset(datafolder + 'LIAIS.2.SEG36.001.nc', 
                      decode_coords="coordinates",
#                      coordinates=['latitude_u', 'longitude_u'],
#                      grid_mapping=latitude
                      )

fig1 = plt.figure(figsize=(13,9))

#%% WIND SPEED COLORMAP
ws = mpcalc.wind_speed(ds1['UT'] * units.meter_per_second, 
                       ds1['VT'] * units.meter_per_second)
#wd = mpcalc.wind_direction(ds1['UT'] * units.meter_per_second, 
#                           ds1['VT'] * units.meter_per_second)

ws_layer = ws[0, ilevel, :, :]

plt.pcolormesh(ds1.longitude, ds1.latitude, ws_layer,
#               cbar_kwargs={"orientation": "horizontal", "shrink": 0.7}
               cmap='BuPu'
              )
cbar = plt.colorbar()
cbar.set_label('Wind speed [m/s]')
#%% WIND BARBS

X = ds1.longitude
Y = ds1.latitude
U = ds1.UT[0, 10, :,:]
V = ds1.VT[0, 10, :,:]

plt.barbs(X[::skip_barbs, ::skip_barbs], Y[::skip_barbs, ::skip_barbs], 
          U[::skip_barbs, ::skip_barbs], V[::skip_barbs, ::skip_barbs],
          pivot='middle',
          length=6,     #length of barbs
          sizes={
#                 'spacing':1, 
#                 'height':1,
#                 'width':1,
                 'emptybarb':0.01}
          )


#%% IRRIGATED COVERS

pgd = xr.open_dataset(
    '/cnrm/surface/lunelt/NO_SAVE/nc_out/2.01_pgds_irr/' + \
    'PGD_400M_CovCor_v26_ivars.nc')

#from scipy.ndimage.filters import gaussian_filter
#sigma = 0.1     #default is 0.1
#irr_covers = gaussian_filter(pgd.COVER369.data, sigma)
irr_covers = pgd.COVER369.data

plt.contour(pgd.longitude.data, 
          pgd.latitude.data, 
          irr_covers,
          levels=0,   #+1 -> number of contour to plot 
          linestyles='solid',
          linewidths=1.,
          colors='g'
#          colors=['None'],
#          hatches='-'
          )

#%% POINTS SITES
sites = {'cendrosa': {'lat': 41.6925905,
                      'lon': 0.9285671},
         'preixana': {'lat': 41.59373,
                      'lon': 1.07250},
         'elsplans': {'lat': 41.590111,
                      'lon': 1.029363},
        }

for site in sites:
    plt.scatter(sites[site]['lon'],
                sites[site]['lat'],
                color='r',
                s=15        #size of markers
                )
    plt.text(sites[site]['lon']+0.01,
             sites[site]['lat']+0.01, 
             site, 
             fontsize=12)


#%% FIGURE OPTIONS
plot_title = 'Winds at {0}m on {1} for simu {2}'.format(
        np.round(ws_layer.level, decimals=1), 
        pd.to_datetime(ws_layer.time.values).strftime('%Y-%m-%dT%H%M'),
        model)
plt.title(plot_title)


if save_plot:
    filename = (plot_title)
    filename = filename.replace('=', '').replace('(', '').replace(')', '')
    filename = filename.replace(' ', '_').replace(',', '').replace('.', '_')
    plt.savefig('./figures/winds/'+str(filename))
