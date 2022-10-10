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
from tools import indices_of_lat_lon, get_simu_filename

###############################################
model = 'irr'
#ilevel = 2  #0 is Halo, 1->2m, 2->6.12m, 3->10.49m, 10->49.3m
save_plot = True
save_folder = './figures/pgd_maps/'

color_map = 'RdYlGn'    # BuPu, coolwarm, viridis, RdYlGn

var_name = 'LAI_ISBA'   #LAI_ISBA, ZO_ISBA, PATCHP7, ALBNIR_S
##############################################


filename = get_simu_filename(model)

# load dataset, default datetime okay as pgd vars are all the same along time
ds1 = xr.open_dataset(filename)


#%% DATA SELECTION and ZOOM

var2d = ds1[var_name]
# remove 999 values, and replace by nan
var2d = var2d.where(~(var2d == 999))


#%% PLOT OF VAR_NAME
fig1 = plt.figure(figsize=(7,5))

plt.pcolormesh(var2d.longitude, var2d.latitude, var2d,
#               cbar_kwargs={"orientation": "horizontal", "shrink": 0.7}
               cmap=color_map,
               vmin=-1, vmax=4
               )
cbar = plt.colorbar()
cbar.set_label(var2d.long_name)

#%% IRRIGATED COVERS
domain = '400m'
if domain == '400m':
    pgd = xr.open_dataset(
        '/cnrm/surface/lunelt/NO_SAVE/nc_out/2.01_pgds_irr/' + \
        'PGD_400M_CovCor_v26_ivars.nc')
elif domain == '2km':
    pgd = xr.open_dataset(
        '/cnrm/surface/lunelt/NO_SAVE/nc_out/2.01_pgds_irr/' + \
        'PGD_2KM_CovCor_v26_ivars.nc')

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
                s=10        #size of markers
                )
    if site == 'elsplans':
        plt.text(sites[site]['lon']-0.1,
                 sites[site]['lat']-0.03, 
                 site, 
                 fontsize=9)
    else:
        plt.text(sites[site]['lon']+0.01,
                 sites[site]['lat']+0.01, 
                 site, 
                 fontsize=9)


#%% FIGURE OPTIONS and ZOOM
plot_title = '{0} for simu {1}'.format(
        var_name,
        model)
plt.title(plot_title)

plt.ylim([41.4, 42.0])
plt.xlim([0.4, 1.4])


if save_plot:
    filename = (plot_title)
    filename = filename.replace('=', '').replace('(', '').replace(')', '')
    filename = filename.replace(' ', '_').replace(',', '').replace('.', '_')
    plt.savefig(save_folder+str(filename))
