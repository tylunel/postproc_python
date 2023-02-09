#!/usr/bin/env python3
"""
@author: Tanguy LUNEL
Creation : 07/01/2021

Plot budgets from synchronous files 000.nc from MNH
"""
#import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tools
import xarray as xr
import global_variables as gv


########## Independant parameters ###############
wanted_date = '20210722-2300'

budget_type = 'TH'  #or TK for turbulence, TH for heat, RV for humidity

model = 'std_d2'
budget_range = [-0.004, 0.004]
total_range = [300, 320]

save_plot = True
save_folder = 'figures/budgets_zones/{0}/{1}/'.format(budget_type, model)
##################################################

fig, ax = plt.subplots(ncols=2, 
                       gridspec_kw={'width_ratios': [7, 3]},
                       figsize=(10, 6),)
#fig = plt.figure(figsize=(8, 6))

#%% FIRST GRAPH: BUDGET TERMS

# retrieve and open file
filename_simu = tools.get_simu_filename_000(model, wanted_date)
ds = xr.open_dataset(filename_simu, group="Budgets/{0}".format(budget_type))

# find indices from lat,lon values 
if 'SEG' in filename_simu:  # corresponds to d1, with name format LIAIS.1.SEG{day}.{hour}.000.nc
    dati_arr = pd.date_range(pd.Timestamp(wanted_date).date(),
                             periods=len(ds.time_budget), 
                             freq='30T')
else:  # corresponds to d2, with name format LIAIS.1.S{day}{hour}.000.nc
    dati_arr = pd.date_range(pd.Timestamp(wanted_date),
                             periods=len(ds.time_budget), 
                             freq='30T')
ds['time_budget'] = dati_arr
ds['cart_level'] = gv.layers_height_MNH_LIAISE[:len(ds['cart_level'])]
ds = ds.rename({'time_budget': 'time', 'cart_level': 'level'})

ds_sel = ds.where(ds.time == pd.Timestamp(wanted_date), drop=True)
# add evolution in the budget:
#ds_sel['TOTAL'] = (ds_sel['ENDF'] - ds_sel['INIF']) / 1800
#ds_sel['TOTAL'].attrs['long_name'] = 'total evolution'
try:
    ds_sel['DIFF'] = ds_sel['MAFL'] - ds_sel['ADV']
    ds_sel['DIFF'].attrs['long_name'] = 'molecular diffusion'
except KeyError:
    pass

for key in list(ds_sel.keys()):
    var1d = ds_sel[key]
#        description = ds_sel[key].comment.split(':')[-1]
    description = ds_sel[key].long_name
    if np.all(var1d.data[0]==0):
        print(description + ' is 0')
    else:
        if 'state' in description:
            continue
        if 'correction' in description:
            continue
        if 'pressure' in description:
            continue
        else:
            ax[0].plot(var1d.data[0], 
                     var1d.level.data,
                     label=ds_sel[key].long_name
                     )

#GRAPH ESTHETICS
plot_title = 'Budget of {0} on {1}'.format(budget_type, wanted_date)
fig.suptitle(plot_title)
ax[0].set_ylabel('height AGL (m)')
ax[0].set_xlabel(ds.ADV.units)
ax[0].set_xlim(budget_range)
ax[0].grid()
ax[0].legend(loc='upper right')

#%% SECOND GRAPH: TOTAL
for key in ['INIF', 'ENDF']:
    var1d = ds_sel[key]

    ax[1].plot(var1d.data[0], 
             var1d.level.data,
             label=ds_sel[key].long_name
             )

subplt_title = 'Total {0}'.format(budget_type)
ax[1].title.set_text(subplt_title)
ax[1].set_ylabel('height AGL (m)')
ax[1].set_xlabel(var1d.units)
ax[1].set_xlim(total_range)
ax[1].legend()
ax[1].grid()


#%% MAP BUDGET ZONE
#filename_pgd = gv.simu_folders[model] + 'PGD_2KM.nc'
filename_pgd = tools.get_simu_filename(model, wanted_date)
    
# load dataset, default datetime okay as pgd vars are all the same along time
ds1 = xr.open_dataset(filename_pgd)
varNd = ds1['LAI_ISBA']
#remove single dimensions
var2d = varNd.squeeze()
# remove 999 values, and replace by nan
var2d = var2d.where(~(var2d == 999))
# filter the outliers
#var2d = var2d.where(var2d <= vmax)

left, bottom, width, height = [0.15, 0.65, 0.15, 0.2]
inset = fig.add_axes([left, bottom, width, height])
#inset.contourf(var2d.longitude, var2d.latitude, var2d,
inset.pcolormesh(var2d,
               cmap='RdYlGn',  # 'RdYlGn'
               vmin=0, vmax=4,)
# add sites
points = ['cendrosa', 
          'elsplans', 'irta-corn',
          ]
sites = {key:gv.whole[key] for key in points}
for site in sites:
    j_lat, i_lon = tools.indices_of_lat_lon(ds1, sites[site]['lat'], sites[site]['lon'], )
    plt.scatter(i_lon,
                j_lat,
                color='k',
                s=8        #size of markers
                )
    plt.text(i_lon,
             j_lat,
             site,
             fontsize=7)
# Zoom
inset.set_xlim(ds.min_I_index_in_physical_domain-10, 
               ds.max_I_index_in_physical_domain+10)
inset.set_ylim(ds.min_J_index_in_physical_domain-10, 
               ds.max_J_index_in_physical_domain+10)

# Create a Rectangle patch to indicate the zone for Budget
rec_width = ds.max_I_index_in_physical_domain - ds.min_I_index_in_physical_domain
rec_height = ds.max_J_index_in_physical_domain - ds.min_J_index_in_physical_domain
rect = patches.Rectangle((ds.min_I_index_in_physical_domain, 
                          ds.min_J_index_in_physical_domain), 
                         rec_width, rec_height, linewidth=1, 
                         edgecolor='r', facecolor='none')
# Add the patch to the Axes
inset.add_patch(rect)


#%% Save plot
if save_plot:
    tools.save_figure(plot_title, save_folder)
    
