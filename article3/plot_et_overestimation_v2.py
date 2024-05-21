#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lunelt

This script reads the pickled files.
For the script that makes the pickled files, cf plot_time_series_sfx_v2.py

"""

# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import xarray as xr
import tools
import global_variables as gv
# import pickle


#%% Definition

varname = 'LE_ISBA'  # LE_ISBA or LETR_ISBA

# cleaner if only on site is kept
sites_list = [
    'cendrosa',
    # 'irta-corn',
    ]

vegtype_list = [
    'C4',  # drought tolerant
    'C3',  # drought avoiding
    ]

cphoto_list = [
    'AST', 
    'NON',
    ]

cphoto_name = {
    'AST': 'A-$g_{s}$',
    'NON': 'Jarvis',
    }
vegtype_name = {
    'C4': 'drought-tolerant',
    'C3': 'drought-avoiding',
    }
# site_name = {
#     'irta-corn': 'IRTA corn',
#     'cendrosa': 'La Cendrosa',
#     }

pickle_folder = '/home/lunelt/postproc_python/article3/'

save_plot = True
save_folder = './fig/compa_forcings/'


colordict={
        'C3_AST': 'darkviolet',  # darkorange, purple, darkviolet
        'C3_NON': 'darkviolet', 
        'C4_AST': 'g', 
        'C4_NON': 'g',
        }
linedict={
        'C3_AST': ':', 
        'C3_NON': '--', 
        'C4_AST': ':', 
        'C4_NON': '--',
        }
markerdict={
        'C3_AST': 'o',  # filled circle
        'C3_NON': 'd',  # filled square
        'C4_AST': 'o', 
        'C4_NON': 'd',
        }


#%% Load pickled data
res_dict = {}
for site in sites_list:
    fn = f'diff_percent_df_{site}_{varname}.pkl'
    df = pd.read_pickle(pickle_folder + fn)
    print(f'Opened: {fn}')
    # since ETtr values are null for swi = 0, remove it
    del df['IRRSWI00']
    # load in dict
    res_dict[site] = df


#%% Extract and plot

fig, ax = plt.subplots(1, 4, 
                       sharey=True, 
                       figsize=[10, 6],
                       gridspec_kw={'width_ratios': [1, 4, 2, 1]})

for site in sites_list:
    for vegtype in vegtype_list:
        for cphoto in cphoto_list:
        
            # simu = f'{site[-2:]}_{cphoto}'
            # label = f'{site[-2:]}_{cphoto_name[cphoto]}'
            simu = f'{vegtype}_{cphoto}'
            label = f'{vegtype_name[vegtype]}_{cphoto_name[cphoto]}'
            # label = f'{site_name[site]}_{cphoto_name[cphoto]}'
            
            # get swi of IRRSWI..
            fixed_swi = []
            diff_percent_val_abs = []
            for veruser in res_dict[site].keys():
                if 'IRRSWI' in veruser:
                    fixed_swi.append(float(veruser[-2:])/10)
                    diff_percent_val_abs.append(-res_dict[site][veruser][cphoto][vegtype])
            
            # plot corresponding line
            ax[1].plot(fixed_swi, diff_percent_val_abs,
                   marker = markerdict.get(simu),
                   color = colordict.get(simu),
                   linestyle = linedict.get(simu),
                   label = label)     
            
            # MASTER
            if cphoto == 'AST':
                xpos = 0.1
            else:
                xpos = -0.1
            ax[0].plot(xpos, -res_dict[site]['MASTER'][cphoto][vegtype],
                   marker = markerdict[simu],
                   color = colordict[simu],
                   linestyle = linedict[simu],
                   label = label)
            
            # IRRLAGRIP30
            if vegtype != 'C3':
                ax[2].plot(0, -res_dict[site]['IRRLAGRIP30_THLD05'][cphoto][vegtype],
                       marker = markerdict[simu],
                       color = colordict[simu],
                       linestyle = linedict[simu],
                       label = label)
                ax[2].plot(1, -res_dict[site]['IRRLAGRIP100_THLD05'][cphoto][vegtype],
                       marker = markerdict[simu],
                       color = colordict[simu],
                       linestyle = linedict[simu],
                       label = label)
            
            # FAO56
            ax[3].plot(0, res_dict[site]['FAO56'][cphoto],
                    marker = '*',
                    color = colordict[simu],
                    label = label)
        


#%% Plot aesthetics

# ax[0].set_ylim([-2, 32])

ax[0].set_xticks([0], ['.'],)
ax[0].set_xlabel('NOIRR')
ax[0].grid(axis='y')
ax[0].set_xlim([-1,1])
ax[0].legend(loc='upper left')
ax[0].set_zorder(3)  # put ax[0] to forefront so as to see the legend properly

ax[1].set_xlabel('fixed SWI value')
ax[1].set_xlim([0, 1.3])
ax[1].grid()

ax[2].set_xticks([0, 1], ['30 mm', '100 mm'],)
ax[2].set_xlabel('IRR_THLD')
ax[2].set_xlim([-0.5, 1.5])
ax[2].grid(axis='y')

ax[3].set_xticks([0], ['.'],)
ax[3].set_xlabel('FAO56')
ax[3].set_xlim([-0.5, 0.5])
ax[3].grid(axis='y')

# global
# fig.legend()
#     lines, labels, loc = (0.5, 0), ncol=5)
for ax_i in ax:
    ax_i.hlines(0, -1, 2,
               colors='k')

ax[0].set_ylabel('Difference [%]')
ax[0].set_ylim([-6, 31])
fig.suptitle(gv.sites[site]['longname'])

# ax[3].set_xticks([0], ['MASTER'],)
# ax[3].grid(axis='y')

# TEST for broken yaxis: cf https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
# fig = plt.figure(figsize=(9, 6),
#                  )
# ax1 = fig.add_subplot(2, 4, 1,)
#                       # subplotspec={'width': 1})
# ax2 = fig.add_subplot(2, 4, 5)
# ax3 = fig.add_subplot(1, 4, 2)
# ax4 = fig.add_subplot(1, 4, 3)
# ax5 = fig.add_subplot(1, 4, 4)


#%%
if save_plot:
    plot_title = f'et_overestimation_{site}'
    tools.save_figure(plot_title, save_folder)
