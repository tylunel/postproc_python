#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lunelt

This script reads the pickled files.
For the script that makes the pickled files, cf plot_time_series_sfx_v2.py

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import tools
import global_variables as gv
import pickle



sites_list = [
    # 'irta-corn_C3',
    # 'irta-corn_C4', 
    'cendrosa_C3',
    'cendrosa_C4',
    ]

# sites_list = [
#     'cendrosa',
#     # 'irta-corn',
#     ]
vegtype_list = [
    # 'C4',
    # 'C3',
    '',
    ]

cphoto_list = [
    'AST', 
    'NON',
    ]

cphoto_name = {
    'AST': 'A-$g_{s}$',
    'NON': 'Jarvis',
    }
# site_name = {
#     'irta-corn': 'C4_IRTA',
#     'cendrosa': 'C3_Cendrosa',
#     }

pickle_folder = '/home/lunelt/postproc_python/article3/'

save_plot = False
save_folder = './fig/compa_forcings/'


colordict={
        'C3_AST': 'g', 
        'C3_NON': 'g', 
        'C4_AST': 'b', 
        'C4_NON': 'b',
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


fig, ax = plt.subplots(1, 4, 
                       sharey=True, 
                       figsize=[9, 6],
                       gridspec_kw={'width_ratios': [1, 4, 1, 1]})

# Load pickled data
res_dict = {}
for site in sites_list:
    fn = f'diff_percent_df_{site}.pkl'
    df = pd.read_pickle(pickle_folder + fn)
    print(f'Opened: {fn}')
    # since ETtr values are null for swi = 0, remove it
    del df['IRRSWI00']
    # load in dict
    res_dict[site] = df


for site in sites_list:
    for vegtype in vegtype_list:
        for cphoto in cphoto_list:
        
            simu = f'{site[-2:]}_{cphoto}'
            # simu = f'{vegtype}_{cphoto}'
            
            # label = f'{site_name[site]}_{cphoto_name[cphoto]}'
            label = f'{site[-2:]}_{cphoto_name[cphoto]}'
            
            # get swi of IRRSWI..
            fixed_swi = []
            diff_percent_val_abs = []
            for key in res_dict[site].keys():
                if 'IRRSWI' in key:
                    fixed_swi.append(float(key[-2:])/10)
                    diff_percent_val_abs.append(-res_dict[site][key][cphoto])
            
            # plot corresponding line
            ax[1].plot(fixed_swi, diff_percent_val_abs,
                   marker = markerdict[simu],
                   color = colordict[simu],
                   linestyle = linedict[simu],
                   label = label)     
            
            # MASTER
            if cphoto == 'AST':
                xpos = 0.1
            else:
                xpos = -0.1
            ax[0].plot(xpos, -res_dict[site]['MASTER'][cphoto],
                   marker = markerdict[simu],
                   color = colordict[simu],
                   linestyle = linedict[simu],
                   label = label)
            
            # IRRLAGRIP30
            ax[2].plot(0, -res_dict[site]['IRRLAGRIP30'][cphoto],
                   marker = markerdict[simu],
                   color = colordict[simu],
                   linestyle = linedict[simu],
                   label = label)
            
            # FAO56
            ax[3].plot(0, res_dict[site]['FAO56'][cphoto],
                    marker = '*',
                    color = colordict[simu],
                    label = label)
        


#%%

# ax[0].set_ylim([-2, 32])

ax[0].set_xticks([0], ['NWP'],)
ax[0].grid(axis='y')
ax[0].set_xlim([-1,1])

ax[1].legend()
ax[1].set_xlabel('fixed SWI value')
ax[1].grid()

ax[2].set_xticks([0], ['IRR_THLD05'],)
ax[2].grid(axis='y')

ax[3].set_xticks([0], ['FAO56'],)
ax[3].grid(axis='y')

# global
ax[0].set_ylabel('Difference [%]')
# fig.suptitle('Difference of ET between irrigated and standard atmospheric files')

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
    plot_title = 'et_overestimation'
    tools.save_figure(plot_title, save_folder)
