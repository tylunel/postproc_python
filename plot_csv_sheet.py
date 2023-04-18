#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:08:02 2023

@author: lunelt
"""

import pandas as pd
import matplotlib.pyplot as plt

filename = '/home/lunelt/Documents/presentations/23-03-27_Lleida_LIAISE_ET/tableau_compa_forcing.csv'

forcfiles = ['irrswi10']

simus = [
        'cendrosa_AST', 
        'cendrosa_JARVIS', 
        'irta-corn_AST', 
        'irta-corn_JARVIS',
        ]

colordict={
        'cendrosa_AST': 'g', 
        'cendrosa_JARVIS': 'g', 
        'irta-corn_AST': 'b', 
        'irta-corn_JARVIS': 'b',
        }
linedict={
        'cendrosa_AST': ':', 
        'cendrosa_JARVIS': '--', 
        'irta-corn_AST': ':', 
        'irta-corn_JARVIS': '--',
        }
markerdict={
        'cendrosa_AST': 'o',  # filled circle
        'cendrosa_JARVIS': 'd',  # filled square
        'irta-corn_AST': 'o', 
        'irta-corn_JARVIS': 'd',
        }

#df = pd.read_csv(file)

res = {}

with open(filename) as file:
    for line in file:
        elts = line.split(',')
        if 'SITE' in elts:
            site = elts[1]
            res[site] = {}
        elif 'CPHOTO' in elts:
            cphoto = elts[1]
            res[site][cphoto] = {}
        elif elts[0] == '' or elts[0] == 'Nota':
            pass
        else:
            res[site][cphoto][elts[1]] = {
                'std': float(elts[2]), 
                'irrswi10': float(elts[3]),
                'irrlagrip30': float(elts[4]),
                }

res_df = {}
sites = list(res.keys())
cphotos = list(res[sites[0]].keys())



for forcfile in forcfiles:
    for site in sites:
        for cphoto in cphotos:
            res_df[site+'_'+cphoto+'_'+forcfile] = pd.DataFrame(res[site][cphoto]).T
     
            diff_percent = res_df[site+'_'+cphoto+'_'+forcfile]['std']/res_df[site+'_'+cphoto+'_'+forcfile][forcfile] -1
            res_df[site+'_'+cphoto+'_'+forcfile]['std-{0}'.format(forcfile)] = diff_percent * 100
        
#PLOT

fig, ax = plt.subplots(1, 3, 
                       sharey=True, 
                       figsize=[9, 6],
                       gridspec_kw={'width_ratios': [4, 2, 1]})


for forcfile in forcfiles:
    for simu in simus:
        ax[0].plot(
            [0, 0.3, 0.5, 0.6, 0.7, 0.8, 1.05],
            res_df[simu+'_'+forcfile]['std-{0}'.format(forcfile)][:7],
            marker = markerdict[simu],
            color = colordict[simu],
            linestyle = linedict[simu],
    #        label = 'LE_{0}'.format(simu),
            label=simu+'_'+forcfile
            )
        ax[1].plot(
            [1, 2],
            res_df[simu+'_'+forcfile]['std-{0}'.format(forcfile)][7:9],
            marker = markerdict[simu],
            color = colordict[simu],
            linestyle = '',
#            label = 'LE_{0}'.format(simu)
            )
        ax[2].plot(
            res_df[simu+'_'+forcfile]['std-{0}'.format(forcfile)].index[9],
            res_df[simu+'_'+forcfile]['std-{0}'.format(forcfile)][9],
            color = colordict[simu],
            marker = markerdict[simu],
    #        label = 'LE_{0}'.format(simu),
            label=simu+'_'+forcfile
            )

ax[0].legend(loc='lower right')
ax[0].set_xticks(
        [0, 0.3, 0.5, 0.6, 0.7, 0.8, 1.05],
        list(res_df['cendrosa_AST_{0}'.format(forcfiles[0])].index[0:7]),
        rotation=30,
        )
ax[0].set_xticks(
    [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
    minor=True 
    )
ax[0].tick_params(axis='x', which='minor', bottom=True, top=True )
ax[0].tick_params(axis='x', which='major', bottom=True, top=True )
#ax[0].set_ylim([-12,35])

ax[1].set_xticks(
        [1, 2],
        list(res_df['cendrosa_AST_{0}'.format(forcfiles[0])].index[7:9]),
        rotation=30,
        )
ax[1].set_xlim([0,3])

ax[2].tick_params(axis='x',
        labelrotation=30
        )
ax[2].tick_params('y', labelright=True)
ax[2].set_ylabel('Difference [%]')
ax[2].yaxis.set_label_position("right")

for subplt in ax:
    subplt.grid(which='both')
    subplt.axhline(color='k')
    
ax[0].set_ylabel('Difference [%]')
fig.suptitle('Difference of ET between irrigated and standard atmospheric files')

ax2 = ax[0].twiny()
ax2.set_xlim(ax[0].get_xlim())
ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1], )
ax2.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], )
ax2.tick_params('x', direction='in', pad=-15)
ax2.set_xlabel('Soil water index')