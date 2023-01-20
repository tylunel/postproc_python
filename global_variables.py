#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lunelt

Gathers global variables for use in scripts. 
"""

global_simu_folder = '/cnrm/surface/lunelt/NO_SAVE/nc_out/'

simu_folders = {
        'irr_d2': '2.16_irr_d2_21-22/',
        'std_d2': '1.16_std_d2_21-22/',
        'irr_d2_old': '2.13_irr_d1d2_21-24/', 
        'std_d2_old': '1.11_std_d1d2_21-24/',
        'irr_d1': '2.15_irr_d1_15-30/',
        'std_d1': '1.15_std_d1_15-30/'
         }

format_filename_simu = {            
        'irr_d2':     'LIAIS.1.S????.001dg.nc',
        'std_d2':     'LIAIS.1.S????.001dg.nc',
        'irr_d2_old': 'LIAIS.2.SEG??.0??dg.nc', 
        'std_d2_old': 'LIAIS.2.SEG??.0??dg.nc',
        'irr_d1':     'LIAIS.1.SEG??.0??dg.nc',
        'std_d1':     'LIAIS.1.SEG??.0??dg.nc'
        }

sites = {'cendrosa': {'lat': 41.6925905,
                      'lon': 0.9285671},
         'preixana': {'lat': 41.59373,
                      'lon': 1.07250},
         'elsplans': {'lat': 41.590111,
                      'lon': 1.029363},
         'irta-corn': {
#                         'lat': 41.619079, # real position, but is not in 100% irr zone in model
#                         'lon': 0.875333,} # real position, but is not in 100% irr zone in model
#                       'lon': 0.805333},  # wrong position, but 100% irr zone in model
                         'lat': 41.5922,  # wrong position, but 100% irr zone in model
                         'lon': 0.8585},  # wrong position, but 100% irr zone in model
        }
         
towns = {'arbeca': {'lat': 41.54236,
                    'lon': 0.9232},
         'verdu': {'lat': 41.6107,
                    'lon': 1.1428},
         'tarragona': {'lat': 41.1188,
                       'lon': 1.2456},
         'tarrega': {'lat': 41.6502,
                     'lon': 1.1389},
         'barcelona': {'lat': 41.4025,
                       'lon': 1.1870},
         'lleida': {'lat': 41.6186,
                    'lon': 0.6257},
         'zaragoza': {'lat': 41.6547,
                      'lon': -0.8784},
              }

mountains = {'tossal baltasana': {'lat': 41.3275,
                                  'lon': 1.00336},
             'puig formigosa': {'lat': 41.42179,
                                'lon': 1.44177},
             'tossal gros': {'lat': 41.47857,
                             'lon': 1.12942},
             'tossal torretes': {'lat': 42.02244,
                                 'lon': 0.93800},
             'moncayo': {'lat': 41.7871,
                         'lon': -1.8396},
             'tres mojones': {'lat': 40.75887,
                              'lon': -0.63924},
             'guara': {'lat': 42.28865,
                       'lon': -0.22992},
             'caro': {'lat': 40.80332,
                      'lon': 0.34325},
             'montserrat': {'lat': 41.6052,
                            'lon': 1.81182},
             'joar': {'lat': 42.6345,
                      'lon': -2.34898},
             'coll_lilla': {'lat': 41.34072,
                            'lon': 1.22014},
             'puig_pelat': {'lat': 41.26571,
                            'lon': 1.05502},           
            }

whole = {**sites, **towns, **mountains}

field_capa = {'cendrosa': {1: 0.28, 2: 0.18, 3: 0.23},  # To determine via plot on August
              'preixana': {1: 0.25, 2: 0.30, 3: 0.187}, # To determine via plot on May
              'irta-corn': {1: 0.39, 2:  0.39, 3: 0.33, 4: 0.34, 5: 0.38}  #estimated at 20cm
              }       

wilt_pt = {'cendrosa': {1: 0.141, 2: 0.07, 3: 0.125} ,  # To determine via plot on August
           'preixana': {1: 0.065, 2: 0.135, 3: 0.115}, # To determine via plot on May
           'irta-corn': {1: 0.05, 2:  0.05, 3: 0.05, 4: 0.05, 5: 0.05}  #estimated at 20cm
           }   




