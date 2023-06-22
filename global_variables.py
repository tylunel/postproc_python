#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lunelt

Gathers global variables for use in scripts. 
"""

#global_simu_folder = '/cnrm/surface/lunelt/NO_SAVE/nc_out/'
global_simu_folder = '/media/lunelt/7C2EB31F2EB2D0FE/Tanguy/'
global_temp_folder = '/home/lunelt/Data/temp_outputs/'

#global_data_liaise = '/cnrm/surface/lunelt/data_LIAISE/'
global_data_liaise = '/home/lunelt/Data/data_LIAISE/'


simu_folders = {
        'irr_d2': '2.17_irr_d2_21-22_bugfix/',
        'std_d2': '1.17_std_d2_21-22_bugfix/',
        'irr_d2_old': '2.13_irr_d1d2_21-24/', 
        'std_d2_old': '1.11_std_d1d2_21-24/',
        'irr_d1': '2.15_irr_d1_15-30/',
        'std_d1': '1.15_std_d1_15-30/',
#        'lagrip100_d1': '5.15_lagrip100_d1_15-30/',  #param had issue
        'irrlagrip30_d1': '7.20_irrlagrip30_d1_1-30/',
        'temp': '',
         }

format_filename_simu = {            
        'irr_d2':     'LIAIS.1.S????.001dg.nc',
        'std_d2':     'LIAIS.1.S????.001dg.nc',
        'irr_d2_old': 'LIAIS.2.SEG??.0??dg.nc', 
        'std_d2_old': 'LIAIS.2.SEG??.0??dg.nc',
        'irr_d1':     'LIAIS.1.SEG??.0??dg.nc',
        'std_d1':     'LIAIS.1.SEG??.0??dg.nc',
#        'lagrip100_d1': 'LIAIS.1.SEG??.0??.nc',
        'irrlagrip30_d1': 'LIAIS.1.SEG??.0??dg.nc',
        }

sites = {'cendrosa': {'lat': 41.6925905,
                      'lon': 0.9285671,
                      'alt': 240},
         'preixana': {'lat': 41.59373,
                      'lon': 1.07250,
                      'alt': 354},
         'elsplans': {'lat': 41.590111,
                      'lon': 1.029363,
                      'alt': 334},
#         'irta-corn': {
##                       'lon': 0.805333},  # wrong position, but 100% irr zone in model irr_d1
#                         'lat': 41.5922,  # wrong position, but 100% irr zone in model
         'irta-corn': {
                     'lat': 41.619079, # real position, but is not in 100% irr zone in model irr_d1
                     'lon': 0.875333,
                     'alt': 245}, # real position, but is not in 100% irr zone in model
         'irta': {  # =irta-corn
                     'lat': 41.619079, 
                     'lon': 0.875333,
                     'alt': 245},
         'verdu': {'lat': 41.595278,
                   'lon': 1.127222,
                   'alt': 412},
         'ivars-lake': {'lat': 41.682018,
                      'lon': 0.946951,
                      'alt': 230},
         'border_irrig_noirr': {'lat': 41.611898,
                                'lon': 0.999708},
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
         'torredembarra': {'lat': 41.145077,
                           'lon': 1.4073809},
         'ponts': {'lat': 41.91506,
                   'lon': 1.1851},
         'balaguer': {'lat': 41.788303,
                      'lon': 0.81065},
         'borges_blanques': {'lat': 41.520106,
                             'lon': 0.86827},
         'tornabous': {'lat': 41.70097,
                       'lon': 1.054512},
         'claravalls': {'lat': 41.70237,
                        'lon': 1.12543},
         'bellpuig': {'lat': 41.62501,
                      'lon': 1.01115},
         'els_omellons': {'lat': 41.50174,
                          'lon': 0.95993},
         'sant_marti': {'lat': 41.55991,
                        'lon': 1.05504},
         'fonolleres': {'lat': 41.65754,
                        'lon': 1.20261},
         'villobi': {'lat': 41.43266,
                     'lon': 1.04594},
         'conesa': {'lat': 41.51996,
                'lon': 1.29126},
         'calafell': {'lat': 41.19005,
#                      'lon': 1.57218},  # real value, but issue with budget
                      'lon': 1.52552},  # modified value
         'santa_coloma': {'lat': 41.53407,
                          'lon': 1.38441},
        'pi_sol': {'lat': 41.302951,
                   'lon': 1.52552},
        'el_morell': {'lat': 41.191488,
                      'lon': 1.208416},
#        '__': {'lat': ,
#                'lon': },
              }

mountains = {'tossal_baltasana': {'lat': 41.3275,
                                  'lon': 1.00336},
             'puig_formigosa': {'lat': 41.42179,
                                'lon': 1.44177},
             'tossal_gros': {'lat': 41.47857,
                             'lon': 1.12942},
             'tossal_torretes': {'lat': 42.02244,
                                 'lon': 0.93800},
             'moncayo': {'lat': 41.7871,
                         'lon': -1.8396},
             'tres_mojones': {'lat': 40.75887,
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
            'tossal_purunyo': {'lat': 41.30137,
                               'lon': 1.17129},
           'puig_cabdells': {'lat': 41.40621,
                              'lon': 1.31137},
            }

whole = {**sites, **towns, **mountains}

areas_corners = {
    'irrig': ['lleida', 'balaguer', 
              'claravalls', 'borges_blanques'],
    'dry': ['claravalls', 'borges_blanques', 
            'els_omellons', 'sant_marti', 'fonolleres'],
    'slope_west': ['els_omellons', 'sant_marti', 'fonolleres',
                   'santa_coloma', 'tossal_gros', 'villobi'],
    'barbera': ['santa_coloma', 'tossal_gros', 'villobi',
                   'tossal_purunyo', 'puig_cabdells', 'puig_formigosa'],
    'slope_east': ['tossal_purunyo', 'puig_cabdells', 'puig_formigosa',
                   'pi_sol', 'el_morell',],                
    'coast': ['pi_sol', 'el_morell',
              'tarragona', 'calafell', ],
              }

field_capa = {'cendrosa': {1: 0.28, 2: 0.18, 3: 0.23},  # To determine via plot on August
              'preixana': {1: 0.25, 2: 0.30, 3: 0.187}, # To determine via plot on May
              'irta-corn': {1: 0.39, 2:  0.39, 3: 0.33, 4: 0.34, 5: 0.38},  #estimated at 20cm
              'elsplans': {10: 0.30, 20: 0.30, 30: 0.30, 40: 0.30},  #arbitrary, inspired of preixana
              }       

wilt_pt = {'cendrosa': {1: 0.141, 2: 0.07, 3: 0.125} ,  # To determine via plot on August
           'preixana': {1: 0.065, 2: 0.135, 3: 0.115}, # To determine via plot on May
           'irta-corn': {1: 0.05, 2:  0.05, 3: 0.05, 4: 0.05, 5: 0.05},  #estimated at 20cm
           'elsplans': {10: 0.03, 20: 0.11, 30: 0.24, 40: 0.17},  #estimated on july only (pb)
           }   

barb_size_increments = {
        'very_weak_winds': {'half':0.5, 'full':1, 'flag':5},
        'weak_winds': {'half':1, 'full':2, 'flag':10},
        'standard': {'half':2.57, 'full':5.14, 'flag':25.7},
        'm_per_s': {'half':5, 'full':10, 'flag':50},
        'pgf': {'half':0.0005, 'full':0.001, 'flag':0.005},  # in m/s²
        'pgf_weak': {'half':0.000025, 'full':0.00005, 'flag':0.00025},  # in m/s²
        }

barb_size_description = {
        'very_weak_winds': "barb increments: half=0.5m/s, full=1m/s, flag=5m/s",
        'weak_winds': "barb increments: half=1m/s, full=2m/s, flag=10m/s",
        'standard': "barb increments: half=5kt=2.57m/s, full=10kt=5.14m/s, flag=50kt=25.7m/s",
        'm_per_s': "barb increments: half=5m/s, full=10m/s, flag=50m/s",
        'm_per_s_detailled': "barb increments: dot < 2.5m/s < half barb < 7.5m/s < full < 12.5m/s, flag=50kt=25.7m/s",
        'pgf': "barb increments: half=0.5mm/s², full=1mm/s², flag=5mm/s²",
        'pgf_weak': "barb increments: half=0.025mm/s², full=0.05mm/s², flag=0.25mm/s²",
        }

zoom_domain_prop = {
    'liaise': {
        'skip_barbs': 1, # 1/skip_barbs will be printed
        'barb_length': 5.5,
        'lat_range': [41.45, 41.8],
        'lon_range': [0.7, 1.2],
        'figsize': (9,7),
        },
   'urgell': {
        'skip_barbs': 2, # 1/skip_barbs will be printed
        'barb_length': 4.5,
        'lat_range': [41.1, 42.1],
        'lon_range': [0.2, 1.7],
        'figsize': (11,9),
        },
    'urgell-paper': {
        'skip_barbs': 6,
        'barb_length': 4.5,
        'lat_range': [41.37, 41.92],
        'lon_range': [0.6, 1.4],
        'figsize': (9,7),
        },
    'd2': {
        'skip_barbs': 3,
        'barb_length': 4.5,
        'lat_range': [40.8106, 42.4328],
        'lon_range': [-0.6666, 1.9364],
        'figsize': (11,9),
        },
    'marinada': {
        'skip_barbs': 2,
        'barb_length': 4.5,
        'lat_range': [41.0, 42],
        'lon_range': [0.6, 1.6],
        'figsize': (9,9),
        },
    None: {
        'skip_barbs': 8,
        'barb_length': 4.5,
        'lat_range': [None, None],
        'lon_range': [None, None],
        'figsize': (13,7),
        },
#    skip_barbs = 8 # 1/skip_barbs will be printed
#    barb_length = 4.5
#    if domain_nb == 1:
#        figsize=(13,7)
#    elif domain_nb == 2:
#        figsize=(10,7)
    }

layers_depth_DIF = [-0.01, -0.04, -0.1, -0.2, -0.4, -0.6, 
                    -0.8, -1, -1.5, -2, -3, -5, -8, -12]
#version dict:
#{1: -0.01, 2: -0.04, 3: -0.1, 4: -0.2, 5: -0.4, 6: -0.6, 
# 7: -0.8, 8: -1, 9: -1.5, 10: -2, 11: -3, 12: -5, 13: -8, 14: -12}

layers_height_MNH_LIAISE = [
    -2.0,   2.0,    6.12,   10.48,  15.11,      # 0, 1, 2, 3, 4
     20.02, 25.22,  30.74,  36.58,  42.78,      # 5, 6, 7, 8, 9
     49.34, 56.30,  63.68,  71.50,  79.79,      # 10, 11, 12, 13, 14,
     88.58, 97.89,  107.77, 118.23, 129.33,     # 15, 16, 17, 18, 19,
     141.09,    153.55,     166.77,     180.77,     195.62,  # 20,21,22,23,24,
     211.36,    228.04,     245.72,     264.46,     284.33,  # 25,26,27,28,29,
     305.39,    327.72,     351.38,     376.46,     403.05,  # 30, ...,
     431.23,    461.11,     492.77,     526.34,     561.92,  # 35, ...,
     599.64,    639.61,     681.99,     726.91,     774.53,  # 40, ...,
     825.00,    878.50,     935.21,     995.32,     1059.04,
     1126.58,   1198.18,    1274.07,    1354.52,    1439.79, # 50, ...,
     1530.18,   1625.99,    1727.55,    1835.20,    1949.31,
     2070.27,   2198.48,    2336.70,    2488.05,    2653.78, # 60, ...,
     2835.25,   3033.96,    3251.55,    3489.81,    3750.70,
     4036.38,   4349.20,    4691.74,    5066.82,    5477.53, # 70, ...,
     5927.25,   6419.71,    6958.94,    7549.40,    8195.96,
     8903.94,   9679.17,    10528.06,   11457.58,   12475.42,# 80, ...,
     13589.95,  14810.35,   16118.23,   17458.23,   18798.23
     ]

