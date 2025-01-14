#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lunelt

Gathers global variables for use in scripts. 
"""

#%% ------- Some global folder paths -----------
# global_simu_folder = '/cnrm/surface/lunelt/NO_SAVE/nc_out/'
#global_simu_folder = '/media/lunelt/7C2EB31F2EB2D0FE/Tanguy/'
#global_temp_folder = '/home/lunelt/Data/temp_outputs/'
global_simu_folder = '/home/lunelt/Data/mnh_run/'

#global_data_liaise = '/cnrm/surface/lunelt/data_LIAISE/'
global_data_liaise = '/home/lunelt/Data/data_LIAISE/'
global_data_nemo = '/home/lunelt/Data/data_NEMO/'


simu_folders = {
        'irr_d2': '2.17_irr_d2_21-22_bugfix/',
        'irrswi1_d1': '8.16_irrswi1_d1_15-30_bugfix/',
        '1_0105_1km': '1_0105_1km/',
        'temp': '',
         }

format_filename_simu_wildcards = {            
        'irr_d2': 'LIAIS.1.S????.001dg.nc',
        'irr_d2_old_old': 'LIAIS.2.SEG??.0??dg.nc',
        'irrswi1_d1': 'LIAIS.1.SEG??.0??dg.nc',
        'irrswi1_d1_16_10min': 'LIAIS.1.SEG??.0??.nc',
        }

format_filename_simu = {  
        'std_d2': 'LIAIS.1.S{seg_nb}{output_nb_2f}.001{file_suffix}.nc',
        'irrswi1_d1': 'LIAIS.1.SEG{seg_nb}{out_suffix}.{output_nb_3f}{file_suffix}.nc',
        '1_0105_1km': 'PLANI.1.SEG{seg_nb}{out_suffix}.{output_nb_3f}{file_suffix}.nc',
        }
# Apply .format() after corresponding string
# ex: format_filename_simu_new['std_d2'].format(seg_nb=22, output_nb_2f=12, output_nb_2f='012', file_suffix='dg')
# format_filename_simu_wildcards = {key:val.format(seg_nb='??', output_nb_2f='??', output_nb_2f='012', file_suffix='dg') if ...}

filename_simu_name_rules = {
    '1_0105_1km': {
        'backup': {
            'seg_nb': '01',
            'out_suffix': '',
            'output_nb': 'HH',  # ex: 12h -> 012 or 12
            'file_suffix': '',
            },
        'output': {
            'seg_nb': '01',
            'out_suffix': '.OUT',
            'output_nb': 'Hx6',  #ex: 01h10m = 1.166h -> x6 = 007
            'file_suffix': '',}
            },
        'diachronic': {
            'seg_nb': '01',
            'output_nb': '000',
            'file_suffix': '',
            'out_suffix': ''}
    }

#%%------- Sites, location, towns, areas --------
sites = {
    'planier': {
        'lat': 43.19885,  # rounded
        'lon': 5.22981,   # rounded
        'alt': 0,
        'longname': 'Ile du Planier',
        'acronym': 'Pl.'},
    'marignane': {
        'lat': 43.43693,  # rounded
        'lon': 5.21777,   # rounded
        'alt': 10,
        'longname': 'Marignane',
        'acronym': 'Ma.'},
    }
         
towns = {
    'marseille': {
        'lat': 43.29444,  #= vieux port
        'lon': 5.36879,
        'longname': 'Marseille',},
    'marseille_old_port': {
        'lat': 43.29444,
        'lon': 5.36879},
    'nimes': {
        'lat': 43.8534,
        'lon': 4.4151},
#    '__': {'lat': ,
#           'lon': },
        }

mountains = {
    'marseilleveyre': {
        'lat': 42.22291,
        'lon': 5.37200},
            }

whole = {**sites, **towns, **mountains,}

areas_corners = {
    'sea': ['tarragona', 'calafell', 
            'calafell_offshore', 'tarragona_offshore', ],
              }

#%% ------- Some model matplotlib properties -----------

longnamedict = {
    'irrswi1_d1': 'FC_IRR',
    '1_0105_1km': 'MNH_1km'
    }

colordict = {
    'irr_d2': 'g',
    'obs': 'k'}

styledict = {
    'irr_d2': '--',
    'obs': '-'}

output_freq_dict = {
    'irrswi1_d1': 30,
    }

#%%------- Barbs properties dict --------
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
    'd2': {
        'skip_barbs': 3,
        'barb_length': 4.5,
        'lat_range': [40.8106, 42.4328],
        'lon_range': [-0.6666, 1.9364],
        'figsize': (11,9),
        },
    None: {
        'skip_barbs': 8,
        'barb_length': 4.5,
        'lat_range': [None, None],
        'lon_range': [None, None],
        'figsize': None,
        },
#    skip_barbs = 8 # 1/skip_barbs will be printed
#    barb_length = 4.5
#    if domain_nb == 1:
#        figsize=(13,7)
#    elif domain_nb == 2:
#        figsize=(10,7)
    }


#%%------- Observation data variables --------
    
# Metadata of Lidar at planier
dict_lidar_metadata ={
    'ws': 'Wind speed [m s$^{-1}$]',
    'wd': 'Wind direction [°]',
    'tke': 'global (3D) TKE [m² s$^{-2}$]',
    'tke_along': '1D (along) TKE [m² s$^{-2}$]',
    'tke_cross': '1D (cross) TKE [m² s$^{-2}$]',
    'tke_verti': '1D (verti) TKE [m² s$^{-2}$]',
    'tke_bud_prod': 'verti TKE production [m² s$^{-3}$]',
    'tke_bud_diss': 'TKE dissipation [m² s$^{-3}$]',
    'tke_bud_diff_trans': 'TKE diffusive transport [m² s$^{-3}$]',
    }

# Metadata of Meteo-France Radiosoundings
dict_rs_mf_metadata ={
    'ALTI': 'altitude [m]',
    'P': 'pressure [hpa]', 
    'DD': 'wind direction [°]', 
    'FF': 'wind speed [m/s]', 
    'T': 'air temperature [°C]',
    'TD': 'dew point temperature [°C]', 
    'HU': 'relative humidity [%]',
    'RV': 'rapport de mélange [g/kg]'
    }

# Metadata of Meteo-France in-situ stations
dict_station_mf_metadata = {
    "station": {
        "Indicatif": "13054001",
        "Nom": "MARIGNANE",
        "Altitude": "9 mètres",
        "Coordonnées": "lat : 43°26'16\"N - lon : 5°12'58\"E",
        "Coordonnées lambert": "X : 8334 hm - Y : 18304 hm",
        "Producteurs": "De 2023 a 2024 : METEO-FRANCE"
    },
    "variables": {
        "RR1": {"label": "HAUTEUR DE PRECIPITATIONS HORAIRE", "unit": "MILLIMETRES ET 1/10"},
        "T": {"label": "TEMPERATURE SOUS ABRI HORAIRE", "unit": "DEG C ET 1/10"},
        "TD": {"label": "TEMPERATURE DU POINT DE ROSEE HORAIRE", "unit": "DEG C ET 1/10"},
        "PSTAT": {"label": "PRESSION STATION HORAIRE", "unit": "HPA ET 1/10"},
        "PMER": {"label": "PRESSION MER HORAIRE", "unit": "HPA ET 1/10"},
        "FF": {"label": "VITESSE DU VENT HORAIRE", "unit": "M/S ET 1/10"},
        "DD": {"label": "DIRECTION DU VENT A 10 M HORAIRE", "unit": "ROSE DE 360"},
        "FXI": {"label": "VITESSE DU VENT INSTANTANE MAXI HORAIRE", "unit": "M/S ET 1/10"},
        "FXY": {"label": "VITESSE DU VENT MOYEN SUR 10 MN MAXI HORAIRE", "unit": "M/S ET 1/10"},
        "DXY": {"label": "DIRECTION DU VENT MOYEN SUR 10 MN MAXIMAL HORAIRE", "unit": "ROSE DE 360"},
        "FXI3S": {"label": "VITESSE DU VENT INSTANTANÉ SUR 3 SECONDES, MAXI DANS L’HEURE", "unit": "M/S ET 1/10"},
        "DXI3S": {"label": "DIRECTION DU VENT INSTANTANÉ SUR 3 SECONDES", "unit": "ROSE DE 360"},
        "U": {"label": "HUMIDITE RELATIVE HORAIRE", "unit": "%"},
        "INS": {"label": "DUREE D'INSOLATION HORAIRE", "unit": "MINUTES"},
        "GLO": {"label": "RAYONNEMENT GLOBAL HORAIRE", "unit": "JOULES/CM2"},
        "DIR": {"label": "RAYONNEMENT DIRECT HORAIRE", "unit": "JOULES/CM2"},
        "DIF": {"label": "RAYONNEMENT DIFFUS HORAIRE", "unit": "JOULES/CM2"},
        "N": {"label": "NEBULOSITE TOTALE HORAIRE", "unit": "OCTAS"},
        "NBAS": {"label": "NEBUL. DE LA COUCHE NUAG. PRINCIPALE LA PLUS BASSE HORAIRE", "unit": "OCTAS"},
        "CL": {"label": "CODE SYNOP NUAGES BAS HORAIRE", "unit": "CODE METEO"},
        "CM": {"label": "CODE SYNOP NUAGES MOYEN HORAIRE", "unit": "CODE METEO"},
        "CH": {"label": "CODE SYNOP NUAGES ELEVE HORAIRE", "unit": "CODE METEO"},
        "N1": {"label": "NEBULOSITE DE LA 1ERE COUCHE NUAGEUSE", "unit": "OCTAS"},
        "C1": {"label": "TYPE DE LA 1ERE COUCHE NUAGEUSE", "unit": "CODE METEO"},
        "B1": {"label": "BASE DE LA 1ERE COUCHE NUAGEUSE", "unit": "METRES"},
        "N2": {"label": "NEBULOSITE DE LA 2EME COUCHE NUAGEUSE", "unit": "OCTAS"},
        "C2": {"label": "TYPE DE LA 2EME COUCHE NUAGEUSE", "unit": "CODE METEO"},
        "B2": {"label": "BASE DE LA 2EME COUCHE NUAGEUSE", "unit": "METRES"},
        "N3": {"label": "NEBULOSITE DE LA 3EME COUCHE NUAGEUSE", "unit": "OCTAS"},
        "B3": {"label": "BASE DE LA 3EME COUCHE NUAGEUSE", "unit": "METRES"},
        "C3": {"label": "TYPE DE LA 3EME COUCHE NUAGEUSE", "unit": "CODE METEO"},
        "N4": {"label": "NEBULOSITE DE LA 4EME COUCHE NUAGEUSE", "unit": "OCTAS"},
        "C4": {"label": "TYPE DE LA 4EME COUCHE NUAGEUSE", "unit": "CODE METEO"},
        "B4": {"label": "BASE DE LA 4EME COUCHE NUAGEUSE", "unit": "METRES"},
        "TMER": {"label": "TEMPERATURE DE LA MER", "unit": "DEG C ET 1/10"},
        "VVMER": {"label": "VISIBILITE VERS LA MER", "unit": "CODE METEO"},
        "ETATMER": {"label": "ETAT DE LA MER HORAIRE", "unit": "CODE METEO"}
    }
}


#%%------- Models properties --------

variable_correspondance_mnh_arome = {  # key is mnh, value is arome
    'TEMP': 't',
    'THT': 'theta',  # diag in arome
    'RVT': 'q',
    'TKET': 'tke',
    'UT': 'u',
    'VT': 'v',
    'WT': 'w',
    'WS': 'ws', # diag in both
    'WD': 'wd', # diag in both
    }

variable_correspondance_arome_mnh = {
    v: k for k, v in variable_correspondance_mnh_arome.items()}


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

# grid not true, depends on pressure    
level_height_AROME_mean = {
    90: 5.0,    89: 16.8,   88: 32.0,
    87: 50.7,   86: 72.6,   85: 97.9,
    84: 126.4,  83: 158.0,  82: 192.7,
    81: 230.4,  80: 271.0,  79: 314.5,
    78: 360.8,  77: 409.8,  76: 461.5,
    75: 515.9,  74: 573.1,  73: 633.2,
    72: 696.4,  71: 762.7,  70: 832.1,
    69: 904.8,  68: 980.8,  67: 1060.1,
    66: 1142.6, 65: 1228.5, 64: 1317.6,
    63: 1410.0, 62: 1505.7, 61: 1604.9,
    60: 1707.8, 59: 1814.4, 58: 1925.0,
    57: 2039.5, 56: 2158.1, 55: 2280.8,
    54: 2407.6, 53: 2538.5, 52: 2673.5,
    51: 2812.5, 50: 2955.6, 49: 3102.5,
    48: 3253.2, 47: 3407.6, 46: 3565.7,
    45: 3727.5, 44: 3892.9, 43: 4062.0,
    42: 4234.8, 41: 4411.5, 40: 4592.0,
    39: 4776.6, 38: 4965.3, 37: 5158.5,
    36: 5356.1, 35: 5558.7, 34: 5766.5,
    33: 5980.0, 32: 6199.8, 31: 6426.5,
    30: 6660.8, 29: 6903.3, 28: 7154.8,
    27: 7416.2, 26: 7688.4, 25: 7972.3,
    24: 8268.1, 23: 8575.2, 22: 8893.6,
    21: 9223.5, 20: 9565.6, 19: 9920.9,
    18: 10290.7, 17: 10677.1, 16: 11082.1,
    15: 11510.3, 14: 11966.2, 13: 12454.8,
    12: 12982.0, 11: 13554.7, 10: 14180.1,
    9: 14866.5, 8: 15627.5, 7: 16485.0,
    6: 17467.8, 5: 18610.9, 4: 19955.6,
    3: 21550.0, 2: 23450.1, 1: 34461.1,
    0: 34461.1,  # not a true level, but useful for indexing
}


