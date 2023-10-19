#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:55:40 2023

@author: lunelt
"""

import os
import re
import xarray as xr
import numpy as np



def indices_of_lat_lon(ds, lat, lon, verbose=True):
    """ 
    Find indices corresponding to latitude and longitude values
    for a given file.
    
    ds: xarray.DataSet
        Dataset containing fields of netcdf file
    lat: float, 
        Latitude
    lon: float,
        Longitude
    
    Return:
        tuple [index_lat, index_lon]
    
    """
    # Get latitude and longitude data
    try:
        lat_dat = ds['latitude'].data
        lon_dat = ds['longitude'].data
    except KeyError:
        try:
            lat_dat = ds['latitude_u'].data
            lon_dat = ds['longitude_u'].data
        except KeyError:
            try:
                lat_dat = ds['latitude_v'].data
                lon_dat = ds['longitude_v'].data
            except KeyError:
                try:
                    lat_dat = ds['latitude_w'].data
                    lon_dat = ds['longitude_w'].data
                except KeyError:
                    raise AttributeError("""this dataset does not have 
                                         latitude-longitude coordinates""")
    
    # Gross evaluation of lat, lon (because latitude lines are curved)
    distance2lat = np.abs(lat_dat - lat)
    index_lat = np.argwhere(distance2lat <= np.nanmin(distance2lat))[0,0]
    
    distance2lon = np.abs(lon_dat - lon)
    index_lon = np.argwhere(distance2lon <= np.nanmin(distance2lon))[0,1]
    
#    print("Before refinement : index_lat={0}, index_lon={1}".format(
#            index_lat, index_lon))
    
    # refine as long as optimum not reached
    opti_reached = False
    n = 0  #count of iteration
    while opti_reached is False:
        if (np.abs(lon_dat[index_lat, index_lon] - lon) > \
            np.abs(lon_dat[index_lat, index_lon+1] - lon) ):
            index_lon = index_lon+1
        elif (np.abs(lon_dat[index_lat, index_lon] - lon) > \
            np.abs(lon_dat[index_lat, index_lon-1] - lon) ):
            index_lon = index_lon-1
        elif (np.abs(lat_dat[index_lat, index_lon] - lat) > \
            np.abs(lat_dat[index_lat+1, index_lon] - lat) ):
            index_lat = index_lat+1
        elif (np.abs(lat_dat[index_lat, index_lon] - lat) > \
            np.abs(lat_dat[index_lat-1, index_lon] - lat) ):
            index_lat = index_lat-1
        elif n > 20:
            raise ValueError("""loop does not converge, 
                             check manually for indices.""")
        else:
            opti_reached = True
    if verbose:
        print("For lat={0}, lon={1} : index_lat={2}, index_lon={3}".format(
            lat, lon, index_lat, index_lon))
    
    return index_lat, index_lon



folder_path = './'

linux_wildcard = 'LIAIS.1.SEG??.00?.nc'
# Convert Linux wildcard to Python regex
regex_pattern = re.escape(linux_wildcard).replace(r'\?', r'[^/]').replace(r'\*', r'.*')
print(regex_pattern)

lat, lon = 41.69, 0.93

for root, _, files in os.walk(folder_path):
    for file_name in files:
        if re.match(regex_pattern, file_name):
            file_path = os.path.join(root, file_name)
            
            ds = xr.open_dataset(file_path)
            
            index_lat, index_lon = indices_of_lat_lon(ds, lat, lon)
            
            value = ds['WG3_P9'][index_lat, index_lon]
            
            if value is not None:
                print(f"File: {file_name}, Value at ({latitude}, {longitude}): {value}")

