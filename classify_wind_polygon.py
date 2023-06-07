#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 16:53:06 2023

@author: lunelt

get scalar or wind variable and select part inside of a polygon
"""

#import numpy as np
import xarray as xr
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import tools
import global_variables as gv

#############################

#varname='LE_ISBA'

corners_irrig = ['lleida', 'balaguer', 
                 'claravalls', 'borges_blanques']
corners_dry = ['claravalls', 'borges_blanques', 
               'els_omellons', 'sant_marti', 'fonolleres']
corners_slope_west = ['els_omellons', 'sant_marti', 'fonolleres',
                      'conesa', 'tossal_gros', 'villobi',]
corners_slope_east = ['conesa', 'tossal_gros', 'villobi',
                      'tossal_purunyo', 'puig_cabdells', 'puig_formigossa']
corners_coast = ['tossal_purunyo', 'puig_cabdells', 'puig_formigossa',
                 'tarragona', 'calafell']

ilevel=10
figsize = (13,9)
#############################


filename = tools.get_simu_filename('irr_d1', '20210716-1600',)
ds = xr.open_dataset(filename,
                      decode_coords="coordinates",
                      )

# for wind
#ds = ds1[['UT','VT','RVT']].squeeze()
ds_cen = tools.center_uvw(ds)
ds_cen['WS'], ds_cen['WD'] = tools.calc_ws_wd(ds_cen['UT'], ds_cen['VT'])
data_in = ds_cen[['WS', 'WD']]


# CREATE Polygon
polygon_irrig_coords = [
        (gv.whole[corners_irrig[0]]['lon'], gv.whole[corners_irrig[0]]['lat']),
        (gv.whole[corners_irrig[1]]['lon'], gv.whole[corners_irrig[1]]['lat']),
        (gv.whole[corners_irrig[2]]['lon'], gv.whole[corners_irrig[2]]['lat']),
        (gv.whole[corners_irrig[3]]['lon'], gv.whole[corners_irrig[3]]['lat']),
        ]
polygon = Polygon(polygon_irrig_coords)

# Classify points within the polygon
classified_points = tools.get_points_in_polygon(data_in, polygon, 
#                                                ilevel=ilevel
                                                )
# concatenate data
extracted_ds = xr.concat(classified_points, 'ind')
# keep layer of interest
extracted_da = extracted_ds['WS'][:, ilevel]
extracted_layer = extracted_ds.isel(level = ilevel)

# filter:
filtered_da = extracted_da.where(extracted_da > 2)
filtered_ds = extracted_layer.where(90 < extracted_layer['WD']).where(extracted_layer['WD'] < 270)
layer_for_fig = filtered_ds['WS']


# ---------- PLOTs -----------------

plt.figure(figsize=figsize)
# global plot
var2d = ds_cen['WS'][ilevel, :, :]
plt.pcolormesh(var2d.longitude, var2d.latitude, var2d)

# extrated area plot
plt.scatter(layer_for_fig.longitude, layer_for_fig.latitude,
            c=layer_for_fig.values, 
            cmap='hot', s=10)
plt.plot(*polygon.exterior.xy)
