#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for plotting vertical profiles from AROME .fa files

@author: lunelt
"""

import epygram
import matplotlib.pyplot as plt
import global_variables as gv
# from dask import compute
# from dask.delayed import delayed
# import dask
import time
import tools


folder_path = '/home/lunelt/Data/analysis/'
wanted_date = '20230724.08'

site = 'planier'
varname = 'tke'     # tke, t, u, v, q

vmin, vmax = 0, 3

wanted_height = 15  #[m]

nearest_height, ind = tools.nearest(list(gv.level_height_AROME_mean.values()), 
                                  wanted_height)
level_arome = list(gv.level_height_AROME_mean.keys())[ind]
print(f"Level and height in AROME are: {level_arome}, {nearest_height}")

#%% MISC

filename = f'arome.AN.{wanted_date}'

file = epygram.formats.resource(filename=f'{folder_path}/{filename}',
                                openmode='r')

# list of variable availables
simplified_field_dict = {}
for field in file.listfields():
    simplified_field_dict[field['shortName']]=field['name']

# for maps
land_sea_mask = file.find_fields_in_resource('shortName:lsm')
lsm = file.readfields(land_sea_mask[0])

# unk = file.find_fields_in_resource('shortName:unknown')

fid3D = file.find_fields_in_resource(f'shortName:{varname}')
# file.readfields(fid3D)
# f = file.readfield({'shortName':'t', 'typeOfFirstFixedSurface':119})  # temperature on pressure levels

fid = file.find_fields_in_resource(f'shortName:{varname}, level:{level_arome}')
field = file.readfields(fid[0])[0]

#%% PLOT verti profile

# t1 = time.time()
# verti_profile = tools.extract_profile_arome(site, wanted_date, varname)
# t2 = time.time()
# print('exec time: ', t2-t1)


# fig = plt.figure()
# plt.plot(verti_profile.values(), verti_profile.keys(), 
#          label='arome', marker='+', linestyle='--')
# plt.xlabel = varname
# plt.ylim(0, 2000)


#%% PLOT 2D map
# fig, ax = field[0].cartoplot()  # does not work

fig = plt.figure()
plt.contour(lsm[0].data, color='k')
plt.pcolormesh(field.data, vmin=vmin, vmax=vmax)
plt.colorbar()
