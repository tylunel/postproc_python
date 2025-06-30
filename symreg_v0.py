#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 27 12:12:51 2025

@author: lunelt
"""

import pysr
import sympy
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

import tools
import global_variables as gv

# model = '2_planier_0122/0411_40m'
model = '2_planier_0122/04_1km_COARE'

output_type = 'backup'

hour = '11'
hour_bu = int(hour)
XCED = 0.84

wanted_date = f'20230122-{hour}00'

#%% Load ds
filename = tools.get_simu_filepath(f'{model}', wanted_date,
                                   output_type = output_type,
                                   global_simu_folder=gv.global_simu_folder,
                                   verbose=True)
ds = xr.open_dataset(filename)

filename_bu = tools.get_simu_filepath(f'{model}', wanted_date,
                                    output_type='diachronic',
                                    global_simu_folder=gv.global_simu_folder)

ds_bu = tools.open_budget_file(filename_bu, 'TK')

dZ = xr.DataArray(np.gradient(ds['level']), dims='level')
dTH = xr.DataArray(np.gradient(ds['THT'].squeeze(), axis=0), 
                    dims=['level', 'nj', 'ni'])

dTHdZ = (dTH/dZ)[10:20, 1:-1:5, 1:-1:5]
LM = ds.LM.squeeze()[10:20, 1:-1:5, 1:-1:5]
TKET = ds.TKET.squeeze()[10:20, 1:-1:5, 1:-1:5]
DISS = ds.TKE_DISS.squeeze()[10:20, 1:-1:5, 1:-1:5]

# DISS = ds_bu.DISS.isel(time_budget=hour_bu)[10:20, ::5, ::5]

# X = np.array([LM.values.flatten(), TKET.values.flatten()])
# print(X.size)

#%% Sym Reg

X = np.array([LM.values.flatten(), TKET.values.flatten(), dTHdZ.values.flatten()])
Y = np.array(DISS.values.flatten())

model = PySRRegressor(
    maxsize=20,
    niterations=30,  # < Increase me for better results
    binary_operators=["*", "/", "+"],
    unary_operators=[
        "cube",
        "sqrt",
        # "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
    ],
    # extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
    )

model.fit(X.T, Y)

print(model)
print('------------------------')
print('Best score is found for:')
print(model.sympy())

#%% plot

# print('DISS time:  ', DISS.time_budget.values)
# print('TKE time:  ', TKET.time.values)

# diss_mano = -XCED*(TKET**1.5)/LM

# X = diss_mano.values.flatten()
# Y = DISS.values.flatten()

# plt.figure()
# plt.scatter(X, Y)
# plt.plot([-0.14, 0], [-0.14, 0], color='r')
# plt.xlabel("diss_mano")
# plt.ylabel('diss_bu')

# #%% alt plot

# import datashader as ds
# import colorcet as cc

# df = pd.DataFrame(data=np.array([X,Y]).T, columns=["x", "y"])  # create a DF from array

# cvs = ds.Canvas(plot_width=1000, plot_height=1000)  # auto range or provide the `bounds` argument
# agg = cvs.points(df, 'x', 'y')  # this is the histogram
# img = ds.tf.set_background(ds.tf.shade(agg, how="log", cmap=cc.fire), 
#                            "black").to_pil()  # create a rasterized image

# img_arr = np.array(img)

# plt.imshow(img_arr)

# plt.xticks(ticks=np.linspace(0, 1000, num=5), 
#            labels=np.round(np.linspace(df['x'].min(), 0, num=5), 2))
# plt.yticks(ticks=np.linspace(0, 1000, num=5), 
#            labels=np.round(np.linspace(df['y'].min(), 0, num=5), 2))

# # plt.plot([200,0], [200, 0], color='b')
# plt.show()


