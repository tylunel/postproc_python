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

model = '2_planier_0122/0411_40m'
# model = '2_planier_0122/04_1km_COARE'

output_type = 'backup'

hour = '11'
hour_bu = int(hour)
XCED = 0.84

wanted_date = f'20230122-{hour}00'

#%% Load ds
# filename = tools.get_simu_filepath(f'{model}', wanted_date,
#                                    output_type = output_type,
#                                    global_simu_folder=gv.global_simu_folder,
#                                    verbose=True)
# ds = xr.open_dataset(filename)

filename_bu = tools.get_simu_filepath(f'{model}', wanted_date,
                                    output_type='diachronic',
                                    global_simu_folder=gv.global_simu_folder)

# ds_bu = tools.open_budget_file(filename_bu, 'TK')

LES_budgets_type = ['Mean', 'Resolved', 'Miscellaneous']
LES_budgets_averaging = 'Time_averaged'
# groupname_les_budget = f"LES_budgets/{LES_budgets_type}/Cartesian/{LES_budgets_averaging}/Not_normalized/cart"
# ds_lesbu = xr.open_dataset(filename_bu, group=groupname_les_budget)
ds_lesbu_orig = tools.open_les_budget_file(filename_bu, 
                                      LES_budgets_type=LES_budgets_type,
                                      LES_budgets_averaging=LES_budgets_averaging)

ds_lesbu = ds_lesbu_orig.where(ds_lesbu_orig['RES_KE']>1)

ds_lesbu['CepsOverLeps'] = -ds_lesbu['MEAN_DISS'] / ds_lesbu['RES_KE']**(3/2)
ds_lesbu['Leps'] = 0.84/ds_lesbu['CepsOverLeps']
# ds_lesbu['MEAN_THL']
dZ = np.gradient(ds_lesbu['level_les'])

MEAN_dTHLdZ = (np.gradient(ds_lesbu['MEAN_THL'], axis=1)/dZ)

MEAN_WS, _ = tools.calc_ws_wd(ds_lesbu['MEAN_U'], ds_lesbu['MEAN_V'])
MEAN_dWSdZ = (np.gradient(MEAN_WS, axis=1)/dZ)[:,2:50]

g = 9.81
MEAN_RIg = (g/ds_lesbu['MEAN_THL'][:,2:50])*(MEAN_dTHLdZ/(MEAN_dWSdZ**2))

MEAN_LM = ds_lesbu['MEAN_LM'][:,2:50]
CepsLeps = ds_lesbu['CepsOverLeps'][:,2:50]
Leps = ds_lesbu['Leps'][:,2:50]

# LM = ds.LM.squeeze()[10:20, 1:-1:5, 1:-1:5]
# TKET = ds.TKET.squeeze()[10:20, 1:-1:5, 1:-1:5]
# DISS = ds.TKE_DISS.squeeze()[10:20, 1:-1:5, 1:-1:5]

# DISS = ds_bu.DISS.isel(time_budget=hour_bu)[10:20, ::5, ::5]

# X = np.array([LM.values.flatten(), TKET.values.flatten()])
# print(X.size)
LM_flat = MEAN_LM.values.flatten()
LM = LM_flat[~np.isnan(LM_flat)]

Leps_flat = Leps.values.flatten()
Leps = Leps_flat[~np.isnan(Leps_flat)]

THL_flat = ds_lesbu['MEAN_THL'][:,2:50].values.flatten()
THL = THL_flat[~np.isnan(THL_flat)]

RIg_flat = MEAN_RIg.values.flatten()
RIg = RIg_flat[~np.isnan(RIg_flat)]

dTHLdZ_flat = MEAN_dTHLdZ.flatten()
dTHLdZ = dTHLdZ_flat[~np.isnan(dTHLdZ_flat)]

dWSdZ_flat = MEAN_dWSdZ.flatten()
dWSdZ = dWSdZ_flat[~np.isnan(dWSdZ_flat)]

#%% Sym Reg

X = np.array([LM, 
              RIg,
              dTHLdZ,
              dWSdZ,
              ])
Y = np.array(Leps)

#%%
model = PySRRegressor(
    maxsize=20,
    niterations=40,  # < Increase me for better results
    binary_operators=["*", "/", "+", '-'],
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


