#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:03:48 2025

@author: lunelt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics.pairwise import euclidean_distances

import global_variables as gv

global_data_nemo = gv.global_data_nemo
site = 'planier'

df = pd.read_pickle(f'{global_data_nemo}/{site}/metocean_conditions_planier.pickle')
df['arome_wd_error_140'] = ((df['arome_wd_140'] - df['wd_140']) + 180) % 360 - 180
df = df[df['arome_wd_error_140'].abs() < 20]
df['arome_ws_error_140'] = (df['arome_ws_140'] - df['ws_140'])
df = df[df['arome_ws_error_140']/df['arome_ws_140'] < 0.5]

df['arome_tke_error_140'] = df['arome_tke_140'] - df['tke_140']
# df = df.dropna()

#TODO: switch to ri_bulk for analysis

df['stability_class_ri_bulk'] = df['arome_ri_bulk_140']
# classification of stability based on doi.org/10.5194/wes-7-221-2022 - Table 4
# class Very Unstable 
df['stability_class_ri_bulk'].where(
    ~(df['arome_ri_bulk_140'] < -0.023), 
    1, inplace=True)
# class Unstable 
df['stability_class_ri_bulk'].where(
    ~(np.logical_and(-0.023 < df['arome_ri_bulk_140'],
                     df['arome_ri_bulk_140'] < -0.0036)),
    2, inplace=True)
# class Neutral 
df['stability_class_ri_bulk'].where(
    ~(np.logical_and(-0.0036 < df['arome_ri_bulk_140'],
                     df['arome_ri_bulk_140'] < 0.0072)),
    3, inplace=True)
# class Stable 
df['stability_class_ri_bulk'].where(
    ~(np.logical_and(0.0072 < df['arome_ri_bulk_140'],
                     df['arome_ri_bulk_140'] < 0.084)),
    4, inplace=True)
# class Very Stable 
df['stability_class_ri_bulk'].where(
    ~(0.084 < df['arome_ri_bulk_140']),
    5, inplace=True)

df['hour'] = df.index.hour

print('Occurence of stability classes: ')
print(df['stability_class_ri_bulk'].value_counts())

# df_verystable = df[df['stability_class_ri_bulk']==5]
# df_veryunstable = df[df['stability_class_ri_bulk']==1]

vars_clustering = [
    'arome_tke_140', 
    'arome_ws_140', 
    'arome_wd_140', 
    # 'arome_ri_bulk_140',
    # 'arome_abl_top_ri',
    # 'sst_planier', 
    # 'nebul_marignane',
    'stability_class_ri_bulk'
    ]
df_subset = df[vars_clustering]

for var in vars_clustering:
    if var in ['arome_tke_140', 'arome_ws_140', 'tke_140', 'ws_140']:
        df_subset[var] = (df_subset[var] - df_subset[var].mean())/df_subset[var].std()
    
    elif var in ['arome_wd_140', 'wd_140']:
        df_subset[var] = ((df_subset[var] - scipy.stats.circmean(df_subset[var], high=360, low=0)) / \
                          scipy.stats.circstd(df_subset[var], high=360, low=0))
#%%
if 'tke_140' in vars_clustering:
    df = df[~df_subset['tke_140'].isna()]
    
df_subset = df_subset.dropna()

#%% Détermination de la valeur optimale de K

# inertia = []
# print('Search for K optimal value')
# for i in range(1, 10):
#     kmeans = KMeans(n_clusters=i)
#     kmeans.fit(df_subset)
#     inertia.append(kmeans.inertia_)
#     print('K = ', i)
#     print('inertia: ', kmeans.inertia_)

# plt.figure()
# plt.plot(range(1,10), inertia)
# plt.title("La méthode Elbow")
# plt.xlabel("nombre de cluster")
# plt.ylabel("Inertie intra-classe")
# plt.show()


#%% Application de KMeans
kmeans = KMeans(n_clusters = 4)
kmeans.fit(df_subset)

df['cluster'] = kmeans.labels_
df_subset['cluster'] = kmeans.labels_

print('Sizes of clusters: ')
for i_cluster in np.unique(kmeans.labels_):
    print(f'group {i_cluster}: ', len(np.where(kmeans.labels_ == i_cluster)[0]))


#%% Visualisation
plt.figure()
xvar = 'arome_tke_140'
yvar = 'stability_class_ri_bulk'
colormap = np.array(["red", "green", "blue", 'k', 'orange'])
for i_cluster in np.unique(kmeans.labels_):
    df_clus = df[df['cluster'] == i_cluster]
    plt.scatter(df_clus[xvar], 
                df_clus[yvar]+0.1*i_cluster,
                c = colormap[i_cluster],
                label='cluster {0} / size {1}'.format(
                    i_cluster, len(np.where(kmeans.labels_ == i_cluster)[0])),
                # c = colormap[df_subset['stability_class_ri_grad']], 
                s=10)
plt.xlabel(xvar)
plt.ylabel(yvar)
plt.legend()
plt.show()

#%% Characteristics of each cluster

df['cluster'] = kmeans.labels_
df_subset['cluster'] = kmeans.labels_
cluster_features = {}
cluster_features_norm = {}

keys_list = list(df.keys())
keys_list.remove('cluster')
keys_list = [
    'arome_tke_140',
    'arome_ws_140',
    'arome_wd_140',
    'tke_140',
    'ws_140',
    'wd_140',
    'arome_tke_error_140',
    'arome_ri_bulk_140',
    # 'arome_abl_top_ri',
    # 'sst_planier',
    'nebul_marignane',
    'stability_class_ri_bulk',
    'arome_ri_bulk_140']

for i_cluster in np.unique(kmeans.labels_):
    df_clus = df[df['cluster'] == i_cluster]
    cluster_features[i_cluster] = {}
    print('--------------------')
    print(f'Features of cluster {i_cluster} are: ')
    for key in keys_list:
        
        new_key = f'{key}_mean'
        if 'wd' in key:
            cluster_features[i_cluster][new_key] = np.round(
                scipy.stats.circmean(df_clus[key], high=360, low=0), 2)
        else:
            cluster_features[i_cluster][new_key] = np.round(
                df_clus[key].mean(), 2)
    # for key in keys_list:
    #     new_key = f'{key}_std'
    #     if 'wd' in key:
    #         cluster_features[i_cluster][new_key] = np.round(
    #             scipy.stats.circstd(df_clus[key], high=360, low=0), 2)
    #     else:
    #         cluster_features[i_cluster][new_key] = np.round(
    #             df_clus[key].std(), 2)
    
        print(f'{new_key}: ', cluster_features[i_cluster][new_key])
    

for i_cluster in np.unique(kmeans.labels_):
    df_clus_subset = df_subset[df_subset['cluster'] == i_cluster]
    cluster_features_norm[i_cluster] = {}   
    
    for key in keys_list:
        if key in vars_clustering:
            new_key = f'{key}_norm'
            if 'wd' in key:
                cluster_features_norm[i_cluster][new_key] = np.round(
                    scipy.stats.circstd(df_clus_subset[key], high=360, low=0), 2)
            else:
                cluster_features_norm[i_cluster][new_key] = np.round(
                    df_clus_subset[key].std(), 2)

#%% Draft
# df_reftest = pd.DataFrame({'ws': [2], 'wd':[4]})
# df_test = pd.DataFrame({'ws': [3,4,5], 'wd':[6,8,10]})

# euclidean_distances(df_reftest, df_test)

#%% find most typical representative of each cluster

# df_subset['cluster'] = kmeans.labels_  # already done above
df_ref = pd.DataFrame(cluster_features_norm)

for i_cluster in np.unique(kmeans.labels_):
    # df_ref_i = df_ref.T.iloc[0]
    df_ref_i = pd.DataFrame(df_ref[i_cluster]).T
    # df_ref = pd.DataFrame(cluster_features[i_cluster], 
    #                       columns=cluster_features[i_cluster].keys())
    df_clus_all = df[df['cluster'] == i_cluster]
    df_clus_subset_all = df_subset[df_subset['cluster'] == i_cluster]
    df_respecting_conditions = (df_clus_subset_all['cluster'] == i_cluster)
    # ---- add conditions on date of interest:
    # -- month:
    # df_clus_subset = df_clus_subset_all[df_clus_all.index.month > 1]  # removes indexes not respecting the conditions
    df_respecting_conditions = df_respecting_conditions.where(  # do not remove False value
        (df_clus_all.index.month > 0), other=False)
    # -- condition on agreement between arome and measures
    df_respecting_conditions = df_respecting_conditions.where(
        (df_clus_all['arome_wd_error_140'] < 20), other=False)
    # -- condition on low nebulosity
    df_respecting_conditions = df_respecting_conditions.where(
        (df_clus_all['nebul_marignane'] < 2), other=False)
    # -- condition on hour, to be close from the analysis time (every 3h, rest is interpolation)
    df_respecting_conditions = df_respecting_conditions.where(
        np.logical_or((df_clus_all.index.hour % 3 == 0), 
                      ((df_clus_all.index.hour+1) % 3 == 0)),
        other=False)
    # -----------------------------------
    df_clus_subset = df_clus_subset_all[df_respecting_conditions]
    df_clus_subset = df_clus_subset.drop(columns=['cluster'])
    
    dist = euclidean_distances(df_ref_i, df_clus_subset)
    df_clus_subset['dist'] = dist[0]
    typic_event = df_clus_subset[df_clus_subset['dist'] == df_clus_subset['dist'].min()]
    print('---------------------------------')
    print(f'most typical event of cluster {i_cluster} is: ')
    print('Date: ', str(typic_event.index[0]))
    prop_event = df.loc[typic_event.index[0]]
    for key in ['wd_140', 'ws_140', 'tke_140', 'arome_wd_140', 'arome_ws_140', 'arome_tke_140', 'nebul_marignane',
                'stability_class_ri_bulk']:
        # if key == 'dist':
        #     continue
        print(key, ' : ', float(prop_event[key]))
