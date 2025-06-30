#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corrected version of clustering
Gives different results than first classification !!!!

@author: lunelt
"""

import pandas as pd
pd.options.mode.copy_on_write = True
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.cluster import KMeans
# from sklearn import datasets
from sklearn.metrics.pairwise import euclidean_distances

import global_variables as gv

global_data_nemo = gv.global_data_nemo
site = 'planier'

df = pd.read_pickle(f'{global_data_nemo}/{site}/metocean_conditions_planier.pickle')

#%% Tri des données pour garder celles avec bonne corrélation AROME vs OBS
# keep only cases where arome and obs agree enough
df['arome_wd_error_140'] = ((df['arome_wd_140'] - df['wd_140']) + 180) % 360 - 180
df = df[df['arome_wd_error_140'].abs() < 20]
df['arome_ws_error_140'] = (df['arome_ws_140'] - df['ws_140'])
df = df[df['arome_ws_error_140']/df['arome_ws_140'] < 0.5]
# df['arome_tke_error_140'] = df['arome_tke_140'] - df['tke_140']
# df = df.dropna()

#%% Standardisation des données
# ri_bulk for analysis seems best
df['stability_class_ri_bulk'] = df['arome_ri_bulk_140']
# classification of stability based on doi.org/10.5194/wes-7-221-2022 - Table 4

# class Very Unstable 
df['stability_class_ri_bulk'] = df['stability_class_ri_bulk'].where(
    ~(df['arome_ri_bulk_140'] < -0.023), 1)
# class Unstable 
df['stability_class_ri_bulk'] = df['stability_class_ri_bulk'].where(
    ~(np.logical_and(-0.023 < df['arome_ri_bulk_140'],
                     df['arome_ri_bulk_140'] < -0.0036)), 2)
# class Neutral 
df['stability_class_ri_bulk'] = df['stability_class_ri_bulk'].where(
    ~(np.logical_and(-0.0036 < df['arome_ri_bulk_140'],
                     df['arome_ri_bulk_140'] < 0.0072)), 3)
# class Stable 
df['stability_class_ri_bulk'] = df['stability_class_ri_bulk'].where(
    ~(np.logical_and(0.0072 < df['arome_ri_bulk_140'],
                     df['arome_ri_bulk_140'] < 0.084)), 4)
# class Very Stable 
df['stability_class_ri_bulk'] = df['stability_class_ri_bulk'].where(
    ~(0.084 < df['arome_ri_bulk_140']), 5)


# # OLD - raise FutureWarning
# # class Very Unstable 
# df['stability_class_ri_bulk'].where(
#     ~(df['arome_ri_bulk_140'] < -0.023), 
#     1, inplace=True)
# # class Unstable 
# df['stability_class_ri_bulk'].where(
#     ~(np.logical_and(-0.023 < df['arome_ri_bulk_140'],
#                      df['arome_ri_bulk_140'] < -0.0036)),
#     2, inplace=True)
# # class Neutral 
# df['stability_class_ri_bulk'].where(
#     ~(np.logical_and(-0.0036 < df['arome_ri_bulk_140'],
#                      df['arome_ri_bulk_140'] < 0.0072)),
#     3, inplace=True)
# # class Stable 
# df['stability_class_ri_bulk'].where(
#     ~(np.logical_and(0.0072 < df['arome_ri_bulk_140'],
#                      df['arome_ri_bulk_140'] < 0.084)),
#     4, inplace=True)
# # class Very Stable 
# df['stability_class_ri_bulk'].where(
#     ~(0.084 < df['arome_ri_bulk_140']),
#     5, inplace=True)

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
        # df_subset[var] = (df_subset[var] - df_subset[var].mean())/df_subset[var].std()
        df_subset[var] = (df[var] - df[var].mean()) / df[var].std()
        # mean = df[var].mean()
        # stdev = df[var].std()
        # df_subset[var] = (df[var] - mean) / stdev
    
    elif var in ['arome_wd_140', 'wd_140']:
        # df_subset[var] = (  # old: ERROR (mean not close to zero, because lack of modulo)
        #     (df_subset[var] - scipy.stats.circmean(df_subset[var], high=360, low=0)) / \
        #     scipy.stats.circstd(df_subset[var], high=360, low=0))
        # df_subset[var] = (
        #     (((df_subset[var] - scipy.stats.circmean(df_subset[var], high=360, low=0)) + 180) % 360 - 180) / \
        #      scipy.stats.circstd(df_subset[var], high=360, low=0))
        df_subset[var] = (
            (((df[var] - scipy.stats.circmean(df[var], high=360, low=0)) + 180) % 360 - 180) / \
             scipy.stats.circstd(df[var], high=360, low=0))

#%% Remove timestamp with missing measurement values
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
kmeans = KMeans(n_clusters=4, random_state=10)
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
colormap = np.array(["red", "green", "blue", 'orange', 'k', ])
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

plt.figure()
xvar = 'arome_ws_140'
yvar = 'arome_wd_140'
colormap = np.array(["red", "green", "blue", 'orange', 'k', ])
for i_cluster in np.unique(kmeans.labels_):
    df_clus = df[df['cluster'] == i_cluster]
    plt.scatter(df_clus[xvar], 
                df_clus[yvar]+0.1*i_cluster,
                c = colormap[i_cluster],
                label='cluster {0} / size {1}'.format(
                    i_cluster, len(np.where(kmeans.labels_ == i_cluster)[0])),
                # c = colormap[df_subset['stability_class_ri_grad']], 
                s=1)
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
    # 'arome_tke_error_140',
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


#%% Most typical event representative of each cluster

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
    
    # ---- Add conditions on date of interest:
    # -- month:
    # df_clus_subset = df_clus_subset_all[df_clus_all.index.month > 1]  # removes indexes not respecting the conditions
    # df_respecting_conditions = df_respecting_conditions.where(  # do not remove False value
    #     (df_clus_all.index.month > 1), other=False)
    # -- condition on agreement between arome and measures
    # df_respecting_conditions = df_respecting_conditions.where(
    #     (df_clus_all['arome_wd_error_140'] < 20), other=False)
    # -- condition on low nebulosity
    df_respecting_conditions = df_respecting_conditions.where(
        (df_clus_all['nebul_marignane'] < 2), other=False)
    # -- condition on hour, to be close from the analysis time (every 3h, rest is interpolation)
    # df_respecting_conditions = df_respecting_conditions.where(
    #     np.logical_or((df_clus_all.index.hour % 3 == 0), 
    #                   ((df_clus_all.index.hour+1) % 3 == 0)),
    #     other=False)
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
        
#%% Extra - Wind rose at 140m

# from windrose import WindroseAxes

# ax = WindroseAxes.from_ax()
# # AROME
# ax.bar(df['arome_wd_140'], df['arome_ws_140'], 
#        normed=True, opening=0.8, edgecolor="white", nsector=32)
# # OBS
# ax.bar(df['wd_140'], df['ws_140'], 
#        normed=True, opening=0.8, edgecolor="white", nsector=32)
# ax.set_legend()
