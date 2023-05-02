#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to visualize cover number of a .dat file, where values are stored like:
lon  lat  cover_nb
0.5  41.1  450  
0.5  41.0  506
0.5  40.9  501
etc  

Cover number is written in upper right corner of the figure 
when using plt.imshow() function (pcolormesh or equivalent do not have this
feature)

@author: lunel tanguy
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Open ascii file (needs to be returned by modif_ECOII.f90)
df = pd.read_csv(
        '/cnrm/surface/lunelt/ECOCLIMAP/irrig_ECOII/' +\
        'extrait_ECOII.dat',
#        'code_modif_ECOII_dir/Cover_nb_correction/extrait_ECOII_V2.3_CovCorr.dat',
        sep='\s+', header=None, index_col=False)

df.columns=['lon', 'lat', 'cover']

#%% with MATPLOTLIB ONLY

x_list = df['lon'].drop_duplicates()
y_list = df['lat'].drop_duplicates()

x, y = np.meshgrid(x_list, y_list)

z_list = df['cover']

z = np.array_split(z_list, len(y_list))

#fig = plt.figure(figsize=[12, 8])
fig = plt.figure()
ax = plt.subplot(1, 1, 1)

# IMSHOW allows to see value of cover with mouse pointer
ax.imshow(z, cmap='flag') #other cmap: gist_ncar, flag, prism
# PCOLORMESH does not allow to see values of covers with pointer
#ax.pcolormesh(x, y, z, cmap='flag') #other cmap: gist_ncar, flag, prism

plt.show()