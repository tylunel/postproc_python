#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tanguy Lunel

Script to load and process tif image 
(e.g. from Theia - data soil moist 10mx10m) 
and rewrite array into tiff

if needed to crop the tiff file before, follow this: 
https://gis.stackexchange.com/questions/301420/how-to-crop-tiff-image-without-losing-classes-in-qgis-3-2

Pour cropper les images comme les 2 premières, utiliser les coordonnées 
suivantes (xmin, xmax, ymin, ymax):
0.61, 1.27, 41.47, 41.89

to convert from numpy array to tiff, une piste ici:
https://gis.stackexchange.com/questions/290776/how-to-create-a-tiff-file-using-gdal-from-a-numpy-array-and-specifying-nodata-va

"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#import copy

############# Independant parameters #######################

father_folder = '/cnrm/surface/lunelt/data_LIAISE/theia_MV_catalogne/'

files_list = [
#              'MV_cropped_20210627T1746_S1A.tif',
#              'MV_cropped_20210702T1754_S1A.tif',
#              'MV_cropped_20210708T1754_S1B.tif',
#              'MV_cropped_20210709T1746_S1A.tif', 
#              'MV_cropped_20210714T1754_S1A.tif',
              'MV_cropped_20210715T1746_S1B.tif',
              'MV_cropped_20210720T1754_S1B.tif',
#              'MV_cropped_20210721T1746_S1A.tif',
#              'MV_cropped_20210726T1754_S1A.tif',
#              'MV_cropped_20210727T1746_S1B.tif',
#              'MV_cropped_20210801T1754_S1B.tif',
#              'MV_cropped_20210802T1746_S1A.tif'
              ]

plot = True

#############################################################

sm_cendrosa = []
date_list = []

for i, file in enumerate(files_list):
#for i in [3]: #7 for 21 to 26/07
    #null:  1, 3
    imtif1 = Image.open(father_folder + files_list[i])
    try:
        imtif2 = Image.open(father_folder + files_list[i+1])
    except IndexError:
        break
    date = files_list[i][15:19] + 'to' + files_list[i+1][15:19]
    #division by 5 to get vol soil moisture [%]
    sm_arr1 = np.array(imtif1, dtype='float') / 5
    sm_arr2 = np.array(imtif2, dtype='float') / 5
    
#    alpha1 = copy.deepcopy(imarray1)
#    alpha = 1 - alpha1 / np.nanmax(alpha1)
#    plt.imshow(alpha, cmap = 'binary')
    
    sm_arr1[sm_arr1 == 0] = np.nan
    sm_arr2[sm_arr2 == 0] = np.nan
    sm_diff = sm_arr2 - sm_arr1
    
    print('cendrosa sm1 and diff {0} = '.format(date))
    print(sm_arr1[2190, 3560])
    print(sm_diff[2190, 3560])
    
    sm_cendrosa.append(sm_arr1[2190, 3560])
    date_list.append(np.datetime64('2021-'+date[0:2]+'-'+date[2:4]))
    
    if plot:
        ##PLOT SM day1
        plt.figure(figsize=(13, 7))
        plt.imshow(sm_arr1, cmap = 'coolwarm_r', 
                   interpolation='nearest',  #better visualization
                   vmin=0, vmax=40, 
    #               alpha=alpha
                   )
        plt.title(date[0:4])
        plt.ylim(2600, 1900)
        plt.xlim(3200, 4000)
        cbar = plt.colorbar()
        cbar.set_label('volumetric moisture [% m3/m3]')
        
        ##PLOT SM day2
        plt.figure(figsize=(13, 7))
        plt.imshow(sm_arr2, cmap = 'coolwarm_r', 
                   interpolation='nearest',  #better visualization
                   vmin=0, vmax=40)
        plt.title(date[6:10])
        plt.ylim(2600, 1900)
        plt.xlim(3200, 4000)
        cbar = plt.colorbar()
        cbar.set_label('volumetric moisture [% m3/m3]')
        
        ##PLOT DIFF between day1 and day2
        plt.figure(figsize=(13, 7))
        plt.imshow(sm_diff, cmap = 'seismic_r',
                   interpolation='nearest',  #better visualization
                   vmin=-50, vmax=50)
        plt.title(date + '_interp')
        plt.ylim(2600, 1900)
        plt.xlim(3200, 4000)
        cbar = plt.colorbar()
        cbar.set_label('volumetric moisture [% m3/m3]')
    
# run plot_multiple_time_series.py before to compare with SM in la cendrosa
# and then:
#plt.plot(date_list, np.array(sm_cendrosa)/100, linestyle='none', marker='o')

# -> does not seem relevant in la cendrosa...
    