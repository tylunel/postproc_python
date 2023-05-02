#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 11:47:49 2022

@author: lunelt

CF LAB IA TOOLBOX: http://gitlab.meteo.fr/datascience/toolbox_datascience
"""

# ---------------------------------
# import
# ---------------------------------
import epygram
#import vortex	 #(will import 'footprints' & 'bronx' dependencies)
#import vtk

epygram.init_env()


# ----------------------------------
# read GRIB file (arpifs, ecmwf, arome)
# ----------------------------------
# import the GRIB file to read: 
fcstARP = epygram.formats.resource(
        filename='../MODELES/arpifs.AN.20160717.00', 
        openmode='r')
# list the fields of the file
headerARP = fcstARP.listfields()
# read a particular field:
fieldARP = fcstARP.readfield({'name':'Temperature', 'level':0})
#field_temp.dump_to_nc('soiltempl1')
#lons, lats = field.geometry.get_lonlat_grid()
# plot the field:
# - if field is H2DField type:
#field_snow.cartoplot()

fcstARP21 = epygram.formats.resource(
        filename='../MODELES/arpifs.AN.20210715.00', 
        openmode='r')
# list the fields of the file
headerARP21 = fcstARP21.listfields()
# read a particular field:
fieldARP21 = fcstARP21.readfield({'name':'Temperature', 'level':0})




# import the GRIB file to read: 
fcstCEP = epygram.formats.resource(
        filename='../MODELES/ecmwf.OD.20160717.00', 
        openmode='r')
# list the fields of the file
headerCEP = fcstCEP.listfields()
# read a particular field:
fieldCEP = fcstCEP.readfield({'name':'Temperature', 'level':1})



fcstARO = epygram.formats.resource(
        filename='../MODELES/arome.AN.20160717.00', 
#        filename='../MODELES/arome.AN.20210715.00',
        openmode='r')
# list the fields of the file
headerARO = fcstARO.listfields()
# read a particular field:
#fieldARO = fcstARO.readfield({'name':'Temperature', 'level':1})
#field_temp.dump_to_nc('soiltempl1')
#lons, lats = field.geometry.get_lonlat_grid()

surf = epygram.formats.resource(
        filename='../MODELES/INIT_SURF.20160717.00.lfi', 
        openmode='r')

PGD_oper = epygram.formats.resource(
        filename='../MODELES/PGD_oper_41t1.01km30.lfi', 
        openmode='r')

#%% With XARRAY or METVIEW - Pb
import xarray as xr
#import metview as mv

#ds = (mv.read("../MODELES/ecmwf.OD.20160717.00")).to_dataset()

data = xr.open_dataset('../MODELES/ecmwf.OD.20160717.00',
                       engine='cfgrib')





