#!/usr/bin/env python3
"""
@author: tylunel
Creation : 07/01/2021

Script for plotting simple colormaps 
"""

import matplotlib.pyplot as plt
import xarray as xr
import tools
import shapefile
import pandas as pd
import global_variables as gv

###############################################
models = [
    '1_planier_0105/02_1km/', 
    '1_planier_0105/04_1km_COARE/',
    ]
folder_res = 'diff_1km-1km_COARE'

wanted_date = '20230105-2300'

color_map = 'coolwarm'    # jet, seismic, BuPu, coolwarm, viridis, RdYlGn, 
var_name = 'WS'

# level, only useful if var 3D
ilevel = 10  #0 is Halo, 1:2m, 2:6.12m, 3:10.49m, 10:49.3m, 20:141m, 30:304m, 40:600m, 50:1126m, 60:2070m

zoom_on = None  #None for no zoom, 'liaise' or 'urgell'

# Topo features
sea_borders = True
france_borders = False
sites_to_plot = [
    'planier',
    'marseille',
    ]
isoalti_list = [200, 400, 600,]

# Winds
add_winds = True
barb_size_option = 'weak_winds'  # 'weak_winds' or 'standard'

# Save
save_plot = True
save_folder = './figures/scalar_maps/{0}/{1}/{2}/'.format(
    folder_res, var_name, ilevel)

vmax_dict = {
    'HU2M_SEA': 0.01,
    'T2M_SEA': 1,
    'RI_SEA': 1,
    'WS': 1,
    'WSME': 1,
    'TKET': 0.5,
    }
vmax = vmax_dict[var_name]
vmin = -vmax

##############################################

prop = gv.zoom_domain_prop[zoom_on]
skip_barbs = prop['skip_barbs']
barb_length = prop['barb_length']
lat_range = prop['lat_range']
lon_range = prop['lon_range']
figsize = prop['figsize']

#%% LOAD DATA

var_layer = {}

for model in models:

    filename = tools.get_simu_filepath(
        model, wanted_date, global_simu_folder=gv.global_simu_folder)
    
    # load dataset, default datetime okay as pgd vars are all the same along time
    ds = xr.open_dataset(filename)
    
    #%% DIAG
    ds1 = tools.subset_ds(ds, 
                          zoom_on=zoom_on,
                          nb_indices_exterior=5,
                          )
    ds1 = tools.flux_pt_to_mass_pt(ds1, only_basic_vars=False)
    ds1['WS'], ds1['WD'] = tools.calc_ws_wd(ds1['UT'], ds1['VT'])
    # ds1['WSME'], ds1['WDME'] = tools.calc_ws_wd(ds1['UMME'], ds1['VMME'])
    
    #%% DATA SELECTION and ZOOM
    varNd = ds1[var_name]
    #remove single dimensions
    varNd = varNd.squeeze()
    
    if len(varNd.shape) == 2:
        var2d = varNd
    elif len(varNd.shape) == 3:
        var2d = varNd[ilevel,:,:]
        
    # remove 999 values, and replace by nan
    var2d = var2d.where(~(var2d == 999))
    # filter the outliers
    #var2d = var2d.where(var2d <= vmax)

    var_layer[model] = var2d


var_diff = var_layer[models[0]] - var_layer[models[1]]
var_diff.coords.update(var_layer[models[0]].coords)

#%% PLOT OF VAR_NAME

if figsize is None:
    xaxis_dist = float(ds.ni.max() - ds.ni.min())  # in km
    yaxis_dist = float(ds.nj.max() - ds.nj.min())  # in km
    scale = 1/20000
    figsize = (xaxis_dist*scale*1.2, yaxis_dist*scale)

fig1 = plt.figure(figsize=figsize)

plt.contourf(var_diff.longitude, var_diff.latitude, var_diff,
#plt.pcolormesh(var_diff.longitude, var_diff.latitude, var_diff,
#               cbar_kwargs={"orientation": "horizontal", "shrink": 0.7}
             cmap=color_map,
             vmin=vmin, vmax=vmax,
             levels=40
             )

cbar = plt.colorbar(boundaries=[vmin, vmax])
try:
    cbar.set_label(var2d.long_name)
except AttributeError:
    cbar.set_label(var2d.name)
#cbar.set_clim(vmin, vmax)

#%% WIND BARBS

barb_size_increments = gv.barb_size_increments
barb_size_description = gv.barb_size_description

if add_winds:
    X = ds.longitude
    Y = ds.latitude
    U = ds.UT[0, ilevel, :,:]
    V = ds.VT[0, ilevel, :,:]
    
    plt.barbs(X[::skip_barbs, ::skip_barbs], Y[::skip_barbs, ::skip_barbs], 
              U[::skip_barbs, ::skip_barbs], V[::skip_barbs, ::skip_barbs],
              pivot='middle',
              length=barb_length,     #length of barbs
              sizes={
    #                 'spacing':1, 
    #                 'height':1,
    #                 'width':1,
                     'emptybarb':0.01},
              barb_increments=barb_size_increments[barb_size_option]
              )
    plt.annotate(barb_size_description[barb_size_option],
                 xy=(0.1, 0.05),
                 xycoords='subfigure fraction'
                 )

#%% TOPO FEATURES

pgd_filename = f'/home/lunelt/Data/mnh_run/{model}/PGD_1KM.nc'
pgd = xr.open_dataset(pgd_filename)

# --- Sea borders
if sea_borders:
    sea_covers = pgd.COVER001.data
    plt.contour(pgd.longitude.data, 
                pgd.latitude.data, 
                sea_covers,
                levels=0,   #+1 -> number of contour to plot 
                linestyles='solid',
                linewidths=2,
                colors='w'
        #        colors=['None'],
        #        hatches='-'
                )

# --- France borders
if france_borders:
    sf = shapefile.Reader("TM-WORLD-BORDERS/TM_WORLD_BORDERS-0.3.sph")
    shapes=sf.shapes()
    france = shapes[64].points
    france_df = pd.DataFrame(france, columns=['lon', 'lat'])
    # france_temp = france_df[france_df.lat < 43.35]
    # france_subset = france_temp[france_S.lon < 2.95]
    france_subset = france_df
    plt.plot(france_subset.lon, france_subset.lat,
             color='k',
             linewidth=1)

# --- ISOLINES FOR ALTI
# contour line of isoaltitude
isoalti = ds['ZS']
plt.contour(isoalti.longitude.data, 
            isoalti.latitude.data, 
            isoalti,
            levels=isoalti_list,   #+1 -> number of contour to plot 
            linestyles=':',
            linewidths=1.,
            colors='k',
            )

# --- POINTS SITES
sites = {key:gv.whole[key] for key in sites_to_plot}

for site in sites:
    plt.scatter(sites[site]['lon'],
                sites[site]['lat'],
                color='k',
                s=12        #size of markers
                )
    # print site name on fig:
    try:
        sitename = sites[site]['longname']  # 'acronym', 'longname'
    except KeyError:
        sitename = site
        
    plt.text(sites[site]['lon']+0.01,
             sites[site]['lat']+0.01, 
             sitename, 
             fontsize=14)


#%% FIGURE OPTIONS and ZOOM
if len(varNd.shape) == 2:
    plot_title = '{0} - {1} diff \nbetween {2} and {3}'.format(
    wanted_date, var_name, models[0], models[1])
elif len(varNd.shape) == 3:
    plot_title = '{0} - {1} diff between {2} and {3} at {4}m'.format(
        wanted_date, var_name, models[0], models[1], var2d.level.round())
        # wanted_date, var_name, models[0], models[1], var2d.level_w.round())

plt.suptitle(plot_title)

if zoom_on is None:
    plt.ylim([var2d.latitude.min(), var2d.latitude.max()])
    plt.xlim([var2d.longitude.min(), var2d.longitude.max()])
else:
    plt.ylim(lat_range)
    plt.xlim(lon_range)

if save_plot:
    save_title = f'{wanted_date} - {var_name} diff'
    tools.save_figure(save_title, save_folder)
