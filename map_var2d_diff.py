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
    '2_planier_0122/05_1km_RM17',
    # '1_planier_0105/02_1km'
    '2_planier_0122/0411_40m',
    ]

folder_res = '{expe}/diff_{simu1}-{simu2}'.format(
    expe = models[0].split('/')[0],
    simu1 = models[0].split('/')[1],
    simu2 = models[1].split('/')[1])

wanted_date = '20230122-1200'

color_map = 'coolwarm'    # jet, seismic, BuPu, coolwarm, viridis, RdYlGn, 

var_name_model_dict = {
    # '2_planier_0122/04_1km_COARE': 'TKEMME',
    '2_planier_0122/05_1km_RM17': 'TKEMME',
    # '1_planier_0105/02_1km': 'WS',
    '2_planier_0122/0411_40m': 'TKEBAVG025',
    }
var_name_main = list(var_name_model_dict.values())[0]
var_name_combined = '-'.join(list(var_name_model_dict.values()))

filetype_model_dict = {
    '2_planier_0122/04_1km_COARE': 'backup',
    '2_planier_0122/05_1km_RM17': 'backup',
    '2_planier_0122/0411_40m': 'diag',
    }

# level, only useful if var 3D
# ilevel = 10  #0 is Halo, 1:2m, 2:6.12m, 3:10.49m, 10:49.3m, 20:141m, 30:304m, 40:600m, 50:1126m, 60:2070m
wanted_alti = 140

zoom_on = 'planier_40m'  #None for no zoom, 'liaise', 'planier_40m'

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
save_folder = './figures/scalar_maps/{0}/{1}/{2}m/'.format(
    folder_res, var_name_combined, wanted_alti)

vmax_dict = {
    'HU2M_SEA': 0.01,
    'T2M_SEA': 1,
    'RI_SEA': 1,
    'WS': 1,
    'WSME': 1,
    'TKET': 1,
    'TKEMME': 2,
    }
vmax = vmax_dict[var_name_main]
vmin = -vmax

##############################################

prop = gv.zoom_domain_prop[zoom_on]
# skip_barbs = prop['skip_barbs']
skip_barbs = 2
# barb_length = prop['barb_length']
barb_length = 6
lat_range = prop['lat_range']
lon_range = prop['lon_range']
figsize = prop['figsize']

#%% LOAD DATA

var_layer = {}

for model in models:

    filename = tools.get_simu_filepath(
        model, wanted_date, global_simu_folder=gv.global_simu_folder,
        output_type=filetype_model_dict[model])
    
    # load dataset, default datetime okay as pgd vars are all the same along time
    ds = xr.open_dataset(filename)
    
    #%% DIAG
    ds1 = tools.subset_ds(ds, 
                          zoom_on=zoom_on,
                          nb_indices_exterior=5,
                          )
    ds1 = tools.flux_pt_to_mass_pt(ds1, only_basic_vars=True)
    try:
        ds1['WS'], ds1['WD'] = tools.calc_ws_wd(ds1['UT'], ds1['VT'])
    except KeyError:  # typically: no UT, VT in diag file
        pass
    # ds1['WSME'], ds1['WDME'] = tools.calc_ws_wd(ds1['UMME'], ds1['VMME'])
    
    #%% DATA SELECTION and ZOOM
    var_name = var_name_model_dict[model]
    varNd = ds1[var_name]
    #remove single dimensions
    varNd = varNd.squeeze()
    
    if len(varNd.shape) == 2:
        var2d = varNd
    elif len(varNd.shape) == 3:
        # var2d = varNd[ilevel,:,:]
        var2d = varNd.interp(level=wanted_alti)
        
    # remove 999 values, and replace by nan
    var2d = var2d.where(~(var2d == 999))
    # filter the outliers
    #var2d = var2d.where(var2d <= vmax)

    var_layer[model] = var2d

data_coarse = var_layer[models[0]]
data_fine = var_layer[models[1]]
data_coarse_refined = data_coarse.interp(ni=data_fine.ni, nj=data_fine.nj)

var_diff = data_coarse_refined - data_fine
# reset lat/lon coords
var_diff.coords.update(data_fine.coords)

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
cbar.set_label(var_name_combined)
#cbar.set_clim(vmin, vmax)

#%% WIND BARBS

barb_size_increments = gv.barb_size_increments
barb_size_description = gv.barb_size_description

if add_winds:
    try:
        # U = ds.UT[0, ilevel, :,:]
        # V = ds.VT[0, ilevel, :,:]
        U = ds['UT'].squeeze().interp(level=wanted_alti)
        V = ds['VT'].squeeze().interp(level=wanted_alti)
    except KeyError:  # typically: no UT, VT in second (diag) file
        # reload dataset of first model
        model = models[0]
        filename = tools.get_simu_filepath(
            model, wanted_date, global_simu_folder=gv.global_simu_folder,
            output_type=filetype_model_dict[model])
        ds = xr.open_dataset(filename)
        # and get U and V
        U = ds['UT'].squeeze().interp(level=wanted_alti)
        V = ds['VT'].squeeze().interp(level=wanted_alti)
    
    X = ds.longitude
    Y = ds.latitude

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
    plt.annotate(f'barbs correspond to abs. wind of model {model} (not diff)',
                 xy=(0.1, 0.025),
                 xycoords='subfigure fraction'
                 )

#%% TOPO FEATURES
try:
    if '_1km' in model:
        pgd_filename = f'{gv.global_simu_folder}/{model}/PGD_1KM.nc'
    elif '_200m' in model:
            pgd_filename = f'{gv.global_simu_folder}/{model}/PGD_200M.nc'
    elif '_40m' in model:
        pgd_filename = f'{gv.global_simu_folder}/{model}/PGD_40M.nc'
    else:
        raise KeyError
except KeyError:
    pgd_filename = f'{gv.global_simu_folder}/{model}/{gv.pgd_filename_dict[model]}'

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
    # plot_title = '{0} - {1} diff \nbetween {2} and {3}'.format(
    #     wanted_date, var_name, models[0], models[1])
    plot_title = '{0} - {1} - {2}'.format(
        wanted_date, var_name, folder_res)
    
elif len(varNd.shape) == 3:
    # plot_title = '{0} - {1} diff between {2} and {3} at {4}m'.format(
    #     wanted_date, var_name, models[0], models[1], int(var2d.level))
    plot_title = '{0} - {1} - {2} at {3}m'.format(
        wanted_date, var_name, folder_res, int(var2d.level))

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
