#!/usr/bin/env python3
"""
@author: tylunel
Creation : 07/01/2021

Script for plotting simple colormaps 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import tools
import shapefile
import pandas as pd
import global_variables as gv
import metpy.calc as mcalc
from metpy.units import units
from shapely.geometry import Polygon
import os
from metpy.plots import StationPlot

###############################################
model = 'irrswi1_d1'  # irrswi1_d1, std_d1

domain_nb = 1

wanted_date = '20210722-1200'

color_map = 'coolwarm'   # BuPu, coolwarm, viridis, RdYlGn, jet,... (add _r to reverse)
                    # YlOrBr for orography
                    # RdYlGn, YlGn for LAI
                    # coolwarm_r for SWI

var_name = 'TI'   #LAI_ISBA, ZO_ISBA, PATCHP7, ALBNIR_S, MSLP, TG1_ISBA, RAINF_ISBA, CLDFR, TSWI_T_ISBA, SWI3_ISBA
vmin = 0
vmax = 0.2
var_name_long = 'TI [%]'

# level, only useful if var 3D
ilevel = 20  #0 is Halo, 1:2m, 2:6.12m, 3:10.49m, 10:49.3m, 20:141m, 30:304m, 40:600m, 50:1126m, 60:2070m, 66:2930m
ilevel_wind = 20

zoom_on = None  #None for no zoom, 'liaise' or 'urgell'

add_winds = True
add_smc_obs = False
add_pgf = False
marinada_areas = False
sea_borders = True
france_borders = False
barb_size_option = 'weak_winds'  # 'weak_winds' or 'standard'
sites_to_plot = [
    'planier',
    'marseille',
    ]

isoalti_list = [200, 400, 600,]
obs_circle_size = 50
cbar_loc = 'right'  # 'left', 'right', 'top', 'bottom'

save_plot = True
#save_folder = './figures/scalar_maps/pgd/'
#save_folder = './figures/scalar_maps/domain{0}/{1}/{2}/'.format(
#        domain_nb, model, var_name)

save_folder = f'./figures/planier/{var_name}/{ilevel}/'
##############################################

wanted_month = str(pd.Timestamp(wanted_date).month).zfill(2)  # format with 2 figures
wanted_day = str(pd.Timestamp(wanted_date).day).zfill(2)

prop = gv.zoom_domain_prop[zoom_on]
skip_barbs = prop['skip_barbs']
barb_length = prop['barb_length']
lat_range = prop['lat_range']
lon_range = prop['lon_range']
figsize = prop['figsize']
# OR: #locals().update(gv.zoom_domain_prop[zoom_on])

skip_barbs = 12
barb_length = 7


if add_smc_obs:
    alpha = 0.4  # transparency of barbs
else:
    alpha = 0.5

# size of font on figure
plt.rcParams.update({'font.size': 11})

# filename = tools.get_simu_filepath(model, wanted_date,
#                                    global_simu_folder=gv.global_simu_folder)

filename = '/home/lunelt/Data/mnh_run/PLANI.1.SEG01.013.nc'
pgd_filename = '/home/lunelt/Data/mnh_run/PGD_1KM.nc'
# load dataset, default datetime okay as pgd vars are all the same along time
ds = xr.open_dataset(filename)
#ds1 = xr.open_dataset(
#        gv.global_simu_folder + \
#        '2.01_pgds_irr/PGD_400M_CovCor_v26_ivars.nc')

#%% DIAGs
ds1 = tools.subset_ds(ds, 
                      zoom_on=zoom_on,
                      nb_indices_exterior=5,
#                      lat_range=[lat_range[0],lat_range[1]+0.1], 
#                      lon_range=[lon_range[0],lon_range[1]+0.1],
                      )

ds1 = tools.center_uvw(ds1)

ds1['WS'], ds1['WD'] = tools.calc_ws_wd(ds1['UT'], ds1['VT'])
#TI
ds1['TI'] = np.sqrt(1.5*ds1['TKET'])/ds1['WS']

#ds_diag = tools.diag_lowleveljet_height_5percent(ds1[['WS', 'ZS']])

# subset_ds = ds1[['WS', 'ZS', 'TKET', 'HBLTOP']]

#ds1['DIV'] = mcalc.divergence(ds1['UT'], ds1['VT'])
#
#ds1['DENS'] = mcalc.density(
#    ds1['PRES']*units.hectopascal,
#    ds1['TEMP']*units.celsius, 
#    ds1['RVT']*units.gram/units.gram)
#
#ds1['MSLP3D'] = tools.calc_mslp(ds1, ilevel=None)


#%% DATA SELECTION and ZOOM

#varNd = ds_diag[var_name]
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


#%% PLOT OF VAR_NAME

if figsize is None:
    xaxis_dist = float(ds.ni.max() - ds.ni.min())  # in km
    yaxis_dist = float(ds.nj.max() - ds.nj.min())  # in km
    scale = 1/20000
    figsize = (xaxis_dist*scale*1.2, yaxis_dist*scale)
    
if cbar_loc == 'bottom':
    figsize = (figsize[0]-2, figsize[1]+1)
    cbar_frac = 0.05
else:
    cbar_frac = 0.15  # default value


fig1 = plt.figure(figsize=figsize)

cmap = plt.cm.get_cmap(color_map).copy()

if var_name == 'ZS':
    vmin=0.1
    cmap.set_under('c')  # for plotting the sea in cyan

if var_name == 'TKET':
    norm_cm=mpl.colors.LogNorm(vmin=0.01, vmax=vmax)  # for TKE
else:
    norm_cm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)  # default normalization



# --- COLORMAP
#plt.contourf(var2d.longitude, var2d.latitude, var2d,
#             levels=20,
#               levels=np.linspace(vmin, vmax, (vmax-vmin)*2+1), # fixed colorbar
plt.pcolormesh(var2d.longitude, var2d.latitude, var2d,
#               cbar_kwargs={"orientation": "horizontal", "shrink": 0.7}
               cmap=cmap,
               norm=norm_cm,
#               extend = 'both',  #highlights the min and max in edges values
               # vmin=vmin, vmax=vmax,
               )

#cbar = plt.colorbar(boundaries=[vmin, vmax])
cbar = plt.colorbar(location=cbar_loc,
#                    orientation=cbar_orientation,
                    fraction=cbar_frac
                    )

try:
    cbar.set_label(var_name_long)
except AttributeError:
    cbar.set_label(var_name)

#cbar.set_label('LAI [$m^2 m^{-2}$]')
    
#cbar.set_clim(vmin, vmax)

# --- WIND BARBS

barb_size_increments = gv.barb_size_increments
barb_size_description = gv.barb_size_description

if (add_winds & add_pgf):
    raise ValueError('Only add_winds or add pgf can be true')

if add_winds or add_pgf:
    X = ds1.longitude
    Y = ds1.latitude
    if add_winds:
        U = ds1['UT'].squeeze()[ilevel_wind, :,:]
        V = ds1['VT'].squeeze()[ilevel_wind, :,:]
    elif add_pgf:
        U = ds1['PGF_U'].squeeze()[ilevel_wind, :,:]
        V = ds1['PGF_V'].squeeze()[ilevel_wind, :,:]
    
    plt.barbs(X[::skip_barbs, ::skip_barbs], Y[::skip_barbs, ::skip_barbs], 
              U[::skip_barbs, ::skip_barbs], V[::skip_barbs, ::skip_barbs],
              pivot='middle',
              length=barb_length,     #length of barbs
              sizes={
    #                 'spacing':1, 
    #                 'height':1,
    #                 'width':1,
                     'emptybarb':0.01},
              barb_increments=barb_size_increments[barb_size_option],
              alpha=alpha,
              )
    plt.annotate(barb_size_description[barb_size_option],
                 xy=(0.1, 0.01),
                 xycoords='subfigure fraction',
                 
                 )

#%% STATION PLOT
plot_barb_wind_obs = False

if add_smc_obs:
    ax = plt.gca()
    
    if var_name == 'T2M_ISBA':
        var_name_obs = 'T_kelvin'
    else:
        var_name_obs = var_name
        
    # --- with global SMC data ---
    datafolder = gv.global_data_liaise + '/SMC/ALL_stations_july/'
    for filename in os.listdir(datafolder):
#        filename = 'C6.nc'
        file_path = os.path.join(datafolder, filename)
        if os.path.isfile(file_path):
            try:
                # --- open and pre-process the station data ---
                obs = xr.open_dataset(file_path)
                # struggling with the datetime formats in the 3 next lines...
                obs['datetime'] = [pd.Timestamp(str((elt.data))) for elt in obs['datetime']]
                obs['datetime64'] = [elt.data.astype('int64') for elt in obs['datetime']]
                obs['datetime64'] = obs['datetime64'].swap_dims({'datetime64' :'datetime'})
                # compute distance to wanted datetime
                obs['time_dist'] = np.abs(obs['datetime64'] - pd.Timestamp(wanted_date).to_datetime64().astype('int64'))
                # keep the data closest to wanted datetime
                obs_t = obs.where(obs['time_dist'] == obs['time_dist'].min(), 
                                  drop=True).squeeze()
                # get height of wind measurement
                wind_height = int((obs_t['obs_wind_height'].data))
                obs_t['UT'], obs_t['VT'] = tools.calc_u_v(
                    obs_t[f'VV{wind_height}'], obs_t[f'DV{wind_height}'])
                obs_t['T_kelvin'] = obs_t['T'] + 273.15

                obs_t['P_pa'] = tools.height_to_pressure_std(
                        obs_t['altitude'], p0=ds1['MSLP'].mean()*100)
                obs_t['THT'] = tools.potential_temperature_from_temperature(
                    obs_t['P_pa'], obs_t['T_kelvin'])
                obs_t['RVT'] = tools.psy_ta_rh(
                        obs_t['T'], obs_t['HR'])['mixing_ratio']  # in kg/kg
                obs_t['MRV'] = obs_t['RVT']*1000    # in g/kg

                # --- plot station data ---
                if (lon_range[0] < obs['lon'] < lon_range[1] and \
                    lat_range[0] < obs['lat'] < lat_range[1]):
                    if plot_barb_wind_obs:
                        # Create the station object and set the location of it
                        location = StationPlot(ax, obs['lon'], obs['lat'])
                        # plot the wind
                        location.plot_barb(
                            obs_t['UT'].data, obs_t['VT'].data,
                               pivot='tip',  # 'tip' or 'middle'
                               length=barb_length*barb_length_coeff,     #length of barbs
                               sizes={'emptybarb':0.1},
                               barb_increments=barb_size_increments[barb_size_option]
                               )
                        # add wind measurement height if different from 10m
                        if wind_height != 10:
                            ax.text(obs['lon']+0.008, obs['lat']-0.016, 
                                     wind_height, 
                                     fontsize=7)
                    # plot a scalar variable in the circle
                    ax.scatter(obs['lon'], obs['lat'],
                                color=cmap((obs_t[var_name_obs]-vmin)/(vmax-vmin)),
                                edgecolors='k', s=obs_circle_size)
#                    # plot a scalar variable in the circle
#                    ax.scatter(obs['lon'], obs['lat'],
#                                color=cmap((obs_t[var_name]-vmin)/(vmax-vmin)),
#                                edgecolors='k')
                    # state that this station was plotted
                    print(f'{filename} plotted')
            except (FileNotFoundError, ValueError, IndexError, TypeError) as e:
                print(f"Error for {obs['station_name']}:")
                if hasattr(e, 'message'):
                    print(e.message)
                else:
                    print(e)
                continue
        
    # --- with Els Plans and Cendrosa data ---
    for site in ['elsplans', 'cendrosa']:
        if site == 'elsplans':
            datafolder = gv.global_data_liaise + '/elsplans/mat_50m/5min_v4/'
            out_filename_obs = f'LIAISE_ELS-PLANS_UKMO_MTO-05MIN_L2_2021{wanted_month}{wanted_day}_V4.0.nc'
            obs = xr.open_dataset(datafolder + out_filename_obs)
            
            # keep time of interest
            obs['time_dist'] = np.abs(obs.time - pd.Timestamp(wanted_date).to_datetime64())
            obs_t = obs.where(obs['time_dist'] == obs['time_dist'].min(), 
                                 drop=True).squeeze()
            # if two datetime are as close to required datetime, keep the first
            try:
                obs_t = obs_t.isel(time=0)
                print("""Warning: Multiple data found close to wanted_date -
                          only first is kept""")
            except ValueError:
                pass
            # check that time dist is ok
            if obs_t['time_dist'] > pd.Timedelta(35, 'min'):
                obs_t = obs_t * np.nan
                
            obs_t['UT'], obs_t['VT'] = tools.calc_u_v(
                    obs_t[f'UTOT_10m'], obs_t[f'DIR_10m'])
            obs_t['T_kelvin'] = obs_t['TEMP_2m'] + 273.15
            obs_t['MRV'] = obs_t['RHO_2m']      # in g/kg
            obs_t['RVT'] = obs_t['MRV']/1000    # in kg/kg
            
        elif site == 'cendrosa':
            freq = 30
            datafolder = gv.global_data_liaise + f'/cendrosa/{freq}min/'
            filename = f'LIAISE_LA-CENDROSA_CNRM_MTO-FLUX-{freq}MIN_L2_2021-{wanted_month}-{wanted_day}_V2.nc'
            obs = xr.open_dataset(datafolder + filename)
            # keep time of interest
            obs['time_dist'] = np.abs(obs.time - pd.Timestamp(wanted_date).to_datetime64())
            obs_t = obs.where(obs['time_dist'] == obs['time_dist'].min(), 
                                 drop=True).squeeze()
            # check that time dist is ok
            if obs_t['time_dist'] > pd.Timedelta(35, 'min'):
                obs_t = obs_t * np.nan
            
            obs_t['UT'], obs_t['VT'] = tools.calc_u_v(
                    obs_t[f'ws_2'], obs_t[f'wd_2'])
            obs_t['T_kelvin'] = obs_t['ta_2'] + 273.15
            obs_t['MRV'] = obs_t['hus_2']      # in g/kg
            obs_t['RVT'] = obs_t['MRV']/1000    # in kg/kg

        obs_t['P_pa'] = tools.height_to_pressure_std(
                gv.sites[site]['alt'], p0=ds1['MSLP'].mean()*100)
        obs_t['THT'] = tools.potential_temperature_from_temperature(
                obs_t['P_pa'], obs_t['T_kelvin'])
        
        # Create the station object and set the location of it
        
        # plot the wind
        if plot_barb_wind_obs:
            location = StationPlot(ax, gv.sites[site]['lon'], gv.sites[site]['lat'])
            location.plot_barb(
                obs_t['UT'].data, obs_t['VT'].data,
                   pivot='tip',  # 'tip' or 'middle'
                   length=barb_length*barb_length_coeff,     #length of barbs
                   sizes={'emptybarb':0.1},
                   barb_increments=barb_size_increments[barb_size_option]
                   )
        # plot a scalar variable in the circle
        ax.scatter(gv.sites[site]['lon'], gv.sites[site]['lat'],
                    color=cmap((obs_t[var_name_obs]-vmin)/(vmax-vmin)),
                    edgecolors='k', s=obs_circle_size)
        # state that this station was plotted
        print(f'{site} plotted')


# --- IRRIGATED, SEA and COUNTRIES BORDERS

pgd = xr.open_dataset(pgd_filename)

# Sea borders
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

#France borders
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
isoalti = ds1['ZS']
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
    if add_smc_obs and site in ['elsplans', 'cendrosa', 'V1', 'WL', 'C6',]:
        pass
    else:
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

# --- AREAS FOR MARINADA STUDY
if marinada_areas:
    areas_corners = gv.areas_corners
    polygon_dict = {}
    
    for it_area, area in enumerate(areas_corners):
        print(area)
        corners = areas_corners[area]
        corners_coordinates = []
        for corner in corners:
            corners_coordinates.append(
                (gv.whole[corner]['lon'], gv.whole[corner]['lat']))
    
        polygon = Polygon(corners_coordinates)
        polygon_dict[area] = polygon
        plt.plot(*polygon.exterior.xy)
    #    
    #    # add name in the middle of the shapes:
    #    plt.text(polygon.centroid.xy[0][0]-0.05,
    #             polygon.centroid.xy[1][0]-0.05, 
    #             area, 
    #             fontsize=12,
    #             fontstyle='italic')


# --- FIGURE OPTIONS and ZOOM
    
if len(varNd.shape) == 2:
    plot_title = '{0} - {1} for simu {2}'.format(
        wanted_date, var_name, model)
elif len(varNd.shape) == 3:
    plot_title = '{0} - {1} for simu {2} at {3}m'.format(
        wanted_date, var_name, model, var2d.level.round())
    
plot_title = f'{var_name} at {var2d.level.round()}m'

plt.title(plot_title)
plt.xlabel('longitude', fontsize=12)
plt.ylabel('latitude', fontsize=12)

plt.ylim(lat_range)
plt.xlim(lon_range)

plt.subplots_adjust(left=0.1, right=1, top=0.90, bottom=0.1)

if save_plot:
    tools.save_figure(plot_title, save_folder)

#%%
#    
#plt.figure()
#p9_filt = ds1.PATCHP9.where(ds1.PATCHP9 < 2, other=0)
#plt.pcolormesh(p9_filt.longitude.data,p9_filt.latitude.data,p9_filt, vmin=0, vmax=1)
#
#sea_covers = pgd.COVER001.data
#p9_filt['sea'] = p9_filt*0 + sea_covers
#p9_sea = p9_filt.where(p9_filt.sea == 1, np.nan)
#plt.pcolormesh(p9_sea.longitude.data,p9_sea.latitude.data,p9_sea, vmin=0, vmax=1, cmap='binary')
