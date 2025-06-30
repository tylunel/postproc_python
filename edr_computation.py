#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 16:04:06 2025

@author: lunelt

"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import tools
import global_variables as gv
# import epygram; epygram.init_env()
# from scipy.fft import dct
import spectra_mod



#%% Parameters
model = '2_planier_0122/0411_40m'
# model = '5_planier_1020/041_200m'
# model = '7_planier_1017/041_200m'

wanted_date = '20230122-0230'

# var_name = 'TOT_TKE'   #LAI_ISBA, ZO_ISBA, PATCHP7, ALBNIR_S, MSLP, TG1_ISBA, RAINF_ISBA, CLDFR, TSWI_T_ISBA, SWI3_ISBA
# var_name_long = var_name

save_plot = False
save_folder = f'./figures/edr_from_les/{model}/'

#%% Load file

# size of font on figure
plt.rcParams.update({'font.size': 11})

filename = tools.get_simu_filepath(f'{model}', wanted_date,
                                   output_type='backup',
                                   global_simu_folder=gv.global_simu_folder,
                                   verbose=True)
    
# load dataset, default datetime okay as pgd vars are all the same along time
ds_orig = xr.open_dataset(filename)

wind_ds = ds_orig[['UT', 'VT', 'WT']]
wind_ds_cen = tools.flux_pt_to_mass_pt(wind_ds)

#%% Spectrum computation

# based on level
ilevel = 15
ds = wind_ds_cen.isel(level=ilevel).squeeze()
alti_agl = int(ds.level)

# based on alti
wanted_alti = 10  # alti agl
# ds = wind_ds_cen.interp(level=wanted_alti).squeeze()
ds = wind_ds_cen.sel(level=wanted_alti, method='nearest').squeeze()
alti_agl = wanted_alti

u1, v1, w1 = ds['UT'].data, ds['VT'].data, ds['WT'].data

# Subsetting:
iinf, isup, jinf, jsup = 100, 370, 10, 300
# ds_sub = ds[jinf:jsup, iinf:isup]
u1sub, v1sub, w1sub = u1[jinf:jsup, iinf:isup], v1[jinf:jsup, iinf:isup], w1[jinf:jsup, iinf:isup]

u1mean, v1mean, w1mean = u1sub.mean(), v1sub.mean(), w1sub.mean()
u_p = u1sub - u1mean
v_p = v1sub - v1mean
w_p = w1sub - w1mean
# plt.figure()
# plt.pcolormesh(u1)
ws, wd = tools.calc_ws_wd(u1sub, v1sub)
# ec_layer = 1/2*(u_p*u_p + v_p*v_p + w_p*w_p)
# ec_1 = 1/2*(u1*u1 + v1*v1 + w1*w1)
ec_1 = 1/2*(u1*u1 + v1*v1)
# ec_1 = 1/2*(w1*w1)
ec_layer = ec_1[jinf:jsup, iinf:isup]
# ec1 = 1/2*(u1*u1 + v1*v1)
# ec1 = 1/2*(w1*w1)
sigma_2u = np.var(u1sub)
sigma_2v = np.var(v1sub)
sigma_2w = np.var(w1sub)

# plt.pcolormesh(ec_layer)

# ec_layer = ec1[0:601, 0:401]
# ec_layer = ec1[0:301, 100:401]   # keep area with well developped turbulence

# Compute the 2D DCT and more
variances = spectra_mod.dctspectrum(ec_layer, verbose=True)
# dct_2d = dct(dct(ec_layer, axis=0, norm='ortho'), axis=1, norm='ortho')

spectres=[]

spectra = spectra_mod.Spectrum(
    variances,
    resolution=40,
    name=f"alti_{alti_agl}m")
spectres.append(spectra)

variances_mavr = tools.moving_average(spectra.variances, window_size=7)
spectra_mavr = spectra_mod.Spectrum(
    variances_mavr,
    resolution=40,
    name=f"alti_{alti_agl}m_mavr")
spectres.append(spectra_mavr)


abl_max_wavelength_index = np.where(spectra.wavelengths_m < 2000)[0][0]
res_min_wavelength_index = np.where(spectra.wavelengths_m > 3*40)[0][-1]

spectra_abl_variances = variances_mavr[abl_max_wavelength_index:res_min_wavelength_index]
spectra_abl_wavelength = spectra_mavr.wavelengths_m[abl_max_wavelength_index:res_min_wavelength_index]
spectra_abl_wavenumber = spectra_mavr.wavenumbers_m[abl_max_wavelength_index:res_min_wavelength_index]


#% Plot
plot_title = f"spectres_{model}_{wanted_date}_alti{alti_agl}m"

# fig, ax = spectra_mod.plotspectra(
#     spectres, 
#     over=(None, None), 
#     slopes=[{'exp': -5/3, 'offset': 10, 'label': '-5/3'}], 
#     zoom=None, 
#     unit='SI', 
#     title=plot_title, 
#     figsize=(10,8))

# ax.plot(spectra_abl_wavelength, 
#         spectra_abl_variances, 
#         color='g',
#         # linestyle=linestyles[i // len(colors)], 
#         label='abl_only')


fig, ax = spectra_mod.plotspectra_kS(
    spectres, 
    over=(None, None), 
    slopes=[{'exp': -2/3, 'offset': 0.001, 'label': '-2/3'}], 
    zoom=None,
    unit='SI', 
    title=None, 
    figsize=(10,8))

ax.plot(spectra_abl_wavelength, 
        spectra_abl_wavenumber*spectra_abl_variances, 
        color='g',
        # linestyle=linestyles[i // len(colors)], 
        label='abl_only')

if save_plot:
    tools.save_figure(plot_title, save_folder)

#%% Compute integral length scale L

x = spectra.wavenumbers_m
E = spectra.variances
kE = spectra.kvariances

integral_tot_E = np.trapezoid(E[:], x[:])
integral_tot_kE = np.trapezoid(kE[:], x[:])
L_i = integral_tot_E/integral_tot_kE
print(f'L_i = {L_i}')

#%% Kristensen fit
from scipy.optimize import curve_fit
from scipy.special import gamma

lamb = spectra_abl_wavelength
k = spectra_abl_wavenumber
S = spectra_abl_variances
kS = spectra_abl_wavenumber*spectra_abl_variances

sigma_2 = sigma_2u
sigma_2 = 0.01

def func_kristensen(k, L, mu):
    # mu=1.5
    a_mu = 3.1416*mu*gamma(5/(6*mu))/(gamma(1/(2*mu))*gamma(1/(3*mu)))
    coeff = L*(sigma_2)/3.1416
    numerator = 1 + (8/3)*(2*L*k/a_mu)**(2*mu)
    denominator = (1 + (2*L*k/a_mu)**(2*mu))**(1+5/(6*mu))
    return coeff*numerator/denominator

# params, cov = curve_fit(func_kristensen, k, S, p0=[L_i, 1.5])
params, cov = curve_fit(func_kristensen, k, S, p0=[300, 1.0], method='lm')
# # L, mu, sigma = params[0], params[1], params[2]
L, mu = params[0], params[1]
# L = params[0]
print('-----------')
print(f'L, mu = {params}')

#TODO: why does curve_fit doesnt provide covariance matrix? Issue?

# fitted_spectrum = func_kristensen(k, )
fitted_spectrum = func_kristensen(k, L, mu)

# L_perso = 500
# mu_perso = 1.5
# fitted_spectrum = func_kristensen(k, L_perso)

plt.plot(lamb, 
        k*fitted_spectrum, 
        color='r',
        # linestyle=linestyles[i // len(colors)], 
        label='abl_only')

#%% test own EDR retrieval method

# Define your x and y arrays
k = spectra.wavenumbers_m
y = spectra.variances

# Define the integration limits
x_start = 600   # waveLENGTH, in [m] for start of inertial subrange
x_end = 100      # waveLENGTH, in [m] for end of inertial subrange
k_start = 1/x_start
k_end = 1/x_end

# Find the indices corresponding to the integration limits
start_index = np.searchsorted(k, k_start)
end_index = np.searchsorted(k, k_end)

# Ensure the indices are within bounds
if start_index < 0 or end_index > len(x):
    raise ValueError("Integration limits are out of bounds.")

# Compute the integral using the trapezoidal rule
integral_K = np.trapezoid(y[start_index:end_index], x[start_index:end_index])

C = 0.5
edr = C * integral_K**1.5 / L_i
print(f'edr = {edr}')

#%% test edrlib
# /!\ Cf article of Nijhuis 2019, it seems not all methods can be applied to spatial turbulence

# import edrlib

# velocity_series                 = {}        #dictionary containing everything
# velocity_series['domain']       = 'space'   #either 'space' or 'time'
# velocity_series['dx']           = 40       #units: m

# #place here the velocity series, units: m/s
# velocity_series['y'] = ec_layer.flatten()
# # velocity_series['y'] = u1[300:302,:-1]

# edrlib.kolmogorov_constants(velocity_series, 'full')           #set Kolmogorov constants 
# edrlib.do_edr_retrievals(velocity_series, 
#                          methods=['variance', 'power_spectrum', '2ndorder'])           #do edr retrievals with different methods
# edrlib.printstats(velocity_series)                  #print retrieved edr values
# edrlib.makeplots(velocity_series)                   #make plots of it all

