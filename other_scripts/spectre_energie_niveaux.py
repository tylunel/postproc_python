import matplotlib as mpl ; mpl.use('Agg')
import matplotlib.pyplot as plt
import epygram ; epygram.init_env()
import tools

############################
model = 'std_d1'

wanted_date = '20210722-1300'

save_folder = './figures/spectres/{0}/'.format(model)

save_plot = True  # if false, nothing prints...

zoom_on = False
izoom_lat = [150, 300]
izoom_lon = [300, 400]
##########################

filename = tools.get_simu_filename(model, wanted_date)

f = epygram.formats.resource(filename=filename, openmode='r')

varu='UT'
varv='VT'
varw='WT'
#ilevel = 40

utemp = f.readfield(varu)
vtemp = f.readfield(varv)
wtemp = f.readfield(varw)

niveaux=[10, 20, 30, 40, 45, 50, 55, 60, 65]
spectres=[]

for ilevel in niveaux:
    hauteur=str(int(round(f.geometry.vcoordinate.grid['gridlevels'][ilevel-1][1]['Ai'],0)))
    print(hauteur)
    
    if zoom_on is True:
        u1 = utemp.data[ilevel, izoom_lat[0]:izoom_lat[1], izoom_lon[0]:izoom_lon[1]]
        v1 = vtemp.data[ilevel, izoom_lat[0]:izoom_lat[1], izoom_lon[0]:izoom_lon[1]]
        w1 = wtemp.data[ilevel, izoom_lat[0]:izoom_lat[1], izoom_lon[0]:izoom_lon[1]]
    else:
        u1 = utemp.data[ilevel, :, :]
        v1 = vtemp.data[ilevel, :, :]
        w1 = wtemp.data[ilevel, :, :]
    
#    ec1 = 1/2*(w1*w1)
    ec1 = 1/2*(u1*u1 + v1*v1)
#    ec1 = 1/2*(u1*u1 + v1*v1 + w1*w1)
    ec_layer = ec1
#    ec_layer = ec1.getlevel(k=niveau).getdata()
    variances = epygram.spectra.dctspectrum(ec_layer)
    spectres.append(epygram.spectra.Spectrum(
        variances,
        resolution=1,
        name="Hauteur : "+hauteur+" m (niveau : "+str(ilevel)+")"))

#%%

fig, ax = epygram.spectra.plotspectra(
    spectres, 
    over=(None, None), 
    slopes=[{'exp': -3, 'offset': 1, 'label': '-3'}, 
            {'exp': -1.66667, 'offset': 1, 'label': '-5/3'}], 
    zoom=None, 
    unit='SI', 
    title=None, 
    figsize=None)

if zoom_on is True:
    plot_title = "spectres_{0}_{1}_zoomed_on_{2}-{3}".format(
            model, wanted_date, izoom_lat, izoom_lon)
else:
    plot_title = "spectres_{0}_{1}".format(model, wanted_date)
    
plt.title(plot_title)

if save_plot:
    tools.save_figure(plot_title, save_folder)



