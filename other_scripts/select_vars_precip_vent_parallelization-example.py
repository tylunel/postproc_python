# -*- coding: utf-8 -*-
"""
Arnaud Forster   -   arnaud.forster@meteo.fr

Original : 23/06/2023
"""

import os 
import sys
from multiprocessing import Pool

########################################
######  A COMPLETER ####################
########################################

domaine = '2'
VILLE = 'True'

#liste_membres = ['mb001','mb002','mb003','mb004','mb005','mb006']#,'mb007','mb008','mb009','mb010','mb011','mb012','mb013','mb014','mb015']
#liste_membres = ['mb007','mb008','mb009','mb010','mb011','mb012','mb013','mb014','mb015']
liste_membres = ['mb013']# sert de boucle mais n'est pas utilise pour ARO12 
#liste_echs = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15']
liste_echs = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25']

variables = 'ACPRT,INPRT,UT,VT,latitude_u,latitude_v,longitude_u,longitude_v'

nb_proc = 15


if VILLE == 'True':
   runs = 'UR9'
   type_simu = 'URB'
else :
   runs = 'NU9'
   type_simu = 'NOURB'

os.system('CDO')



#fonction rapide pour selectionner les variables en parallele pour les echeances :

def select(echeance):
    print(echeance)
    os.system('ncks -O -h -v {a} -d level,12.8 -d level,513.949,3565.73 {b}.{c}.SEG01.0{d}.nc LIGHT/PRECIP_VENT_{b}.{c}.SEG01.0{d}_light.nc'.format(a=variables,b=run,c=domaine,d=echeance))
    



for membre in liste_membres:
    #run = runs+membre[-2:]
    run = type_simu[:3]+'12'
    #adresse_dossier = '/scratch/work/forstera/07052022/AROME_EPS/RESEAU_9UTC_NEW_PGD/'+membre+'/'+type_simu+'/008_runmod12/'
    adresse_dossier = '/home/cnrm_other/ge/mrmu/forstera/init12h/'+type_simu+'/008_runmod12/'
    os.chdir(adresse_dossier)
    if os.path.exists('./LIGHT'):
       print('file OK')
    else:
       print('file not available')
       os.system('mkdir ./LIGHT')


    if __name__ == '__main__':
       # create the process pool
       with Pool() as pool:
           # call the same function with different data in parallel
           p = Pool(nb_proc)
           #for result in p.map(task, range(20)):
           # report the value to show progress
           #    print(result)

           L = p.map(select, liste_echs)
           p.close()
  




