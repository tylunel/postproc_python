#!/bin/bash
# Code running python script plot_time_series.py for multiple variables

script=plot_time_series.py

#For SIMU and OBS:

arr_obs_var=(ta_2 hus_2 soil_temp_1 soil_temp_2 swi_1 swi_2 shf_1 lhf_1)
#arr_obs_var=(TEMP_2m RHO_2m ST04 ST10 SWI10_subsoil H_2m LE_2m)
arr_sim_var=(T2M_ISBA Q2M_ISBA TG2_ISBA TG3_ISBA SWI3_ISBA H_ISBA LE_ISBA)
#arr_sim_var=()

len_arr=${#arr_sim_var[@]}  #retrieves list length

i=0
while [ $i -lt ${len_arr} ]; do
  
  obs_var=${arr_obs_var[$i]}
  sim_var=${arr_sim_var[$i]}
  echo '--------------'
  echo $obs_var $sim_var
  sed -i -e "s/varname_obs = .*/varname_obs = \'${obs_var}\'/" ${script}
  sed -i -e "s/varname_sim = .*/varname_sim = \'${sim_var}\'/" ${script}
  
  python3 ${script}

  i=$(expr $i + 1)
done

#For SIMU only:
#arr_obs_var=(TEMP_2m RHO_2m ST10)
arr_sim_var=(SWI_T_ISBA GFLUX_ISBA RN_ISBA TG1_ISBA TG4_ISBA TG5_ISBA TG6_ISBA SWI1_ISBA SWI2_ISBA SWI3_ISBA SWI4_ISBA SWI5_ISBA FMU_ISBA FMV_ISBA)
#arr_sim_var=()

len_arr=${#arr_sim_var[@]}  #retrieves list length

i=0
while [ $i -lt ${len_arr} ]; do

  sim_var=${arr_sim_var[$i]}
  echo '--------------'
  echo $sim_var
  sed -i -e "s/varname_obs = .*/varname_obs = \'\'/" ${script}
  sed -i -e "s/varname_sim = .*/varname_sim = \'${sim_var}\'/" ${script}
  
  python3 ${script}

  i=$(expr $i + 1)
done

