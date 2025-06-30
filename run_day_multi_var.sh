#!/bin/bash
# Code running python script for whole day

script=$1

list_day='20230105- '
list_time='0100 0200 0300 0400 0500 0600 0700 0800 0900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100 2200 2300'

list_var='T2M_SEA HU2M_SEA RI_SEA TKET WS'

for var in ${list_var}; do
  echo '---- var = -----'
  echo $var
  sed -i -e "s/var_name = .*/var_name = \'${var}\'/" ${script}
  for day in ${list_day}; do
    for time in ${list_time}; do
      datetime=\'${day}${time}\'
      echo 'datetime: ' $datetime
      sed -i -e "s/wanted_date = .*/wanted_date = ${datetime}/" ${script}
      python3 ${script}
    done
  done
done
