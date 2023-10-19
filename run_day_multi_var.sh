#!/bin/bash
# Code running python script for whole day

script=$1

list_day='20210721- '
list_time='0000 0100 0200 0300 0400 0500 0600 0700 0800 0900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100 2200 2300'

list_bu='UU VV TK TH RV WW'
list_nbi='0 1 2 3 4 5'

for bu in ${list_bu}; do
  budget=\'${bu}\'
  echo '------ budget_type = ------'
  echo $budget
  sed -i -e "s/budget_type = .*/budget_type = ${budget}/" ${script}

  for nbi in ${list_nbi}; do
    echo '------ i = ------'
    echo $nbi
    sed -i -e "s/nb_var = .*/nb_var = ${nbi}/" ${script}

    for day in ${list_day}; do
      for time in ${list_time}; do
        datetime=\'${day}${time}\'
        echo '---- datetime = -------'
        echo $datetime
        sed -i -e "s/wanted_date = .*/wanted_date = ${datetime}/" ${script}
        python3 ${script}
      done
    done
  done
done
