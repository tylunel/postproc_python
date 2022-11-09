#!/bin/bash
# Code running python script for whole day

script=$1

day='20210722-'
list_time='0600 0700 0800 0900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000'

# sign & allows for parallel runs
for time in ${list_time}; do
  datetime=\'${day}${time}\'
  echo '--------------'
  echo $datetime
  sed -i -e "s/wanted_date = .*/wanted_date = ${datetime}/" ${script}
  python3 ${script}
done
