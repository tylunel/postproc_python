#!/bin/bash
# Code running python script for whole day

script=$1

#list_day='20210715- 20210716- 20210717- 20210718- 20210719- 20210720- 20210721- 20210722- 20210723- 20210724- 20210725- 20210726- 20210727- 20210728- 20210729-'
list_day='20210721- 20210722- 20210723- 20210724-'
#list_time='0100 0400 0500 2100'
list_time='0100 0300 0400 0500 0600 0700 0800 0900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2100 2300'

for day in ${list_day}; do
  for time in ${list_time}; do
    datetime=\'${day}${time}\'
    echo '--------------'
    echo $datetime
    sed -i -e "s/wanted_date = .*/wanted_date = ${datetime}/" ${script}
    python3 ${script}
  done
done
