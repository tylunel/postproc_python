#!/bin/bash
# Code running python script for different models and whole days

script=$1
list_models='irr_d1 std_d1 irr_d2 std_d2'

for model in ${list_models}; do
  echo '--------------'
  echo $model
  sed -i -e "s/model = .*/model = \'${model}\'/" ${script}
  bash run_day.sh ${script}
done

