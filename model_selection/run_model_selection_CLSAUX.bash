#!/bin/bash

dataset='CLSAUX'
main_metric_label='aucpr'
cp_list_file='final_CP_list_CLSAUX.csv'

{
tstart=`date +%s.%N`
date1=`date`
echo $date1

python -u select_optimal_model.py \
  --dataset $dataset \
  --main_metric_label $main_metric_label \
  --cp_list_file $cp_list_file

tend=`date +%s.%N`
date2=`date`

echo $date2

awk -v tstart=$tstart -v tend=$tend 'BEGIN{time=tend-tstart; printf "TIME [s]: %12.2f\n", time}'
} > out1.dat 2> out2.dat


