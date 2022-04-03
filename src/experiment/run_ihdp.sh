#!/usr/bin/env bash

options=(
    dragonnet
    tarnet
)

for i in ${options[@]}
do
  echo "$i"
  python -m experiment.ihdp_main --data_base_dir ../dat/ihdp/csv --knob $i \
                                 --output_base_dir ../result/ihdp
done