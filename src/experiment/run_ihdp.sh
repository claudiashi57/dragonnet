#!/usr/bin/env bash


options=(
    dragonnet
    nednet
    tarnet
)

for i in ${options[@]}; do
    echo $i
    python -m experiment.dragonnet --data_base_dir ../data/ihdp/csv\
                                 --knob $i\
                                 --output_base_dir ../result/ihdp_paper/dep
done