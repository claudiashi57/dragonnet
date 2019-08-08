#!/usr/bin/env bash


options=(

    tarnet

)

for i in ${options[@]}; do
    echo $i
    python -m experiment.dragonnet --data_base_dir /Users/claudiashi/data/ihdp_csv_1-1000/csv\
                                 --knob $i\
                                 --output_base_dir /Users/claudiashi/result/ihdp_dep\
                                 --dataset 'ihdp'\

done