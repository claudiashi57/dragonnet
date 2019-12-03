#!/usr/bin/env bash

# script should be run from 'src'

list=(
    tarnet
    nednet
    dragonnet

)

folders=(
    a
    b
    c
    d
)

for i in ${list[@]}; do
    for folder in ${folders[@]}; do
        echo $i
        echo $folder
        python -m experiment.acic_main --data_base_dir /Users/claudiashi/data/ACIC_2018/\
                                     --knob $i\
                                     --folder $folder\
                                     --output_base_dir /Users/claudiashi/ml/dragonnet/result/acic\

    done

done
