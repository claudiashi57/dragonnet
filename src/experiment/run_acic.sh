#!/usr/bin/env bash

# script should be run from 'src'

list=(
    nednet
    dragonnet
    tarnet
)

folders=(
    a
)

for i in ${list[@]}; do
    for folder in ${folders[@]}; do
        echo $i
        echo $folder
        python -m experiment.dragonnet --data_base_dir ~/dat/acic\
                                     --knob $i\
                                     --folder $folder\
                                     --output_base_dir ~/result/acic_2018
    done

done
