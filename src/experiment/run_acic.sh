#!/usr/bin/env bash

# script should be run from 'src'

list=(
    nednet

)

folders=(
    a
)

for i in ${list[@]}; do
    for folder in ${folders[@]}; do
        echo $i
        echo $folder
        python -m experiment.dragonnet --data_base_dir /Users/claudiashi/data/small_sweep\
                                     --knob $i\
                                     --folder $folder\
                                     --output_base_dir /Users/claudiashi/result/acic_2018\
                                     --dataset "acic"
    done

done
