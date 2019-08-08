#!/usr/bin/env bash

# script should be run from 'src'

list=(

    tarnet

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
        python -m experiment.dragonnet --data_base_dir /Users/claudiashi/data/ACIC_2018_SPLIT/scaling\
                                     --knob $i\
                                     --folder $folder\
                                     --output_base_dir /Users/claudiashi/result/acic_2018/dep\
                                     --dataset "acic"
    done

done
