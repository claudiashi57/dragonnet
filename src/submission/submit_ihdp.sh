#!/usr/bin/env bash
#!/bin/sh

list=(
    dragonnet
    nednet
    tarnet

)

mkdir ssu-p output


for SET in ${list[@]}; do

    export SET=$SET
    NAME=${SET}
    sbatch --job-name=${NAME} \
        --output=output/${NAME}.out \
        ~/dragonnet/src/submission/run_ihdp.sh
done
