#!/usr/bin/env bash
#!/bin/sh

folders=(
    a
    b
    c
    d
    )
settings=(
    tarnet
    )


mkdir -p output

for SIM in  ${settings[@]}; do
    for FOLDER in ${folders[@]}; do
        export SIM=$SIM
        export FOD=$FOLDER
        NAME=${SIM}
        sbatch --job-name=${NAME} \
            --output=output/${NAME}.out \
             /submission/run_acic.sh
    done
done

