#!/bin/bash

#SBATCH -A sml
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-user=victorveitch@gmail.com
#SBATCH --mail-type=ALL

source activate ce-dev


DATA_DIR=/proj/sml_netapp/dat/undocumented/ihdp/csv/

OUTPUT_DIR=/proj/sml_netapp/projects/targeted_regularization/ihdp_notest

echo "  python -m experiment.dragonnet --data_base_dir $DATA_DIR\
                                 --knob $SET\
                                 --output_base_dir $FOD"


python -m experiment.dragonnet --data_base_dir $DATA_DIR\
                                 --knob $SET\
                                 --output_base_dir $OUTPUT_DIR
