#!/bin/bash

#SBATCH -A sml
#SBATCH -c 8
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-user=victorveitch@gmail.com
#SBATCH --mail-type=ALL

source activate ce-dev


DATA_DIR=/proj/sml_netapp/dat/undocumented/ACIC/2000_acic/scaling

OUTPUT_DIR=/proj/sml_netapp/projects/targeted_regularization/2000_acic/



echo "  python -m experiment.dragonnet --data_base_dir $DATA_DIR\
                                  --output_base_dir $OUTPUT_DIR\
                                 --knob $SIM
                                 --folder $FOD"



python -m experiment.dragonnet --data_base_dir $DATA_DIR\
                               --output_base_dir $OUTPUT_DIR\
                                 --knob $SIM\
                                 --folder $FOD
