# Introduction

This repository contains software and data for "Adapting Neural Networks for the Estimation of Treatment Effects" (`arxiv:TBD`).
The paper describes This paper addresses the use of neural networks for the estimation of treatment effects from observational data.
This software builds on


# Requirements and setup
You will need to install tensorflow 1.13, sklearn, numpy 1.15, keras 2.2.4 and, pandas 0.24.1

# Data

1. IHDP
This dataset is based on a randomized experiment investigating the effect of home visits by specialists on future cognitive scores.
It is generated via the npci package `https://github.com/vdorie/npci`.
We also uploaded a portion of the simulated data in the dat folder.


2. ACIC
ACIC is a collection of semi-synthetic datasets derived from the linked birth and infant death data (LBIDD)
Here is the full dataset description `https://www.researchgate.net/publication/11523952_Infant_Mortality_Statistics_from_the_1999_Period_Linked_BirthInfant_Death_Data_Set`
Here is the github repo associated with the competition with some sample data `https://github.com/IBM-HRL-MLHLS/IBM-Causal-Inference-Benchmarking-Framework/blob/master/data/LBIDD/scaling_params.csv`
For access to the ACIC 2018 competition data: Please see here `https://www.synapse.org/#!Synapse:syn11294478/wiki/486304`

# Reproducing the IHDP experiments
The default setting would let you run dragonnet, tarnet, and nednet under targeted regularization and default mode

You'll run the from `src` code as 
`./experiment/run_ihdp.sh`
Before doing this, you'll need to edit `run_ihdp.sh` to change 
`data_base_dir= where you stored the data`
to
`output_base_dir=wherer you want the result to be`.

Change under the option if you only want to run one of the frameworks

# Reproducing the ACIC experiment
Same as above except You'll run the from `src` code as `./experiment/run_acic.sh`
