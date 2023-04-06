#!/bin/bash
#SBATCH --tasks-per-node=1   # Tasks per node
#SBATCH --nodes=1                # Number of nodes requested
#SBATCH --time=30:00:00         # walltime
#SBATCH --job-name=one-D-domain

method=$1
stochastic=$2
tag=$3
python one_D_domain.py --method $method --stochastic $stochastic  --tag ${tag}