#!/bin/bash

#PBS -l select=1:ncpus=20:mem=20gb
#PBS -q short_cpuQ
#PBS -l walltime=02:00:00
#PBS -o output

module load singularity-3.4.0

singularity exec --bind ~/$folder/:/data --pwd /data ~/$folder/singularity/test_0.sif python3 /data/GA_RA_PST.py $population_size $number_traces $id $ftol