#!/bin/bash

#PBS -l select=1:ncpus=40:mem=40gb
#PBS -q short_cpuQ
#PBS -l walltime=05:55:00
#PBS -o output
#PBS -e error

module load singularity-3.4.0

singularity exec --bind ~/$folder/:/data --pwd /data ~/cluster/singularity/container.sif python3 /data/core/GA_RA_PST.py $population_size $number_traces $id $ftol