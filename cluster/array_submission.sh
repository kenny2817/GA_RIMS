#!/bin/bash

LOG_FILE="log.txt"

# Number of jobs to submit
NUMBER_JOBS=${1:-10}

# Folder
FOLDER=${2:-$HOME/GA_RIMS_0}

# Parameters
POPULATION_SIZE=${3:-10}
NUMBER_TRACES=${4:-10}
FTOL=${5:-0.0025}

# Initial job submission (first job runs without dependency)
JOB_ID=$(qsub -v population_size=$POPULATION_SIZE,number_traces=$NUMBER_TRACES,id=0,folder=$FOLDER,ftol=$FTOL cluster/GA_RIMS.sh | awk '{print $1}')
echo "Submitted job 0 with ID: $JOB_ID [$POPULATION_SIZE, $NUMBER_TRACES]" | tee -a $LOG_FILE

# Submit the rest of the jobs sequentially
for ((i=1; i<NUMBER_JOBS; i++)); do
    JOB_ID=$(qsub -v population_size=$POPULATION_SIZE,number_traces=$NUMBER_TRACES,id=$i,folder=$FOLDER,ftol=$FTOL -W depend=afterany:$JOB_ID cluster/GA_RIMS.sh | awk '{print $1}')
    echo "Submitted job $i with ID: $JOB_ID [$POPULATION_SIZE, $NUMBER_TRACES]" | tee -a $LOG_FILE
done

qstat -u quentin.meneghini
