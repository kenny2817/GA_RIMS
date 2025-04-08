#!/bin/bash

LOG_FILE="log.txt"

# Number of jobs to submit
NUMBER_JOBS=${1:-10}

if ((NUMBER_JOBS * 3 > 30)); then
    echo "too many man"
    exit 0
fi

# Folder
FOLDER=${2:-$HOME/GA_RIMS_0}

# Parameters
POPULATION_SIZE=50
FTOL=0.0025

NUMBER_TRACES=1000
JOB_ID=$(qsub -v population_size=$POPULATION_SIZE,number_traces=$NUMBER_TRACES,id=0,folder=$FOLDER,ftol=$FTOL cluster/GA_RIMS.sh | awk '{print $1}')
echo "Submitted job 0 with ID: $JOB_ID [$POPULATION_SIZE, $NUMBER_TRACES]" | tee -a $LOG_FILE
for ((i=1; i<NUMBER_JOBS; i++)); do
    JOB_ID=$(qsub -v population_size=$POPULATION_SIZE,number_traces=$NUMBER_TRACES,id=$i,folder=$FOLDER,ftol=$FTOL -W depend=afterany:$JOB_ID cluster/GA_RIMS.sh | awk '{print $1}')
    echo "Submitted job $i with ID: $JOB_ID [$POPULATION_SIZE, $NUMBER_TRACES]" | tee -a $LOG_FILE
done

NUMBER_TRACES=5000
for ((i=0; i<NUMBER_JOBS; i++)); do
    JOB_ID=$(qsub -v population_size=$POPULATION_SIZE,number_traces=$NUMBER_TRACES,id=$i,folder=$FOLDER,ftol=$FTOL -W depend=afterany:$JOB_ID cluster/GA_RIMS.sh | awk '{print $1}')
    echo "Submitted job $i with ID: $JOB_ID [$POPULATION_SIZE, $NUMBER_TRACES]" | tee -a $LOG_FILE
done

NUMBER_TRACES=10000
for ((i=0; i<NUMBER_JOBS; i++)); do
    JOB_ID=$(qsub -v population_size=$POPULATION_SIZE,number_traces=$NUMBER_TRACES,id=$i,folder=$FOLDER,ftol=$FTOL -W depend=afterany:$JOB_ID cluster/GA_RIMS.sh | awk '{print $1}')
    echo "Submitted job $i with ID: $JOB_ID [$POPULATION_SIZE, $NUMBER_TRACES]" | tee -a $LOG_FILE
done

# qstat -u quentin.meneghini
