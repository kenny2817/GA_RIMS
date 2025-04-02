#!/bin/bash

FOLDER=${1:-"GA_RIMS"}
FILE_NAME=${2:-"diagram_0"}

git clone --branch cluster --single-branch https://github.com/kenny2817/GA_RIMS.git ./$FOLDER
rm -r cluster/
mkdir output
mkdir output/output_$FILE_NAME
cd ..

if [[ -n "$3" ]]; then
    cp -R $3 $FOLDER/$FILE_NAME
else
    echo "missing input file"
fi