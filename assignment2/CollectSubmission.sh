#!/bin/bash

if [ -z "$1" ]; then
    echo "team name is required.
Usage: ./CollectSubmission team_*"
    exit 0
fi

files="./Utils/model_checkpoints/*.meta
./Utils/layer_utils.py
Assignment2-1_Implementing_CNN.ipynb
Assignment2-2_Training_CNN.ipynb
Assignment2-3_Visualizing_CNN.ipynb"

for file in $files
do
    if [ ! -f $file ]; then
        echo "Required $file not found."
        exit 0
    fi
done

rm -f $1.tar.gz
mkdir $1
cp -r ./Utils/model_checkpoints ./Utils/layer_utils.py ./*.ipynb $1/
tar cvzf $1.tar.gz $1
