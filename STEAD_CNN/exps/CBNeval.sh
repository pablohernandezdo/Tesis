#!/bin/bash

# ANN Classifier comparison

mkdir -p ../logs/CBNeval
mkdir -p ../models

trnpath="../Train_data.hdf5"
tstpath="../Test_data.hdf5"

# Classifier_XXL
echo "Starting evaluation #1"
python ../eval.py --train_path $trnpath --test_path $tstpath --classifier CBN --model_name CBN_10epch > ../logs/CBNeval/CBN_10epch.txt

echo "Finished"