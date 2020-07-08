#!/bin/bash

# ANN Classifier comparison

mkdir -p ../logs/XXLeval
mkdir -p ../models

trnpath="../Train_data.hdf5"
tstpath="../Test_data.hdf5"

# Classifier_XXL
echo "Starting evaluation #1"
python ../eval.py --train_path $trnpath --test_path $tstpath --classifier XXL --model_name XXL_lr000001_bs32 > ../logs/XXLeval/XXL_lr000001_bs32.txt

echo "Finished"