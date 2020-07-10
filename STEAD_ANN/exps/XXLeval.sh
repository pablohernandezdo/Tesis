#!/bin/bash

mkdir -p ../logs/XXL_eval
mkdir -p ../models

trn="Train_data.hdf5"
tst="Test_data.hdf5"


# Classifier_XXL
echo "Starting evaluation #1"
python ../eval.py --train_path $trn --test_path $tst --classifier XXL --model_name XXL_lr000001_bs32 > ../logs/XXL_eval/XXL_lr000001_bs32.txt

echo "Finished"