#!/bin/bash

# CNN Classifier evaluation

mkdir -p ../logs/CBN_eval
mkdir -p ../models

trn="Train_data.hdf5"
tst="Test_data.hdf5"


# Classifier CBN
echo "Starting evaluation #1"
python ../eval.py --train_path $trn --test_path $tst --classifier CBN --model_name CBN_10epch > ../logs/CBN_eval/CBN_10epch.txt

echo "Finished"