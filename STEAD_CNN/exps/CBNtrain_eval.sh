#!/bin/bash

# ClassConv_BN Classifier training

mkdir -p ../logs/CBN_train
mkdir -p ../logs/CBN_eval
mkdir -p ../models

trn="Train_data.hdf5"
tst="Test_data.hdf5"

# Training model

echo "Training CBN model on $trn and $tst datasets"
python ../train.py --train_path $trn --test_path $tst --classifier CBN --model_name CBN_10epch --n_epochs 10 > ../logs/CBN_train/CBN_10epch.txt

# Evaluating model
echo "Starting evaluation on $trn and $tst datasets"
python ../eval.py --train_path $trn --test_path $tst --classifier CBN --model_name CBN_10epch > ../logs/CBN_eval/CBN_10epch.txt

echo "Finished"
